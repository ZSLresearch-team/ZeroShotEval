"""Main script-launcher for training of ZSL models."""


# region IMPORTS
import argparse
import os
import sys
import copy
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

import torch
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler
from torch.utils.data import DataLoader, TensorDataset

from config import default
from config import generate_config

from src.dataset_loaders.data_loader import load_dataset
from src.modalities_feature_extractors.modalities_feature_extractor import compute_embeddings
from src.zeroshot_networks.cada_vae.cada_vae_model import VAEModel 
from src.zeroshot_networks.cada_vae.cada_vae_train import train_VAE, test_VAE
from src.dataloader.dataset import ObjEmbeddingDataset
from src.evaluation_procedures.classification import classification_procedure

# TODO: from gan_net import gan_model
# endregion


# region ARGUMENTS PROCESSING
def init_arguments():
    """Initialize arguments replacing the default ones."""
    parser = argparse.ArgumentParser(
        description='Main script-launcher for training of ZSL models')

    general_args = parser.add_argument_group(title='General configs')
    general_args.add_argument('--model', default=default.model,
                              help='Name of model to use for ZSL training.')
    general_args.add_argument('--datasets', default=default.datasets,
                              help='Comma-separated list of names of datasets to use for ZSL training.')
    general_args.add_argument('--modalities', default=default.modalities,
                              help='Comma-separated list of modalities (e.g. "img,cls_attr") to use for ZSL training.')
    general_args.add_argument('--generalized-zsl', action='store_true', default=default.generalized_zsl,
                              help='Whether to perform Generalized ZSL training.')

    nets_args = parser.add_argument_group(title='Networks configs')
    nets_args.add_argument('--img-net', default=default.img_net,
                           help='Name of the network to use for images embeddings extraction.')
    nets_args.add_argument('--cls-attr-net', default=default.cls_attr_net,
                           help='Name of the network to use for class attributes embeddings extraction.')

    saveload_args = parser.add_argument_group(
        title='Configs for saving/loading')
    saveload_args.add_argument('--saved-obj-embeddings-path', default=default.saved_obj_embeddings_path,
                               help='Path to stored object embeddings to load')
    saveload_args.add_argument('--obj-embeddings-save-path', default=default.obj_embeddings_save_path,
                               help='Path to save computed embeddings.')

    # place here other arguments, that are not in config.py if necessary

    return parser


def check_arguments(args):
    """Check arguments compatibility and correctness."""
    pass


def load_arguments():
    """Initialize, check and pass arguments."""
    parser = init_arguments()
    args = parser.parse_args()
    args = split_commasep_arguments(args)
    check_arguments(args)

    # TODO: process all paths in a loop
    args.saved_obj_embeddings_path = Path(args.saved_obj_embeddings_path)
    args.obj_embeddings_save_path = Path(args.obj_embeddings_save_path)

    return args


def split_commasep_arguments(args):
    args_dict = vars(args)
    for key, value in args_dict.items():
        try:
            value = value.split(',')
        except:
            continue
        if len(value) == 1:
            value = value[0]
        args_dict[key] = value
    return args


def load_configurations(args):
    return generate_config(parsed_model=args.model,
                           parsed_datasets=args.datasets)
# endregion


# region SAVING/LOADING COMPUTED DATA
def save_embeddings(embeddings, labels, splits, save_dir):
    print('\n[info] Saving all data to "{}"'.format(save_path))
    for dataset_name in embeddings:
        embs_path = save_dir / (dataset_name + '_obj_embeddings.pickle')
        lab_path = save_dir / (dataset_name + '_obj_labels.pickle')
        splits_path = save_dir / (dataset_name + '_splits.csv')

        print('Saving embeddings to "{}"'.format(embs_path))
        with open(embs_path, 'wb') as f:
            pickle.dump(embeddings[dataset_name], f)

        print('Saving labels to "{}"'.format(lab_path))
        with open(lab_path, 'wb') as f:
            pickle.dump(labels[dataset_name], f)

        if splits[dataset_name]:  # if splits are defined for the dataset
            print('Save data splits to "{}"'.format(splits_path))
            splits[dataset_name].to_csv(splits_path, index=False)

    print('Done!')


def load_embeddings(load_dir):
    print('[INFO] Loading computed embeddings and data splits from "{}"...'.format(
        load_dir), end=' ')
    embeddings_dict = {}
    dataset_labels = {}
    dataset_splits = {}
    for filename in os.listdir(load_dir):
        path = os.path.join(load_dir, filename)
        if filename.endswith('_obj_embeddings.pickle'):
            dataset_name = filename.rsplit('_', 2)[0]
            with open(path, 'rb') as f:
                embeddings = pickle.load(f)
            embeddings_dict[dataset_name] = embeddings

        elif filename.endswith('_obj_labels.pickle'):
            dataset_name = filename.rsplit('_', 2)[0]
            with open(path, 'rb') as f:
                labels = pickle.load(f)
            dataset_labels[dataset_name] = labels

        elif filename.endswith('_splits.csv'):
            dataset_name = filename.rsplit('_', 1)[0]
            splits_df = pd.read_csv(path)
            dataset_splits[dataset_name] = splits_df

    if not dataset_splits:  # if there are no predefined splits for this dataset
        dataset_splits = None
    print('Done!')
    return embeddings_dict, dataset_labels, dataset_splits
# endregion


# region DATA PROCESSING
def split_dataset(modalities_dict, splits_df):
    train_df = splits_df.loc[splits_df['is_train'] == 1].astype(int)
    test_unseen_df = splits_df.loc[(splits_df['is_train'] == 0) & (
        splits_df['is_seen'] == 0)].astype(int)
    test_seen_df = splits_df.loc[(splits_df['is_train'] == 0) & (
        splits_df['is_seen'] == 1)].astype(int)

    train_data = {}
    test_unseen_data = {}
    test_seen_data = {}
    for dataset_name, data_array in modalities_dict.items():
        # NOTE: here is an assumption that in data_array ids are arranged in ascending order
        # in other case it can be a weak point for bugs
        train_data[dataset_name] = data_array[train_df['obj_id'].values]
        test_unseen_data[dataset_name] = data_array[test_unseen_df['obj_id'].values]
        test_seen_data[dataset_name] = data_array[test_seen_df['obj_id'].values]
    if test_seen_df.empty:
        test_seen_data = None
    return train_data, test_unseen_data, test_seen_data

def get_indice_split(splits_df):
    """
    Get indices for data split using pandas dataframe with split info.

    Args:
        splits(pandas DataFrame): dataframe with object Id, and ``is_train`` is_seen`` binary columns.

    Returns:
        train_indice, test_seen_indice, test_unseen_indice: train, test seen and inseen indicies.
    """
    train_indice = splits_df[splits_df['is_train'] == 1]['obj_id'].to_numpy()
    test_seen_indice = splits_df[(splits_df['is_train'] == 0) & (splits_df['is_seen'] == 1)]['obj_id'].to_numpy()
    test_unseen_indice = splits_df[splits_df['is_seen'] == 0]['obj_id'].to_numpy()

    return train_indice, test_seen_indice, test_unseen_indice

def check_loaded_modalities(data_dict, modalities):
    for dataset_name, data in data_dict.items():
        keys = set(data.keys())
        modalities = set(modalities)
        if not modalities.issubset(keys):
            raise ValueError('You have specified modalities that are not presented in dataset!\
                                \nAvaliable modalities for {} dataset: {}'.format(dataset_name, keys))


def filter_modalities(data_dict, modalities):
    check_loaded_modalities(data_dict, modalities)
    filtered_dict = copy.deepcopy(data_dict)
    for dataset_name, data in data_dict.items():
        for key in data:
            if key not in modalities:
                filtered_dict[dataset_name].pop(key)
    return filtered_dict


def extend_cls_attributes(data_dict, labels):
    for dataset_name, data in data_dict.items():
        counts_list = []
        labels_set = []
        for label in labels[dataset_name].tolist():
            if label not in labels_set:
                counts_list.append(labels[dataset_name].tolist().count(label))
                labels_set.append(label)

        cls_attr_list = []
        for attr, count in zip(data['cls_attr'].tolist(), counts_list):
            cls_attr_list.extend([attr for _ in range(count)])
        data_dict[dataset_name]['cls_attr'] = np.array(cls_attr_list)
    return data_dict


def get_data_dimensions(modalities_dict):
    dims = {}
    for modality_name, data in modalities_dict.items():
        if isinstance(data, np.ndarray):
            dims[modality_name] = data.shape[1]
        elif isinstance(data, list):
            dims[modality_name] = len(data)
        else:
            raise NotImplementedError(
                'Please write a way to get dimension for {}'.format(type(data)))

    return dims
# endregion


def main():
    args = load_arguments()
    model_config, datasets_config = load_configurations(args)

    print('\n========== DATA LOADING ==========\n')
    if not args.saved_obj_embeddings_path:

        # region DATA LOADING
        datasets = {}
        datasets_labels = {}
        datasets_splits = {}
        for dataset_name, config in datasets_config.items():
            # modalities_dict contains modalities names as keys and data as values
            modalities_dict, labels, splits_df = load_dataset(dataset_name,
                                                              modalities=args.modalities,
                                                              path=config.path)
            datasets[dataset_name] = modalities_dict
            datasets_labels[dataset_name] = labels
            datasets_splits[dataset_name] = splits_df
        # endregion

        # region DATA OBJ EMBEDDINGS EXTRACTION
        embeddings = {}
        for dataset_name, data in datasets.items():
            dataset_embeddings = compute_embeddings(modalities_dict=data)
            embeddings[dataset_name] = dataset_embeddings
        # endregion

        # region OBJ EMBEDDINGS SAVING
        if args.obj_embeddings_save_path:
            save_embeddings(embeddings=embeddings,
                            labels=datasets_labels,
                            splits=datasets_splits,
                            save_dir=args.obj_embeddings_save_path)
        # endregion

    # region OBJ EMBEDDINGS READING/SAVING
    if args.saved_obj_embeddings_path:
        embeddings, datasets_labels, datasets_splits = load_embeddings(
            load_dir=args.saved_obj_embeddings_path)
    # endregion

    # region MODALITIES PREPARATION
    embeddings = filter_modalities(embeddings, args.modalities)
    embeddings = extend_cls_attributes(embeddings, datasets_labels)
    # endregion

    # region ZERO-SHOT MODELS TRAINING / INFERENCE
    for dataset_name, dataset_embeddings in embeddings.items():
        # TODO: add loop for regenerating splits and retraining ZSL net
        splits_df = datasets_splits[dataset_name]
        labels = datasets_labels[dataset_name]
        if splits_df is None:
            # TODO: !!! Do not forget to handle GZSL parameter and make additional data split
            splits_df = generate_splits()

        train_embeddings, test_unseen_embeddings, test_seen_embeddings = split_dataset(
             modalities_dict=dataset_embeddings, splits_df=splits_df)
        train_indice, test_seen_indice, test_unseen_indice = get_indice_split(splits_df)

        modalities_dimensions = get_data_dimensions(train_embeddings)
        test_modality = 'img'
        # NOTE: now all loaded modalities are passed to the model, but
        # it shouldn't be a restriction! There can be a situation, where we want to
        # compare two different models, that are trained on different modalities
        if args.model == 'cada_vae':
            feature_dimensions = [modalities_dimensions[key] for key in args.modalities]
            data = ObjEmbeddingDataset(dataset_embeddings, labels)

            model = VAEModel(hidden_size_encoder=model_config.specific_parameters.hidden_size_encoder,
                             hidden_size_decoder=model_config.specific_parameters.hidden_size_decoder,
                             latent_size=model_config.specific_parameters.latent_size,
                             modalities=args.modalities,
                             feature_dimensions=feature_dimensions)

            model.to(model_config.device)
            # Model training
            optimizer = torch.optim.Adam(model.parameters(), lr=model_config.specific_parameters.lr_gen_model,
                                         betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=True)

            train_sampler = SubsetRandomSampler(train_indice)
            train_loader = DataLoader(data, batch_size=model_config.batch_size, sampler=train_sampler)
            loss_history = train_VAE(config=model_config,
                                     model=model,
                                     train_loader=train_loader,
                                     optimizer=optimizer,
                                     verbose=2)
            # Model evaluating
            sampler = SequentialSampler(data)
            loader = DataLoader(data, batch_size=model_config.batch_size, sampler=sampler)

            zsl_emb, _labels = test_VAE(model, loader, 'img')
            # Save zsl embeddings if needed
            if default.cache_zsl_embeddings:
                target_dir = default.obj_embeddings_save_path
                zsl_emb_file = os.path.join(target_dir, 'zsl_emb_cada_vae.pt')
                torch.save(zsl_emb, zsl_emb_file)

            # Train classifier
            labels_tensor = torch.Tensor(labels).long().to(model_config.device)
            # labels_tensor = labels
            zsl_emb_dataset = TensorDataset(zsl_emb, labels_tensor)

            train_loss_hist, acc_seen_hist, acc_unseen_hist, acc_H_hist = \
                classification_procedure(data=zsl_emb_dataset,
                                        in_features=model_config.specific_parameters.latent_size ,
                                        num_classes=datasets_config[dataset_name].num_classes,
                                        batch_size=model_config.batch_size,
                                        device=model_config.device,
                                        n_epoch=model_config.specific_parameters.cls_train_epochs,
                                        lr=model_config.specific_parameters.lr_cls,
                                        train_indicies=train_indice,
                                        test_seen_indicies=test_seen_indice,
                                        test_unseen_indicies=test_seen_indice)

        elif args.model == 'clswgan':
            pass  # TODO: initialize the model with configs

        else:
            raise NotImplementedError('Unknown network')

        # zsl_unseen_embeddings = model.transform(test_unseen_embeddings)
        # zsl_seen_embeddings = model.transform(test_seen_embeddings)

        # if args.compute_zsl_train_embeddings:
        #     zsl_train_embeddings = model.transform(train_embeddings)
    # endregion

    # region ZSL EMBEDDINGS CACHING
    # if default.cache_zsl_embeddings:
    #     train_embeddings_out = model.transform(train_embeddings).detach()
    #     test_unseen_out = model.transform(test_unseen_embeddings).detach()
    #     test_seen_out = model.transform(test_seen_embeddings).detach()
    #     target_dir = default.obj_embeddings_save_path
    #     file_train = os.path.join(target_dir, 'train.npy')
    #     ile_unseen = os.path.join(target_dir, 'test_unseen.npy')
    #     file_seen = os.path.join(target_dir, 'test_seen.npy')
    #     np.save(file_train, train_embeddings_out.cpu().numpy())
    #     np.save(file_unseen, test_unseen_out.cpu().numpy())
    #     np.save(file_seen, test_unseen_out.cpu().numpy())
    # endregion


if __name__ == '__main__':
    main()
