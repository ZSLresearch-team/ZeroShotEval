"""
"""
from ray import tune

# from training import load_embeddings, filter_modalities
from src.zeroshot_networks.cada_vae.cada_vae_train import VAE_train_procedure
from src.dataloader.dataset import ObjEmbeddingDataset
from src.evaluation_procedures.classification import classification_procedure

import os
import copy
import pickle
import pandas as pd 
from easydict import EasyDict as edict


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

def experiment(model_config, reporter):
    """
    
    """
    num_classes=200


    
    model_config = edict(model_config)
    # model_config.nepoch = int(model_config.nepoch)
    # model_config.specific_parameters.cls_train_epochs = int(model_config.cls_train_epochs)

    # model_config.specific_parameters.warmup.beta.start_epoch = int(model_config.beta_start_epoch)
    # model_config.specific_parameters.warmup.beta.end_epoch = int(model_config.beta_end_epoch)
    # model_config.specific_parameters.warmup.distance.start_epoch = int(model_config.distance_start_epoch)
    # model_config.specific_parameters.warmup.distance.end_epoch = int(model_config.distance_end_epoch)
    # model_config.specific_parameters.warmup.cross_reconstruction.start_epoch = int(model_config.cross_reconstruction_start_epoch)
    # model_config.specific_parameters.warmup.cross_reconstruction.end_epoch = int(model_config.cross_reconstruction_end_epoch)

    # model_config.specific_parameters.warmup.beta.factor = model_config.beta_factor
    # model_config.specific_parameters.warmup.distance.factor = model_config.distance_factor
    # model_config.specific_parameters.warmup.cross_reconstruction.factor = model_config.cross_reconstruction_factor

    model_config.specific_parameters.lr_cls = model_config.lr_cls
    # model_config.specific_parameters.lr_gen_model = model_config.lr_gen_model

    # model_config.specific_parameters.samples_per_modality_class.cls_attr = int(model_config.seen_unseen_factor *
    #                                                                            model_config.specific_parameters.samples_per_modality_class.img)
    
    embeddings, datasets_labels, datasets_splits = load_embeddings(load_dir='/home/mlitvinov/zsl/ZSLConstructor/data/CUB')

    embeddings = filter_modalities(embeddings, ['img', 'cls_attr'])

    data = ObjEmbeddingDataset(embeddings['cub'], datasets_labels['cub'], datasets_splits['cub'], ['img'])

    zsl_emb_dataset, csl_train_indice, csl_test_indice = VAE_train_procedure(model_config, data)


    # # Train classifier
    _train_loss_hist, _acc_seen_hist, _acc_unseen_hist, acc_H_hist = \
    classification_procedure(data=zsl_emb_dataset,
                             in_features=model_config.specific_parameters.latent_size,
                             num_classes=num_classes,
                             batch_size=model_config.batch_size,
                             device=model_config.device,
                             n_epoch=model_config.specific_parameters.cls_train_epochs,
                             lr=model_config.specific_parameters.lr_cls,
                             train_indicies=csl_train_indice,
                             test_indicies=csl_test_indice,
                             seen_classes=data.seen_classes,
                             unseen_classes=data.unseen_classes,
                             verbose=model_config.verbose
                             )


    for i in range(len(acc_H_hist)):
        # acc_seen = _acc_seen_hist[i]
        # acc_unseen = _acc_unseen_hist[i] 
        reporter(acc_H=acc_H_hist[i], acc_seen=_acc_seen_hist[i], acc_unseen=_acc_unseen_hist[i])

    return None
