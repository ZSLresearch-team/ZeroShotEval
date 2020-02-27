"""Main script-launcher for training of ZSL models.
"""

# TODO: replace all template variables with actual values
# after completing all modules


#region IMPORTS
import argparse
import sys
import copy

from config import default, generate_config

from src.dataset_loaders.data_loader import load_dataset
from src.modalities_feature_extractors.modalities_feature_extractor import ModalitiesFeatureExtractor
from src.zeroshot_networks.cada_vae.cada_vae_model import Model as CadaVaeModel

# TODO: from gan_net import gan_model
#endregion


#region ARGUMENTS PROCESSING
def init_arguments():
    """Initialize arguments replacing the default ones.
    """
    parser = argparse.ArgumentParser(description='Main script-launcher for training of ZSL models')
    
    # general configs
    parser.add_argument('--model', default=default.model,
                        help='Name of model to use for ZSL training.')
    parser.add_argument('--datasets', default=default.datasets,
                        help='Comma-separated list of names of datasets to use for ZSL training.')
    parser.add_argument('--modalities', default=default.modalities,
                        help='Comma-separated list of modalities (e.g. "img,cls_attr") to use for ZSL training.')
    parser.add_argument('--img-net', default=default.img_net,
                        help='Name of the network to use for images embeddings extraction.')
    parser.add_argument('--cls_attr_net', default=default.cls_attr_net,
                        help='Name of the network to use for class attributes embeddings extraction.')
    parser.add_argument('--load-dataset-precomputed-embeddings', action='store_true', default=default.load_dataset_precomputed_embeddings,
                        help='Whether to load precomputed embeddings for datasets.')
    # TODO: complete parsing of arguments with default values from config.py

    # place here other arguments, that are not in config.py if necessary

    return parser


def check_arguments(args):
    """Check arguments compatibility and correctness.
    """
    pass


def load_arguments():
    """Initialize, check and pass arguments.
    """
    parser = init_arguments()
    args = parser.parse_args()
    args = split_commasep_arguments(args)
    check_arguments(args)
    return args


def split_commasep_arguments(args):
    args_dict = vars(args)
    for key, value in args_dict.items():
        try:
            value = value.split(',')
        except: continue
        if len(value) == 1:
            value = value[0]
        args_dict[key] = value 
    return args


def load_configurations(args):

    return generate_config(parsed_model=args.model, 
                           parsed_datasets=args.datasets)

#endregion


def save_embeddings(save_dir):
    pass


def load_embeddings(load_dir):
    pass


def main():
    args = load_arguments()
    model_config, datasets_config = load_configurations(args)

    #region DATA LOADING
    datasets = {}
    for dataset_name, config in datasets_config:
        # dataset_dict contains modalities names as keys and data as values
        dataset_dict = load_dataset(dataset_name, 
                                    modalities=args.modalities, 
                                    path=config.path,
                                    load_embeddings=args.load_dataset_precomputed_embeddings)
        datasets[dataset_name] = dataset_dict
    #endregion


    #region DATA EMBEDDINGS EXTRACTION
    if config.load_dataset_precomputed_embeddings:
        embeddings = copy.deepcopy(datasets)
    else:
        embeddings = []
        for dataset in datasets:
            extractor = ModalitiesFeatureExtractor(modalities_dict=dataset)
            dataset_embeddings = extractor.compute_embeddings()
            embeddings.append(dataset_embeddings)
    #endregion


    #region OBJ EMBEDDINGS READING/CACHING
    # if config.load_cached_obj_embeddings:
    #     pass  # TODO: load cached objects embeddings
    
    # if config.cache_obj_embeddings:
    #     pass  # TODO: cache computed embeddings on the disk
    #endregion


    #region ZERO-SHOT MODELS TRAINING / INFERENCE

    # NOTE: don't forget to handle multiple datasets!
    if args.model == 'cada_vae':
        model = CadaVaeModel(config)
        model.to(config.device)

        model.fit()

    elif args.model == 'clswgan':
        pass  # TODO: initialize the model with configs

    else:
        raise NotImplementedError('Unknown network')

    zsl_embeddings = model.transform()
    if config.compute_zsl_train_embeddings:
        zsl_train_embeddings = model.transform()
    #endregion


    #region ZSL EMBEDDINGS CACHING
    if config.cache_zsl_embeddings:
        pass  # TODO: cache computed embeddings on the disk
    #endregion


if __name__ == '__main__':
    main()
    