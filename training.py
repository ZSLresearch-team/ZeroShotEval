"""Main script-launcher for training of ZSL models.
"""

# TODO: replace all template variables with actual values
# after completing all modules


#region IMPORTS
import argparse
import sys

from config import config, default, generate_config

from src.dataset_loaders.data_loader import load_dataset
from src.modalities_feature_extractors.modalities_feature_extractor import ModalitiesFeatureExtractor
from src.zeroshot_networks.cada_vae.cada_vae_model import Model as CadaVaeModel

# TODO: from gan_net import gan_model
#endregion


#region ARGUMENTS
def init_arguments():
    """Initialize arguments replacing the default ones.
    """
    parser = argparse.ArgumentParser(description='Main script-launcher for training of ZSL models')
    
    # general configs
    parser.add_argument('--model', default=config.model,
                        help='Name of model to use for ZSL training.')
    parser.add_argument('--datasets', default=config.datasets,
                        help='Name of datasets to use for ZSL training.')
    args, rest = parser.parse_known_args()

    datasets = args.datasets.split(',')
    generate_config(parsed_model=args.model, parsed_datasets=datasets)

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
    check_arguments(args)
    return args
#endregion


def save_embeddings(save_dir):
    pass


def load_embeddings(load_dir):
    pass


def main():
    args = load_arguments()

    #region DATA LOADING
    datasets = []
    for dataset_name in config.datasets:
        # dataset_dict contains modalities names as keys and data as values
        dataset_dict = load_dataset(dataset_name, 
                                    modalities=config.modalities, 
                                    path=config[dataset_name].path,
                                    load_embeddings=config.load_dataset_precomputed_embeddings)
        datasets.append(dataset_dict)
    #endregion


    #region DATA EMBEDDINGS EXTRACTION
    embeddings = []
    if not config.load_dataset_precomputed_embeddings:
        for dataset in datasets:
            extractor = ModalitiesFeatureExtractor(modalities_dict=dataset)
            dataset_embeddings = extractor.compute_embeddings()
            embeddings.append(dataset_embeddings)
    #endregion


    #region OBJ EMBEDDINGS READING/CACHING
    if config.load_cached_obj_embeddings:
        pass  # TODO: load cached objects embeddings
    
    if config.cache_obj_embeddings:
        pass  # TODO: cache computed embeddings on the disk
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
    