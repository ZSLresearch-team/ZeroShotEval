"""Main script-launcher for training of ZSL models.
"""

# TODO: replace all template variables with actual values
# after completing all modules


#region IMPORTS
import argparse
import sys

from config import config, default, generate_config

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

    # datasets = args.datasets.split(',')
    generate_config(parsed_model=args.model, parsed_datasets=args.datasets)

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


def main():
    args = load_arguments()

    #region DATA LOADING
    if config.load_raw_modalities:
        pass  # TODO: load raw images and text attributes from dataset
    if config.load_dataset_precomputed_embeddings:
        pass  # TODO: load precomputed embeddings from dataset
    #endregion


    #region DATA EMBEDDINGS EXTRACTION
    if config.load_raw_modalities:
        pass  # TODO: pass loaded images and text attributes from the DATA LOADING region 
              # for obj embedding extraction
    #endregion


    #region OBJ EMBEDDINGS READING/CACHING
    if config.load_cached_obj_embeddings:
        pass  # TODO: load cached objects embeddings
    
    if config.cache_obj_embeddings:
        pass  # TODO: cache computed embeddings on the disk
    #endregion

    # hyperparameters = {}  # TODO: load hyperparameters from config file

    #region ZERO-SHOT MODELS TRAINING
    if args.model == 'cada_vae':
        cada_vae_model = CadaVaeModel(config)
        cada_vae_model.to(config.device)
        cada_vae_model.train_vae()
        pass  # TODO: initialize the model with configs
    elif args.model == 'clswgan':
        pass  # TODO: initialize the model with configs
    else:
        raise NotImplementedError('Unknown network')

    # model.train(*args)  # TODO: complete model training call

    #endregion


    #region ZERO-SHOT MODEL INFERENCE
    # model.predict(*args)  # TODO: implement computing and 
                            # caching of ZSL embeddings with trained model
    #endregion


if __name__ == '__main__':
    main()
    