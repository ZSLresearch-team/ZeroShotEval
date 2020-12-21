"""Main script-launcher for training of ZSL models."""

import os
from argparse import Namespace

from fvcore.common.config import CfgNode
from torch.nn.modules.module import Module

from zeroshoteval.evaluation.classification import classification_procedure
from zeroshoteval.utils.defaults import default_setup
from zeroshoteval.utils.parser import load_config, parse_args
from zeroshoteval.zeroshotnets.build import build_zsl
from zeroshoteval.zeroshotnets.cada_vae.cada_vae_inference import CADA_VAE_inference_procedure

# os.chdir("../")


def setup(args: Namespace) -> CfgNode:
    cfg: CfgNode = load_config(args)
    cfg.freeze()
    default_setup(cfg, args)

    return cfg


def experiment(cfg: CfgNode) -> None:
    """
    Start single experiment with the specified configs.

    Args:
        cfg(CfgNode): Configs. Details can be found in
                      zeroshoteval/config/defaults.py
    """

    # STEP 0 - DATA PREPARATION
    # Prepare dataset CSV with file names to use for raw data loading.
    # --------------------------------------------------------------------------

    # STEP 1 - EMBEDDINGS EXTRACTION
    # Extract embeddings from raw data with specified NN and save it to
    # the disk. Note that embedding extractor can load data as a whole or
    # using iterators (to fit into RAM). This step also creates a CSV with
    # paths to all embedding files.
    # --------------------------------------------------------------------------

    pass

    # STEP 2 - MODEL TRAINING
    # Pass extracted embeddings to a zero-shot neural network to train it as an
    # embedding extractor.
    # --------------------------------------------------------------------------

    # Embeddings loading from disk
    # TODO: Replace ModalitiesEmbeddingDataset with loader from separate files

    # Training
    train_procedure = build_zsl(cfg)
    model: Module = train_procedure(cfg)

    # STEP 3 - MODEL INFERENCE
    # Apply trained model to test data and (similar to the prev step) extract
    # zero-shot embeddings.
    # --------------------------------------------------------------------------

    # TODO: extract gen_synthetic_data from CADA-VAE and move it to inference section
    # TODO: replace CADA-VAE with general model
    zsl_data = CADA_VAE_inference_procedure(cfg, model)

    # STEP 4 - MODEL EVALUATION
    # Pass extracted zero-shot embeddings to one of evaluation tasks (
    # classification, clustering, verification, etc.)
    # --------------------------------------------------------------------------

    classification_procedure(cfg=cfg, data=zsl_data)


if __name__ == "__main__":
    args: Namespace = parse_args()
    cfg: CfgNode = setup(args)

    experiment(cfg)
