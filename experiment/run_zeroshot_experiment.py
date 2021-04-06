"""Main script-launcher for training of ZSL models."""

import os
from argparse import Namespace
from typing import Callable, Tuple

from fvcore.common.config import CfgNode
from torch.nn import Module

from zeroshoteval.evaluation.classification import classification_procedure
from zeroshoteval.utils.defaults import default_setup
from zeroshoteval.utils.parser import load_config, parse_args
from zeroshoteval.zeroshotnets.build import (
    build_zeroshot_train,
    build_zeroshot_inference
)
from zeroshoteval.zeroshotnets.cada_vae.cada_vae_inference import CADA_VAE_inference_procedure

# os.chdir("../")


def experiment(cfg: CfgNode) -> None:
    """
    Start single experiment with the specified configs. Note that this procedure
    requires data embeddings, extracted previously. If you don't have the
    extracted embeddings, use the `run_embedding_extractor.py` script.

    Args:
        cfg(CfgNode): Configs. Details can be found in
                      zeroshoteval/config/defaults.py
    """

    # MODEL TRAINING
    # ==============
    # Pass extracted embeddings to a zero-shot neural network to train it as an
    # embedding extractor.
    # --------------------------------------------------------------------------

    # Embeddings loading from disk
    # TODO: Replace ModalitiesEmbeddingDataset with loader from separate files

    # Get training procedure function from registry
    train_procedure: Callable[[CfgNode], Module] = build_zeroshot_train(cfg)

    # Training
    zsl_model: Module = train_procedure(cfg)

    # MODEL INFERENCE
    # ===============
    # Apply trained model to test data and (similar to the prev step) extract
    # zero-shot embeddings.
    # --------------------------------------------------------------------------

    # TODO: replace CADA-VAE with general model

    # Get inference procedure function from registry
    inference_procedure: Callable[[CfgNode, Module], Tuple[Tuple, Tuple]] = build_zeroshot_inference(cfg)

    # Inference
    train_data, test_data = inference_procedure(cfg, zsl_model)

    # MODEL EVALUATION
    # ================
    # Pass extracted zero-shot embeddings to one of evaluation tasks (
    # classification, clustering, verification, etc.)
    # --------------------------------------------------------------------------

    # TODO: replace classification procedure with more general evaluation calling
    classification_procedure(cfg, train_data, test_data)


def main() -> None:
    # Parse arguments
    experiment_options: Namespace = parse_args()

    # Load default configuration and merge with specific configs from args
    cfg: CfgNode = load_config(experiment_options)

    # Freeze current config to avoid arbitrary changes
    cfg.freeze()

    # Perform some basic common setups at the beginning of a job
    default_setup(cfg, experiment_options)

    # Start the experiment
    experiment(cfg)


if __name__ == "__main__":
    main()
