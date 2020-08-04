"""Main script-launcher for training of ZSL models."""


# region IMPORTS
import numpy as np
import torch

from zeroshoteval.utils.defaults import default_setup
from zeroshoteval.utils.parser import load_config, parse_args

from single_experiment import experiment


def setup(args):
    cfg = load_config(args)
    cfg.freeze()
    default_setup(cfg, args)

    return cfg


def main():
    args = parse_args()
    cfg = setup(args)

    experiment(cfg)


if __name__ == "__main__":
    main()
