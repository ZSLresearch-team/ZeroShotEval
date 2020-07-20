"""Main script-launcher for training of ZSL models."""


# region IMPORTS
import numpy as np
import torch

from zeroshoteval.utils.parser import load_config, parse_args

from single_experiment import experiment


def main():
    args = parse_args()
    cfg = load_config(args)
    cfg.freeze()

    # Set random generator seed if need
    if cfg.RNG_SEED > 0:
        np.random.seed(cfg.RNG_SEED)
        torch.manual_seed(cfg.RNG_SEED)

        # Set CuDNN to deterministic mode
        if cfg.CUDNN_DETERMINISTIC:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    experiment(cfg)


if __name__ == "__main__":
    main()
