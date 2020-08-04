import numpy as np
import torch


def RNG_seed_setup(seed=None):
    """
    Set up random number generators seed
    """
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Set CuDNN to deterministic mode
        if cfg.CUDNN_DETERMINISTIC:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
