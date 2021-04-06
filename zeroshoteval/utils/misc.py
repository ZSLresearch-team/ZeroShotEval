import numpy as np
import psutil
import torch
from fvcore.nn.activation_count import activation_count
from fvcore.nn.flop_count import flop_count

import logging
import os

logger = logging.getLogger(__name__)


def setup_random_number_generator_seed(cfg):
    """
    Set up random number generators seed

    Args:
        cfg(CfgNode): configs. Details can be found in
            zeroshoteval/config/defaults.py
    """
    seed = None if (cfg.RNG_SEED < 0) else cfg.RNG_SEED
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Set CuDNN to deterministic mode
        if cfg.CUDNN_DETERMINISTIC:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def params_count(model):
    """
    Compute the number of parameters.

    Args:
        model (model): model to count the number of parameters.
    """
    return np.sum([p.numel() for p in model.parameters()]).item()


def gpu_mem_usage():
    """
    Compute the GPU memory usage for the current device (GB).
    """
    if torch.cuda.is_available():
        mem_usage_bytes = torch.cuda.max_memory_allocated()
    else:
        mem_usage_bytes = 0
    return mem_usage_bytes / 1024 ** 3


def RAM_usage():
    """
    Compute the system memory (RAM) usage for the current device (GB).

    Returns:
        usage (float): used memory (GB).
        total (float): total memory (GB).
    """
    vram = psutil.virtual_memory()
    usage = (vram.total - vram.available) / 1024 ** 3
    total = vram.total / 1024 ** 3

    return usage, total


"""
Below are some model logging helpers.
They are mainly adopted from
https://github.com/facebookresearch/SlowFast/tree/master/slowfast/utils/misc.py
"""


def get_model_stats(model, mode):
    """
    Temoraly incomplite due to large variety of model inputs.
    Compute statistics for the current model given the config.

    Args:
        model (model): model to perform analysis.
        mode (str): Options include `flop` or `activation`. Compute either flop
            (gflops) or activation count (mega).

    Returns:
        float: the total number of count of the given model.
    """
    assert mode in [
        "flop",
        "activation",
    ], "'{}' not supported for model analysis".format(mode)
    if mode == "flop":
        model_stats_fun = flop_count
    elif mode == "activation":
        model_stats_fun = activation_count

    pass


def log_model_info(model, model_name):
    """
    Log info, includes number of parameters, gpu usage.
        The model info is computed when the model is in validation mode.

    Args:
        model (model): model to log the info.
        model_name (str): Name of model to log info.
    """
    logger.info(f"Model {model_name}:\n{model}")
    logger.info(f"Params: {params_count(model):,}")
    logger.info(f"Mem: {gpu_mem_usage():.2} GB")

    # Logging model stats temporaly unavailable

    logger.info("nvidia-smi")
    os.system("nvidia-smi")
