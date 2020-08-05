import torch
from fvcore.common.file_io import PathManager

from zeroshoteval.utils.collect_env import collect_env_info
from zeroshoteval.utils.logger import setup_logger

import argparse
import logging
import os
import sys
from collections import OrderedDict


def default_setup(cfg, args):
    """
    Perform some basic common setups at the beginning of a job, including:
    1. Set up the zeroshoteval logger
    2. Log basic information about environment, cmdline arguments, and config
    3. Save config to the file

    Args:
        cfg (CfgNode): the full config to be used
    """
    output_dir = cfg.OUTPUT_DIR
    PathManager.mkdirs(output_dir)

    setup_logger(output_dir, name="fvcore")
    logger = setup_logger(output_dir)

    logger.info("Environment info:\n" + collect_env_info())

    logger.info("Command line arguments: " + str(args))
    if hasattr(args, "config_file") and args.config_file != "":
        logger.info(
            "Contents of args.config_file={}:\n{}".format(
                args.config_file, PathManager.open(args.config_file, "r").read()
            )
        )

    logger.info("Running with full config:\n{}".format(cfg))
    if output_dir:
        path = os.path.join(output_dir, "config.yaml")
        with PathManager.open(path, "w") as f:
            f.write(cfg.dump())
        logger.info("Full config saved to {}".format(path))


def get_model_stats(model, cfg, mode, use_train_input):
    """
    Compute statistics for the current model given the config.
    Args:
        model (model): model to perform analysis.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        mode (str): Options include `flop` or `activation`. Compute either flop
            (gflops) or activation count (mega).
        use_train_input (bool): if True, compute statistics for training. Otherwise,
            compute statistics for testing.
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

    # Set model to evaluation mode for analysis.
    # Evaluation mode can avoid getting stuck with sync batchnorm.
    model_mode = model.training
    model.eval()
    inputs = _get_model_analysis_input(cfg, use_train_input)
    count_dict, _ = model_stats_fun(model, inputs)
    count = sum(count_dict.values())
    model.train(model_mode)
    return count

    
def log_model_info(model, cfg, use_train_input=True):
    """
    Log info, includes number of parameters, gpu usage.
        The model info is computed when the model is in validation mode.
    Args:
        model (model): model to log the info.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        use_train_input (bool): if True, log info for training. Otherwise,
            log info for testing.
    """
    logger.info("Model:\n{}".format(model))
    logger.info("Params: {:,}".format(params_count(model)))
    logger.info("Mem: {:,} MB".format(gpu_mem_usage()))

    logger.info(
        "Activations: {:,} M".format(
            get_model_stats(model, cfg, "activation", use_train_input)
        )
    )
    logger.info("nvidia-smi")
    os.system("nvidia-smi")