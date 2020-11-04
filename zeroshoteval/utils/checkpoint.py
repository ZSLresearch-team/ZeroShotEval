#!/usr/bin/env python3

"""Function that handle saving and loading checkpoints, embeddings, etc."""
import torch
from fvcore.common.file_io import PathManager

import os


def get_embeddings_dir(path_to_job):
    """
    Get path for storing embeddings.

    Args:
        path_to_job (string): the path to the folder of the current job.
    """
    return os.path.join(path_to_job, "embeddings")


def get_path_to_data(path_to_job, cfg):
    """
    Get the full path to a data file.

    Args:
        path_to_job (string): the path to the folder of the current job.
        cfg(CfgNode): configs. Details can be found in
            zeroshoteval/config/defaults.py
    """
    name = f"{cfg.ZSL_MODEL_NAME}_{cfg.DATA.DATASET_NAME}_emb.pyth"
    return os.path.join(get_embeddings_dir(path_to_job), name)


def save_embeddings(path_to_job, data, cfg):
    """
    Save a checkpoint.

    Args:
        model (model): model to save the weight to the checkpoint.
        data (dict): dictionary with embeddings dataset and extra data.
        cfg (CfgNode): configs to save.
    """
    # Ensure that the checkpoint dir exists.
    PathManager.mkdirs(get_embeddings_dir(path_to_job))

    # Record the state.
    data["cfg"] = {"cfg": cfg.dump()}
    # Write the checkpoint.
    path_to_data = get_path_to_data(path_to_job, cfg)
    with PathManager.open(path_to_data, "wb") as f:
        torch.save(data, f)
    return path_to_data
