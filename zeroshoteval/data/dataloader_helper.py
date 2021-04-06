from typing import Optional

import torch
from fvcore.common.config import CfgNode
from torch.utils.data.dataloader import DataLoader

from zeroshoteval.data.modalities_embedding_dataset import ModalitiesEmbeddingDataset
from zeroshoteval.data.synthetic_dataset import GenEmbeddingDataset


def construct_loader(cfg: CfgNode, split, shuffle: Optional[bool] = None, drop_last: Optional[bool] = None) -> DataLoader:
    """
    Constructs the dataloader for zsl train/test procedure and the given
        dataset.

    Args:
        cfg(CfgNode): configs. Detail can de found in
            zeroshoteval/config/defaults.py
        split(str): the split of the data loader. Include `trainval` and
            `test` for now.
        drop_last(bool): whether to drop the last incomplite batch

    Returns:
        loader(DataLoader): data loader for zsl train/test procedure
    """
    if shuffle is None:
        if split in ["train"]:
            shuffle = True
        else:
            shuffle = False

    if drop_last is None:
        if split in ["train"]:
            drop_last = True
        else:
            drop_last = False

    modalities_dataset = ModalitiesEmbeddingDataset(root_dir=cfg.DATA.FEAT_EMB.PATH, modalities=["IMG"], split=split)  # TODO: replace hardcoded modalities with proper value from config.

    modalities_dataloader = DataLoader(dataset=modalities_dataset,
                                       batch_size=cfg.ZSL.BATCH_SIZE,
                                       shuffle=shuffle,
                                       num_workers=cfg.DATA_LOADER.NUM_WORKERS,
                                       pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
                                       drop_last=drop_last)

    return modalities_dataloader


def build_gen_loaders(cfg):
    """
    Builds data loaders to generate embeddigs for different splits
    and modalities

    Args:
        cfg(CfgNode): configs. Details can be found in
            zeroshoteval/config/defaults.py
    """

    data = {}
    if cfg.GENERALIZED:
        data["train_img_loader"] = _construct_gen_loader(cfg, "trainval", "IMG")

    data["train_attr_loader"] = _construct_gen_loader(
        cfg, "test_unseen", "CLSATTR"
    )

    data["test_loader"] = _construct_gen_loader(cfg, "test", "IMG")

    return data


def _construct_gen_loader(cfg, split, mod):
    """
    Construct dataloader for generating zsl embeddings

    Args:
        cfg: configs. Details can be found in
            zeroshoteval/config/defaults.py
        split(str): data split, e.g. `train`, `trainval`, `test`.
        mod(str): modality name

    Returns:
        loader(DataLoader): data loader for generating zsl embeddings
    """
    dataset = GenEmbeddingDataset(cfg, split, mod)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.ZSL.BATCH_SIZE,
        num_workers=cfg.DATA_LOADER.NUM_WORKERS,
        pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
    )

    return loader
