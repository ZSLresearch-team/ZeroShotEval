# Script for dataloader construction
import torch

from zeroshoteval.data.dataset import (
    GenEmbeddingDataset,
    ObjEmbeddingDataset,
)


def construct_loader(cfg, split):
    """
    Constructs the dataloader for zsl train/test procedure and the given
        dataset.

    Args:
        cfg(CfgNode): configs. Detail can de found in
            zeroshoteval/config/defaults.py
        split(str): the split of the data loader. Include `trainval` and
            `test` for now.

    Returns:
        loader(DataLoader): data loader for zsl train/test procedure
    """
    # Note that val train and val split temporaly unavailable
    assert split in ["trainval", "test"]
    if split in ["train", "trainval"]:
        shuffle = True
        drop_last = True
    elif split in ["test"]:
        shuffle = False
        drop_last = False

    loader_extras = {}
    dataset = ObjEmbeddingDataset(cfg, ["IMG"], split)
    loader_extras["loader"] = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.ZSL.BATCH_SIZE,
        shuffle=shuffle,
        num_workers=cfg.DATA_LOADER.NUM_WORKERS,
        pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
        drop_last=drop_last,
    )
    loader_extras["num_classes"] = (
        dataset.num_classes if cfg.GENERALIZED else dataset.num_unseen_classes
    )
    loader_extras["seen_classes"] = dataset.seen_classes
    loader_extras["unseen_classes"] = dataset.unseen_classes

    return loader_extras


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
