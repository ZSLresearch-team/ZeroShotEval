# Script for dataloader construction
import torch

from zeroshoteval.dataset.dataset import ObjEmbeddingDataset


def construct_loader(cfg, split):
    """
    Constructs the dataloader for zsl procedure and the given dataset.
    
    Args:
        cfg(CfgNode): configs. Detail can de found in 
            zeroshoteval/config/defaults.py
        split(str): the split of the data loader. Include `trainval` and
            `test` for now.
    """
    # Note that val train and val split temporaly unavailable
    assert split in ["trainval", "test"]
    if split in ["train", "trainval"]:
        shuffle = True
        drop_last = True
    elif split in ["test"]:
        shuffle = False
        drop_last = False

    dataset = ObjEmbeddingDataset(cfg.DATA.FEAT_EMB.PATH, ["IMG"], split)
    loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=cfg.ZSL.BATCH_SIZE,
            shuffle=shuffle,
            num_workers=cfg.DATA_LOADER.NUM_WORKERS,
            pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
            drop_last=drop_last,
        )
    
    return loader