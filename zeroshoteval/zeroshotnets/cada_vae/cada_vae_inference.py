from typing import Tuple, Dict

import torch
from fvcore.common.config import CfgNode
from torch import Tensor
from torch.nn.modules.module import Module
from torch.utils.data.dataloader import DataLoader

from zeroshoteval.data.dataloader_helper import construct_loader
from zeroshoteval.utils import checkpoint


def CADA_VAE_inference_procedure(cfg: CfgNode, model: Module) -> Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]]:
    """
    Generates synthetic dataset via trained zsl model to cls training

    Args:
        cfg(CfgNode): configs. Details can be found in
            zeroshoteval/config/defaults.py
        model: pretrained CADA-VAE model.

    Returns:
        zsl_emb_dataset: sythetic dataset for classifier.
        csl_train_indice: train indicies.
        csl_test_indice: test indicies.
    """
    # Data loader building
    # --------------------
    train_loader = construct_loader(cfg, split="train")
    test_loader = construct_loader(cfg, split="test")

    train_zsl_data: Dict[str, Tuple[Tensor, Tensor]] = dict()
    test_zsl_data: Dict[str, Tuple[Tensor, Tensor]] = dict()

    # Generate ZSL embeddings for train (seen) images
    # -----------------------------------------------
    train_zsl_data["IMG"] = CADA_VAE_inference(model=model,
                                               data_loader=train_loader,
                                               modality="IMG",
                                               device=cfg.DEVICE)

    # Generate ZSL embeddings for additional modality (CLSATTR) to use for classifier TRAINING
    # (give classifier info about unseen classes with another modality)
    # ----------------------------------------------------------------------------------------
    test_zsl_data["CLSATTR"] = CADA_VAE_inference(model=model,
                                                  data_loader=test_loader,
                                                  modality="CLSATTR",
                                                  device=cfg.DEVICE)

    # Leave unseen instances only as additional info for the classifier
    test_embeddings, test_labels = test_zsl_data["CLSATTR"]

    test_embeddings = test_embeddings[test_loader.dataset.unseen_indexes]
    test_labels = test_labels[test_loader.dataset.unseen_indexes]

    test_zsl_data["CLSATTR"] = (test_embeddings, test_labels)

    # Generate ZSL embeddings for test data
    # -------------------------------------
    test_zsl_data["IMG"] = CADA_VAE_inference(model=model,
                                              data_loader=test_loader,
                                              modality="IMG",
                                              device=cfg.DEVICE,
                                              reparametrize_with_noise=False)

    # For Generalized ZSL setting leave only unseen indexes
    if not cfg.GENERALIZED:
        test_embeddings, test_labels = test_zsl_data["IMG"]

        test_embeddings = test_embeddings[test_loader.dataset.unseen_indexes]
        test_labels = test_labels[test_loader.dataset.unseen_indexes]

        test_zsl_data["IMG"] = (test_embeddings, test_labels)

    # Creation of the final data layout
    # ---------------------------------
    train_data: Tuple[Tensor, Tensor] = (
        torch.cat((train_zsl_data["IMG"][0], test_zsl_data["CLSATTR"][0]), dim=0),
        torch.cat((train_zsl_data["IMG"][1], test_zsl_data["CLSATTR"][1]), dim=0)
    )

    test_data: Tuple[Tensor, Tensor] = test_zsl_data["IMG"]

    if cfg.ZSL.SAVE_EMB:
        data: Tuple[Tensor, Tensor] = (
            torch.cat((train_data[0], test_data[0]), dim=0),
            torch.cat((train_data[1], test_data[1]), dim=0)
        )
        # TODO: change embeddings saving to saving of 3 files: embeddings, labels, splits
        checkpoint.save_embeddings(cfg.OUTPUT_DIR, data, cfg)

    return train_data, test_data


def CADA_VAE_inference(model: Module,
                       data_loader: DataLoader,
                       modality: str,
                       device: str,
                       reparametrize_with_noise: bool = True) -> Tuple[Tensor, Tensor]:
    """
    Calculate zsl embeddings for given VAE model and data.

    Args:
        model:
        data_loader:
        modality:
        device:
        reparametrize_with_noise:

    Returns:
        zsl_emb: zero shot learning embeddings for given data and model
    """
    model.eval()

    with torch.no_grad():
        zsl_emb = torch.Tensor().to(device)
        labels = torch.Tensor().long().to(device)

        for _i_step, (x, y) in enumerate(data_loader):

            x = x[modality]

            x = x.float().to(device)
            z_mu, z_logvar, z_noize = model.encoder[modality](x)

            if reparametrize_with_noise:
                zsl_emb = torch.cat((zsl_emb, z_noize.to(device)), 0)
            else:
                zsl_emb = torch.cat((zsl_emb, z_mu.to(device)), 0)

            labels: Tensor = torch.cat((labels, y.long().to(device)), 0)

    return zsl_emb.to(device), labels
