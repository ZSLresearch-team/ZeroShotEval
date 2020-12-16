from typing import Tuple

import torch
import numpy as np
from torch import Tensor
from torch.nn.modules.module import Module
from torch.utils.data import TensorDataset
from torch.utils.data.dataloader import DataLoader


def CADA_VAE_inference_procedure(cfg, model):
    r"""
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

    # Set CADA-Vae model to evaluate mode
    model.eval()

    loader = build_gen_loaders(cfg)
    # Generate zsl embeddings for train seen images
    if cfg.GENERALIZED:
        zsl_emb_img, zsl_emb_labels_img = CADA_VAE_inference(
            model, loader["train_img_loader"], "IMG", cfg.DEVICE
        )
    else:
        zsl_emb_img = torch.FloatTensor()
        zsl_emb_labels_img = torch.LongTensor()

    # Generate zsl embeddings for unseen classes
    zsl_emb_cls_attr, labels_cls_attr = CADA_VAE_inference(
        model, loader["train_attr_loader"], "CLS_ATTR", cfg.DEVICE
    )
    # if not cfg.GENERALIZED:
    #     labels_cls_attr = remap_labels(
    #         labels_cls_attr.cpu().numpy(), dataset.unseen_classes
    #     )
    #     labels_cls_attr = torch.from_numpy(labels_cls_attr)

    # Generate zsl embeddings for test data
    zsl_emb_test, zsl_emb_labels_test = CADA_VAE_inference(
        model,
        loader["test_loader"],
        "IMG",
        cfg.DEVICE,
        reparametrize_with_noise=False,
    )

    # Create zsl embeddings dataset
    zsl_emb = torch.cat((zsl_emb_img, zsl_emb_cls_attr, zsl_emb_test), 0)

    zsl_emb_labels_img = zsl_emb_labels_img.long().to(cfg.DEVICE)
    labels_cls_attr = labels_cls_attr.long().to(cfg.DEVICE)
    zsl_emb_labels_test = zsl_emb_labels_test.long().to(cfg.DEVICE)

    labels_tensor = torch.cat(
        (zsl_emb_labels_img, labels_cls_attr, zsl_emb_labels_test), 0
    )

    # Getting train and test indices
    n_train = len(zsl_emb_labels_img) + len(labels_cls_attr)
    csl_train_indice = np.arange(n_train)
    csl_test_indice = np.arange(n_train, n_train + len(zsl_emb_labels_test))

    zsl_emb_dataset = TensorDataset(zsl_emb, labels_tensor)

    data = {
        "dataset": zsl_emb_dataset,
        "train_indicies": csl_train_indice,
        "test_indicies": csl_test_indice,
    }
    return data


def CADA_VAE_inference(model: Module,
                       dataloader: DataLoader,
                       modality: str,
                       device: str,
                       reparametrize_with_noise: bool = True) -> Tuple[Tensor, np.ndarray]:
    """
    Calculate zsl embeddings for given VAE model and data.

    Args:
        model:
        dataloader:
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

        for _i_step, (x, y) in enumerate(dataloader):

            x = x.float().to(device)
            z_mu, z_logvar, z_noize = model.encoder[modality](x)

            if reparametrize_with_noise:
                zsl_emb = torch.cat((zsl_emb, z_noize.to(device)), 0)
            else:
                zsl_emb = torch.cat((zsl_emb, z_mu.to(device)), 0)

            labels: Tensor = torch.cat((labels, y.long().to(device)), 0)
            labels: np.ndarray = labels.cpu().detach().numpy()

    return zsl_emb.to(device), labels
