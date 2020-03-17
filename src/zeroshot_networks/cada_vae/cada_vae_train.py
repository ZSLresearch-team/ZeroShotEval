import itertools
import torch
import torch.nn as nn


def train_CADA_VAE(config, data):
    """
    Train CADA-VAE model.

    Args:
        config:
        data:
    """
    pass

def compute_cada_losses(x, x_recon, z_mu, z_var, recon_loss_norm="l2"):
    """
    Computes reconstruction loss, Kullback–Leibler divergence loss, and distridution allignment loss.

    Args:
        x(dict: {string: Tensor}): dictionary mapping modalities names to modalities input.
        x_recon(dict: {string: Tensor}): dictionary mapping modalities names to modalities input reconstruction.
        z_mu(dict: {string: Tensor}): dictionary mapping modalities names to mean.
        z_var(dict: {string: Tensor}): dictionary mapping modalities names to variance.
        recon_loss(string, optional): specifies the norm to apply to calculate reconstuction loss:
        'l1'|'l2'. 'l1': using l1-norm. 'l2'-using l2-norm.

    Returns:
        loss_recon: reconstructioon loss.
        loss_kld: Kullback–Leibler divergence loss
        loss_da: distridution allignment loss, using Wasserstien distance as distance measure.
    """
    loss_recon = torch.zeros(())
    loss_kld = torch.zeros(())
    loss_da = torch.zeros(())

    for modality in z_mu.keys():
        if recon_loss_norm == "l1":
            loss_recon_modality = nn.functional.l1_loss(x[modality], x_recon[modality])
        elif recon_loss_norm == "l2":
            loss_recon_modality = nn.functional.mse_loss(x[modality], x_recon[modality])

        loss_recon += loss_recon_modality

        loss_kld += (1 + z_var[modality] - z_mu[modality].pow(2) - z_var[modality].exp()).mean()

    for (modality_1, modality_2) in itertools.combinations(z_mu.keys(), 2):
        loss_da += compute_da_loss(z_mu[modality_1], z_mu[modality_2], z_var[modality_1], z_var[modality_2])

    n_modalities = len(z_mu)
    loss_kld /= n_modalities
    loss_da /= n_modalities * (n_modalities - 1) / 2

    return loss_recon, loss_kld, loss_da


def compute_da_loss(z_mu_1, z_var_1, z_mu_2, z_var_2):
    """
    Computes Distribution Allignment loss.
    Using Wasserstein distance.

    Args:
        z_mu_1(dict: {string: Tensor}): mean for first modality endoderer out
        z_var_1(dict: {string: Tensor}): variance for first modality endoderer out
        z_mu_2(dict: {string: Tensor}): mean for second modality endoderer out
        z_var_2(dict: {string: Tensor}): variance for second modality endoderer out

    Return:
        loss_da: Distribution Allignment loss
    """
    loss_mu = (z_mu_1 - z_mu_2).pow(2).sum(dim=1)
    loss_var = ((z_var_1 / 2.0).exp() + (z_var_2 / 2.0).exp()).pow(2).sum(dim=1)

    loss_da = torch.sqrt(loss_mu + loss_var).mean()

    return loss_da
