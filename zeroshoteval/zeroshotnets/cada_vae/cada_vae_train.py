import itertools
import logging

import torch
from fvcore.common.config import CfgNode
from torch import nn as nn
from torch.nn.modules.module import Module
from torch.optim.optimizer import Optimizer
from torch.utils.data.dataloader import DataLoader

from zeroshoteval.data.dataloader_helper import construct_loader
from zeroshoteval.solver.optimizer_helper import build_optimizer
from zeroshoteval.utils.misc import log_model_info
from zeroshoteval.zeroshotnets.build import ZSL_MODEL_REGISTRY
from zeroshoteval.zeroshotnets.trainer_base import TrainerBase
from .cada_vae_model import VAEModel

logger = logging.getLogger(__name__)


@ZSL_MODEL_REGISTRY.register()
def CADA_VAE_train_procedure(cfg: CfgNode) -> Module:
    """
    Starts CADA-VAE model training and generates zsl_embedding dataset for
    classifier training.

    Args:
        cfg(CfgNode): configs. Details can be found in
            zeroshoteval/config/defaults.py.

    Returns:
        data(dict): dictionary with zero-shot embeddings dataset and extra data.

    """
    logger.info("Building CADA-VAE model")

    # Model building
    # --------------
    model = VAEModel(hidden_size_encoder=cfg.CADA_VAE.HIDDEN_SIZE.ENCODER,
                     hidden_size_decoder=cfg.CADA_VAE.HIDDEN_SIZE.DECODER,
                     latent_size=cfg.CADA_VAE.LATENT_SIZE,
                     feature_dimensions=cfg.DATA.FEAT_EMB.DIM,
                     use_bn=cfg.CADA_VAE.USE_BN,
                     use_dropout=False)

    model.to(cfg.DEVICE)
    log_model_info(model, cfg.ZSL_MODEL_NAME)

    # Optimizer building
    # ------------------
    optimizer = build_optimizer(model, cfg, procedure="ZSL")

    # Data loader building
    # --------------------
    train_loader = construct_loader(cfg, split="train")

    # Model training
    # --------------
    trainer = CADAVAETrainer(model=model, data_loader=train_loader, optimizer=optimizer, cfg=cfg)

    trainer.train(max_iter=cfg.ZSL.EPOCH)

    return model


class CADAVAETrainer(TrainerBase):

    def __init__(self, model: Module, data_loader: DataLoader, optimizer: Optimizer, cfg: CfgNode) -> None:
        """
        Args:
            cfg(CfgNode): configs. Details can be found in
                zeroshoteval/config/defaults.py
            model(nn.Module): model to train.
            data_loader: Loads train data
            optimizer: optimizer to be used
        """
        super().__init__(model, data_loader, optimizer)

        self.cfg = cfg

        self.loss_history = []
        self.loss_vae = []
        self.loss_ca = []
        self.loss_da = []

    def run_step(self):
        assert self.model.training, "Model must be set to training mode!"
        loss_accum: float = 0.0
        loss_vae_accum: float = 0.0
        loss_ca_accum: float = 0.0
        loss_da_accum: float = 0.0

        beta, cross_reconstruction_factor, distance_factor = (
            self.__calculate_loss_factors(self.current_iter, self.cfg.CADA_VAE.WARMUP)
        )

        # TODO: move the cycle below to the run_step separately
        for _i_step, (x, _) in enumerate(self.data_loader):

            for modality, modality_tensor in x.items():
                x[modality] = modality_tensor.to(self.cfg.DEVICE).float()
                x[modality].requires_grad = False

            x_recon, z_mu, z_logvar, z_noize = self.model(x)

            loss_vae, loss_ca, loss_da = self.__compute_cada_losses(
                self.model.decoder,
                x,
                x_recon,
                z_mu,
                z_logvar,
                z_noize,
                beta
            )

            self.optimizer.zero_grad()

            loss = loss_vae
            loss_vae_accum += loss_vae.item()
            loss_ca_accum += loss_ca.item() * cross_reconstruction_factor
            loss_da_accum += loss_da.item() * distance_factor

            if self.cfg.CADA_VAE.CROSS_RECONSTRUCTION:
                if cross_reconstruction_factor < 0:
                    logger.warning('Cross-reconstruction factor is less than zero!')
                loss += loss_ca * cross_reconstruction_factor
            if self.cfg.CADA_VAE.DISTRIBUTION_ALLIGNMENT and (distance_factor > 0):
                if distance_factor < 0:
                    logger.warning('Distribution alignment factor is less than zero!')
                loss += loss_da * distance_factor

            # self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss_accum += loss.item()

        loss_accum_mean = loss_accum / (_i_step + 1)
        loss_vae_accum = loss_vae_accum / (_i_step + 1)
        loss_ca_accum = loss_ca_accum / (_i_step + 1)
        loss_da_accum = loss_da_accum / (_i_step + 1)
        logger.info(
            f"Epoch: {self.current_iter+1} "
            f"Loss: {loss_accum_mean:.1f} "
            f"Loss VAE: {loss_vae_accum} "
            f"Loss CA: {loss_ca_accum} "
            f"Loss DA: {loss_da_accum}"
        )

        self.loss_history.append(loss_accum_mean)

    def __calculate_loss_factors(self, epoch: int, warmup: dict):
        r"""
        Calculates cross-allignment, distance allignment and beta factors.

        Args:
            epoch(Int): current epoch number.
            warmup(dict): dict of dicts mapping

        Returns:
            beta, cross_reconstruction_factor, distance_factor
        """
        # Beta factor
        if epoch < warmup.BETA.START_EPOCH:
            beta = 0
        elif epoch >= warmup.BETA.END_EPOCH:
            beta = warmup.BETA.FACTOR
        else:
            beta = (
                1.0
                * (epoch - warmup.BETA.START_EPOCH)
                / (warmup.BETA.END_EPOCH - warmup.BETA.START_EPOCH)
                * warmup.BETA.FACTOR
            )

        # Cross-reconstruction factor
        if epoch < warmup.CROSS_RECONSTRUCTION.START_EPOCH:
            cross_reconstruction_factor = 0
        elif epoch >= warmup.CROSS_RECONSTRUCTION.END_EPOCH:
            cross_reconstruction_factor = warmup.CROSS_RECONSTRUCTION.FACTOR
        else:
            cross_reconstruction_factor = (
                1.0
                * (epoch - warmup.CROSS_RECONSTRUCTION.START_EPOCH)
                / (
                    warmup.CROSS_RECONSTRUCTION.END_EPOCH
                    - warmup.CROSS_RECONSTRUCTION.START_EPOCH
                )
                * warmup.CROSS_RECONSTRUCTION.FACTOR
            )

        # Distribution alignment factor
        if epoch < warmup.DISTANCE.START_EPOCH:
            distance_factor = 0
        elif epoch >= warmup.DISTANCE.END_EPOCH:
            distance_factor = warmup.DISTANCE.FACTOR
        else:
            distance_factor = (
                1.0
                * (epoch - warmup.DISTANCE.START_EPOCH)
                / (warmup.DISTANCE.END_EPOCH - warmup.DISTANCE.START_EPOCH)
                * warmup.DISTANCE.FACTOR
            )

        return beta, cross_reconstruction_factor, distance_factor

    def __compute_cada_losses(self, decoder, x, x_recon, z_mu, z_logvar, z_noize, beta):
        r"""
        Computes reconstruction loss, Kullback–Leibler divergence loss, and
            distridution allignment loss.

        Args:
            x(dict: {string: Tensor}): dictionary mapping modalities names to
                modalities input.
            x_recon(dict: {string: Tensor}): dictionary mapping modalities names
                to modalities input reconstruction.
            z_mu(dict: {string: Tensor}): dictionary mapping modalities names to
                mean.
            z_logvar(dict: {string: Tensor}): dictionary mapping modalities names
                to variance logarithm.
            z_noize(dict: {string: Tensor}): dictionary mapping modalities names to
                encoder out.
            beta(float): KL loss factor in VAE loss.
            recon_loss(string, optional): specifies the norm to apply to calculate
                reconstuction loss: 'l1'|'l2'. 'l1': using l1-norm.
                'l2'-using l2-norm.

        Returns:
            loss_vae: VAE loss.
            loss_ca: cross allignment reconstruction loss.
            loss_da: distridution allignment loss, using Wasserstien distance as
                distance measure.
        """
        loss_recon = 0
        loss_kld = 0
        loss_da = 0
        loss_ca = 0

        for modality in z_mu.keys():
            # Calculate reconstruction and KLD loss for each modality
            loss_recon += self.__reconstruction_loss(x[modality], x_recon[modality], recon_loss_norm=self.cfg.CADA_VAE.NORM_TYPE)
            loss_kld += (0.5 * (1 + z_logvar[modality] - z_mu[modality].pow(2) - z_logvar[modality].exp()).sum(dim=1).mean())

        # Calculate standart vae loss as sum of reconstion loss and
        # Kullback–Leibler divergence
        loss_vae = loss_recon - beta * loss_kld

        for (modality_1, modality_2) in itertools.combinations(z_mu.keys(), 2):
            # Calulate cross allignment and distribution allignment loss for each
            # pair of modalities
            loss_da += self.__compute_da_loss(
                z_mu[modality_1],
                z_logvar[modality_1],
                z_mu[modality_2],
                z_logvar[modality_2]
            )
            loss_ca += self.__compute_ca_loss(
                decoder[modality_1],
                decoder[modality_2],
                x[modality_1],
                x[modality_2],
                z_noize[modality_1],
                z_noize[modality_2]
            )

        return loss_vae, loss_ca, loss_da

    def __compute_da_loss(self, z_mu_1, z_logvar_1, z_mu_2, z_logvar_2):
        r"""
        Computes Distribution Allignment loss.
        Using Wasserstein distance.

        Args:
            z_mu_1(Tensor): mean for first modality endoderer out
            z_logvar_1(Tensor): variance logarithm for first modality endoderer out
            z_mu_2(Tensor): mean for second modality endoderer out
            z_logvar_2(Tensor): variance logarithm for second modality endoderer out

        Return:
            loss_da: Distribution Allignment loss
        """

        loss_mu = (z_mu_1 - z_mu_2).pow(2).sum(dim=1)
        loss_var = (
            # ((z_logvar_1 / 2).exp() - (z_logvar_2 / 2).exp()).pow(2).sum(dim=1)
            (z_logvar_1.exp().pow(1.0 / 2) - z_logvar_2.exp().pow(1.0 / 2)).pow(2).sum(dim=1)
        )

        loss_da = (loss_mu + loss_var).sqrt().sum()
        # loss_da = torch.sqrt(loss_mu + loss_var).mean()

        return loss_da

    def __compute_ca_loss(self, decoder_1, decoder_2, x_1, x_2, z_sample_1, z_sample_2):
        r"""
            Computes cross alignment loss.
            First modality original input compares to reconstrustion wich uses first
            modality decoder and second modality endoder out. And visa versa: x1_input
            vs decoder1(z2)

            Args:
                decoder_1(nn.module): decoder for fist modality.
                decoder_2(nn.module): decoder for second modality.
                x_1(Tensor): first modality original input.
                x_2(Tensor): second modality original input.
                z_sample_1(Tensor): first modality latent representation sample.
                z_sample_2(Tensor): second modality latent representation sample.

            Returns:
                loss_ca: cross alignment loss over two given modalities.
            """
        decoder_1.eval()
        decoder_2.eval()

        x_recon_1 = decoder_1(z_sample_2)
        x_recon_2 = decoder_2(z_sample_1)

        recon_loss_1 = self.__reconstruction_loss(
            x_1,
            x_recon_1,
            recon_loss_norm=self.cfg.CADA_VAE.NORM_TYPE
        )

        recon_loss_2 = self.__reconstruction_loss(
            x_2,
            x_recon_2,
            recon_loss_norm=self.cfg.CADA_VAE.NORM_TYPE
        )

        loss_ca = recon_loss_1 + recon_loss_2

        return loss_ca

    def __reconstruction_loss(self, x, x_recon, recon_loss_norm="l1"):
        r"""
        Computes reconstruction loss.

        Args:
            x(Tensor): original input.
            x_recon(Tensor): reconstructed input.
            recon_loss(string, optional): specifies the norm to apply to calculate
                reconstuction loss:
            'l1'|'l2'. 'l1': using l1-norm. 'l2'-using l2-norm.

        Returns:
            loss_recon: reconstruction loss.
        """
        if recon_loss_norm == "l1":
            loss_recon = nn.L1Loss(size_average=False)(x_recon, x)
            # loss_recon = (nn.functional.l1_loss(x, x_recon, reduction="sum") / x.shape[0])
        elif recon_loss_norm == "l2":
            loss_recon = nn.MSELoss(size_average=False)(x_recon, x)
            # loss_recon = (nn.functional.mse_loss(x, x_recon, reduction="sum") / x.shape[0])

        return loss_recon
