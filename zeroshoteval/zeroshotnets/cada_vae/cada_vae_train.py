import time
import numpy as np
import torch
from torch import nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.sampler import SubsetRandomSampler

from zeroshoteval.zeroshotnets.trainer_base import TrainerBase

from zeroshoteval.utils.misc import log_model_info
from zeroshoteval.solver.optimizer_helper import build_optimizer

import itertools
import logging

from zeroshoteval.zeroshotnets.build import ZSL_MODEL_REGISTRY
from .cada_vae_model import VAEModel


logger = logging.getLogger(__name__)


@ZSL_MODEL_REGISTRY.register()
def CADA_VAE_train_procedure(
    cfg, dataset, gen_syn_data=True, save_model=False, save_dir="checkpoints",
):
    """
    Starts CADA-VAE model training and generates zsl_embedding dataset for
    classifier training.

    Args:
        cfg(CfgNode): configs. Details can be found in
            zeroshoteval/config/defaults.py
        dataset(Dataset): dataset for training and evaluating.
        gen_sen_data(bool): if ``True`` generates synthetic data for classifier
            after training.
        save_model(bool):  if ``True`` saves model.
        save_dir(str): root to models save dir.

    Returns:
        model: trained model.

    """
    logger.info("Building CADA-VAE model")

    # Model building
    model = VAEModel(
        hidden_size_encoder=cfg.CADA_VAE.HIDDEN_SIZE.ENCODER,
        hidden_size_decoder=cfg.CADA_VAE.HIDDEN_SIZE.DECODER,
        latent_size=cfg.CADA_VAE.LATENT_SIZE,
        modalities=dataset.modalities,
        feature_dimensions=cfg.DATA.FEAT_EMB.DIM,
        use_bn=cfg.CADA_VAE.USE_BN,
        use_dropout=False,
    )

    model.to(cfg.DEVICE)
    log_model_info(model, cfg.ZSL_MODEL_NAME)

    # Optimizer building
    optimizer = build_optimizer(model, cfg, "ZSL")

    # Data loader building
    train_sampler = SubsetRandomSampler(dataset.train_indices)
    train_loader = DataLoader(
        dataset,
        batch_size=cfg.ZSL.BATCH_SIZE,
        sampler=train_sampler,
        drop_last=cfg.DATA_LOADER.DROP_LAST,
        num_workers=cfg.DATA_LOADER.NUM_WORKERS,
        pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
    )

    # Model training
    trainer = CadaVaeTrainer(model=model,
                             data_loader=train_loader,
                             optimizer=optimizer,
                             cfg=cfg)

    trainer.train(max_iter=cfg.ZSL.EPOCH)

    # Post-training actions
    if save_model:
        # TODO: implement model saving
        raise NotImplementedError("Model saving is not implemented!")

    # TODO: unify return value of the fucntion CADA_VAE_train_procedure
    if gen_syn_data:
        return generate_synthetic_dataset(cfg, dataset, model)

    return model


class CadaVaeTrainer(TrainerBase):

    def __init__(self, model, data_loader, optimizer, cfg) -> None:
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

    def train(self, max_iter: int) -> None:
        logger = logging.getLogger(__name__)
        logger.info("Starting training...")
        start_time: float = time.perf_counter()

        self.current_iter: int = 0
        self.max_iter: int = max_iter

        while self.current_iter < self.max_iter:
            self.run_step()
            self.current_iter += 1

        end_time: float = time.perf_counter()
        logger.info(f"Time elapsed for training: {start_time - end_time} seconds.")

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

            loss = loss_vae
            loss_vae_accum += loss_vae.item()
            loss_ca_accum += loss_ca.item() * cross_reconstruction_factor
            loss_da_accum += loss_da.item() * distance_factor

            if self.cfg.CADA_VAE.CROSS_RECONSTRUCTION:
                loss += loss_ca * cross_reconstruction_factor
            if self.cfg.CADA_VAE.DISTRIBUTION_ALLIGNMENT and (distance_factor > 0):
                loss += loss_da * distance_factor

            self.optimizer.zero_grad()
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
            # Calculate reconstruction and kld loss for each modality
            loss_recon += self.__reconstruction_loss(
                x[modality],
                x_recon[modality],
                recon_loss_norm=self.cfg.CADA_VAE.NORM_TYPE
            )
            loss_kld += (
                0.5
                * (
                    1
                    + z_logvar[modality]
                    - z_mu[modality].pow(2)
                    - z_logvar[modality].exp()
                )
                .sum(dim=1)
                .mean()
            )

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
            ((z_logvar_1 / 2).exp() - (z_logvar_2 / 2).exp()).pow(2).sum(dim=1)
        )

        loss_da = torch.sqrt(loss_mu + loss_var).mean()

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
            loss_recon = (
                nn.functional.l1_loss(x, x_recon, reduction="sum") / x.shape[0]
            )
        elif recon_loss_norm == "l2":
            loss_recon = (
                nn.functional.mse_loss(x, x_recon, reduction="sum") / x.shape[0]
            )

        return loss_recon


def eval_VAE(model, test_loader, test_modality, device, reparametrize_with_noise=True):
    """
    Calculate zsl embeddings for given VAE model and data.

    Args:
        model: VAE model.
        test_loader: test dataloader.
        test_modality: modality name for modality to test.
        device: device to use for inference.

    Returns:
        zsl_emb: zero shot learning embeddings for given data and model
    """
    model.eval()

    with torch.no_grad():
        zsl_emb = torch.Tensor().to(device)
        labels = torch.Tensor().long().to(device)

        for _i_step, (x, _y) in enumerate(test_loader):

            if test_modality == "IMG":
                x = x[test_modality].float().to(device)
            else:
                x = x.float().to(device)
            z_mu, _z_logvar, z_noize = model.encoder[test_modality](x)

            if reparametrize_with_noise:
                zsl_emb = torch.cat((zsl_emb, z_noize.to(device)), 0)
            else:
                zsl_emb = torch.cat((zsl_emb, z_mu.to(device)), 0)

            labels = torch.cat((labels, _y.long().to(device)), 0)

    return zsl_emb.to(device), labels


def generate_synthetic_dataset(cfg, dataset, model):
    r"""
    Generates synthetic dataset via trained zsl model to cls training

    Args:
        cfg(dict): dictionary with setting for model.
        dataset: original dataset.
        model: pretrained CADA-VAE model.

    Returns:
        zsl_emb_dataset: sythetic dataset for classifier .
        csl_train_indice: train indicies.
        csl_test_indice: test indicies.
    """
    logger.info("ZSL embedding generation")
    (
        zsl_emb_object_indice,
        zsl_emb_class,
        zsl_emb_class_label,
    ) = dataset.get_zsl_emb_indice(cfg.ZSL.SAMPLES_PER_CLASS)

    # Set CADA-Vae model to evaluate mode
    model.eval()

    # Generate zsl embeddings for train seen images
    if cfg.GENERALIZED:
        sampler = SubsetRandomSampler(zsl_emb_object_indice)
        loader = DataLoader(
            dataset, batch_size=cfg.ZSL.BATCH_SIZE, sampler=sampler
        )

        zsl_emb_img, zsl_emb_labels_img = eval_VAE(
            model, loader, "IMG", cfg.DEVICE
        )
    else:
        zsl_emb_img = torch.FloatTensor()
        zsl_emb_labels_img = torch.LongTensor()

    # Generate zsl embeddings for unseen classes
    zsl_emb_class = torch.from_numpy(zsl_emb_class)
    zsl_emb_class_label = torch.from_numpy(zsl_emb_class_label)
    zsl_emb_dataset = TensorDataset(zsl_emb_class, zsl_emb_class_label)

    loader = DataLoader(zsl_emb_dataset, batch_size=cfg.ZSL.BATCH_SIZE)

    zsl_emb_cls_attr, labels_cls_attr = eval_VAE(
        model, loader, "CLS_ATTR", cfg.DEVICE
    )
    if not cfg.GENERALIZED:
        labels_cls_attr = remap_labels(
            labels_cls_attr.cpu().numpy(), dataset.unseen_classes
        )
        labels_cls_attr = torch.from_numpy(labels_cls_attr)

    # Generate zsl embeddings for test data
    if cfg.GENERALIZED:
        sampler = SubsetRandomSampler(dataset.test_indices)
    else:
        sampler = SubsetRandomSampler(dataset.test_unseen_indices)

    loader = DataLoader(dataset, batch_size=cfg.ZSL.BATCH_SIZE, sampler=sampler)

    zsl_emb_test, zsl_emb_labels_test = eval_VAE(
        model, loader, "IMG", cfg.DEVICE, reparametrize_with_noise=False,
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

    return zsl_emb_dataset, csl_train_indice, csl_test_indice


def remap_labels(labels, classes):
    """
    Remapping labels

    Args:
        labels(np.array): array of labels
        classes:

    Returns:
        Remapped labels
    """
    remapping_dict = dict(zip(classes, list(range(len(classes)))))

    return np.vectorize(remapping_dict.get)(labels)
