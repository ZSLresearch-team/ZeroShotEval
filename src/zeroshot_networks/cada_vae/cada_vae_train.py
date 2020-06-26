import itertools

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.sampler import SequentialSampler, SubsetRandomSampler

from tqdm import tqdm, trange

from .cada_vae_model import VAEModel


def train_VAE(
    config,
    model,
    train_loader,
    optimizer,
    use_ca_loss=True,
    use_da_loss=True,
    verbose=1,
    *args,
    **kwargs,
):
    r"""
    Train VAE model.

    Args:
        config(dict): dict with different model, dataset and train parameters.
        model: model to train.
        datrain_loader: trainloader - loads train data.
        optimizer: optimizer to be used.
        use_ca_loss: if ``True`` will add cross allignment loss
        use_da_loss: if ``True`` will add distance allignment loss
        verbose: boolean or Int. The higher value verbose is - the more
            info you get.

    Returns:
        loss_history(list): loss history for 
    """
    if config.verbose > 1:
        print("\nTrain CADA-VAE model")

    tqdm_epoch = trange(
        config.nepoch,
        desc="Loss: None. Epoch",
        unit="epoch",
        disable=(verbose <= 0),
        leave=True,
    )

    loss_history = []
    model.train()

    for epoch in tqdm_epoch:

        loss_accum = 0

        beta, cross_reconstruction_factor, distance_factor = loss_factors(
            epoch, config.specific_parameters.warmup
        )
        tqdm_train_loader = tqdm(
            train_loader,
            desc="Loss: None. Batch",
            unit="batch",
            disable=(verbose <= 1),
            leave=False,
        )

        for i_step, (x, _) in enumerate(tqdm_train_loader):

            for modality, modality_tensor in x.items():
                x[modality] = modality_tensor.to(config.device).float()

            x_recon, z_mu, z_logvar, z_noize = model(x)

            loss_vae, loss_ca, loss_da = compute_cada_losses(
                model.decoder,
                x,
                x_recon,
                z_mu,
                z_logvar,
                z_noize,
                beta,
                *args,
                **kwargs,
            )

            loss = loss_vae

            if use_ca_loss:
                loss += loss_ca * cross_reconstruction_factor
            if use_da_loss and (distance_factor > 0):
                loss += loss_da * distance_factor

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_accum += loss.item()
            tqdm_train_loader.set_description(
                f"Loss: {loss_accum / (i_step + 1):.1f}. Batch"
            )
            tqdm_train_loader.refresh()

        loss_accum_mean = loss_accum / (i_step + 1)

        tqdm_epoch.set_description(f"Loss: {loss_accum_mean:.1f}. Epoch")
        tqdm_epoch.refresh()

        loss_history.append(loss_accum_mean)

    return loss_history


def eval_VAE(model, test_loader, test_modality, reparametrize_with_noise=True):
    """
    Calculate zsl embeddings for given VAE model and data.

    Args:
        model: VAE model.
        test_loader: test dataloader.
        test_modality: modality name for modality to test.

    Returns:
        zsl_emb: zero shot learning embeddings for given data and model
    """
    model.eval()

    with torch.no_grad():
        zsl_emb = torch.Tensor().to("cpu")
        labels = torch.Tensor().long().to("cpu")

        for _i_step, (x, _y) in enumerate(test_loader):
            # !
            if test_modality == "img":
                x = x[test_modality].float().to("cuda:0")
            else:
                x = x.float().to("cuda:0")
            z_mu, _z_logvar, z_noize = model.encoder[test_modality](x)

            if reparametrize_with_noise:
                zsl_emb = torch.cat((zsl_emb, z_noize.to("cpu")), 0)
            else:
                zsl_emb = torch.cat((zsl_emb, z_mu.to("cpu")), 0)

            labels = torch.cat((labels, _y.long()), 0)

    return zsl_emb.to("cpu"), labels


def compute_cada_losses(
    decoder, x, x_recon, z_mu, z_logvar, z_noize, beta, *args, **kwargs
):
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
        loss_recon += reconstruction_loss(
            x[modality], x_recon[modality], **kwargs
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
        loss_da += compute_da_loss(
            z_mu[modality_1],
            z_logvar[modality_1],
            z_mu[modality_2],
            z_logvar[modality_2],
        )
        loss_ca += compute_ca_loss(
            decoder[modality_1],
            decoder[modality_2],
            x[modality_1],
            x[modality_2],
            z_noize[modality_1],
            z_noize[modality_2],
            *args,
            **kwargs,
        )

    return loss_vae, loss_ca, loss_da


def compute_da_loss(z_mu_1, z_logvar_1, z_mu_2, z_logvar_2):
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


def compute_ca_loss(
    decoder_1, decoder_2, x_1, x_2, z_sample_1, z_sample_2, *args, **kwargs
):
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

    loss_ca = reconstruction_loss(
        x_1, x_recon_1, *args, **kwargs
    ) + reconstruction_loss(x_2, x_recon_2, *args, **kwargs)

    return loss_ca


def reconstruction_loss(x, x_recon, recon_loss_norm="l1", **kwargs):
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


def loss_factors(current_epoch, warmup):
    r"""
    Calculates cross-allignment, distance allignment and beta factors.

    Args:
        curent_epoch(Int): current epoch number.
        warmup(dict): dict of dicts mapping

    Returns:
        beta, cross_reconstruction_factor, distance_factor
    """
    # Beta factor
    if current_epoch < warmup.beta.start_epoch:
        beta = 0
    elif current_epoch >= warmup.beta.end_epoch:
        beta = warmup.beta.factor
    else:
        beta = (
            1.0
            * (current_epoch - warmup.beta.start_epoch)
            / (warmup.beta.end_epoch - warmup.beta.start_epoch)
            * warmup.beta.factor
        )

    # Cross-reconstruction factor
    if current_epoch < warmup.cross_reconstruction.start_epoch:
        cross_reconstruction_factor = 0
    elif current_epoch >= warmup.cross_reconstruction.end_epoch:
        cross_reconstruction_factor = warmup.cross_reconstruction.factor
    else:
        cross_reconstruction_factor = (
            1.0
            * (current_epoch - warmup.cross_reconstruction.start_epoch)
            / (
                warmup.cross_reconstruction.end_epoch
                - warmup.cross_reconstruction.start_epoch
            )
            * warmup.cross_reconstruction.factor
        )

    # Distribution alignment factor
    if current_epoch < warmup.distance.start_epoch:
        distance_factor = 0
    elif current_epoch >= warmup.distance.end_epoch:
        distance_factor = warmup.distance.factor
    else:
        distance_factor = (
            1.0
            * (current_epoch - warmup.distance.start_epoch)
            / (warmup.distance.end_epoch - warmup.distance.start_epoch)
            * warmup.distance.factor
        )

    return beta, cross_reconstruction_factor, distance_factor


def generate_synthetic_dataset(model_config, dataset, model):
    r"""
    Generates synthetic dataset via trained zsl model to cls training

    Args:
        model_config(dict): dictionary with setting for model.
        dataset: original dataset.
        model: pretrained CADA-VAE model.

    Returns:
        zsl_emb_dataset: sythetic dataset for classifier .
        csl_train_indice: train indicies.
        csl_test_indice: test indicies.
    """
    (
        zsl_emb_object_indice,
        zsl_emb_class,
        zsl_emb_class_label,
    ) = dataset.get_zsl_emb_indice(
        model_config.specific_parameters.samples_per_modality_class
    )

    # Set CADA-Vae model to evaluate mode
    model.eval()

    # Generate zsl embeddings for train seen images
    if model_config.generalized:
        sampler = SubsetRandomSampler(zsl_emb_object_indice)
        loader = DataLoader(
            dataset, batch_size=model_config.batch_size, sampler=sampler
        )

        zsl_emb_img, zsl_emb_labels_img = eval_VAE(model, loader, "img")
    else:
        zsl_emb_img = torch.FloatTensor()
        zsl_emb_labels_img = torch.LongTensor()

    # Generate zsl embeddings for unseen classes
    zsl_emb_class = torch.from_numpy(zsl_emb_class)
    zsl_emb_class_label = torch.from_numpy(zsl_emb_class_label)
    zsl_emb_dataset = TensorDataset(zsl_emb_class, zsl_emb_class_label)

    loader = DataLoader(zsl_emb_dataset, batch_size=model_config.batch_size)

    zsl_emb_cls_attr, labels_cls_attr = eval_VAE(model, loader, "cls_attr")
    if not model_config.generalized:
        labels_cls_attr = remap_labels(
            labels_cls_attr.cpu().numpy(), dataset.unseen_classes
        )
        labels_cls_attr = torch.from_numpy(labels_cls_attr)

    # Generate zsl embeddings for test data
    if model_config.generalized:
        sampler = SubsetRandomSampler(dataset.test_indices)
    else:
        sampler = SubsetRandomSampler(dataset.test_unseen_indices)

    loader = DataLoader(
        dataset, batch_size=model_config.batch_size, sampler=sampler
    )

    zsl_emb_test, zsl_emb_labels_test = eval_VAE(
        model, loader, "img", reparametrize_with_noise=False
    )

    # Create zsl embeddings dataset
    zsl_emb = torch.cat((zsl_emb_img, zsl_emb_cls_attr, zsl_emb_test), 0)

    zsl_emb_labels_img = zsl_emb_labels_img.long().to(model_config.device)
    labels_cls_attr = labels_cls_attr.long().to(model_config.device)
    zsl_emb_labels_test = zsl_emb_labels_test.long().to(model_config.device)

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


def VAE_train_procedure(
    model_config,
    dataset,
    gen_syn_data=True,
    save_model=False,
    save_dir="../../../model/",
):
    """
    Starts CADA-VAE model training and generates zsl_embedding dataset for
    classifier training.

    Args:
        model_config(dict): dictionary with setting for model.
        dataset: dataset for training and evaluating.
        gen_sen_data(bool): if ``True`` generates synthetic data for classifier
            after training.
        save_model(bool):  if ``True``saves model. 
        save_dir(str): root to models save dir.

    Returns:
        model: trained model.

    """
    model = VAEModel(
        hidden_size_encoder=model_config.specific_parameters.hidden_size_encoder,
        hidden_size_decoder=model_config.specific_parameters.hidden_size_decoder,
        latent_size=model_config.specific_parameters.latent_size,
        modalities=dataset.modalities,
        feature_dimensions=model_config.dataset.feature_dimensions,
        use_bn=model_config.specific_parameters.use_bn,
        use_dropout=model_config.specific_parameters.use_dropout,
    )

    if model_config.verbose > 2:
        print(model)
    model.to(model_config.device)
    # Model training
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=model_config.specific_parameters.lr_gen_model,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=0,
        amsgrad=True,
    )

    train_sampler = SubsetRandomSampler(dataset.train_indices)
    train_loader = DataLoader(
        dataset,
        batch_size=model_config.batch_size,
        sampler=train_sampler,
        drop_last=True,
    )

    loss_history = train_VAE(
        config=model_config,
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        recon_loss_norm=model_config.specific_parameters.loss,
        use_ca_loss=model_config.cross_resonstuction,
        use_da_loss=model_config.distibution_allignment,
        verbose=model_config.verbose,
    )
    if save_model:
        # TODO: implement model saving
        pass

    if gen_syn_data:

        return generate_synthetic_dataset(model_config, dataset, model)

    return model
