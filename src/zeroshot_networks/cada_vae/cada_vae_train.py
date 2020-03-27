import itertools
import torch
import torch.nn as nn
from tqdm import tqdm, trange


def train_VAE(config, model, train_loader, optimizer, use_ca_loss=True, use_da_loss=True, verbose=1, *args, **kwargs):
    r"""
    Train VAE model.

    Args:
        config(dict): dict with different model, dataset and train parameters.
        model: model to train.
        datrain_loader: trainloader - loads train data.
        optimizer: optimizer to be used.
        use_ca_loss: if ``True`` will add cross allignment loss
        use_da_loss: if ``True`` will add distance allignment loss
        verbose: boolean or Int. The higher value verbose is - the more info you get.

    Returns:
        loss_history(list): loss history for 
    """
    model.to(config.device)

    tqdm_epoch = trange(config.nepoch, desc='Loss: None. Epoch', unit='epoch', disable=(verbose<=0), leave=True)
    # tqdm_train_loader = tqdm(train_loader, desc='Batch:', unit='batch', disable=(verbose<=1), leave=False)

    loss_history = []
    for epoch in tqdm_epoch:

        model.train()

        loss_accum = 0
        beta, cross_reconstruction_factor, distance_factor = loss_factors(epoch, config.specific_parameters.warmup)
        tqdm_train_loader = tqdm(train_loader, desc='Loss: None. Batch', unit='batch', disable=(verbose<=1), leave=False)

        for i_step, (x, _) in enumerate(tqdm_train_loader):

            for modality, modality_tensor in x.items():
                x[modality] = modality_tensor.to(config.device).float()
            x_recon, z_mu, z_logvar, z_noize = model(x)

            loss_vae, loss_ca, loss_da = compute_cada_losses(model.decoder, x, x_recon, z_mu, z_logvar, z_noize,
                                                             beta, *args, **kwargs)

            loss_value = loss_vae

            if use_ca_loss and (loss_ca > 0):
                loss_value += loss_ca * cross_reconstruction_factor
            if use_da_loss and (loss_da > 0):
                loss_value += loss_da * distance_factor

            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()

            loss_accum += loss_value.item()
            tqdm_train_loader.set_description(f'Loss: {loss_accum / (i_step + 1):.1f}. Batch')
            tqdm_train_loader.refresh()

        loss_accum_mean = loss_accum / (i_step + 1)
        tqdm_epoch.set_description(f'Loss: {loss_accum_mean:.1f}. Epoch')
        tqdm_epoch.refresh()

        loss_history.append(loss_accum_mean)

    return loss_history

def test_VAE(model, test_loader, test_modality):
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
        zsl_emb = torch.Tensor().to('cpu')
        labels = torch.Tensor().long().to('cpu')
        for _i_step, (x, _y) in enumerate(test_loader):
            _z_mu, _z_logvar, z_noize = model.encoder[test_modality](x[test_modality].float().to('cuda:0'))
            zsl_emb = torch.cat((zsl_emb, z_noize.to('cpu')), 0)
            labels = torch.cat((labels, _y.long()), 0)
    return zsl_emb.to('cpu'), labels

def compute_cada_losses(decoder, x, x_recon, z_mu, z_logvar, z_noize, beta, *args, **kwargs):
    r"""
    Computes reconstruction loss, Kullback–Leibler divergence loss, and distridution allignment loss.

    Args:
        x(dict: {string: Tensor}): dictionary mapping modalities names to modalities input.
        x_recon(dict: {string: Tensor}): dictionary mapping modalities names to modalities input reconstruction.
        z_mu(dict: {string: Tensor}): dictionary mapping modalities names to mean.
        z_logvar(dict: {string: Tensor}): dictionary mapping modalities names to variance logarithm.
        z_noize(dict: {string: Tensor}): dictionary mapping modalities names to encoder out.
        beta(float): KL loss factor in VAE loss
        recon_loss(string, optional): specifies the norm to apply to calculate reconstuction loss:
        'l1'|'l2'. 'l1': using l1-norm. 'l2'-using l2-norm.

    Returns:
        loss_vae: VAE loss.
        loss_ca: cross allignment reconstruction loss.
        loss_da: distridution allignment loss, using Wasserstien distance as distance measure.
    """
    loss_recon = 0
    loss_kld = 0
    loss_da = 0
    loss_ca = 0

    for modality in z_mu.keys():
        # Calculate reconstruction and kld loss for each modality
        loss_recon += reconstruction_loss(x[modality], x_recon[modality], **kwargs)
        loss_kld = (z_mu[modality].pow(2) + z_logvar[modality].exp().sqrt() - 1 - z_logvar[modality]/2).sum(dim=1).mean()

    loss_vae = loss_recon + beta * loss_kld

    for (modality_1, modality_2) in itertools.combinations(z_mu.keys(), 2):
        # Calulate cross allignment and distribution allignment loss for each pair of modalities
        loss_da += compute_da_loss(z_mu[modality_1], z_mu[modality_2], z_logvar[modality_1], z_logvar[modality_2])
        loss_ca += compute_ca_loss(decoder[modality_1], decoder[modality_2], x[modality_1], x[modality_2],
                                   z_noize[modality_1], z_noize[modality_2], *args, **kwargs)

    n_modalities = len(z_mu)
    # loss_kld /= n_modalities
    loss_da /= n_modalities * (n_modalities - 1) / 2

    return loss_vae, loss_ca, loss_da

def compute_da_loss(z_mu_1, z_logvar_1, z_mu_2, z_logvar_2):
    r"""
    Computes Distribution Allignment loss.
    Using Wasserstein distance.

    Args:
        z_mu_1(dict: {string: Tensor}): mean for first modality endoderer out
        z_logvar_1(dict: {string: Tensor}): variance logarithm for first modality endoderer out
        z_mu_2(dict: {string: Tensor}): mean for second modality endoderer out
        z_logvar_2(dict: {string: Tensor}): variance logarithm for second modality endoderer out

    Return:
        loss_da: Distribution Allignment loss
    """

    loss_mu = (z_mu_1 - z_mu_2).pow(2).sum(dim=1)
    loss_var = (z_logvar_1.exp() - z_logvar_2.exp()).pow(2).sum(dim=1)

    loss_da = torch.sqrt(loss_mu + loss_var).mean()

    return loss_da

def compute_ca_loss(decoder_1, decoder_2, x_1, x_2, z_noize_1, z_noize_2, *args, **kwargs):
    r"""
    Computes cross alignment loss.
    First modality original input compares to reconstrustion wich uses first modality decoder 
    and second modality endoder out. And visa versa: x1_input vs decoder1(z2)

    Args:
        decoder_1(nn.module): decoder for fist modality.
        decoder_2(nn.module): decoder for second modality.
        x_1(Tensor): first modality original input.
        x_2(Tensor): second modality original input.
        z_noize_1(Tensor): first modality noize encoder out.
        z_noize_2(Tensor): first modality noize encoder out.

    Returns:
        loss_ca: cross alignment loss over two given modalities.
    """
    decoder_1.eval()
    decoder_2.eval()

    x_recon_1 = decoder_1(z_noize_2)
    x_recon_2 = decoder_2(z_noize_1)

    loss_ca = reconstruction_loss(x_1, x_recon_1, *args, **kwargs) + \
                reconstruction_loss(x_2, x_recon_2, *args, **kwargs)

    return loss_ca

def reconstruction_loss(x, x_recon, recon_loss_norm="l2", **kwargs):
    r"""
    Computes reconstruction loss.

    Args:
        x(Tensor): original input.
        x_recon(Tensor): reconstructed input.
        recon_loss(string, optional): specifies the norm to apply to calculate reconstuction loss:
        'l1'|'l2'. 'l1': using l1-norm. 'l2'-using l2-norm.

    Returns:
        loss_recon: reconstruction loss.
    """
    if recon_loss_norm == "l1":
        loss_recon = nn.functional.l1_loss(x, x_recon, reduction='sum') / x.shape[0]
    elif recon_loss_norm == "l2":
        loss_recon = nn.functional.mse_loss(x, x_recon, reduction='sum') / x.shape[0]

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
    if current_epoch < warmup.cross_reconstruction.start_epoch:
        cross_reconstruction_factor = 0
    elif current_epoch >= warmup.cross_reconstruction.end_epoch:
        cross_reconstruction_factor = warmup.cross_reconstruction.factor
    else:
        cross_reconstruction_factor = 1.0 * (current_epoch - warmup.cross_reconstruction.start_epoch) / (
            warmup.cross_reconstruction.end_epoch - warmup.cross_reconstruction.start_epoch) * \
                warmup.cross_reconstruction.factor

    if current_epoch < warmup.beta.start_epoch:
        beta = 0
    elif current_epoch >= warmup.beta.end_epoch:
        beta = warmup.beta.factor
    else:
        beta = 1.0 * (current_epoch - warmup.beta.start_epoch) / \
            (warmup.beta.end_epoch - warmup.beta.start_epoch) * warmup.beta.factor

    if current_epoch < warmup.distance.start_epoch:
        distance_factor = 0
    elif current_epoch >= warmup.distance.end_epoch:
        distance_factor = warmup.distance.factor
    else:
        distance_factor = 1.0 * (current_epoch - warmup.distance.start_epoch) / \
              (warmup.distance.end_epoch - warmup.distance.start_epoch) * warmup.distance.factor

    return beta, cross_reconstruction_factor, distance_factor
