import itertools
import torch
import torch.nn as nn
from tqdm import tqdm, trange


def train_VAE(config, model, train_loader, optimizer, use_ca_loss=True, use_da_loss=True, verbose=1, **kwargs):
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
    model.to(device)

    tqdm_epoch = trange(config.nepoch, desc='Loss: None. Epoch:', unit='epoch', disable=not(verbose>0), leave=True)
    tqdm_train_loader = tqdm(train_loader, desc='Batch:', unit='batch', disable=not(verbose>1), leave=False)

    loss_history =[]
    for epoch in tqdm_epoch:
        
        model.train()

        loss_accum = 0
        beta, cross_reconstruction_factor, distance_factor = loss_factors(epoch, config.warmup)

        for i_step, (x, _) in enumerate(tqdm_train_loader):
            x_recon, z_mu, z_var = model(x)

            loss_vae, loss_ca, loss_da = compute_cada_losses(x, x_recon, z_mu, z_var, beta, **kwargs)
            
            loss_value = loss_vae

            if use_ca_loss and (loss_ca > 0):
                loss_value += loss_ca * cross_reconstruction_factor
            if use_da_loss and (loss_da > 0):
                loss_value += loss_da * distance_factor

            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()

            loss_accum += loss_value.item()
        
        loss_accum_mean /= (i_step + 1)
        tqdm_epoch.set_description('Loss: {loss_accum_mean}. Epoch:')
        tqdm_epoch.refresh()

        loss_history.append(loss_accum_mean)

    return loss_history

def compute_cada_losses(x, x_recon, z_mu, z_var, beta, **kwargs):
    r"""
    Computes reconstruction loss, Kullback–Leibler divergence loss, and distridution allignment loss.

    Args:
        x(dict: {string: Tensor}): dictionary mapping modalities names to modalities input.
        x_recon(dict: {string: Tensor}): dictionary mapping modalities names to modalities input reconstruction.
        z_mu(dict: {string: Tensor}): dictionary mapping modalities names to mean.
        z_var(dict: {string: Tensor}): dictionary mapping modalities names to variance.
        beta(): KL loss factor in VAE loss
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
        loss_kld += (1 + z_var[modality] - z_mu[modality].pow(2) - z_var[modality].exp()).mean()
    
    loss_vae = loss_recon - beta * loss_kld

    for (modality_1, modality_2) in itertools.combinations(z_mu.keys(), 2):
        # Calulate cross allignment and distribution allignment loss for each pair of modalities
        loss_da += compute_da_loss(z_mu[modality_1], z_mu[modality_2], z_var[modality_1], z_var[modality_2]
        
        loss_ca += reconstruction_loss(x[modality_1], x_recon[modality_2], **kwargs) \
                    + reconstruction_loss(x[modality_2], x_recon[modality_1], **kwargs)

    n_modalities = len(z_mu)
    # loss_kld /= n_modalities
    loss_da /= n_modalities * (n_modalities - 1) / 2

    return loss_vae, loss_ca, loss_da


def compute_da_loss(z_mu_1, z_var_1, z_mu_2, z_var_2):
    r"""
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
            loss_recon = nn.functional.l1_loss(x[modality], x_recon[modality])
        elif recon_loss_norm == "l2":
            loss_recon = nn.functional.mse_loss(x[modality], x_recon[modality])
    
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
        cross_reconstruction_factor = 1.0 * (current_epoch - warmup.cross_reconstruction.start_epoch) / \
                                      (1.0 *(warmup.cross_reconstruction.end_epoch - warmup.cross_reconstruction.start_epoch)) \ 
                                      *  warmup.cross_reconstruction.factor

    if current_epoch < warmup.beta.start_epoch:
        beta = 0
    elif current_epoch >= warmup.beta.end_epoch:
        beta = warmup.beta.factor
    else:
        beta = 1.0 * (current_epoch - warmup.beta.start_epoch) / \
              (1.0 *(warmup.beta.end_epoch - warmup.beta.start_epoch)) \ 
               *  warmup.beta.factor

    if current_epoch < warmup.distance_factor.start_epoch:
        distance_factor = 0
    elif current_epoch >= warmup.distance_factor.end_epoch:
        distance_factor = warmup.distance_factor.factor
    else:
        distance_factor = 1.0 * (current_epoch - warmup.distance_factor.start_epoch) / \
              (1.0 *(warmup.distance_factor.end_epoch - warmup.distance_factor.start_epoch)) \ 
               *  warmup.distance_factor.factor

    return beta, cross_reconstruction_factor, distance_factor
