"""
"""

# region IMPORTS
import copy
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.utils import data
from src.zeroshot_networks.cada_vae.vae_networks import EncoderTemplate, DecoderTemplate
# endregion


class CADAVaeModel(nn.Module):
    """
    Model performs CADA-VAE ZSL approach

    .. _CADA-VAE arxiv papper:
        https://arxiv.org/pdf/1812.01784.pdf
    """

    def __init__(self, hidden_size_encoder, hidden_size_decoder, latent_size, modalities, feature_dimensions):
        """
        Args:
            latent_size: size of models latent space
            modalities: list of modalities to be used
            feature_dimensions: dictionary mapping modalities names to modalities embedding size.
            For example:
            {'img': 1024,
            'cls_attr': 312}
        """
        super(CADAVaeModel, self).__init__()
        self.modalities = modalities

        self.encoder = {}
        for datatype, dim in zip(self.modalities, feature_dimensions):
            self.encoder[datatype] = EncoderTemplate(dim, self.hidden_size_encoder[datatype], self.latent_size)
            # print(str(datatype) + ' ' + str(dim))

        self.decoder = {}
        for datatype, dim in zip(self.modalities, feature_dimensions):
            self.decoder[datatype] = DecoderTemplate(self.latent_size, self.hidden_size_decoder[datatype], dim)

    def forward(self, x):
        """
        Returns:
            z_mu: dictionary mapping modalities names to mean.
            z_var: dictionary mapping modalities names to z_variance.
            x_recon: dictionary mapping modalities names to decoder out.
        """
        z_mu = {}
        z_var = {}
        x_recon = {}

        for modality in self.modalities:
            z_mu[modality], z_var[modality] = self.encoder[modality](x[modality])

            std = (z_var[modality] / 2.0).exp()
            eps = torch.randn_like(std)
            x_recon[modality] = self.decoder[modality](eps * std + z_mu[modality])

        return x_recon, z_mu, z_var
