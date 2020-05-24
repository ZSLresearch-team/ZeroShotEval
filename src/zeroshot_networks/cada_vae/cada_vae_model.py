"""
"""
# region IMPORTS
import torch
import torch.nn as nn

from .vae_networks import EncoderTemplate, DecoderTemplate
# endregion


class VAEModel(nn.Module):
    """
    Model performs CADA-VAE ZSL approach

    .. _CADA-VAE arxiv papper:
        https://arxiv.org/pdf/1812.01784.pdf
    """

    def __init__(self,
                 hidden_size_encoder,
                 hidden_size_decoder,
                 latent_size,
                 modalities,
                 feature_dimensions,
                 *args, **kvargs):
        """
        Args:
            latent_size: size of models latent space
            modalities: list of modalities to be used
            feature_dimensions: dictionary mapping modalities names to modalities embedding size. 
        """
        super(VAEModel, self).__init__()
        self.modalities = modalities
        self.hidden_size_encoder = hidden_size_encoder
        self.hidden_size_decoder = hidden_size_decoder
        self.latent_size = latent_size

        self.encoder = nn.ModuleDict()
        for modality in self.modalities:
            self.encoder.update({modality: EncoderTemplate(feature_dimensions[modality],
                                                           self.hidden_size_encoder[modality],
                                                           self.latent_size, *args, **kvargs)})

        self.decoder = nn.ModuleDict()
        for modality in self.modalities:
            self.decoder.update({modality: DecoderTemplate(self.latent_size, self.hidden_size_decoder[modality],
                                                           feature_dimensions[modality])})

    def forward(self, x):
        """
        Returns:
            z_mu: dictionary mapping modalities names to mean layer out.
            z_logvar: dictionary mapping modalities names to variance layer out.
            x_recon: dictionary mapping modalities names to decoder out.
            z_noize: dictionary mapping modalities names to latent space representation.
        """
        z_mu = {}
        z_logvar = {}
        x_recon = {}
        z_noize = {}

        for modality in self.modalities:
            z_mu[modality], z_logvar[modality], z_noize[modality] = self.encoder[modality](x[modality])

            x_recon[modality] = self.decoder[modality](z_noize[modality])

        return x_recon, z_mu, z_logvar, z_noize
