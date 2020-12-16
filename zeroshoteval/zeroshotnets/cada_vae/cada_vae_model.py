from typing import Any

from torch.nn import ModuleDict
from torch.nn.modules.module import Module

from zeroshoteval.layers.vae_encoder import EncoderTemplate
from zeroshoteval.layers.vae_decoder import DecoderTemplate


class VAEModel(Module):
    """
    Model performs CADA-VAE ZSL approach

    .. _CADA-VAE arxiv papper:
        https://arxiv.org/pdf/1812.01784.pdf
    """

    def __init__(
        self,
        hidden_size_encoder,
        hidden_size_decoder,
        latent_size,
        feature_dimensions,
        *args,
        **kwargs,
    ):
        """
        Args:
            latent_size(int): size of models latent space
            modalities: list of modalities to be used
            feature_dimensions(dict): dictionary mapping modalities names to
                modalities embedding size.
        """
        super(VAEModel, self).__init__()
        self.modalities = ["IMG", "CLSATTR"]
        self.hidden_size_encoder = hidden_size_encoder
        self.hidden_size_decoder = hidden_size_decoder
        self.latent_size = latent_size

        self.encoder = ModuleDict()
        for modality in self.modalities:
            self.encoder.update(
                {
                    modality: EncoderTemplate(
                        feature_dimensions[modality],
                        self.hidden_size_encoder[modality],
                        self.latent_size,
                        *args,
                        **kwargs,
                    )
                }
            )

        self.decoder = ModuleDict()
        for modality in self.modalities:
            self.decoder.update(
                {
                    modality: DecoderTemplate(
                        self.latent_size,
                        self.hidden_size_decoder[modality],
                        feature_dimensions[modality],
                    )
                }
            )

    def forward(self, x):
        """
        Returns:
            z_mu(dict): dictionary mapping modalities names to mean layer out.
            z_logvar(dict): dictionary mapping modalities names to variance
                layer out.
            x_recon(dict): dictionary mapping modalities names to decoder out.
            z_sample(dict): dictionary mapping modalities names to latent space
                representation sample.
        """
        z_mu = {}
        z_logvar = {}
        x_recon = {}
        z_sample = {}

        for modality in self.modalities:
            z_mu[modality], z_logvar[modality], z_sample[modality] = self.encoder[modality](x[modality])

            x_recon[modality] = self.decoder[modality](z_sample[modality])

        return x_recon, z_mu, z_logvar, z_sample
