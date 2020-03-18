""""""

# region IMPORTS
import torch
import torch.nn as nn
# endregion


def weights_init(m):
    """Weight init."""
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.bias.data.fill_(0)
        nn.init.xavier_uniform_(m.weight, gain=0.5)

    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class EncoderTemplate(nn.Module):
    """
    Encoder part for CADA-VAE model.

    Args:
        input_dim: size of input layer
        hidden size_rule: list of sizes of hidden layers
        output_dim: size of output layer
    """

    def __init__(self, input_dim, hidden_size_rule, output_dim):
        super(EncoderTemplate, self).__init__()

        self.layer_sizes = [input_dim] + hidden_size_rule + [output_dim]

        modules = []
        for i in range(len(self.layer_sizes)-2):

            modules.append(
                nn.Linear(self.layer_sizes[i], self.layer_sizes[i+1]))
            modules.append(nn.ReLU())

        self.feature_encoder = nn.Sequential(*modules)

        self._mu = nn.Linear(
            in_features=self.layer_sizes[-2], out_features=output_dim)

        self._var = nn.Linear(
            in_features=self.layer_sizes[-2], out_features=output_dim)

        self.apply(weights_init)

    def forward(self, x):
        hidden = self.feature_encoder(x)
        z_mu = self._mu(hidden)
        z_var = self._var(hidden)

        return z_mu, z_var


class DecoderTemplate(nn.Module):
    """Decoder part for CADA-VAE model.

    Args:
        input_dim: size of input layer
        hidden_size_rule: list of sizes of hidden layers
        output_dim: size of output layer
    """

    def __init__(self, input_dim, hidden_size_rule, output_dim):
        super(DecoderTemplate, self).__init__()

        self.layer_sizes = [input_dim] + hidden_size_rule + [output_dim]

        # self.feature_decoder = nn.Sequential(
        #                                      nn.Linear(input_dim, self.layer_sizes[1]),
        #                                      nn.ReLU(), nn.Linear(self.layer_sizes[1], output_dim))
        modules = []
        for i in range(len(self.layer_sizes)-2):
            modules.append(
                nn.Linear(self.layer_sizes[i], self.layer_sizes[i+1]))
            modules.append(nn.ReLU())

        modules.append(nn.Linear(self.layer_sizes[-2], self.layer_sizes[-1]))
        self.feature_decoder = nn.Sequential(*modules)

        self.apply(weights_init)

    def forward(self, x):
        return self.feature_decoder(x)
