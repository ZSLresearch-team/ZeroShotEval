""""""

# region IMPORTS
import torch
import torch.nn as nn
# endregion


def weights_init(m):
    """
    Weight init.

    To do:
        -Try another gain(1.41 )
        -Try another weight initialization methods.
    """
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.bias.data.fill_(0)
        nn.init.xavier_uniform_(m.weight, gain=0.7)

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

    def __init__(self, input_dim, hidden_size_rule, output_dim, use_bn=True, use_dropout=False):
        super(EncoderTemplate, self).__init__()

        self.layer_sizes = [input_dim] + hidden_size_rule + [output_dim]

        modules = []
        for i in range(len(self.layer_sizes)-2):
            if use_dropout and i > 0:
                modules.append(nn.Dropout(p=0.3))
            modules.append(nn.Linear(self.layer_sizes[i], self.layer_sizes[i+1]))
            if use_bn:
                modules.append(nn.BatchNorm1d(self.layer_sizes[i+1]))
            modules.append(nn.ReLU())

        self.feature_encoder = nn.Sequential(*modules)

        self.mu = nn.Linear(self.layer_sizes[-2], self.layer_sizes[-1])
        self.var = nn.Linear(self.layer_sizes[-2], self.layer_sizes[-1])

        self.apply(weights_init)

    def forward(self, x):
        hidden = self.feature_encoder(x)

        z_mu = self.mu(hidden)
        z_var = self.var(hidden)

        std = z_var.exp()
        eps = torch.randn_like(std)
        z_noize = eps * std + z_mu

        return z_mu, z_var, z_noize


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
