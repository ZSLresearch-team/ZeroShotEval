from torch import nn
from zeroshoteval.layers.utils import weights_init


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
        for i in range(len(self.layer_sizes) - 2):
            modules.append(
                nn.Linear(self.layer_sizes[i], self.layer_sizes[i + 1])
            )
            modules.append(nn.ReLU())

        modules.append(nn.Linear(self.layer_sizes[-2], self.layer_sizes[-1]))
        self.feature_decoder = nn.Sequential(*modules)

        self.apply(weights_init)

    def forward(self, x):
        """
        Returns:
            Decoder out
        """

        return self.feature_decoder(x)
