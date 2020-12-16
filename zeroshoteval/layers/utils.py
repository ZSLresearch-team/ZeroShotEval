from torch import nn as nn


def weights_init(m):
    """
    Weight init.

    To do:
        -Try another gain(1.41 )
        -Try another weight initialization methods.
    """
    class_name = m.__class__.__name__
    if class_name.find("Linear") != -1:
        m.bias.data.fill_(0)
        nn.init.xavier_uniform_(m.weight, gain=0.5)

    elif class_name.find("BatchNorm") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
