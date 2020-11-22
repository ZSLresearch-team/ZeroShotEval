import torch.nn as nn
import torch

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class Model_discriminator(nn.Module):
    def __init__(self, sample_feature_size, class_feachere_size, hidden_layer_size): 
        super(Model_discriminator, self).__init__()
        self.fc1 = nn.Linear(sample_feature_size + class_feachere_size, hidden_layer_size)
        self.fc2 = nn.Linear(hidden_layer_size, 1)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.apply(weights_init)

    def forward(self, x, class_feature):
        h = torch.cat((x, class_feature), 1) 
        h = self.lrelu(self.fc1(h))
        h = self.fc2(h)
        return h


class Model_generator(nn.Module):
    def __init__(self, sample_feature_size, class_feachere_size, hidden_layer_size, 
                noise_size):
        super(Model_generator, self).__init__()
        self.fc1 = nn.Linear(class_feachere_size + noise_size, hidden_layer_size)
        self.fc2 = nn.Linear(hidden_layer_size, sample_feature_size)
        self.lrelu = nn.LeakyReLU(0.2, True)
        #self.prelu = nn.PReLU()
        self.relu = nn.ReLU(True)
        self.apply(weights_init)

    def forward(self, noise, class_feature):
        h = torch.cat((noise, class_feature), 1)
        h = self.lrelu(self.fc1(h))
        sample = self.relu(self.fc2(h))
        return sample