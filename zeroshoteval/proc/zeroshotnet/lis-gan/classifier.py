import numpy as np
import torch
from torch import nn as nn
import torch.optim as optim


class LogSoftmaxClassifier(nn.Module):
    def __init__(self, input_dim, nclass):
        super(LogSoftmaxClassifier, self).__init__()
        self.fc = nn.Linear(input_dim, nclass)
        self.logic = nn.LogSoftmax(dim=1)
    def forward(self, x): 
        o = self.logic(self.fc(x))
        return o  

def weights_init(m):
    """
    Weight init.
    """
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        m.bias.data.fill_(0)
        nn.init.xavier_uniform_(m.weight, gain=0.5)

    elif classname.find("BatchNorm") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class Classifier_train:
    def __init__(self, device, num_epoch, input_dim, num_classes, train_loader, lr=0.001, beta1=0.5, pretrain_classifer=''):
        self.train_loader =  train_loader 
        self.num_epoch = num_epoch
        self.num_classes = num_classes
        self.device = device
        self.input_dim = input_dim
        
        self.model = LogSoftmaxClassifier(self.input_dim, self.num_classes)
        self.model.apply(weights_init)
        self.criterion = nn.NLLLoss()
        self.model.to(device)
        self.model = self.model.double()
        self.lr = lr
        self.beta1 = beta1
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, betas=(beta1, 0.999))

    
    def fit(self):
        loss_hist = []
        for _epoch in range(self.num_epoch):
            self.model.train()
            for i_step, (x, y) in enumerate(self.train_loader):
                x = x['img'].to(self.device)
                y = y.to(self.device)

                self.optimizer.zero_grad()

                predictions =  self.model(x.double())
                loss =  self.criterion(predictions, y)
                loss.backward()
                self.optimizer.step()

                loss_accum += loss.item()

            loss_accum_mean = loss_accum / (i_step + 1)
            loss_hist.append(loss_accum_mean)
        return loss_hist


    def get_loss(self, input_embeddings, input_label):
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(input_embeddings.double())
            loss = nn.NLLLoss(reduction='sum')(predictions, input_label)
        return loss.item()
    
    def save_trained_classifier(self, path_to_save):
        pass
    
    def load_trained_classifier(self, path_to_classifier)