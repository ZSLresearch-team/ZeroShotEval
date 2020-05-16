"""
"""
import torch
import torch.nn as nn
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader

import numpy as np
from sklearn.metrics import confusion_matrix
from tqdm import trange

class SoftmaxClassifier(nn.Module):
    """
    Simple softmax classifier
    """

    def __init__(self, cls_in, num_classes):
        super(SoftmaxClassifier, self).__init__()
        self.fc = nn.Linear(cls_in, num_classes)

        self.apply(weights_init)

    def forward(self, x):
        x = self.fc(x)
        return x

def weights_init(m):
    """
    Weight init.

    To do:
        -Try another gain(1.41 )
        -Try 
    """
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.bias.data.fill_(0)
        nn.init.xavier_uniform_(m.weight, gain=0.5)

    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def train_cls(classifier, optimizer, device, n_epoch, num_classes, seen_classes,
              unseen_classes, train_loader, test_loader, verbose=1):
    """
    Train Softmax classifier model

    Args:
        Classifier: classifier model to train.
        optimizer: optimizer to use.
        device: device to use.
        n_epoch: number of train epochs.
        num_seen(int): number of test seen objects.
        num_unseen(int): number of test unseen objects
        train_loader: loaer of the train data.
        test_seen_loader: loader of the test seen data.
        test_unseen_loader: loader of the test unseen data.
        verbose: boolean or Int. The higher value verbose is - the more info you get.

    Returns:
        loss_hist(list): train loss history.
        acc_seen_hist(list): accuracy for seen classes.
        acc_unseen_hist(list): accuracy for unseen classes.
        acc_H_hist(list): harmonic mean of seen and unseen accuracies.
    """
    classifier.to(device)

    loss_hist = []
    acc_seen_hist = []
    acc_unseen_hist = []
    acc_H_hist = []

    if verbose >= 1:
        print("Train Classifier")

    tqdm_epoch = trange(n_epoch, desc='Accuracy Seen: None. Unseen: None. H', unit='epoch', disable=(verbose<=0), leave=True)

    for epoch in tqdm_epoch:
        classifier.train()

        loss_accum = 0


        for i_step, (x, y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            predictions = classifier(x)
            loss = nn.functional.cross_entropy(predictions, y)
            loss.backward()
            optimizer.step()

            loss_accum += loss.item()

        loss_hist.append(loss_accum / i_step)

        # Calculate accuracies
        acc_seen, acc_unseen, acc_H = compute_mean_per_class_accuracies(classifier, test_loader, seen_classes,
                                                                        unseen_classes, device)
        # To be reworked
        acc_seen_hist.append(acc_seen)
        acc_unseen_hist.append(acc_unseen)
        acc_H_hist.append(acc_H)

        tqdm_epoch.set_description(f'Seen: {acc_seen:.2f}. Unseen: {acc_unseen:.2f}. H: {acc_H:.2f}')
        tqdm_epoch.refresh()

    return loss_hist, acc_seen_hist, acc_unseen_hist, acc_H_hist

def compute_mean_per_class_accuracies(classifier, loader, seen_classes, unseen_classes, device):
    """
    Computes mean per-class accuracies for both seen and unseen classes.

    Args:
        classifier: classifaer model to eval.
        loader(Dataloader): data loader.
        seen_classes(numpy array): labels of seen classes.
        unseen_classes(numpy array): labels of unseen classes.
        device(String): device to use.

    Returns:
        acc_seen: mean per-class accuracy for seen classes.
        acc_unseen: mean per-class accuracy for unseen classes.
        acc_H: harmonic mean between mean per-class accuracies for seen classes anduanseen classes.
    """

    classifier.eval()
    labels_all = torch.Tensor().long().to(device)
    preds_all = torch.Tensor().long().to(device)
    with torch.no_grad():
        for _, (x, y) in enumerate(loader):
            x = x.to(device)
            y = y.to(device)

            probs = classifier(x)
            _, preds = torch.max(probs, 1)

            labels_all = torch.cat((labels_all, y), 0)
            preds_all = torch.cat((preds_all, preds.long()), 0)

    conf_matrix = confusion_matrix(labels_all.cpu().numpy(), preds_all.cpu().numpy())
    acc_seen = (np.diag(conf_matrix)[seen_classes] / conf_matrix.sum(1)[seen_classes]).mean()
    acc_unseen = (np.diag(conf_matrix)[unseen_classes] / conf_matrix.sum(1)[unseen_classes]).mean()

    if (acc_unseen < 1e-4) or (acc_seen < 1e-4):
        acc_H = 0
    else:
        acc_H = (2 * acc_seen * acc_unseen) / (acc_seen + acc_unseen)

    return acc_seen, acc_unseen, acc_H

def classification_procedure(data, in_features, num_classes, batch_size, device, n_epoch, lr,
                             train_indicies, test_indicies, seen_classes, unseen_classes, verbose):
    """
    Launches classifier training.

    Args:
        data:
        in_features:
        num_classes:
        batch_size(Int): batch size for classifier training.
        device(str): model device.
        n_epoch(int): number of epochs to train.
        lr(float): learning rate.
        train_indicies: indicies of training data.
        test_indices: inicies of testing data.
        seen_classes(numpy array): labels of seen classes.
        unseen_classes(numpy array): labels of unseen classes.
        verbose: boolean or Int. The higher value verbose is - the more info you get.
    
    Returns:
        loss_hist(list): train loss history.
        acc_seen_hist(list): accuracy for seen classes.
        acc_unseen_hist(list): accuracy for unseen classes.
        acc_H_hist(list): harmonic mean of seen and unseen accuracies.
    """
    classifier = SoftmaxClassifier(in_features, num_classes)

    train_sampler = SubsetRandomSampler(train_indicies)
    test_sampler = SubsetRandomSampler(test_indicies)

    train_loader = DataLoader(data, batch_size=batch_size, sampler=train_sampler)
    test_loader = DataLoader(data, batch_size=batch_size, sampler=test_sampler)

    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr, betas=(0.9, 0.999))

    train_loss_hist, acc_seen_hist, acc_unseen_hist, acc_H_hist = \
        train_cls(classifier, optimizer, device, n_epoch, num_classes, seen_classes, 
                  unseen_classes, train_loader, test_loader, verbose=verbose)

    print(f'Best accuracy H: {max(acc_H_hist)}')

    return train_loss_hist, acc_seen_hist, acc_unseen_hist, acc_H_hist
