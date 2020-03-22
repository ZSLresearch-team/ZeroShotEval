"""
"""
import torch
import torch.nn as nn
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader

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
        nn.init.xavier_uniform_(m.weight, gain=1)

    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def train_cls(classifier, optimizer, device, n_epoch, num_seen, num_unseen,
              train_loader, test_seen_loader, test_unseen_loader):
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

    print("Train Classifier")
    tqdm_epoch = trange(n_epoch, desc='Accuracy Seen: None. Unseen: None. H:', unit='epoch', disable=False, leave=True)

    for epoch in tqdm_epoch:
        classifier.train()

        loss_accum = 0
        correct_seen = 0
        correct_unseen = 0 

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

        classifier.eval()
        # Test seen classes
        with torch.no_grad():
            for _, (x, y) in enumerate(test_seen_loader):
                x = x.to(device)
                y = y.to(device)

                predictions = classifier(x)
                correct_seen += (predictions.argmax(dim=1) == y).sum().item()
            # Test unseen classes
            for _, (x, y) in enumerate(test_unseen_loader):
                x = x.to(device)
                y = y.to(device)

                predictions = classifier(x)
                correct_unseen += (predictions.argmax(dim=1) == y).sum().item()

        # Calculate accuracies
        acc_seen = correct_seen / num_seen
        acc_unseen = correct_unseen / num_unseen
        # To be reworked
        if (acc_unseen < 1e-4) or (acc_seen < 1e-4):
            acc_H = 0
        else:
            acc_H = (2 * acc_seen * acc_unseen) / (acc_seen + acc_unseen)

        acc_seen_hist.append(acc_seen)
        acc_unseen_hist.append(acc_unseen)
        acc_H_hist.append(acc_H)

        tqdm_epoch.set_description(f'Seen: {acc_seen:.2f}. Unseen: {acc_unseen:.2f}. H: {acc_H:.2f}')
        tqdm_epoch.refresh()

    return loss_hist, acc_seen_hist, acc_unseen_hist, acc_H_hist

def compute_val_metrics(model, loader):
    pass

def compute_accuracy(model, loader):
    """
    Computes accuracy on the dataset wrapped in a loader

    Returns: accuracy as a float value between 0 and 1
    """
    model.eval()

    correct_samples = 0
    total_samples = 0
    for i_step, (x, y) in enumerate(loader):
        prediction = model(x)
        _, indices = torch.max(prediction, 1)
        correct_samples += torch.sum(indices == y)
        total_samples += y.shape[0]

    return float(correct_samples) / total_samples

def classification_procedure(data, in_features, num_classes, batch_size, device, n_epoch,
                             lr, train_indicies, test_seen_indicies, test_unseen_indicies):

    classifier = SoftmaxClassifier(in_features, num_classes)

    num_seen = len(test_seen_indicies)
    num_unseen = len(test_unseen_indicies)

    train_sampler = SubsetRandomSampler(train_indicies)
    test_seen_sampler = SubsetRandomSampler(test_seen_indicies)
    test_unseen_sampler = SubsetRandomSampler(test_unseen_indicies)

    train_loader = DataLoader(data, batch_size=batch_size, sampler=train_sampler)
    test_seen_loader = DataLoader(data, batch_size=batch_size, sampler=test_seen_sampler)
    test_unseen_loader = DataLoader(data, batch_size=batch_size, sampler=test_unseen_sampler)

    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr, betas=(0.5, 0.999))

    train_loss_hist, acc_seen_hist, acc_unseen_hist, acc_H_hist = \
        train_cls(classifier, optimizer, device, n_epoch, num_seen, num_unseen,
                  train_loader, test_seen_loader, test_unseen_loader)

    print(f'Best accuracy H: {max(acc_H_hist)}')

    return train_loss_hist, acc_seen_hist, acc_unseen_hist, acc_H_hist
