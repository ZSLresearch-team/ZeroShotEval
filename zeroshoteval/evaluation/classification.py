"""
"""
from typing import Tuple, Optional, List

import numpy as np
import torch
from fvcore.common.config import CfgNode
from sklearn.metrics import confusion_matrix
from torch import nn as nn, Tensor
from torch.utils.data import DataLoader
from torch.utils.data.dataset import TensorDataset
from torch.utils.data.sampler import SubsetRandomSampler

from zeroshoteval.utils.misc import setup_random_number_generator_seed, log_model_info
from zeroshoteval.solver.optimizer_helper import build_optimizer
from zeroshoteval.utils.checkpoint import load_embeddings

import logging

logger = logging.getLogger(__name__)


class LINEAR_LOGSOFTMAX(nn.Module):
    def __init__(self, input_dim, nclass):
        super(LINEAR_LOGSOFTMAX, self).__init__()
        self.fc = nn.Linear(input_dim,nclass)
        self.logic = nn.LogSoftmax(dim=1)
        self.lossfunction =  nn.NLLLoss()

    def forward(self, x):
        o = self.logic(self.fc(x))
        return o


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
    if classname.find("Linear") != -1:
        m.bias.data.fill_(0)
        nn.init.xavier_uniform_(m.weight, gain=0.5)

    elif classname.find("BatchNorm") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def train_cls(
        classifier,
        optimizer,
        device,
        n_epoch,
        seen_classes,
        unseen_classes,
        train_loader,
        test_loader,
):
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

    logger.info("Training Final classifier\n")

    for _epoch in range(n_epoch):
        classifier.train()
        loss_accum = 0

        for _i_step, (x, y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            predictions = classifier(x)
            loss = nn.functional.cross_entropy(predictions, y)
            loss.backward()
            optimizer.step()

            loss_accum += loss.item()

        loss_accum_mean = loss_accum / (_i_step + 1)
        loss_hist.append(loss_accum_mean)

        # Calculate accuracies
        acc_seen, acc_unseen, acc_H = compute_mean_per_class_accuracies(
            classifier, test_loader, seen_classes, unseen_classes, device
        )
        # To be reworked
        acc_seen_hist.append(acc_seen)
        acc_unseen_hist.append(acc_unseen)
        acc_H_hist.append(acc_H)

        logger.info(
            f"Epoch: {_epoch + 1} "
            f"Loss: {loss_accum_mean:.1f} Seen: {acc_seen:.2f} "
            f"Unseen: {acc_unseen:.2f} H: {acc_H:.2f}"
        )

    return loss_hist, acc_seen_hist, acc_unseen_hist, acc_H_hist


def compute_mean_per_class_accuracies(
        classifier, loader, seen_classes, unseen_classes, device
):
    """
    Computes mean per-class accuracies for both seen and unseen classes.

    Args:
        classifier: classifier model to eval.
        loader(Dataloader): data loader.
        seen_classes(numpy array): labels of seen classes.
        unseen_classes(numpy array): labels of unseen classes.
        device(String): device to use.

    Returns:
        acc_seen: mean per-class accuracy for seen classes.
        acc_unseen: mean per-class accuracy for unseen classes.
        acc_H: harmonic mean between mean per-class accuracies for seen classes
        anduanseen classes.
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

    conf_matrix = confusion_matrix(
        labels_all.cpu().numpy(), preds_all.cpu().numpy()
    )
    acc_seen = (np.diag(conf_matrix)[seen_classes-1] / conf_matrix.sum(1)[seen_classes-1]).mean()
    acc_unseen = (np.diag(conf_matrix)[unseen_classes-1] / conf_matrix.sum(1)[unseen_classes-1]).mean()

    if (acc_unseen < 1e-4) or (acc_seen < 1e-4):
        acc_H = 0
    else:
        acc_H = (2 * acc_seen * acc_unseen) / (acc_seen + acc_unseen)

    return acc_seen, acc_unseen, acc_H


def classification_procedure(cfg: CfgNode,
                             train_data: Optional[Tuple[Tensor, Tensor]] = None,
                             test_data: Optional[Tuple[Tensor, Tensor]] = None) -> Tuple[List, List, List, List]:
    """
    Launches classifier training.

    Args:
        cfg(CfgNode): configs. Details can be found in
            zeroshoteval/config/defaults.py
        data(dict): dictionary with zsl dataset and another extra data.

    Returns:
        loss_hist(list): train loss history.
        acc_seen_hist(list): accuracy for seen classes.
        acc_unseen_hist(list): accuracy for unseen classes.
        acc_H_hist(list): harmonic mean of seen and unseen accuracies.
    """
    setup_random_number_generator_seed(cfg)

    if cfg.CLS.LOAD_DATA:
        # TODO: refactor embeddings saving/loading (see `saving`)
        data = load_embeddings(cfg)

    assert train_data is not None and test_data is not None, "Input data were not passed or loaded from disk."

    logger.info("Building final classifier model")
    classifier = SoftmaxClassifier(cls_in=train_data[0].size()[1],
                                   num_classes=train_data[1].unique().size()[0])
    classifier.to(cfg.DEVICE)
    log_model_info(classifier, "Final Classifier")

    # train_sampler = SubsetRandomSampler(data["train_indicies"])
    # test_sampler = SubsetRandomSampler(data["test_indicies"])

    num_seen_classes = 200
    num_unseen_classes = 50

    train_dataset = TensorDataset(train_data[0], train_data[1])
    test_dataset = TensorDataset(test_data[0], test_data[1])

    train_loader = DataLoader(dataset=train_dataset, batch_size=cfg.CLS.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=cfg.CLS.BATCH_SIZE, shuffle=True)

    optimizer = build_optimizer(model=classifier, cfg=cfg, procedure="CLS")

    train_loss_hist, acc_seen_hist, acc_unseen_hist, acc_H_hist = train_cls(
        classifier=classifier,
        optimizer=optimizer,
        device=cfg.DEVICE,
        n_epoch=cfg.CLS.EPOCH,
        seen_classes=num_seen_classes,
        unseen_classes=num_unseen_classes,
        train_loader=train_loader,
        test_loader=test_loader
    )

    best_H_idx = acc_H_hist.index(max(acc_H_hist))

    logger.info(
        "Results:\n"
        f"Best accuracy H: {acc_H_hist[best_H_idx]:.4f}, "
        f"Seen: {acc_seen_hist[best_H_idx]:.4f}, "
        f"Unseen: {acc_unseen_hist[best_H_idx]:.4f}"
    )

    return train_loss_hist, acc_seen_hist, acc_unseen_hist, acc_H_hist
