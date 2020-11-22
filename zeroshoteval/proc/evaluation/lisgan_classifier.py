import numpy as np
import torch
from torch import nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.sampler import SubsetRandomSampler
from scipy.stats import entropy
from zeroshoteval.utils.misc import RNG_seed_setup, log_model_info
from zeroshoteval.utils.optimizer_helper import build_optimizer

import logging

logger = logging.getLogger(__name__)

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
    """
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        m.bias.data.fill_(0)
        nn.init.xavier_uniform_(m.weight, gain=0.5)

    elif classname.find("BatchNorm") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def calculate_acc(true_label, pred_label, seen_classes, unseen_classes):
    """
    Computes accuracies for both seen and unseen classes and their harmonic mean.
    Args:
        true_label(tensor): classifier model to eval.
        pred_label(tensor): data loader.
        seen_classes(numpy array): labels of seen classes.
        unseen_classes(numpy array): labels of unseen classes.
    Returns:
        acc_seen: mean per-class accuracy for seen classes.
        acc_unseen: mean per-class accuracy for unseen classes.
        acc_H: harmonic mean between mean per-class accuracies for seen classes
        anduanseen classes.
    """
    num_seen, num_true_seen, num_unseen, num_true_unseen = 0, 0, 0, 0
    for true, pred in zip(true_label.cpu().numpy(), pred_label.cpu().numpy()):
        if true in seen_classes:
            num_seen += 1
            num_true_seen += int(true == pred)
        else:
            num_unseen += 1
            num_true_unseen += int(true == pred)
    if num_unseen == 0:
        acc_unseen = 1
    else:
        acc_unseen = num_true_unseen / num_unseen
    if num_seen == 0:
        acc_seen = 1
    else:
        acc_seen = num_true_seen / num_seen
    if (acc_unseen < 1e-4) or (acc_seen < 1e-4):
        acc_H = 0
    elif  (num_seen == 0) or (num_unseen == 0):
        acc_H = min(acc_seen, acc_unseen)
    else:
        acc_H = 2 * acc_seen * acc_unseen / (acc_seen + acc_unseen)
    return acc_seen, acc_unseen, acc_H

def train_classifier(classifier, optimizer, n_epoch, dataset, gzsl,
          train_indices, test_indices, ratio, device, batch_size, seen_classes, 
          unseen_classes):
    """
    Training classifier procedure.
    Args:
        classifier(nn.Module): classifier model to eval.
        optimizer: optimizer for training.
        n_epoch(int): number epoch for training.
        dataset(torch.Dataset): dataset with test and trainig embeddings/labels.
        gzsl(bool): if true then evaluate in gzsl mode, else in zsl mode.
        train_indices(np.array): indices train samples in dataset.
        test_indices(np.array): indices test samples in dataset.
        ratio(float): part of test samples for retrainig classifier. 
        device(str): device to be used.
        batch_size(int): batch size for training classifier.
    Returns:
        loss_hist(dict): dictionary containe accuracy, loss, H for each epoch.
    """
    logger.info("Training Final classifier\n")
    all_test_feature = dataset.tensors[0][test_indices]
    all_test_label = dataset.tensors[1][test_indices]
    test_sampler = SubsetRandomSampler(test_indices)
    test_loader = DataLoader(dataset, batch_size=batch_size, 
                             sampler=test_indices)

    loss_hist = []
    acc_unseen_hist = []
    if gzsl:
        best_H = 0
        acc_seen_hist = []
        acc_H_hist = []
    else:
        best_unseen_acc = 0

    for _epoch in range(n_epoch):
        train_sampler = SubsetRandomSampler(train_indices)
        train_loader = DataLoader(dataset, batch_size=batch_size,
                                  sampler=train_sampler)
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

        true_label, pred_label, output_prob = validate_classifier(classifier, 
                                                       test_loader, 
                                                       device)
        acc_seen, acc_unseen, acc_H = calculate_acc(true_label, 
                                                    pred_label, 
                                                    seen_classes, 
                                                    unseen_classes)
        acc_seen_hist.append(acc_seen)
        acc_unseen_hist.append(acc_unseen)
        acc_H_hist.append(acc_H)

        if gzsl:
            logger.info(
            f"Epoch: {_epoch+1} "
            f"Loss: {loss_accum_mean:.1f} Seen: {acc_seen:.2f} "
            f"Unseen: {acc_unseen:.2f} H: {acc_H:.2f}")

            if acc_H >= best_H:
                best_H = acc_H
                best_all_pred = pred_label
                best_all_output = output_prob
        else:
            logger.info(
            f"Epoch: {_epoch+1} "
            f"Loss: {loss_accum_mean:.1f} Unseen: {acc_unseen:.2f} ")

            if acc_unseen >= best_acc_unseen:
                best_acc_unseen = acc_unseen
                best_all_pred = pred_label
                best_all_output = output_prob

    test_easy_length = int(len(test_indices) * ratio)
    test_hard_length = len(test_indices) - test_easy_length
    entropy_value = torch.Tensor(list(map(entropy, best_all_output.cpu())))
    indices = torch.sort(-entropy_value)[1]
    exit_indices = indices[:test_easy_length]
    keep_indices = indices[test_easy_length:]
    best_easy_pred = best_all_pred[exit_indices]
    best_easy_label = all_test_label[exit_indices]
    best_hard_pred = best_all_pred[keep_indices]
    best_hard_label = all_test_label[keep_indices]
    all_easy_hard_label = torch.cat((best_easy_label, best_hard_label), 0)

    acc_seen_easy, acc_unseen_easy, acc_H_easy = calculate_acc(best_easy_label, 
                                                               best_easy_pred,
                                                               seen_classes, 
                                                               unseen_classes)
    acc_seen_hard, acc_unseen_hard, acc_H_hard = calculate_acc(best_hard_label, 
                                                               best_hard_pred, 
                                                               seen_classes, 
                                                               unseen_classes)
    
    if gzsl:
        logger.info(
            f"Easy score Seen: {acc_seen_easy:.2f} Unseen: {acc_unseen_easy:.2f} H: {acc_H_easy:.2f} "
            f"Hard score Seen: {acc_seen_hard:.2f} Unseen: {acc_unseen_hard:.2f} H: {acc_H_hard:.2f} ")
    else:
        logger.info(
            f"Easy score Unseen: {acc_unseen_easy:.2f} "
            f"Hard score Unseen: {acc_unseen_hard:.2f} ")

    train_features  = dataset.tensors[0][train_indices]
    train_labels =  dataset.tensors[1][train_indices]
    extended_features = torch.cat((train_features, 
                                   all_test_feature[exit_indices]), 0)
    extended_labels = torch.cat((train_labels, best_easy_label), 0)
    extended_dataset = TensorDataset(extended_features, extended_labels)
    hard_test_loader = DataLoader(dataset, 
                                  batch_size=batch_size,
                                  sampler=test_indices[keep_indices])

    loss_hist = []
    acc_unseen_hist = []
    if gzsl:
        best_hard_H = 0
        acc_seen_hist = []
        acc_H_hist = []
    else:
        best_acc_unseen = 0

    for epoch in range(n_epoch):
        train_loader = DataLoader(extended_dataset, batch_size=batch_size, 
                                  shuffle=True)
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

        hard_true_label, hard_pred_label, _= validate_classifier(classifier, 
                                                      hard_test_loader,
                                                      device)
        hard_acc_seen, hard_acc_unseen, hard_acc_H = calculate_acc(hard_true_label, 
                                                                   hard_pred_label, 
                                                                   seen_classes, 
                                                                   unseen_classes)
        acc_seen_hist.append(hard_acc_seen)
        acc_unseen_hist.append(hard_acc_unseen)
        acc_H_hist.append(hard_acc_H)

        if gzsl:
            logger.info(
            f"Epoch: {epoch+1} "
            f"Loss: {loss_accum_mean:.1f} Seen: {hard_acc_seen:.2f} "
            f"Unseen: {hard_acc_unseen:.2f} H: {hard_acc_H:.2f}")

            if hard_acc_H >= best_hard_H:
                best_hard_H = hard_acc_H
                best_hard_pred = hard_pred_label
                best_hard_labels = hard_true_label
        else:
            logger.info(
            f"Epoch: {_epoch+1} "
            f"Loss: {loss_accum_mean:.1f} Unseen: {hard_acc_unseen:.2f} ")

            if hard_acc_unseen >= best_acc_unseen:
                best_acc_unseen = hard_acc_unseen
                best_hard_pred = hard_pred_label
                best_hard_labels = hard_true_label
            
    all_pred = torch.cat((best_easy_pred.cpu(), best_hard_pred.cpu()), 0)
    all_true_labels = torch.cat((best_easy_label.cpu(), best_hard_labels.cpu()), 0)
    final_acc_seen, final_acc_unseen, final_acc_H = calculate_acc(all_true_labels,
                                                                  all_pred, 
                                                                  seen_classes, 
                                                                  unseen_classes)
    if gzsl:
        logger.info(
            f"Final score Seen: {final_acc_seen:.2f} Unseen: {final_acc_unseen:.2f} H: {final_acc_H:.2f} ")
        return acc_seen_hist, acc_unseen_hist, acc_H_hist
    else:
        logger.info(
            f"Final score Unseen: {final_acc_unseen:.2f} ")
        return acc_unseen_hist

def validate_classifier(classifier, loader, device):
    """
    Calculate classifier output and prediction.
    Args:
        classifier(nn.Module): classifier model to eval.
        loader(Dataloader): data loader for prediction.
        device(String): device to use.
    Returns:
        labels_all(tensor): true labels.
        preds_all(tensor): predicted labels.
        output_all(tensor): output tensor probability from classifier net.
    """
    classifier.eval()
    labels_all = torch.Tensor().long().to(device)
    preds_all = torch.Tensor().long().to(device)
    output_all =  torch.Tensor().long().to(device)
    with torch.no_grad():
        for _, (x, y) in enumerate(loader):
            x = x.to(device)
            y = y.to(device)

            probs = classifier(x)
            _, preds = torch.max(probs, 1)

            labels_all = torch.cat((labels_all, y), 0)
            preds_all = torch.cat((preds_all, preds.long()), 0)
            output_all = torch.cat((output_all, probs), 0)
    return labels_all, preds_all, output_all

def classification_procedure(
    cfg,
    data,
    in_features,
    num_classes,
    train_indicies,
    test_indicies,
    seen_classes,
    unseen_classes
):
    """
    Launches classifier training.
    Args:
        cfg(CfgNode): configs. Details can be found in
            zeroshoteval/config/defaults.py
        data(Dataset): torch tensor dataset for ZSL embeddings.
        in_features(int): number of input features.
        num_classes(int): number of classes.
        train_indicies: indicies of training data.
        test_indices: inicies of testing data.
        seen_classes(numpy array): labels of seen classes.
        unseen_classes(numpy array): labels of unseen classes.
    Returns:
        loss_hist(list): train loss history.
        acc_seen_hist(list): accuracy for seen classes.
        acc_unseen_hist(list): accuracy for unseen classes.
        acc_H_hist(list): harmonic mean of seen and unseen accuracies.
    """

    RNG_seed_setup(cfg)
    logger.info("Building final lisgan-classifier model")
    classifier = SoftmaxClassifier(in_features, num_classes)
    classifier.to(cfg.DEVICE)
    classifier = classifier.float()
    log_model_info(classifier, "Final Classifier")

    optimizer = build_optimizer(classifier, cfg, "CLS")
    #optimizer = optim.Adam(classifier.parameters(), lr=0.001, betas=(0.5, 0.999))

    acc_seen_hist, acc_unseen_hist, acc_H_hist = train_classifier(classifier, 
                                                                  optimizer, 
                                                                  cfg.CLS.EPOCH,
                                                                  data, 
                                                                  cfg.GENERALIZED, 
                                                                  train_indicies, 
                                                                  test_indicies, 
                                                                  cfg.CLS.RATIO,
                                                                  cfg.DEVICE, 
                                                                  cfg.CLS.BATCH_SIZE, 
                                                                  seen_classes, 
                                                                  unseen_classes)

    best_H_idx = acc_H_hist.index(max(acc_H_hist))

    logger.info(
        "Results:\n"
        f"Best accuracy H: {acc_H_hist[best_H_idx]:.4f}, "
        f"Seen: {acc_seen_hist[best_H_idx]:.4f}, "
        f"Unseen: {acc_unseen_hist[best_H_idx]:.4f}"
    )

    return acc_seen_hist, acc_unseen_hist, acc_H_hist