from typing import Tuple, Dict

import torch
from fvcore.common.config import CfgNode
from torch import Tensor
from torch.nn.modules.module import Module
from torch.utils.data.dataloader import DataLoader

from zeroshoteval.data.dataloader_helper import construct_loader
from zeroshoteval.utils import checkpoint
from zeroshoteval.zeroshotnets.build import ZSL_MODEL_REGISTRY


@ZSL_MODEL_REGISTRY.register()
def CADA_VAE_inference_procedure(cfg: CfgNode, model: Module) -> Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]]:
    """
    Generates synthetic dataset via trained zsl model to cls training

    Args:
        cfg(CfgNode): configs. Details can be found in
            zeroshoteval/config/defaults.py
        model: pretrained CADA-VAE model.

    Returns:
        zsl_emb_dataset: sythetic dataset for classifier.
        csl_train_indice: train indicies.
        csl_test_indice: test indicies.
    """

    # ================
    train_classifier(cfg, model)
    new_inference_procdure(cfg, model)
    # ================

    # Data loader building
    # --------------------
    train_loader = construct_loader(cfg, split="train")
    test_loader = construct_loader(cfg, split="test")

    train_zsl_data: Dict[str, Tuple[Tensor, Tensor]] = dict()
    test_zsl_data: Dict[str, Tuple[Tensor, Tensor]] = dict()

    # Generate ZSL embeddings for train (seen) images
    # -----------------------------------------------
    train_zsl_data["IMG"] = CADA_VAE_inference(model=model,
                                               data_loader=train_loader,
                                               modality="IMG",
                                               device=cfg.DEVICE)

    # Generate ZSL embeddings for additional modality (CLSATTR) to use for classifier TRAINING
    # (give classifier info about unseen classes with another modality)
    # ----------------------------------------------------------------------------------------
    test_zsl_data["CLSATTR"] = CADA_VAE_inference(model=model,
                                                  data_loader=test_loader,
                                                  modality="CLSATTR",
                                                  device=cfg.DEVICE)

    # Leave unseen instances only as additional info for the classifier
    test_embeddings, test_labels = test_zsl_data["CLSATTR"]

    test_embeddings = test_embeddings[test_loader.dataset.unseen_indexes]
    test_labels = test_labels[test_loader.dataset.unseen_indexes]

    test_zsl_data["CLSATTR"] = (test_embeddings, test_labels)

    # Generate ZSL embeddings for test data
    # -------------------------------------
    test_zsl_data["IMG"] = CADA_VAE_inference(model=model,
                                              data_loader=test_loader,
                                              modality="IMG",
                                              device=cfg.DEVICE,
                                              reparametrize_with_noise=False)

    # For Generalized ZSL setting leave only unseen indexes
    if not cfg.GENERALIZED:
        test_embeddings, test_labels = test_zsl_data["IMG"]

        test_embeddings = test_embeddings[test_loader.dataset.unseen_indexes]
        test_labels = test_labels[test_loader.dataset.unseen_indexes]

        test_zsl_data["IMG"] = (test_embeddings, test_labels)

    # Creation of the final data layout
    # ---------------------------------
    train_data: Tuple[Tensor, Tensor] = (
        torch.cat((train_zsl_data["IMG"][0], test_zsl_data["CLSATTR"][0]), dim=0),
        torch.cat((train_zsl_data["IMG"][1], test_zsl_data["CLSATTR"][1]), dim=0)
    )

    test_data: Tuple[Tensor, Tensor] = test_zsl_data["IMG"]

    if cfg.ZSL.SAVE_EMB:
        data: Tuple[Tensor, Tensor] = (
            torch.cat((train_data[0], test_data[0]), dim=0),
            torch.cat((train_data[1], test_data[1]), dim=0)
        )
        # TODO: change embeddings saving to saving of 3 files: embeddings, labels, splits
        checkpoint.save_embeddings(cfg.OUTPUT_DIR, data, cfg)

    return train_data, test_data


# TODO: move the function below to utils
def sample_train_data_on_sample_per_class_basis(features, label, sample_per_class):
    sample_per_class = int(sample_per_class)

    if sample_per_class != 0 and len(label) != 0:

        classes = label.unique()

        for i, s in enumerate(classes):

            features_of_that_class = features[label == s, :]  # order of features and labels must coincide
            # if number of selected features is smaller than the number of features we want per class:
            multiplier = torch.ceil(torch.cuda.FloatTensor(
                [max(1, sample_per_class / features_of_that_class.size(0))])).long().item()

            features_of_that_class = features_of_that_class.repeat(multiplier, 1)

            if i == 0:
                features_to_return = features_of_that_class[:sample_per_class, :]
                labels_to_return = s.repeat(sample_per_class)
            else:
                features_to_return = torch.cat(
                    (features_to_return, features_of_that_class[:sample_per_class, :]), dim=0)
                labels_to_return = torch.cat((labels_to_return, s.repeat(sample_per_class)),
                                                dim=0)

        return features_to_return, labels_to_return
    else:
        return torch.cuda.FloatTensor([]), torch.cuda.LongTensor([])


def new_inference_procdure(cfg, model):
    # Prepare test set (image IMG features only)
    # ------------------------------------------
    test_zsl_embeddings = dict()
    test_zsl_labels = dict()

    test_loader = construct_loader(cfg, split='test')
    test_zsl_embeddings['IMG'], test_zsl_labels['IMG'] = CADA_VAE_inference(model=model,
                                                                            data_loader=test_loader,
                                                                            modality="IMG",
                                                                            device=cfg.DEVICE,
                                                                            reparametrize_with_noise=False)

    # Prepare training set (images from train + additional info from attributes from test)
    # --------------------
    train_zsl_embeddings = dict()
    train_zsl_labels = dict()

    # LOAD
    train_loader = construct_loader(cfg, split='train', shuffle=False, drop_last=False)
    train_zsl_embeddings['IMG'], train_zsl_labels['IMG'] = CADA_VAE_inference(model=model,
                                                                              data_loader=train_loader,
                                                                              modality="IMG",
                                                                              device=cfg.DEVICE)

    z_seen_embeddings = dict()
    z_novel_embeddings = dict()
    z_seen_labels = dict()
    z_novel_labels = dict()

    # ----------------------------------

    # SPLIT (seen / unseen (novel))
    img_seen_feat = train_zsl_embeddings['IMG'][train_loader.dataset.seen_indexes]
    img_seen_label = train_zsl_labels['IMG'][train_loader.dataset.seen_indexes]

    img_unseen_feat = train_zsl_embeddings['IMG'][train_loader.dataset.unseen_indexes]
    img_unseen_label = train_zsl_labels['IMG'][train_loader.dataset.unseen_indexes]

    # ----------------------------------

    # RESAMPLE SEPARATELY
    z_seen_embeddings['IMG'], z_seen_labels['IMG'] = sample_train_data_on_sample_per_class_basis(
        img_seen_feat, img_seen_label, sample_per_class=200
    )  # TODO: remove hardcode 200 samples per class

    z_novel_embeddings['IMG'], z_novel_labels['IMG'] = sample_train_data_on_sample_per_class_basis(
        img_unseen_feat, img_unseen_label, sample_per_class=0
    )  # TODO: remove hardcode 0 samples per class

    # ----------------------------------

    # LOAD
    test_loader = construct_loader(cfg, split='test', shuffle=False, drop_last=False)
    test_zsl_embeddings['CLSATTR'], test_zsl_labels['CLSATTR'] = CADA_VAE_inference(model=model,
                                                                                    data_loader=test_loader,
                                                                                    modality="CLSATTR",
                                                                                    device=cfg.DEVICE)

    # ----------------------------------

    # SPLIT
    att_seen_feat = test_zsl_embeddings['CLSATTR'][test_loader.dataset.seen_indexes]
    att_seen_label = test_zsl_labels['CLSATTR'][test_loader.dataset.seen_indexes]

    att_unseen_feat = test_zsl_embeddings['CLSATTR'][test_loader.dataset.unseen_indexes]
    att_unseen_label = test_zsl_labels['CLSATTR'][test_loader.dataset.unseen_indexes]

    # ----------------------------------

    z_seen_embeddings['CLSATTR'], z_seen_labels['CLSATTR'] = sample_train_data_on_sample_per_class_basis(
        att_seen_feat, att_seen_label, sample_per_class=0
    )  # TODO: remove hardcode 0 samples per class

    z_novel_embeddings['CLSATTR'], z_novel_labels['CLSATTR'] = sample_train_data_on_sample_per_class_basis(
        att_unseen_feat, att_unseen_label, sample_per_class=400
    )  # TODO: remove hardcode 400 samples per class

    # ----------------------------------

    train_Z = [
        z_seen_embeddings['IMG'], 
        z_novel_embeddings['IMG'],
        z_seen_embeddings['CLSATTR'],
        z_novel_embeddings['CLSATTR']
        ]
    train_L = [
        z_seen_labels['IMG'],
        z_novel_labels['IMG'],
        z_seen_labels['CLSATTR'],
        z_novel_labels['CLSATTR']
        ]

    # empty tensors are sorted out
    train_X = [train_Z[i] for i in range(len(train_Z)) if train_Z[i].size(0) != 0]
    train_Y = [train_L[i] for i in range(len(train_L)) if train_Z[i].size(0) != 0]

    train_X = torch.cat(train_X, dim=0)
    train_Y = torch.cat(train_Y, dim=0)

    # =======================================

    ############################################################
    ##### initializing the classifier and train one epoch
    ############################################################

    from zeroshoteval.final_classifier import CLASSIFIER
    from zeroshoteval.evaluation.classification import LINEAR_LOGSOFTMAX
    import numpy as np
    from torch import nn as nn

    clf = LINEAR_LOGSOFTMAX(input_dim=64, nclass=200)

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            m.bias.data.fill_(0)

            nn.init.xavier_uniform_(m.weight,gain=0.5)


        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

    clf.apply(weights_init)

    cls = CLASSIFIER(clf,
                     train_X,
                     train_Y,
                     test_zsl_embeddings['IMG'][test_loader.dataset.seen_indexes], 
                     test_zsl_labels['IMG'][test_loader.dataset.seen_indexes],
                     test_zsl_embeddings['IMG'][test_loader.dataset.unseen_indexes], 
                     test_zsl_labels['IMG'][test_loader.dataset.unseen_indexes],
                     test_zsl_labels['IMG'][test_loader.dataset.seen_indexes].unique(), 
                     test_zsl_labels['IMG'][test_loader.dataset.unseen_indexes].unique(), 
                     200, cfg.DEVICE, cfg.CLS.SOLVER.BASE_LR, 0.5, 1,
                     cfg.CLS.BATCH_SIZE,
                     cfg.GENERALIZED)

    history = []

    for k in range(cfg.CLS.EPOCH):
        if k > 0:
            if cfg.GENERALIZED:
                cls.acc_seen, cls.acc_novel, cls.H = cls.fit()
            else:
                cls.acc = cls.fit_zsl()

        if cfg.GENERALIZED:

            print('[%.1f]     novel=%.4f, seen=%.4f, h=%.4f , loss=%.4f' % (
            k, cls.acc_novel, cls.acc_seen, cls.H, cls.average_loss))

            history.append([torch.tensor(cls.acc_seen).item(), torch.tensor(cls.acc_novel).item(),
                            torch.tensor(cls.H).item()])

        else:
            print('[%.1f]  acc=%.4f ' % (k, cls.acc))
            history.append([0, torch.tensor(cls.acc).item(), 0])

    if cfg.GENERALIZED:
        return torch.tensor(cls.acc_seen).item(), torch.tensor(cls.acc_novel).item(), torch.tensor(
            cls.H).item(), history
    else:
        return 0, torch.tensor(cls.acc).item(), 0, history


def train_classifier(cfg, model):

    # ------------------------------------
    test_zsl_embeddings = dict()
    test_zsl_labels = dict()

    test_loader = construct_loader(cfg, split='test')
    test_zsl_embeddings['IMG'], test_zsl_labels['IMG'] = CADA_VAE_inference(model=model,
                                                                            data_loader=test_loader,
                                                                            modality="IMG",
                                                                            device=cfg.DEVICE,
                                                                            reparametrize_with_noise=False)

    train_loader = construct_loader(cfg, split='train', shuffle=False, drop_last=False)
    # -------------------------------------

    history = []  # stores accuracies

    cls_seenclasses = test_zsl_labels['IMG'][test_loader.dataset.seen_indexes].unique().long().to(cfg.DEVICE)
    cls_novelclasses = test_zsl_labels['IMG'][test_loader.dataset.unseen_indexes].unique().long().to(cfg.DEVICE)

    train_seen_feat = torch.cat([x['IMG'] for x, y in train_loader])
    train_seen_label = torch.cat([y for x, y in train_loader]).long().to(cfg.DEVICE)

    novelclass_aux_data = torch.unique(torch.cat([x['CLSATTR'] for x, y in test_loader])[test_loader.dataset.unseen_indexes], dim=0)
    seenclass_aux_data = torch.unique(torch.cat([x['CLSATTR'] for x, y in train_loader]), dim=0)


    novel_test_feat = torch.cat([x['IMG'] for x, y in test_loader])[test_loader.dataset.unseen_indexes]
    seen_test_feat = torch.cat([x['IMG'] for x, y in test_loader])[test_loader.dataset.seen_indexes]
    test_seen_label = torch.cat([y for x, y in test_loader])[test_loader.dataset.seen_indexes]
    test_novel_label = torch.cat([y for x, y in test_loader])[test_loader.dataset.unseen_indexes]

    train_unseen_feat = torch.cat([x['IMG'] for x, y in train_loader])[train_loader.dataset.unseen_indexes]
    train_unseen_label = torch.cat([y for x, y in train_loader])[train_loader.dataset.unseen_indexes]


    with torch.no_grad():
        test_novel_X = CADA_VAE_inference_array(model=model,
                                                data=novel_test_feat,
                                                modality='IMG',
                                                device=cfg.DEVICE,
                                                reparametrize_with_noise=False)
        test_novel_Y = test_novel_label.to(cfg.DEVICE)

        test_seen_X = CADA_VAE_inference_array(model=model,
                                               data=seen_test_feat,
                                               modality='IMG',
                                               device=cfg.DEVICE,
                                               reparametrize_with_noise=False)
        test_seen_Y = test_seen_label.to(cfg.DEVICE)


        img_seen_feat,   img_seen_label   = sample_train_data_on_sample_per_class_basis(
                train_seen_feat,train_seen_label, 200 )

        img_unseen_feat, img_unseen_label = sample_train_data_on_sample_per_class_basis(
            train_unseen_feat, train_unseen_label, 0 )

        att_unseen_feat, att_unseen_label = sample_train_data_on_sample_per_class_basis(
                novelclass_aux_data,
                cls_novelclasses, 400 )

        att_seen_feat, att_seen_label = sample_train_data_on_sample_per_class_basis(
            seenclass_aux_data,
            cls_seenclasses, 0)

        z_seen_img = CADA_VAE_inference_array(model=model,
                                              data=img_seen_feat,
                                              modality='IMG',
                                              device=cfg.DEVICE)
        z_unseen_img = CADA_VAE_inference_array(model=model,
                                                data=img_unseen_feat,
                                                modality='IMG',
                                                device=cfg.DEVICE)

        z_seen_att = CADA_VAE_inference_array(model=model,
                                              data=att_seen_feat,
                                              modality='CLSATTR',
                                              device=cfg.DEVICE)
        z_unseen_att = CADA_VAE_inference_array(model=model,
                                                data=att_unseen_feat,
                                                modality='CLSATTR',
                                                device=cfg.DEVICE)

        train_Z = [z_seen_img, z_unseen_img, z_seen_att, z_unseen_att]
        train_L = [img_seen_label, img_unseen_label, att_seen_label, att_unseen_label]

        # empty tensors are sorted out
        train_X = [train_Z[i] for i in range(len(train_Z)) if train_Z[i].size(0) != 0]
        train_Y = [train_L[i] for i in range(len(train_L)) if train_Z[i].size(0) != 0]

        train_X = torch.cat(train_X, dim=0)
        train_Y = torch.cat(train_Y, dim=0)

    ############################################################
    ##### initializing the classifier and train one epoch
    ############################################################

    from zeroshoteval.final_classifier import CLASSIFIER
    from zeroshoteval.evaluation.classification import LINEAR_LOGSOFTMAX
    import numpy as np
    from torch import nn as nn

    clf = LINEAR_LOGSOFTMAX(input_dim=64, nclass=200)

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            m.bias.data.fill_(0)

            nn.init.xavier_uniform_(m.weight,gain=0.5)


        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

    clf.apply(weights_init)

    cls = CLASSIFIER(clf, train_X, train_Y, test_seen_X, test_seen_Y, test_novel_X,
                                test_novel_Y,
                                cls_seenclasses, cls_novelclasses,
                                200, cfg.DEVICE, cfg.CLS.SOLVER.BASE_LR, 0.5, 1,
                                cfg.CLS.BATCH_SIZE,
                                cfg.GENERALIZED)

    history = []

    for k in range(cfg.CLS.EPOCH):
        if k > 0:
            if cfg.GENERALIZED:
                cls.acc_seen, cls.acc_novel, cls.H = cls.fit()
            else:
                cls.acc = cls.fit_zsl()

        if cfg.GENERALIZED:

            print('[%.1f]     novel=%.4f, seen=%.4f, h=%.4f , loss=%.4f' % (
            k, cls.acc_novel, cls.acc_seen, cls.H, cls.average_loss))

            history.append([torch.tensor(cls.acc_seen).item(), torch.tensor(cls.acc_novel).item(),
                            torch.tensor(cls.H).item()])

        else:
            print('[%.1f]  acc=%.4f ' % (k, cls.acc))
            history.append([0, torch.tensor(cls.acc).item(), 0])

    if cfg.GENERALIZED:
        return torch.tensor(cls.acc_seen).item(), torch.tensor(cls.acc_novel).item(), torch.tensor(
            cls.H).item(), history
    else:
        return 0, torch.tensor(cls.acc).item(), 0, history


def CADA_VAE_inference(model: Module,
                       data_loader: DataLoader,
                       modality: str,
                       device: str,
                       reparametrize_with_noise: bool = True) -> Tuple[Tensor, Tensor]:
    """
    Calculate zsl embeddings for given VAE model and data.

    Args:
        model:
        data_loader:
        modality:
        device:
        reparametrize_with_noise:

    Returns:
        zsl_emb: zero shot learning embeddings for given data and model
    """
    model.eval()

    with torch.no_grad():
        zsl_emb = torch.Tensor().to(device)
        labels = torch.Tensor().long().to(device)

        for _i_step, (x, y) in enumerate(data_loader):

            x = x[modality]

            x = x.float().to(device)
            z_mu, z_logvar, z_noize = model.encoder[modality](x)

            if reparametrize_with_noise:
                zsl_emb = torch.cat((zsl_emb, z_noize.to(device)), 0)
            else:
                zsl_emb = torch.cat((zsl_emb, z_mu.to(device)), 0)

            labels: Tensor = torch.cat((labels, y.long().to(device)), 0)

    return zsl_emb.to(device), labels


def CADA_VAE_inference_array(model: Module,
                             data: Tensor,
                             modality: str,
                             device: str,
                             reparametrize_with_noise: bool = True) -> Tuple[Tensor, Tensor]:
    """
    Calculate zsl embeddings for given VAE model and data.

    Args:
        model:
        data_loader:
        modality:
        device:
        reparametrize_with_noise:

    Returns:
        zsl_emb: zero shot learning embeddings for given data and model
    """
    model.eval()

    with torch.no_grad():
        zsl_emb = torch.Tensor().to(device)

        for _i_step, x in enumerate(data):

            x = torch.unsqueeze(x, dim=0)

            x = x.float().to(device)
            z_mu, z_logvar, z_noize = model.encoder[modality](x)

            if reparametrize_with_noise:
                zsl_emb = torch.cat((zsl_emb, z_noize.to(device)), 0)
            else:
                zsl_emb = torch.cat((zsl_emb, z_mu.to(device)), 0)

    return zsl_emb.to(device)