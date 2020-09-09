import torch
import torch.nn as nn
import torch.autograd as autograd
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial import distance
import torch.optim as optim

from .classifier import Classifier_train
from .gan import Model_discriminator, Model_generator

from zeroshoteval.utils.misc import log_model_info
from zeroshoteval.utils.optimizer_helper import build_optimizer

import itertools
import logging

from ..build import ZSL_MODEL_REGISTRY
from .cada_vae_model import VAEModel

logger = logging.getLogger(__name__)

def train_LisGAN(config, model, train_loader, optimizer):
    """
    TRain lisgan procedure.
    Args:
        config: config for lisgan.
        model(dict): dict containing generator network and discriminator network. 
        train_loader: train loader with data for train.
        optimizer(dict): dict containing optimizer for generator network and 
                                            optimizer for discriminator network. 
    Returns:
       loss_history(dict): dict with mean loss for each epoch for generator, 
                            discriminator and summary loss.
    """
    logger.info("Train LISGAN model")
    net_discriminator = model['discriminator'].float()
    net_generator = model['generator'].float()
    for net in model:
        model[net].train()
    loss_history = {'discriminator_loss':[], 'generator_loss':[], 'sum_loss':[]}
    train_classes = train_loader.dataset.seen_classes
    class_attributes = train_loader.dataset.data['cls_attr']

    optimizerD = optimizer['discriminator']
    optimizerG = optimizer['generator']
    
    real_soul_sample = generate_real_soul_samples(train_loader.dataset, 
                                                  config.LISGAN.N_CLUSTER, config.DEVICE)
    classifier = Classifier_train(config.DEVICE, config.LISGAN.CLASSIFIER_NUM_EPOCH,
                                  config.DATA.FEAT_EMB.DIM.IMG, config.DATA.NUM_CLASSES,
                                  train_loader)
    classifier.fit()
    for epoch in range(config.LISGAN.N_EPOCH):
        mean_lossD = 0
        mean_lossG = 0
        sum_loss = 0

        for i_step, data in enumerate(train_loader):

            sample_embeddings = data[0]['img'].to(config.DEVICE).float()
            sample_class_attributes = data[0]['cls_attr'].to(config.DEVICE).float()
            labels = torch.tensor(remap_labels(data[1], train_classes), 
                                  device=config.DEVICE)

            noise = torch.zeros((config.ZSL.BATCH_SIZE, config.LISGAN.NOISE_SIZE), 
                                device=config.DEVICE, dtype=torch.float)
            for iter_d in range(config.LISGAN.TRAIN_DISCRIMINATOR_NUM_ITER):
                net_discriminator.zero_grad()
                discriminator_score_real = net_discriminator(sample_embeddings, 
                                                             sample_class_attributes)
                discriminator_score_real = discriminator_score_real.mean()
                noise.normal_(0, 1)
                with torch.no_grad(): #my code, original work did't freeze generator
                    fake_embeddings = net_generator(noise, sample_class_attributes)
                discriminator_score_fake = net_discriminator(fake_embeddings, 
                                                             sample_class_attributes)
                discriminator_score_fake = discriminator_score_fake.mean()

                gradient_penalty = calc_gradient_penalty(net_discriminator, 
                                                         sample_embeddings,
                                                         fake_embeddings,
                                                         sample_class_attributes, 
                                                         config.DEVICE,
                                                         config.ZSL.BATCH_SIZE)
                discriminator_loss = discriminator_score_fake - discriminator_score_real + gradient_penalty*config.LISGAN.BETA

                optimizerD.zero_grad()
                discriminator_loss.backward()
                optimizerD.step()

            net_discriminator.requires_grad = False 
            net_generator.zero_grad()
            noise.normal_(0, 1)
            fake_embeddings = net_generator(noise, sample_class_attributes)
            discriminator_score_fake = net_discriminator(fake_embeddings, 
                                                         sample_class_attributes)
            wasserstein_dist = discriminator_score_fake.mean()
            classification_loss = classifier.get_loss(fake_embeddings, labels)

            labels = labels.reshape(config.ZSL.BATCH_SIZE, 1)
            regularization1 = calculate_regularization_coef(fake_embeddings,
                                                            real_soul_sample, 
                                                            config.DATA.TRAIN_CLS_NUM, 
                                                            config.LISGAN.N_CLUSTER, 
                                                            labels, config.DEVICE)

            syn_samples, syn_labels = generate_synthesis_samples(net_generator, 
                                    train_classes, class_attributes,
                                    config.LISGAN.NUM_SYNTH_SAMPLES, config.DATA.FEAT_EMB.DIM.IMG ,
                                    config.DEVICE, config.DATA.FEAT_EMB.DIM.CLS_ATTR,
                                    config.LISGAN.NOISE_SIZE)
            syn_labels = remap_labels(syn_labels.cpu().numpy(), train_classes)
            syn_labels = torch.LongTensor(syn_labels).to(config.DEVICE)

            transform_matrix = torch.zeros(config.DATA.TRAIN_CLS_NUM, syn_samples.shape[0],
                                       device=config.DEVICE, dtype=torch.float) 
            for class_label in remap_labels(train_classes, train_classes):
                sample_idx = (syn_labels == class_label).nonzero().squeeze()
                if sample_idx.numel() == 0:
                    continue
                else:
                    class_sample_number = sample_idx.numel()
                    transform_matrix[class_label][sample_idx] = 1 / class_sample_number * torch.ones(1, class_sample_number).squeeze()
            synthesis_soul_sample = torch.mm(transform_matrix, syn_samples)
            lbl_idx = torch.LongTensor(remap_labels(train_classes, train_classes))
            lbl_idx = lbl_idx.to(config.DEVICE).unsqueeze(1)

            regularization2 = calculate_regularization_coef(synthesis_soul_sample, 
                                                            real_soul_sample,
                                                            config.DATA.TRAIN_CLS_NUM, 
                                                            config.LISGAN.N_CLUSTER,
                                                            lbl_idx,
                                                            config.DEVICE)

            generator_loss = wasserstein_dist + config.LISGAN.REG2_COEF * regularization2 + config.LISGAN.REG1_COEF * regularization1  + config.LISGAN.CLS_WEIGHT* classification_loss
            generator_loss.backward()
            optimizerG.step()
            mean_lossD += discriminator_loss
            mean_lossG += generator_loss
            sum_loss = mean_lossD + mean_lossG

        mean_lossD = mean_lossD / i_step + 1
        mean_lossG = mean_lossG / i_step + 1
        sum_loss = sum_loss / i_step + 1
        loss_history['discriminator_loss'].append(mean_lossD / i_step + 1)
        loss_history['generator_loss'].append(mean_lossG / i_step + 1)
        loss_history['sum_loss'].append(sum_loss / i_step + 1)
        logger.info(
            f"Epoch: {epoch+1} "
            f"Loss: {sum_loss:.1f} "
            f"Loss discriminator: {mean_lossD} loss generator: {mean_lossG}"
        )
    return loss_history


def generate_real_soul_samples(dataset, n_cluster, device):
    """
    Generate centroid for each clusters of samples each train classes.
    Args:
        dataset: dataset for training and evaluating.
        n_cluster(int): namber of cluster in each class
        decvice(str): device to be used.
    Returns:
       soul_samples(tensor) : tensor consist of centroids for each classes.
    """
    num_classes = dataset.num_classes
    feature_dim = dataset.data['img'].shape[1]
    train_cls_num = dataset.num_seen_classes
    train_cls = dataset.seen_classes
    soul_samples = torch.zeros(n_cluster * train_cls_num,
                               feature_dim, device=device)
    for i in range(train_cls_num):
        cls_label = train_cls[i]
        sample_idx = (dataset.labels == cls_label).nonzero()[0]
    if len(sample_idx) == 0:
        soul_samples[n_cluster * i: n_cluster * (i+1)] = torch.zeros(n_cluster, 
                                                                     feature_dim)
    else:
        real_sample_cls = dataset.data['img'][sample_idx]
        y_pred = KMeans(n_clusters=n_cluster, random_state=3).fit_predict(real_sample_cls)
        for j in range(n_cluster):
            soul_samples[n_cluster*i+j] = torch.tensor(np.mean(real_sample_cls[y_pred == j], axis=0))
    return soul_samples

def calc_gradient_penalty(net_discriminator, sample_embeddings, fake_data, 
                          input_att, device, batch_size):
    """
    calculate gradient penalty.
    Args:
        net_discriminator(nn.Nodule): discriminator networ.
        sample_embeddings(tensor): real sample embeddings. 
        fake_data(tensor): synthesis embeddings same samples.
        input_att(): class featchure of sample's classes. 
        device(str): device to be used.
        batch_size(int): batch_size.
    Returns:
       gradient_penalty(int): gradient penalty.
    """
    alpha = torch.rand((batch_size, 1), device=device)
    alpha = alpha.expand(sample_embeddings.shape)
    alpha = alpha.to(device)
    interpolates = alpha * sample_embeddings + ((1 - alpha) * fake_data)
    interpolates.requires_grad = True
    disc_interpolates = net_discriminator(interpolates,input_att)
    ones = torch.ones(disc_interpolates.shape, device=device)
    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                grad_outputs=ones,create_graph=True, 
                              retain_graph=True, only_inputs=True)[0]
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() 
    return gradient_penalty

def generate_synthesis_samples(netG, classes, attribute, number_samples, 
                            sample_size, device, attribute_size, noize_size):
    """
    generate synthesis samples for calculation synthesis soul samples.
    Args:
        netG(nn.Module): generator network.
        classes(list): list of all train classes.
        attribute(np.array): array of all class attributes, where attribute[i] 
                                                            attribute of i class.
        number_samples(int): number sumples each class to generate. 
        sample_size(int): dimention of sample's embedding.
        device(str): device to use.
        attribute_size(int): dimention of class attributes.
        noize_size(int): dimention of noize tensor.
    Returns:
       syn_samples(tensor): synthesis sample embeddings.
       syn_labels(tensor): corresponding labels for synthesis samples.
    """
    number_classes = len(classes)
    syn_samples = torch.zeros((number_classes * number_samples, sample_size),
                                    device=device, dtype=torch.float)
    syn_labels = torch.zeros((number_classes * number_samples), 
                            device=device, dtype=torch.int)
    syn_att = torch.zeros((number_samples, attribute_size),
                          device=device, dtype=torch.float)
    syn_noise = torch.zeros((number_samples, noize_size),
                            device=device, dtype=torch.float)

    for i in range(number_classes):
        class_index = classes[i]
        class_att = attribute[class_index]
        syn_att[:] = (class_att.repeat(number_samples, 1)).float()
        syn_noise.normal_(0, 1)
        output = netG(syn_noise, syn_att)
        syn_samples[i*number_samples:number_samples*(i+1),:] = output
        syn_labels[i*number_samples:number_samples*(i+1)] = class_index
    return syn_samples, syn_labels

def remap_labels(labels, classes):
    """
    Remapping labels
    Args:
        labels(np.array): array of labels
        classes:
    Returns:
        Remapped labels
    """
    remapping_dict = dict(zip(classes, list(range(len(classes)))))

    return np.vectorize(remapping_dict.get)(labels)

def calculate_regularization_coef(synthesis_sample, real_soul_sample, 
                                  train_cls_num, n_clusters, labels, device):
    """
    calculate regularisation for generate's loss i.e. mean distances between real
    soul samples and synthesis samples or soul samples.
    Args:
        synthesis_sample(tensor): synthesis object's samples or synthesis object's 
                                soul samples.
        real_soul_sample(tensor): soul samples of real samples.
        train_cls_num(int): number of train classes.
        n_clusters(int):  number of soul samples for each class.
        labels(tensor): class label for synthesis samples.
        device(str): device to use.
    Returns:
       regularization(): regularization addition for generator loss.
    """
    np_synthesis_sample = synthesis_sample.cpu().detach().numpy()
    np_real_soul_sample = real_soul_sample.cpu().numpy()
    distances = torch.FloatTensor(distance.cdist(np_synthesis_sample, 
                                                 np_real_soul_sample)).to(device)
    min_index = torch.zeros(len(np_synthesis_sample), train_cls_num,
                               device=device, dtype=torch.int64)
    for i in range(train_cls_num):
        min_index[:,i] = torch.min(distances[:,i*n_clusters:(i+1)*n_clusters],
                                    dim=1)[1] + i*n_clusters
    regularization = distances.gather(1, min_index).gather(1,labels).squeeze()
    regularization = regularization.view(-1).mean()
    return regularization

def generate_dataset_classification(train_dataset, config, net_generator):
    """
    calculate regularisation for generate's loss i.e. mean distances between real
    soul samples and synthesis samples or soul samples.
    Args:
        train_dataset(torch.Dataset): train dataset.
        config: config for lisgan.
        net_generator(nn.Module): generator network.
    Returns:
        dataset(dict): A dicttionary include np.array object embeddings for train
        and test classifier and appropriate labels of the samples.
        train_indexes(np.array):Indices of train samples in dataset.
        test_indexes(np.array):Indices of test samples in dataset.
    """
    logger.info("ZSL embedding dataset generation")
    object_modalities = train_dataset.object_modalities[0]
    class_modalities = train_dataset.class_modalities[0]
    train_emb = train_dataset.data[object_modalities][train_dataset.train_indices]
    train_labels = train_dataset.labels[train_dataset.train_indices]

    class_attributes = train_dataset.data[class_modalities]
    net_generator.eval()
    with torch.no_grad():
        unseen_train_emb, unseen_train_labels = generate_synthesis_samples(net_generator, 
                                                train_dataset.unseen_classes, 
                                                class_attributes, config.LISGAN.CLS_NUM_SAMPLES, 
                                                config.DATA.FEAT_EMB.DIM.IMG, config.DEVICE, 
                                                config.DATA.FEAT_EMB.DIM.CLS_ATTR, 
                                                config.LISGAN.NOISE_SIZE)
    unseen_train_emb = unseen_train_emb.cpu().numpy()
    unseen_train_labels = unseen_train_labels.cpu().numpy()
    
    unseen_test_emb = train_dataset.data[object_modalities][train_dataset.test_unseen_indices]
    unseen_test_labels = train_dataset.labels[train_dataset.test_unseen_indices]
    if config.GENERALIZED:
        seen_test_emb = train_dataset.data[object_modalities][train_dataset.test_seen_indices]
        seen_test_labels = train_dataset.labels[train_dataset.test_seen_indices]
        dataset = {object_modalities:np.concatenate([train_emb, unseen_train_emb,
                                                     seen_test_emb, 
                                                     unseen_test_emb], axis=0), 
               'labels':np.concatenate([train_labels, unseen_train_labels, 
                                        seen_test_labels, unseen_test_labels], axis=0)}
        train_indexes = np.arange(len(train_emb) + len(unseen_train_emb))
        test_indexes = np.arange(len(seen_test_emb) + len(unseen_test_emb)) + len(train_indexes)
    else:
        dataset = {object_modalities:np.concatenate([train_emb, unseen_train_emb,
                                                     unseen_test_emb], axis=0), 
               'labels':np.concatenate([train_labels, unseen_train_labels, 
                                        unseen_test_labels], axis=0)}
        train_indexes = np.arange(len(train_emb) + len(unseen_train_emb))
        test_indexes = np.arange(len(unseen_test_emb)) + len(train_indexes)
    return dataset, train_indexes, test_indexes