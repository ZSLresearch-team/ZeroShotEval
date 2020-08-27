import torch
import torch.nn as nn
import torch.autograd as autograd
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial import distance
import torch.optim as optim

from .classifier import Classifier_train
from .gan import Model_discriminator, Model_generator

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
                                  config.sample_dim, config.num_classes, train_loader)
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

            noise = torch.zeros((config.ZSL.BATCH_SIZE, config.noize_size), 
                                device=config.DEVICE, dtype=torch.float)
            for iter_d in range(config.train_discriminator_num_iter):
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
                discriminator_loss = discriminator_score_fake - discriminator_score_real + gradient_penalty*config.beta

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
            dists1 = torch.FloatTensor(distance.cdist(fake_embeddings.detach().numpy(), 
                                                      real_soul_sample.detach().numpy()))
            min_index1 = torch.zeros(config.ZSL.BATCH_SIZE, config.train_cls_num,
                               device=config.DEVICE, dtype=torch.int64)
            for i in range(len(train_classes)):
                fake_emb_soul_sample_dist = dists1[:,i*config.LISGAN.N_CLUSTER:(i+1)*config.LISGAN.N_CLUSTER]
                min_index1[:,i] = torch.min(fake_emb_soul_sample_dist, dim=1)[1] + i*config.LISGAN.N_CLUSTER
            regularization1 = dists1.gather(1, min_index1).gather(1,labels).squeeze().view(-1).mean()

            syn_samples, syn_labels = generate_synthesis_samples(net_generator, 
                                    train_classes, class_attributes,
                                    config.loss_syn_num, config.sample_dim,
                                    config.DEVICE, config.attribute_size,
                                    config.noize_size)
            syn_labels = torch.tensor(remap_labels(syn_labels, train_classes),
                                      device=config.DEVICE)
            transform_matrix = torch.zeros(config.train_cls_num, syn_samples.shape[0],
                                       device=config.DEVICE, dtype=torch.float) 
            for class_label in remap_labels(train_classes, train_classes):
                sample_idx = (syn_labels == class_label).nonzero().squeeze()
                if sample_idx.numel() == 0:
                    continue
                else:
                    class_sample_number = sample_idx.numel()
                    transform_matrix[class_label][sample_idx] = 1 / class_sample_number * torch.ones(1, class_sample_number).squeeze()
            synthesis_soul_sample = torch.mm(transform_matrix, syn_samples)  
            dists2 = torch.FloatTensor(distance.cdist(synthesis_soul_sample.detach().numpy(),
                                                      real_soul_sample.detach().numpy()))
            min_index2 = torch.zeros(config.train_cls_num, config.train_cls_num,
                               device=config.DEVICE, dtype=torch.int64)
            for i in range(len(train_classes)):
                fake_soul_samle_real_soul_sample_dist = dists2[:,i*config.LISGAN.N_CLUSTER:(i+1)*config.LISGAN.N_CLUSTER]
                min_index2[:,i] = torch.min(fake_soul_samle_real_soul_sample_dist,dim=1)[1] + i*config.LISGAN.N_CLUSTER
            lbl_idx = torch.LongTensor(list(range(len(train_classes))))
            regularization2 = dists2.gather(1,min_index2).gather(1,lbl_idx.unsqueeze(1)).squeeze().mean()

            generator_loss = wasserstein_dist + config.proto_param2 * regularization2 + config.proto_param1 * regularization1  + config.cls_weight * classification_loss
            generator_loss.backward()
            optimizerG.step()
            mean_lossD += discriminator_loss
            mean_lossG += generator_loss
            sum_loss = mean_lossD + mean_lossG
        loss_history['discriminator_loss'].append(mean_lossD / i_step + 1)
        loss_history['generator_loss'].append(mean_lossG / i_step + 1)
        loss_history['sum_loss'].append(sum_loss / i_step + 1)
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
                            device=device, dtype=torch.float)
    syn_att = torch.zeros((number_samples, attribute_size),
                          device=device, dtype=torch.float)
    syn_noise = torch.zeros((number_samples, noize_size),
                            device=device, dtype=torch.float)

    for i in range(number_classes):
        class_index = classes[i]
        class_att = attribute[class_index]
        syn_att = (class_att.repeat(number_samples, 1)).float()
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