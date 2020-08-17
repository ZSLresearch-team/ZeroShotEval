
import torch
import torch.nn as nn
import torch.autograd as autograd
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial import distance
from classifier import Classifier_train

def train_LisGAN(config, model, train_loader, optimizer, verbose=1, *args, **kwargs):
    """
    calculate gradient penalty.
    Args:
        net_discriminator: 
        classes:
        attribute:
        number_samples:
        config:
    Returns:
       syn_samples(tensor):
       syn_labels
    """
    net_discriminator = model['discriminator']
    net_generator = model['generator']
    for net in model:
        net.train()
    loss_history = {'discriminator_loss':[], 'generator_loss':[], 'sum_loss':[]}
    train_classes = train_loader.seen_classes

    optimizerD = optimizer['discriminator'](net_discriminator.parameters(), lr=config.lr, betas=(config.beta1, 0.999))
    optimizerG = optimizer['generator'](net_generator.parameters(), lr=config.lr, betas=(config.beta1, 0.999))
    
    real_soul_sample = generate_real_soul_samples(train_loader.dataset, config.n_cluster)
    classifier = Classifier_train(config.device, config.classifier_num_epoch, config.input_dim, num_classes, train_loader,

    for epoch in range(config.n_epoh):
        mean_lossD = 0
        mean_lossG = 0
        sum_loss = 0

        for i_step, data in enumerate(train_loader):
            sample_embeddings = data[0]['img']
            sample_class_attributes = data[0]['cls_att']
            labels = data[1]
            noise = torch.zeros((config.batch_size, config.noize_size), 
                                device=config.device, dtype=torch.float)
            for iter_d in range(config.trein_discriminator_num_iter):
                net_discriminator.zero_grad()
                discriminator_score_real = net_discriminator(sample_embeddings, sample_class_attributes)
                discriminator_score_real = discriminator_score_real.mean()
                noise.normal_(0, 1)
                with torch.no_grad(): #my code, original work did't freeze generator
                    fake_embeddings = net_generator(noise, sample_class_attributes)
                discriminator_score_fake = net_discriminator(fake_embeddings, sample_class_attributes)
                discriminator_score_fake = discriminator_score_fake.mean()

                gradient_penalty = calc_gradient_penalty(net_discriminator, sample_embeddings,
                                                     fake_embeddings,
                                                     sample_class_attributes, config)
                discriminator_loss = discrimnator_score_fake - discrimnator_score_real + gradient_penalty*config.beta

                optimizerD.zero_grad()
                discriminator_loss.backward()
                optimizerD.step()

            net_discriminator.requires_grad = False 
            net_generator.zero_grad()
            noise.normal_(0, 1)
            fake_embeddings = net_generator(noise, sample_class_attributes)
            discrimnator_score_fake = net_discriminator(fake_embeddings, sample_class_attributes)
            wasserstein_dist = discrimnator_score_fake.mean()
            classification_loss = classifier.get_loss()

            labels = labels.view(config.batch_size, 1)
            dists1 = distance.cdist(fake_embeddings, real_soul_sample)
            min_index1 = torch.zeros(config.batch_size, config.train_cls_num,
                               device=config.device, dtype=torch.long)
            for i in train_classes:
                min_index1[:,i] = torch.min(dists1[:,i*config.n_clusters:(i+1)*config.n_clusters],dim=1)[1] + i*config.n_clusters
            regularization1 = dists1.gather(1, min_index1).gather(1,labels).squeeze().view(-1).mean()

            syn_samples, syn_labels = generate_synthesis_samples(net_generator, train_loader.seenclasses, data.attribute, config.loss_syn_num)
            transform_matrix = torch.zeros(config.train_cls_num, syn_samples.shape[0],
                                       device=config.deivce, dtype=torch.float)  # 150x20*150
            for i in range(len(train_classes)):
                class_label = train_classes[i]
                sample_idx = (syn_labels == class_label).nonzero().squeeze()
                if sample_idx.numel() == 0:
                    continue
                else:
                    class_sample_number = sample_idx.numel()
                    transform_matrix[i][sample_idx] = 1 / class_sample_number * torch.ones(1, class_sample_number).squeeze()
            synthesis_soul_sample = torch.mm(transform_matrix, syn_samples)  # 150;20*150 x 20*150;1024 = 150;1024
            dists2 =  distance.cdist(synthesis_soul_sample, real_proto) # 150 x 450
            min_index2 = torch.zeros(train_loader.train_cls_num, train_loader.train_cls_num,
                               device=config.device, dtype=torch.long)
            for i in range(len(train_classes)):
                min_index2[:,i] = torch.min(dists2[:,i*config.n_clusters:(i+1)*config.n_clusters],dim=1)[1] + i*config.n_clusters
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


def generate_real_soul_samples(dataset, n_cluster, device):
    """
    Generate centroid for each clusters of samples each train classes.
    Args:
        dataset: dataset for training and evaluating.
        n_cluster(int): namber of cluster in each class
    Returns:
       soul_samples(tensor) : tensor consist of .
    """
    num_classes = dataset.num_classes
    feature_dim = dataset.data['img'].shape[1]
    train_cls_num = dataset.num_seen_classes
    train_cls = dataset.seen_classes
    soul_samples = torch.zeros(n_cluster * train_cls_num, feature_dim, device=device)
    for i in range(train_cls_num):
        cls_label = train_cls[i]
        sample_idx = (dataset.labels == cls_label).nonzero()[0]
    if len(sample_idx) == 0:
        soul_samples[n_cluster * i: n_cluster * (i+1)] = torch.zeros(n_cluster, feature_dim)
    else:
        real_sample_cls = dataset.data['img'][sample_idx]
        y_pred = KMeans(n_clusters=n_cluster, random_state=3).fit_predict(real_sample_cls)
        for j in range(n_cluster):
            soul_samples[n_cluster*i+j] = torch.tensor(np.mean(real_sample_cls[y_pred == j], axis=0))
    return soul_samples

def calc_gradient_penalty(net_discriminator, real_data, fake_data, input_att, config):
    """
    calculate gradient penalty.
    Args:
        net_discriminator: 
        real_data:
        fake_data:
        input_att:
        config:
    Returns:
       gradient_penalty(tensor): 
    """
    alpha = torch.rand((config.batch_size, 1), device=config.device)
    alpha = alpha.expand(real_data.shape)
    alpha = alpha.to(config.device)
    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    interpolates.requires_grad = True
    disc_interpolates = net_discriminator(interpolates,input_att)
    ones = torch.ones(disc_interpolate.shape, device=config.device)
    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                grad_outputs=ones,create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() 
    return gradient_penalty

def generate_synthesis_samples(netG, classes, attribute, number_samples, config):
    """
    calculate gradient penalty.
    Args:
        netD: 
        classes:
        attribute:
        number_samples:
        config:
    Returns:
       syn_samples(tensor):
       syn_labels
    """
    number_classes = len(classes)
    syn_samples = torch.Zeros((number_classes * number_samples, config.sample_size),
                                    device=config.device, dtype=torch.float)
    syn_label = torch.Zeros((number_classes * number_samples), 
                            device=config.device, dtype=torch.long)
    syn_att = torch.Zeros((number_samples, config.attribute_size),
                          device=config.device, dtype=torch.float)
    syn_noise = torch.Zeros((number_samples, config.noize_size),
                            device=config.device, dtype=torch.float)

    for i in range(number_classes):
        class_att = attribute[class_index]
        syn_att = class_att.repeat(number_samples, 1)
        syn_noise.normal_(0, 1)
        output = netG(syn_noise, syn_att)
        syn_samples[i:number_samples*(i+1),:] = output
        syn_labels[i:number_samples*(i+1)] = classes[i]
    return syn_samples, syn_labels
