"""
"""

# region IMPORTS
import copy
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.utils import data
from src.zeroshot_networks.cada_vae.vae_networks import EncoderTemplate, DecoderTemplate
# endregion


class Model(nn.Module):
    def __init__(self, config, modalities, feature_dimensions):
        super(Model, self).__init__()

        self.ntrain = feature_dimensions['img'][0]  # TODO: remove this

        # region MODEL HYPERPARAMETERS ASSIGNMENT
        self.device = config.device
        self.batch_size = config.batch_size
        self.nepoch = config.nepoch
        self.num_shots = config.num_shots  # for Few-Shot Learning only, else equals 0

        self.modalities = modalities
        
        self.latent_size = config.specific_parameters.latent_size
        self.hidden_size_rule = config.specific_parameters.hidden_size_rule
        self.warmup = config.specific_parameters.warmup
        self.reco_loss_function = config.specific_parameters.loss
        self.cross_reconstruction = config.specific_parameters.warmup.cross_reconstruction
        # endregion

        # TODO: make all checkings on data loading/preparation stage (training.py)
        if len(feature_dimensions) > 2 \
            or len(modalities) > 2:
            raise ValueError('You have passed more than 2 modalities or feature dims')
        
        feature_dimensions = [feature_dimensions[key][1] for key in modalities]

        # Here, the encoders and decoders for all modalities are created and put into dict
        self.encoder = {}
        for datatype, dim in zip(self.modalities, feature_dimensions):
            self.encoder[datatype] = EncoderTemplate(
                dim, self.latent_size, self.hidden_size_rule[datatype], self.device)
            print(str(datatype) + ' ' + str(dim))

        self.decoder = {}
        for datatype, dim in zip(self.modalities, feature_dimensions):
            self.decoder[datatype] = DecoderTemplate(
                self.latent_size, dim, self.hidden_size_rule[datatype], self.device)

        # An optimizer for all encoders and decoders is defined here
        parameters_to_optimize = list(self.parameters())
        for datatype in self.modalities:
            parameters_to_optimize += list(self.encoder[datatype].parameters())
            parameters_to_optimize += list(self.decoder[datatype].parameters())

        self.optimizer = optim.Adam(parameters_to_optimize, lr=config.specific_parameters.lr_gen_model, betas=(
            0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=True)

        if self.reco_loss_function == 'l2':
            self.reconstruction_criterion = nn.MSELoss(size_average=False)

        elif self.reco_loss_function == 'l1':
            self.reconstruction_criterion = nn.L1Loss(size_average=False)

    def reparameterize(self, mu, logvar):
        if self.reparameterize_with_noise:
            sigma = torch.exp(logvar)
            eps = torch.FloatTensor(logvar.size()[0],1).normal_(0,1)
            eps  = eps.expand(sigma.size())
            return mu + sigma*eps
        else:
            return mu

    def trainstep(self, img, att):

        ##############################################
        # Encode image features and additional
        # features
        ##############################################

        mu_img, logvar_img = self.encoder[self.modalities[0]](img)
        z_from_img = self.reparameterize(mu_img, logvar_img)

        mu_att, logvar_att = self.encoder[self.modalities[1]](att)
        z_from_att = self.reparameterize(mu_att, logvar_att)

        ##############################################
        # Reconstruct inputs
        ##############################################

        img_from_img = self.decoder[self.modalities[0]](z_from_img)
        att_from_att = self.decoder[self.modalities[1]](z_from_att)

        reconstruction_loss = self.reconstruction_criterion(img_from_img, img) \
                              + self.reconstruction_criterion(att_from_att, att)

        ##############################################
        # Cross Reconstruction Loss
        ##############################################
        img_from_att = self.decoder[self.modalities[0]](z_from_att)
        att_from_img = self.decoder[self.modalities[1]](z_from_img)

        cross_reconstruction_loss = self.reconstruction_criterion(img_from_att, img) \
                                    + self.reconstruction_criterion(att_from_img, att)

        ##############################################
        # KL-Divergence
        ##############################################

        KLD = (0.5 * torch.sum(1 + logvar_att - mu_att.pow(2) - logvar_att.exp())) \
              + (0.5 * torch.sum(1 + logvar_img - mu_img.pow(2) - logvar_img.exp()))

        ##############################################
        # Distribution Alignment
        ##############################################
        distance = torch.sqrt(torch.sum((mu_img - mu_att) ** 2, dim=1) + \
                              torch.sum((torch.sqrt(logvar_img.exp()) - torch.sqrt(logvar_att.exp())) ** 2, dim=1))

        distance = distance.sum()

        ##############################################
        # scale the loss terms according to the warmup
        # schedule
        ##############################################

        f1 = 1.0*(self.current_epoch - self.warmup.cross_reconstruction.start_epoch)/(1.0*( self.warmup.cross_reconstruction.end_epoch- self.warmup.cross_reconstruction.start_epoch))
        f1 = f1*(1.0*self.warmup.cross_reconstruction.factor)
        cross_reconstruction_factor = torch.FloatTensor([min(max(f1,0),self.warmup.cross_reconstruction.factor)])

        f2 = 1.0 * (self.current_epoch - self.warmup.beta.start_epoch) / ( 1.0 * (self.warmup.beta.end_epoch - self.warmup.beta.start_epoch))
        f2 = f2 * (1.0 * self.warmup.beta.factor)
        beta = torch.FloatTensor([min(max(f2, 0), self.warmup.beta.factor)])

        f3 = 1.0*(self.current_epoch - self.warmup.distance.start_epoch )/(1.0*( self.warmup.distance.end_epoch- self.warmup.distance.start_epoch))
        f3 = f3*(1.0*self.warmup.distance.factor)
        distance_factor = torch.FloatTensor([min(max(f3,0),self.warmup.distance.factor)])

        ##############################################
        # Put the loss together and call the optimizer
        ##############################################

        self.optimizer.zero_grad()

        loss = reconstruction_loss - beta * KLD

        if cross_reconstruction_loss>0:
            loss += cross_reconstruction_factor*cross_reconstruction_loss
        if distance_factor >0:
            loss += distance_factor*distance

        loss.backward()

        self.optimizer.step()

        return loss.item()

    def fit(self, data):
        losses = []

        # self.dataloader = data.DataLoader(
        #     dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)  # ,num_workers = 4)

        # dataset.novelclasses = dataset.novelclasses.long()
        # dataset.seenclasses = dataset.seenclasses.long()
        # leave both statements
        self.train()
        self.reparameterize_with_noise = True

        print('\ntrain for reconstruction')
        for epoch in range(0, self.nepoch):
            self.current_epoch = epoch

            i = -1
            for iters in range(0, self.ntrain, self.batch_size):
                i += 1

                data_from_modalities = gen_random_batch(
                    data, self.batch_size, self.ntrain)

                for j in range(len(data_from_modalities)):
                    data_from_modalities[j] = data_from_modalities[j].to(
                        self.device)
                    data_from_modalities[j].requires_grad = False

                loss = self.trainstep(
                    data_from_modalities[0], data_from_modalities[1])

                if i % 50 == 0:

                    print('epoch ' + str(epoch) + ' | iter ' + str(i) + '\t' +
                          ' | loss ' + str(loss)[:5])

                if i % 50 == 0 and i > 0:
                    losses.append(loss)

        # turn into evaluation mode:
        for key, value in self.encoder.items():
            self.encoder[key].eval()
        for key, value in self.decoder.items():
            self.decoder[key].eval()

        return losses

    def transform(self, data):
        """Inference mode for the model
        """
        print('\nComputing ZSL embeddings for test data...')
        iter_idx = 0
        embeddings = torch.Tensor()
        for batch in gen_next_batch(data, self.batch_size):
            iter_idx += 1
            data_from_modalities = batch

            for j in range(len(data_from_modalities)):
                data_from_modalities[j] = data_from_modalities[j].to(
                    self.device)
            
            mu_att, logvar_att = self.encoder[self.modalities[1]](data_from_modalities[1])
            z_from_att = self.reparameterize(mu_att, logvar_att)

            embeddings = torch.cat((embeddings, z_from_att), 0)
            # embeddings.append(z_from_att)
        
        return embeddings
                
    # def transfer_features(self, n, num_queries='num_features'):

        # NOTE: this method is used only for Few-Shot Learning
        # In current version of CADA-VAE model this functionality is not implemented
        # For full implementation please refer to original code from repo
        # https://github.com/edgarschnfld/CADA-VAE-PyTorch/blob/master/model/data_loader.py


# TODO: improve generator and place it not in the model
# Pass generator to method fit as in Keras
def gen_next_batch(data, batch_size):
        
        features = torch.from_numpy(data['img']).float()
        attr = torch.from_numpy(data['cls_attr']).float()
        # labels = torch.from_numpy(labels).long().to(self.device)

        iter_len = len(features) // batch_size + 1
        for current_batch in range(iter_len):
            current_idx = current_batch * batch_size
            end_idx = current_idx + batch_size

            batch_features = features[current_idx:end_idx]
            # batch_label = labels[current_idx:end_idx]
            batch_attr = attr[current_idx:end_idx]

            # yield batch_label, [batch_features, batch_attr]
            yield [batch_features, batch_attr]

def gen_random_batch(data, batch_size, ntrain):
    idx = torch.randperm(ntrain)[0:batch_size]
    features = torch.from_numpy(data['img']).float()
    attr = torch.from_numpy(data['cls_attr']).float()
    batch_feature = features[idx]
    # batch_label =  labels[idx]
    batch_attr = attr[idx]
    # return batch_label, [batch_feature, batch_attr]
    return [batch_feature, batch_attr]
