from src.zeroshot_networks.cada_vae.cada_vae_train import VAE_train_procedure
from src.dataloader.dataset import ObjEmbeddingDataset
from src.evaluation_procedures.classification import classification_procedure

from easydict import EasyDict as edict


def experiment(model_config):
    """
    Launches single experiment
    """

    data = ObjEmbeddingDataset(model_config.cub.path, ['img'], model_config.verbose)
    zsl_emb_dataset, csl_train_indice, csl_test_indice = VAE_train_procedure(model_config, data)


    # Train 
    num_classes = data.num_classes if model_config.generalized else data.num_unseen_classes

    _train_loss_hist, _acc_seen_hist, _acc_unseen_hist, acc_H_hist = \
    classification_procedure(data=zsl_emb_dataset,
                             in_features=model_config.specific_parameters.latent_size,
                             num_classes=num_classes,
                             batch_size=model_config.batch_size,
                             device=model_config.device,
                             n_epoch=model_config.specific_parameters.cls_train_epochs,
                             lr=model_config.specific_parameters.lr_cls,
                             train_indicies=csl_train_indice,
                             test_indicies=csl_test_indice,
                             seen_classes=data.seen_classes,
                             unseen_classes=data.unseen_classes,
                             verbose=model_config.verbose
                             )

    return None
