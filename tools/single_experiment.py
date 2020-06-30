from easydict import EasyDict as edict

from load_zsl_embeddings import load_zsl_emb
from zeroshoteval.dataset.dataset import ObjEmbeddingDataset
from zeroshoteval.proc.evaluation.classification import classification_procedure
from zeroshoteval.proc.zeroshotnet.cada_vae.cada_vae_train import (
    VAE_train_procedure,
)


def experiment(model_config):
    """
    Start single experiment
    """

    if not model_config.generate_obj_emb:
        pass
        if model_config.save_obj_emb:
            pass

    data = ObjEmbeddingDataset(
        model_config.dataset.path, ["img"], model_config.verbose
    )

    if model_config.generate_zsl_emb:
        (
            zsl_emb_dataset,
            csl_train_indice,
            csl_test_indice,
        ) = VAE_train_procedure(model_config, data)
    else:
        zsl_emb_dataset, csl_train_indice, csl_test_indice = load_zsl_emb(
            model_config.dataset.path
        )
    # Train
    num_classes = (
        data.num_classes
        if model_config.generalized
        else data.num_unseen_classes
    )

    (
        _train_loss_hist,
        _acc_seen_hist,
        _acc_unseen_hist,
        acc_H_hist,
    ) = classification_procedure(
        data=zsl_emb_dataset,
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
        verbose=model_config.verbose,
    )

    return None
