from zeroshoteval.dataset.dataset import ObjEmbeddingDataset
from zeroshoteval.proc.evaluation.classification import classification_procedure
from zeroshoteval.proc.zeroshotnet.build import build_zsl


def experiment(cfg):
    """
    Start single experiment

    Args:
        cfg(CfgNode): configs. Details can be found in
            zeroshoteval/config/defaults.py

    """

    data = ObjEmbeddingDataset(cfg.DATA.FEAT_EMB.PATH, ["IMG"], cfg.VERBOSE)

    (zsl_emb_dataset, csl_train_indice, csl_test_indice,) = build_zsl(cfg)(
        cfg, data
    )

    # Train
    num_classes = (
        data.num_classes if cfg.GENERALIZED else data.num_unseen_classes
    )

    (
        _train_loss_hist,
        _acc_seen_hist,
        _acc_unseen_hist,
        acc_H_hist,
    ) = classification_procedure(
        cfg=cfg,
        data=zsl_emb_dataset,
        in_features=zsl_emb_dataset.tensors[0].size(1),
        num_classes=num_classes,
        train_indicies=csl_train_indice,
        test_indicies=csl_test_indice,
        seen_classes=data.seen_classes,
        unseen_classes=data.unseen_classes,
    )

    return None
