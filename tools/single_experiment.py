"""Main script-launcher for training of ZSL models."""

from zeroshoteval.utils.defaults import default_setup
from zeroshoteval.utils.parser import load_config, parse_args

from zeroshoteval.data.dataset import ObjEmbeddingDataset
from zeroshoteval.evaluation.classification import classification_procedure
from zeroshoteval.zeroshotnet.build import build_zsl


def setup(args):
    cfg = load_config(args)
    cfg.freeze()
    default_setup(cfg, args)

    return cfg


def experiment(cfg):
    """
    Start single experiment

    Args:
        cfg(CfgNode): configs. Details can be found in
            zeroshoteval/config/defaults.py

    """

    data = ObjEmbeddingDataset(cfg.DATA.FEAT_EMB.PATH, ["IMG"], cfg.VERBOSE)

    train_procedure = build_zsl(cfg)

    zsl_emb_dataset, csl_train_indice, csl_test_indice = train_procedure(cfg, data)

    num_classes = data.num_classes if cfg.GENERALIZED else data.num_unseen_classes

    _train_loss_hist, _acc_seen_hist, _acc_unseen_hist, acc_H_hist = (
        classification_procedure(
            cfg=cfg,
            data=zsl_emb_dataset,
            in_features=zsl_emb_dataset.tensors[0].size(1),
            num_classes=num_classes,
            train_indicies=csl_train_indice,
            test_indicies=csl_test_indice,
            seen_classes=data.seen_classes,
            unseen_classes=data.unseen_classes,
        )
    )


if __name__ == "__main__":
    args = parse_args()
    cfg = setup(args)

    experiment(cfg)
