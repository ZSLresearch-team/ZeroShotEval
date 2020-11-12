from zeroshoteval.proc.evaluation.classification import classification_procedure
from zeroshoteval.proc.zeroshotnet.build import build_zsl


def experiment(cfg):
    """
    Start single experiment

    Args:
        cfg(CfgNode): configs. Details can be found in
            zeroshoteval/config/defaults.py

    """
    # ZSL
    zsl_data = build_zsl(cfg)(cfg)

    # Final CLS
    classification_procedure(cfg=cfg, data=None)

    return None
