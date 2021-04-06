import torch
from fvcore.common.config import CfgNode
from torch.nn import Module
from torch.optim.optimizer import Optimizer

PROCEDURE_TYPES = ['ZSL', 'CLS']
OPTIMIZERS = ['SGD', 'ADAM']


def build_optimizer(model: Module, cfg: CfgNode, procedure: str) -> Optimizer:
    """
    Constructs optimizer, defined by config.

    Args:
        model(Model): model to perform optimization
        cfg(CfgNode): configs. Details can be found in
            zeroshoteval/config/defaults.py
        procedure(str): procedure type, e.g. ``ZSL``, ``CLS``

    Returns:
        optimizer
    """
    assert procedure.upper() in PROCEDURE_TYPES, \
        f"Unknown procedure type in Optimizer building. Available: {PROCEDURE_TYPES}"

    opt_cfg = cfg[procedure].SOLVER

    optimizer: Optimizer
    if opt_cfg.OPTIMIZING_METHOD == "sgd":
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=opt_cfg.BASE_LR,
                                    momentum=opt_cfg.MOMENTUM,
                                    weight_decay=opt_cfg.WEIGHT_DECAY,
                                    dampening=opt_cfg.DAMPENING,
                                    nesterov=opt_cfg.NESTEROV)

    elif opt_cfg.OPTIMIZING_METHOD == "adam":
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=opt_cfg.BASE_LR,
                                     betas=opt_cfg.BETAS,
                                     eps=1e-08,
                                     weight_decay=opt_cfg.WEIGHT_DECAY,
                                     amsgrad=opt_cfg.AMSGRAD)

    else:
        raise ValueError(f"Does not support {cfg.SOLVER.OPTIMIZING_METHOD} optimizer")

    return optimizer
