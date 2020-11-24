import torch


def build_optimizer(model, cfg, proc):
    """
    Constructs optimizer, defined by config.

    Args:
        model(Model): model to perform optimization
        cfg(CfgNode): configs. Details can be found in
            zeroshoteval/config/defaults.py
        proc(str): procedure type, e.g. ``ZSL``, ``CLS``

    Returns:
        optimizer
    """
    opt_cfg = cfg[proc].SOLVER
    if opt_cfg.OPTIMIZING_METHOD == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=opt_cfg.BASE_LR,
            momentum=opt_cfg.MOMENTUM,
            weight_decay=opt_cfg.WEIGHT_DECAY,
            dampening=opt_cfg.DAMPENING,
            nesterov=opt_cfg.NESTEROV,
        )
    elif opt_cfg.OPTIMIZING_METHOD == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=opt_cfg.BASE_LR,
            betas=opt_cfg.BETAS,
            eps=1e-08,
            weight_decay=opt_cfg.WEIGHT_DECAY,
            amsgrad=opt_cfg.AMSGRAD,
        )
    else:
        raise NotImplementedError(
            "Does not support {} optimizer".format(cfg.SOLVER.OPTIMIZING_METHOD)
        )
