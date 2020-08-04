from fvcore.common.registry import Registry

from zeroshoteval.utils.misc import RNG_seed_setup

ZSL_MODEL_REGISTRY = Registry("DATASET")
ZSL_MODEL_REGISTRY.__doc__ = """
Registry for Zero-shot model.
The registered object will be called with `obj(cfg, dataset)`.

"""


def build_zsl(cfg):
    """
    Builds zero shot learning model training and evaluating.
    Also setting RNG seed.

    Args: 
        cfg(CfgNode):configs. Details can be found in
            zeroshoteval/config/defaults.py
    """
    RNG_seed_setup(None if (cfg.RNG_SEED < 0) else RNG_SEED)

    return ZSL_MODEL_REGISTRY.get(f"{cfg.ZSL_MODEL_NAME}_train_procedure")
