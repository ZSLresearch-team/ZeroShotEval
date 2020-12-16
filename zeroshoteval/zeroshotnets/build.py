from fvcore.common.registry import Registry

from zeroshoteval.utils.misc import RNG_seed_setup

ZSL_MODEL_REGISTRY = Registry("DATASET")
ZSL_MODEL_REGISTRY.__doc__ = """
Registry for Zero-shot model.
The registered object will be called with `obj(cfg)`.

"""


def build_zsl(cfg):
    """
    Builds zero shot learning model training and evaluating.
    Also setting RNG seed.

    Args:
        cfg(CfgNode): configs. Details can be found in
            zeroshoteval/config/defaults.py

    Returns:
        Train procedure function to use for model training.
    """
    RNG_seed_setup(cfg)

    return ZSL_MODEL_REGISTRY.get(f"{cfg.ZSL_MODEL_NAME}_train_procedure")
