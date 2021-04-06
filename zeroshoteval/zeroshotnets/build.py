from fvcore.common.config import CfgNode
from fvcore.common.registry import Registry

from zeroshoteval.utils.misc import setup_random_number_generator_seed

ZSL_MODEL_REGISTRY = Registry("ZSL_MODEL")
ZSL_MODEL_REGISTRY.__doc__ = \
    """
    Registry for Zero-shot model.
    The registered object will be called with `obj(cfg)`.
    """


def build_zeroshot_train(cfg: CfgNode):
    """
    Builds zero shot learning model training and evaluating.
    Also setting RNG seed.

    Args:
        cfg(CfgNode): configs. Details can be found in
            zeroshoteval/config/defaults.py

    Returns:
        Train procedure function to use for model training.
    """
    setup_random_number_generator_seed(cfg)

    return ZSL_MODEL_REGISTRY.get(f"{cfg.ZSL_MODEL_NAME}_train_procedure")


def build_zeroshot_inference(cfg: CfgNode):

    # TODO: think about seed setup

    return ZSL_MODEL_REGISTRY.get(f"{cfg.ZSL_MODEL_NAME}_inference_procedure")
