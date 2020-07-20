from fvcore.common.registry import Registry

ZSL_MODEL_REGISTRY = Registry("DATASET")
ZSL_MODEL_REGISTRY.__doc__ = """
Registry for Zero-shot model.
The registered object will be called with `obj(cfg, dataset)`.
.
"""


def build_zsl(zsmodel_name):
    """
    """

    return ZSL_MODEL_REGISTRY.get(f"{zsmodel_name}_train_procedure")
