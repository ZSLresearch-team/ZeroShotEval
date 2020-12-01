from .loader import _construct_gen_loader


def build_gen_loaders(cfg):
    """
    Builds data loaders to generate embeddigs for different splits
    and modalities

    Args:
        cfg(CfgNode): configs. Details can be found in
            zeroshoteval/config/defaults.py
    """

    data = {}
    if cfg.GENERALIZED:
        data["train_img_loader"] = _construct_gen_loader(cfg, "trainval", "IMG")

    data["train_attr_loader"] = _construct_gen_loader(
        cfg, "test_unseen", "CLS_ATTR"
    )

    data["test_loader"] = _construct_gen_loader(cfg, "test", "IMG")

    return data
