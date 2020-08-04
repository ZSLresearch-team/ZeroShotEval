import numpy as np
import torch

import importlib
import os
import PIL
import re
import subprocess
import sys
from collections import defaultdict
from tabulate import tabulate

__all__ = ["collect_env_info"]


def collect_torch_env():
    try:
        import torch.__config__

        return torch.__config__.show()
    except ImportError:
        # compatible with older versions of pytorch
        from torch.utils.collect_env import get_pretty_env_info

        return get_pretty_env_info()


def collect_env_info():
    has_gpu = torch.cuda.is_available()  # true for both CUDA & ROCM
    torch_version = torch.__version__

    from torch.utils.cpp_extension import CUDA_HOME

    data = []
    data.append(("sys.platform", sys.platform))
    data.append(("Python", sys.version.replace("\n", "")))
    data.append(("numpy", np.__version__))

    try:
        import zeroshoteval  # noqa

        data.append(
            (
                "zeroshoteval",
                zeroshoteval.__version__
                + " @"
                + os.path.dirname(zeroshoteval.__file__),
            )
        )
    except ImportError:
        data.append(("zeroshoteval", "failed to import"))

    # data.append(get_env_module())
    data.append(
        ("PyTorch", torch_version + " @" + os.path.dirname(torch.__file__))
    )
    data.append(("PyTorch debug build", torch.version.debug))

    data.append(("GPU available", has_gpu))
    if has_gpu:
        devices = defaultdict(list)
        for k in range(torch.cuda.device_count()):
            devices[torch.cuda.get_device_name(k)].append(str(k))
        for name, devids in devices.items():
            data.append(("GPU " + ",".join(devids), name))
            data.append(("CUDA_HOME", str(CUDA_HOME)))

            cuda_arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST", None)
            if cuda_arch_list:
                data.append(("TORCH_CUDA_ARCH_LIST", cuda_arch_list))
    data.append(("Pillow", PIL.__version__))

    try:
        import fvcore

        data.append(("fvcore", fvcore.__version__))
    except ImportError:
        pass

    env_str = tabulate(data) + "\n"
    env_str += collect_torch_env()
    return env_str


if __name__ == "__main__":
    try:
        import zeroshoteval  # noqa
    except ImportError:
        print(collect_env_info())
    else:
        from zeroshoteval.utils.collect_env import collect_env_info

        print(collect_env_info())
