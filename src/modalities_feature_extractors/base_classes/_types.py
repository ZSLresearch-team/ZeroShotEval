from os import PathLike
from typing import Text, Union

import numpy as np
import torch
from PIL.Image import Image

"""
    This file contains type definitions and aliases
"""
ExtractorType = str
ImageObject = Union[np.ndarray, Image]
TextObject = Text
SourceObject = Union[ImageObject, TextObject]
EmbeddingObject = Union[np.ndarray, torch.tensor]
