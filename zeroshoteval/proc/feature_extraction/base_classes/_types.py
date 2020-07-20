import numpy as np
import torch

from os import PathLike
from PIL.Image import Image
from typing import Text, Union

"""
    This file contains type definitions and aliases
"""
ExtractorType = str
ImageObject = Union[np.ndarray, Image]
TextObject = Text
SourceObject = Union[ImageObject, TextObject]
EmbeddingObject = Union[np.ndarray, torch.tensor]
