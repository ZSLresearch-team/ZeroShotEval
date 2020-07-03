#!/usr/bin/env python3

from .build import ZSL_MODEL_REGISTRY, build_zsl

# import all zsl procedures sothey can be registered
from .cada_vae.cada_vae_train import VAE_train_procedure
