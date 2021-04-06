#!/usr/bin/env python3
from .build import ZSL_MODEL_REGISTRY, build_zeroshot_train

# import all zsl procedures so they can be registered
from .cada_vae.cada_vae_train import CADA_VAE_train_procedure
