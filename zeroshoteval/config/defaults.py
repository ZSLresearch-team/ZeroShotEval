"""
Configuration file for the entire project. Contains all the configurations with
comprehensive documentation and provide sensible defaults for all options.

This file is the one-stop reference point for all configurable options.
Next, you'll create YAML configuration files; typically you'll make one for each
experiment. Each configuration file only overrides the options that are changing
in that experiment. This allows to maintain experiments reproducibility in a
clear way.
"""
from fvcore.common.config import CfgNode

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
_C = CfgNode()


# ---------------------------------------------------------------------------- #
# CADA-VAE model options
# ---------------------------------------------------------------------------- #
_C.CADA_VAE = CfgNode()

# If True train with cross reconstuction loss
_C.CADA_VAE.CROSS_RECONSTRUCTION = True

# If True use train with distribution allignment loss
_C.CADA_VAE.DISTRIBUTION_ALLIGNMENT = True

# Distance norm type for reconstuction loss, option include `l1`, `l2`
_C.CADA_VAE.NORM_TYPE = "l1"

# VAEs latent space size
_C.CADA_VAE.LATENT_SIZE = 64

# If True use batch norm layer, else not
_C.CADA_VAE.USE_BN = False

# Dropout rate, if equal 0 - don't use dropout layers
_C.CADA_VAE.DROPOUT_RATE = 0


# ---------------------------------------------------------------------------- #
# CADA-VAE hidden sizes options
# ---------------------------------------------------------------------------- #
_C.CADA_VAE.HIDDEN_SIZE = CfgNode()

# Encoders_hidden sizes
_C.CADA_VAE.HIDDEN_SIZE.ENCODER = CfgNode()

# Image encoder hidden sizes
_C.CADA_VAE.HIDDEN_SIZE.ENCODER.IMG = [1560]

# Class attributes hidden size
_C.CADA_VAE.HIDDEN_SIZE.ENCODER.CLSATTR = [1450]

# Decoders hidden sizes
_C.CADA_VAE.HIDDEN_SIZE.DECODER = CfgNode()

# Image encoder hidden sizes
_C.CADA_VAE.HIDDEN_SIZE.DECODER.IMG = [1660]

# Class attributes hidden size
_C.CADA_VAE.HIDDEN_SIZE.DECODER.CLSATTR = [665]


# ---------------------------------------------------------------------------- #
# CADA-VAE warmup factors
# ---------------------------------------------------------------------------- #
_C.CADA_VAE.WARMUP = CfgNode()

# Beta warmup - factor for VAE KL divergense loss
_C.CADA_VAE.WARMUP.BETA = CfgNode()

# Beta Base factor
_C.CADA_VAE.WARMUP.BETA.FACTOR = 0.25

# Beta Warmup end epoch
_C.CADA_VAE.WARMUP.BETA.END_EPOCH = 93

# Beta Warmup start epoch
_C.CADA_VAE.WARMUP.BETA.START_EPOCH = 0

# Cross reconstruction warmup - factor for VAEs cross reconstruction loss
_C.CADA_VAE.WARMUP.CROSS_RECONSTRUCTION = CfgNode()

# Cross reconstruction Base factor
_C.CADA_VAE.WARMUP.CROSS_RECONSTRUCTION.FACTOR = 2.37

# Cross reconstruction Warmup end epoch
_C.CADA_VAE.WARMUP.CROSS_RECONSTRUCTION.END_EPOCH = 75

# Cross reconstruction Warmup start epoch
_C.CADA_VAE.WARMUP.CROSS_RECONSTRUCTION.START_EPOCH = 21

# Distance warmup - factor for VAEs distance loss
_C.CADA_VAE.WARMUP.DISTANCE = CfgNode()

# Distance Base factor
_C.CADA_VAE.WARMUP.DISTANCE.FACTOR = 8.13

# Distance Warmup end epoch
_C.CADA_VAE.WARMUP.DISTANCE.END_EPOCH = 22

# Distance Warmup start epoch
_C.CADA_VAE.WARMUP.DISTANCE.START_EPOCH = 6


# ---------------------------------------------------------------------------- #
# General options for all zsl models, e.g. number of train epochs, batch size.
# ---------------------------------------------------------------------------- #
_C.ZSL = CfgNode()

# Number training epochs
_C.ZSL.EPOCH = 100

# Training mini-batch size
_C.ZSL.BATCH_SIZE = 50

# Whether to save embediings or not
_C.ZSL.SAVE_EMB = False


# ---------------------------------------------------------------------------- #
# Defines number of samples generated per class for each modality
# ---------------------------------------------------------------------------- #
_C.ZSL.SAMPLES_PER_CLASS = CfgNode()

# Samples per class for image modality
_C.ZSL.SAMPLES_PER_CLASS.IMG = 200

# Samples per class for class attribut modality
_C.ZSL.SAMPLES_PER_CLASS.CLSATTR = 400


# ---------------------------------------------------------------------------- #
# ZSL models solver options
# ---------------------------------------------------------------------------- #
_C.ZSL.SOLVER = CfgNode()

# Base learning rate
_C.ZSL.SOLVER.BASE_LR = 1e-3

# Learning rate policy(not available now)
_C.ZSL.SOLVER.LR_POLICY = ""

# Gamma - exponential decay factor
_C.ZSL.SOLVER.GAMMA = 0.1

# Momentum
_C.ZSL.SOLVER.MOMENTUM = 0.9

# Momentum dampening.
_C.ZSL.SOLVER.DAMPENING = 0.0

# Nesterov momentum.
_C.ZSL.SOLVER.NESTEROV = True

# Betas - coefficients used for computing running averages
_C.ZSL.SOLVER.BETAS = (0.9, 0.999)

# AMSGrad variant of Adam algorithms
_C.ZSL.SOLVER.AMSGRAD = False

# L2 weight regularization
_C.ZSL.SOLVER.WEIGHT_DECAY = 1e-4

# Optimization method(only adam and sgd available for now)
_C.ZSL.SOLVER.OPTIMIZING_METHOD = "adam"


# ---------------------------------------------------------------------------- #
# Final classifier options
# ---------------------------------------------------------------------------- #
_C.CLS = CfgNode()

# Number training epochs
_C.CLS.EPOCH = 100

# Training mini-batch size
_C.CLS.BATCH_SIZE = 32

# Load emveddings data from file
_C.CLS.LOAD_DATA = False


# ---------------------------------------------------------------------------- #
# Clasifier solver options
# ---------------------------------------------------------------------------- #
_C.CLS.SOLVER = CfgNode()

# Base learning rate
_C.CLS.SOLVER.BASE_LR = 1e-3

# Learning rate policy(not available now)
_C.CLS.SOLVER.LR_POLICY = ""

# Gamma - exponential decay factor
_C.CLS.SOLVER.GAMMA = 0.1

# Momentum
_C.CLS.SOLVER.MOMENTUM = 0.9

# Momentum dampening.
_C.CLS.SOLVER.DAMPENING = 0.0

# Nesterov momentum.
_C.CLS.SOLVER.NESTEROV = True

# Betas - coefficients used for computing running averages
_C.CLS.SOLVER.BETAS = (0.9, 0.999)

# AMSGrad variant of Adam algorithms
_C.CLS.SOLVER.AMSGRAD = False

# L2 weight regularization
_C.CLS.SOLVER.WEIGHT_DECAY = 1e-4

# Optimization method(only adam and sgd available for now)
_C.CLS.SOLVER.OPTIMIZING_METHOD = "adam"


# ---------------------------------------------------------------------------- #
# Data options
# ---------------------------------------------------------------------------- #
_C.DATA = CfgNode()

_C.DATA.DATASET_NAME = "CUB"


# ---------------------------------------------------------------------------- #
# Original data options
# ---------------------------------------------------------------------------- #
_C.DATA.ORIG = CfgNode()

# Path to original data
_C.DATA.ORIG.PATH = ""


# ---------------------------------------------------------------------------- #
# Feature embeddings options
# ---------------------------------------------------------------------------- #
_C.DATA.FEAT_EMB = CfgNode()


# Path to feature embeddings
_C.DATA.FEAT_EMB.PATH = "datasets/CUB_resnet101/"
# _C.DATA.FEAT_EMB.PATH = "datasets/CMUMovies_bert-large-cased/"


_C.DATA.FEAT_EMB.DIM = CfgNode()


_C.DATA.FEAT_EMB.DIM.IMG = 2048


_C.DATA.FEAT_EMB.DIM.CLSATTR = 312


# ---------------------------------------------------------------------------- #
# ZSL embeddings options
# ---------------------------------------------------------------------------- #
_C.DATA.ZSL_EMB = CfgNode()

_C.DATA.ZSL_EMB.PATH = ""


# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #

# ZSL model name
_C.ZSL_MODEL_NAME = "CADA_VAE"

# Output basedir.
_C.OUTPUT_DIR = "."

# Note that non-determinism may still be present due to non-deterministic
# operator implementations in GPU operator libraries.
# If seed is set to negative number, no seed will set.
_C.RNG_SEED = -1

# Sets CuDNN backend to deterministic mode
_C.CUDNN_DETERMINISTIC = False

# Log period in iters.
_C.LOG_PERIOD = 10

# If True, log the model info.
_C.LOG_MODEL_INFO = True

# If = 0 zero-shot mode, else few-shot learning
_C.NUM_SHOTS = 0

# If True GZSL mode, else ZSL
_C.GENERALIZED = True

# Device to use dor training
_C.DEVICE = "cuda:0"


# ---------------------------------------------------------------------------- #
# Common train/test data loader options
# ---------------------------------------------------------------------------- #
_C.DATA_LOADER = CfgNode()

# Number of data loader workers per training process.
_C.DATA_LOADER.NUM_WORKERS = 8

# Load data to pinned host memory.
_C.DATA_LOADER.PIN_MEMORY = True

# Drop last mini-batch
_C.DATA_LOADER.DROP_LAST = True


def get_cfg_defaults() -> CfgNode:
    """
    Get a YACS CfgNode object with default values for the project.
    """
    return _C.clone()
