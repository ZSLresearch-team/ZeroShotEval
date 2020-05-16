"""Config file for ZeroShotEval launcher scripts.

This module contains config settings for all modules in ZeroShotEval toolkit
in the format of easydict dictionaries.


Already contains the following configs:

Neural networks:
    - CADA-VAE (incomplete)  \\TODO: Verify CADA-VAE configs
    -

Datasets:
    - CUB (incomplete)  \\TODO: Verify CUB configs
"""

# TODO:
#     * Add AWA2, SUN datasets conf
#     * Add GAN conf
#     * Add embeddings conf
#     * etc.


from easydict import EasyDict as edict


# region GLOBAL DEFAULT CONFIGS
default = edict()

default.model = 'cada_vae'
default.datasets = 'cub'

default.modalities = 'img,cls_attr'
default.img_net = 'resnet101'
default.cls_attr_net = 'word2vec'

default.generalized_zsl = True

# path to stored object embeddings to load
default.saved_obj_embeddings_path = 'data/CUB/'
default.obj_embeddings_save_path = 'data/CUB/zsl_emb/'  # path to save computed embeddings

default.cache_zsl_embeddings = False
default.compute_zsl_train_embeddings = True
# endregion


# region MODEL CONFIGS
model = edict()

model.general_parameters = edict()  # general hyper for all models
model.general_parameters.device = 'cuda:0'
model.general_parameters.num_shots = 0
model.general_parameters.generalized = True
model.general_parameters.batch_size = 50
model.general_parameters.nepoch = 100
model.general_parameters.fp16_train_mode = False  # for GPUs with tensor cores
model.general_parameters.verbose = 2

# region CADA_VAE CONFIGS
model.cada_vae = edict()
model.cada_vae.model_name = 'cada_vae'
model.cada_vae.distance = 'wasserstein'
model.cada_vae.cross_resonstuction = True
model.cada_vae.distibution_allignment = True

model.cada_vae.specific_parameters = edict()
model.cada_vae.specific_parameters.lr_gen_model = 0.00015
model.cada_vae.specific_parameters.loss = 'l1'
model.cada_vae.specific_parameters.latent_size = 64
model.cada_vae.specific_parameters.use_bn = False
model.cada_vae.specific_parameters.use_dropout = False

# NOTE: probably for classification task only
model.cada_vae.specific_parameters.lr_cls = 0.001
# early stopping nepoch стоит изменить
model.cada_vae.specific_parameters.cls_train_epochs = 30
# для общности следует переделать эту и связанные части
model.cada_vae.specific_parameters.auxiliary_data_source = 'attributes'


model.cada_vae.specific_parameters.samples_per_modality_class = edict()
model.cada_vae.specific_parameters.samples_per_modality_class.img = 200
model.cada_vae.specific_parameters.samples_per_modality_class.cls_attr = 400
# NOTE: эти парамертры стоит извлекать из генераторов эмбедингов/кэшированных эмбедингов.
# Их нужно перенести в dataset или куда-то ещё.
#
# Стоит ли развести скрытые слои для декодера/энкодера?


model.cada_vae.specific_parameters.hidden_layers = edict()
model.cada_vae.specific_parameters.hidden_layers.cnn_features = (1560, 1660)
model.cada_vae.specific_parameters.hidden_layers.attributes = (1450, 665)
model.cada_vae.specific_parameters.hidden_layers.sentences = (1450, 665)
model.cada_vae.specific_parameters.input_features_from_cnn = 2048  # for ResNet101

model.cada_vae.specific_parameters.hidden_size_encoder = edict()
model.cada_vae.specific_parameters.hidden_size_encoder.img = [1560]
model.cada_vae.specific_parameters.hidden_size_encoder.cls_attr = [1450]
model.cada_vae.specific_parameters.hidden_size_encoder.sentences = [1450]

model.cada_vae.specific_parameters.hidden_size_decoder = edict()
model.cada_vae.specific_parameters.hidden_size_decoder.img = [1660]
model.cada_vae.specific_parameters.hidden_size_decoder.cls_attr = [665]
model.cada_vae.specific_parameters.hidden_size_decoder.sentences = [665]

model.cada_vae.specific_parameters.warmup = edict()
model.cada_vae.specific_parameters.warmup.beta = edict()
model.cada_vae.specific_parameters.warmup.beta.factor = 0.25
model.cada_vae.specific_parameters.warmup.beta.end_epoch = 93
model.cada_vae.specific_parameters.warmup.beta.start_epoch = 0

model.cada_vae.specific_parameters.warmup.cross_reconstruction = edict()
model.cada_vae.specific_parameters.warmup.cross_reconstruction.factor = 2.37
model.cada_vae.specific_parameters.warmup.cross_reconstruction.end_epoch = 75
model.cada_vae.specific_parameters.warmup.cross_reconstruction.start_epoch = 21

model.cada_vae.specific_parameters.warmup.distance = edict()
model.cada_vae.specific_parameters.warmup.distance.factor = 8.13
model.cada_vae.specific_parameters.warmup.distance.end_epoch = 22
model.cada_vae.specific_parameters.warmup.distance.start_epoch = 6

# NOTE: parameter below is for classification task only
# TODO: transfer auto selection from original repo
model.cada_vae.specific_parameters.cls_train_steps = 29
# endregion

# region CLSWGAN CONFIGS
model.clswgan = edict()
model.clswgan.model_name = 'clswgan'
# TODO: complete CLSWGAN CONFIGS section
# endregion

# endregion


# region DATASET CONFIGS
dataset = edict()

# region CUB DATASET CONFIGS
dataset.cub = edict()
dataset.cub.dataset_name = 'cub'
dataset.cub.path = 'data/CUB/resnet101/'
dataset.cub.precomputed_embeddings_path = 'data/CUB/res101.mat'

dataset.cub.num_classes = 200
dataset.cub.num_novel_classes = 50
# ! Будет меняться от generalized (num_shots == 0). Данные значение для GZSL
dataset.cub.samples_per_class = (200, 0, 400, 0)
# TODO: Стоит изменть для общности предыдущую строку

dataset.cub.class_embedding = edict()

dataset.cub.class_embedding.description_emb = edict()
dataset.cub.class_embedding.description_emb.module_name = ''
dataset.cub.class_embedding.description_emb.class_name = ''
dataset.cub.class_embedding.description_emb.have_pretrained = True
dataset.cub.class_embedding.description_emb.path = ''

dataset.cub.object_embedding = edict()

dataset.cub.object_embedding.resnet101 = edict()

dataset.cub.object_embedding.resnet101.module_name = ''
dataset.cub.object_embedding.resnet101.class_name = ''
dataset.cub.object_embedding.resnet101.have_pretrained = True
dataset.cub.object_embedding.resnet101.path = ''
# endregion

# region AWA2 DATASET CONFIGS
dataset.awa2 = edict()
dataset.awa2.dataset_name = 'awa2'
dataset.awa2.path = 'data/AWA2/'

dataset.awa2.num_classes = 50
dataset.awa2.num_novel_classes = 10
# TODO: complete AWA2 DATASET CONFIGS section
# endregion

# endregion


def generate_config(parsed_model, parsed_datasets):
    specific_model = model[parsed_model]
    for key, value in model.general_parameters.items():
        specific_model[key] = value

    datasets = {}
    for dataset_name in parsed_datasets:
        # datasets[dataset_name] = dataset[dataset_name]
        # specific_model[dataset_name] = dataset[dataset_name]
        pass
    specific_model['cub'] = dataset.cub

    return specific_model, datasets
