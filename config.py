"""Config file for ZSLConstructor

This module this module contains all config settings for ZSLConstuctor in edicts.


Todo:
    * Add AWA2, SUN, aPY datasets conf
    * Add GAN conf
    * Add embeddings conf
    * etc.

"""


from  easydict import EasyDict as edict

# Model config
model = edict()

#Общие гиперпараметры для всех моделей
model.general_hyperparameters = edict()
# Возможно стоит перетащить эти параметры в default?
model.general_hyperparameters.device = "cuda"
model.general_hyperparameters.num_shots = 0
model.general_hyperparameters.generalized = True
model.general_hyperparameters.batch_size = 32
model.general_hyperparameters.nepoch = 100
model.general_hyperparameters.fp16_train_mode = False #Для *сверхлюдей* обладающих GPU с тензорными-ядрами


model.CADA_VAE = edict()
model.CADA_VAE.module_name = "models.CADA_VAE_net"
model.CADA_VAE.class_name = "CADA_VAE"
model.CADA_VAE.cross_resonstuction = True
model.CADA_VAE.distance = "Wasserstein"

#Специфичные гипперпараметры для CADA_VAE
model.CADA_VAE.specific_paraneters = edict()
model.CADA_VAE.specific_paraneters.lr_gen_model = 0.00015
model.CADA_VAE.specific_paraneters.loss = 'l1'
model.CADA_VAE.specific_paraneters.latent_size = 64
model.CADA_VAE.specific_paraneters.lr_cls = 0.001
model.CADA_VAE.specific_paraneters.cls_train_epochs = 100# early stopping nepoch стоит изменить
model.CADA_VAE.specific_paraneters.auxiliary_data_source = 'attributes'#!для общности следуюет переделать эту и
#связанные части

#######################################################################################################################
# !NB эти парамертры стоит извлекать из генераторов эмбедингов/кэшированных эмбедингов. Их нужно перенести в dataset
# !куда-то ещё. Стоит развести скрытые слои для декодера/энкодера?
#######################################################################################################################
model.CADA_VAE.specific_paraneters.hidden_layers = edict()
model.CADA_VAE.specific_paraneters.hidden_layers.cnn_features = (1560, 1660)
model.CADA_VAE.specific_paraneters.hidden_layers.attributes = (1450, 665)
model.CADA_VAE.specific_paraneters.hidden_layers.sentences = (1450, 665)
model.CADA_VAE.specific_paraneters.input_features_from_cnn = 2048 #для резнета101

model.CADA_VAE.specific_paraneters.warmup = edict()

model.CADA_VAE.specific_paraneters.warmup.beta = edict()
model.CADA_VAE.specific_paraneters.warmup.beta.factor = 0.25
model.CADA_VAE.specific_paraneters.warmup.beta.end_epoch = 93
model.CADA_VAE.specific_paraneters.warmup.beta.start_epoch = 0

model.CADA_VAE.specific_paraneters.warmup.cross_reconstuction = edict()
model.CADA_VAE.specific_paraneters.warmup.cross_reconstuction.factor = 2.37
model.CADA_VAE.specific_paraneters.warmup.cross_reconstuction.end_epoch = 75
model.CADA_VAE.specific_paraneters.warmup.cross_reconstuction.start_epoch = 21

model.CADA_VAE.specific_paraneters.warmup.distance = edict()
model.CADA_VAE.specific_paraneters.warmup.distance.factor = 8.13
model.CADA_VAE.specific_paraneters.warmup.distance.end_epoch = 22
model.CADA_VAE.specific_paraneters.warmup.distance.start_epoch = 6

# Dataset settings
dataset = edict()

dataset.CUB = edict()

dataset.CUB.path = ""
dataset.CUB.num_classes = 200
dataset.CUB.num_novel_classes = 50
dataset.CUB.samples_per_class = (200, 0, 400, 0) # ! Будет меняться от generalized (num_shots == 0)
                                                 # ! данные значение для GZSL. Стоит изменть

dataset.CUB.class_embedding = edict()

dataset.CUB.class_embedding.description_emb = edict()
dataset.CUB.class_embedding.description_emb.module_name = ""
dataset.CUB.class_embedding.description_emb.class_name = ""
dataset.CUB.class_embedding.description_emb.have_pretrained = True
dataset.CUB.class_embedding.description_emb.path = ""

dataset.CUB.object_embedding = edict()

dataset.CUB.object_embedding.resnet101 = edict()

dataset.CUB.object_embedding.resnet101 = edict()
dataset.CUB.object_embedding.resnet101.module_name = ""
dataset.CUB.object_embedding.resnet101.class_name = ""
dataset.CUB.object_embedding.resnet101.have_pretrained = True
dataset.CUB.object_embedding.resnet101.path = ""


default = edict()

default.model = "CADA_VAE"
default.dataset = "CUB"
default.class_embedding = "description_emb"
default.object_embedding = "resnet101"
