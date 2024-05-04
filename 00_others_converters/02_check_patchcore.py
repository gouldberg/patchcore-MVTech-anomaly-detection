
import os
import torch
import torch.nn.functional as F
import timm
import torchvision.models as models
import numpy as np

import pandas as pd
import copy
import PIL

from src2 import *
from src2 import _BACKBONES
from src2.mvtec import IMAGENET_MEAN as normalizer_mean
from src2.mvtec import IMAGENET_STD as normalizer_std


base_path = 'C:\\Users\\kosei-wada\\Desktop\\mvtec_ad\\patchcore'


##################################################################################################################
# ----------------------------------------------------------------------------------------------------------------
# list of backbones and its loading code
# ----------------------------------------------------------------------------------------------------------------

# list of backbones
print(_BACKBONES.keys())


# ----------
# code to load
print(_BACKBONES['wideresnet50'])
print(_BACKBONES['mobilenetv2_100'])


# ----------------------------------------------------------------------------------------------------------------
# set backbone and target layers
# ----------------------------------------------------------------------------------------------------------------

# here set only 1 backbone (backbone_names has only 1 backbone)

# WideResNet50
# backbone_names = ['wideresnet50']
# layers_to_extract_from = ['layer2', 'layer3']


# MobileNetV2_100
backbone_names = ['mobilenetv2_100']
layers_to_extract_from = ['blocks.2', 'blocks.3']
# layers_to_extract_from = ['blocks.2']
# layers_to_extract_from = ['blocks.3']


# ----------------------------------------------------------------------------------------------------------------
# arrange layers
# ----------------------------------------------------------------------------------------------------------------

if len(backbone_names) > 1:
    layers_to_extract_from_coll = [[] for _ in range(len(backbone_names))]
    for layer in layers_to_extract_from:
        idx = int(layer.split(".")[0])
        layer = ".".join(layer.split(".")[1:])
        layers_to_extract_from_coll[idx].append(layer)
else:
    layers_to_extract_from_coll = [layers_to_extract_from]

print(f'layers: {layers_to_extract_from_coll}')


# ----------------------------------------------------------------------------------------------------------------
# check backbone
# ----------------------------------------------------------------------------------------------------------------

# select only 1 backbone
backbone_name = backbone_names[0]


# ----------
# load
print(_BACKBONES[backbone_name])

backbone = eval(_BACKBONES[backbone_name])
backbone.name = backbone_name


# ----------
# check backbone
print(backbone)

for (module_name, module) in backbone.named_modules():
    #  print(module_name, module)
     print(module_name)



##################################################################################################################
# ----------------------------------------------------------------------------------------------------------------
# PatchCore instance
# ----------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------
# load backbone to device
# ----------------------------------------------------------------------------------------------------------------

device = 'cuda' if torch.cuda.is_available() else 'cpu'

backbone = backbone.to(device)


# ----------------------------------------------------------------------------------------------------------------
# tflite extractor
# ----------------------------------------------------------------------------------------------------------------

tflite_model_path = os.path.join(base_path, 'model\\model_ext_f32.tflite')
dim_by_layer = [[1, 32, 28, 28], [1, 64, 14, 14]]

extractor_tflite = feature_extractor_tflite(
    model_path=tflite_model_path,
    layer_name_list=layers_to_extract_from,
    dim_by_layer=dim_by_layer,
    image_normalizer=[normalizer_mean, normalizer_std]
)


# ----------------------------------------------------------------------------------------------------------------
# patch maker
# ----------------------------------------------------------------------------------------------------------------

patchsize = 3
patchstride = 1

patch_maker = PatchMaker(patchsize, stride=patchstride)


# ----------------------------------------------------------------------------------------------------------------
# construct forward modules
# ----------------------------------------------------------------------------------------------------------------

cropsize = 224
input_shape = (3, cropsize, cropsize)


# ----------
# case 1:  only NetworkFeatureAggregator
forward_modules = torch.nn.ModuleDict({})
feature_aggregator = NetworkFeatureAggregator(backbone, layers_to_extract_from, device)
forward_modules["feature_aggregator"] = feature_aggregator


# ------------------------------
# case 2:  NetworkFeatureAggregator + preprocessing
# forward_modules = torch.nn.ModuleDict({})
# feature_aggregator = NetworkFeatureAggregator(backbone, layers_to_extract_from, device)
# forward_modules["feature_aggregator"] = feature_aggregator

# cropsize = 224
# input_shape = (3, cropsize, cropsize)
# feature_dimensions = feature_aggregator.feature_dimensions(input_shape)
# pretrain_embed_dimension = 1024
# preprocessing = Preprocessing(feature_dimensions, pretrain_embed_dimension)

# forward_modules["preprocessing"] = preprocessing


# ------------------------------
# case 3:  NetworkFeatureAggregator + eval + preprocessing
# forward_modules = torch.nn.ModuleDict({})
# feature_aggregator = NetworkFeatureAggregator(backbone, layers_to_extract_from, device)
# feature_aggregator.eval() # added!
# forward_modules["feature_aggregator"] = feature_aggregator
#
# cropsize = 224
# input_shape = (3, cropsize, cropsize)
# feature_dimensions = feature_aggregator.feature_dimensions(input_shape)
# pretrain_embed_dimension = 1024
# preprocessing = Preprocessing(feature_dimensions, pretrain_embed_dimension)
#
# forward_modules["preprocessing"] = preprocessing


# # ----------
# # THIS IS THE FORWARD MODULES
#
# for (module_name, module) in forward_modules.named_modules():
#     #  print(module_name, module)
#      print(module_name)


# ----------------------------------------------------------------------------------------------------------------
# construct others
#  - nearest neighbour scorer
#  - segmentor
#  - sampler
# ----------------------------------------------------------------------------------------------------------------

# faiss_on_gpu = True
faiss_on_gpu = False
# faiss_num_workers = 12
faiss_num_workers = 1
nn_method = FaissNN(faiss_on_gpu, faiss_num_workers)

anomaly_score_num_nn = 5
anomaly_scorer = NearestNeighbourScorer(
    n_nearest_neighbours=anomaly_score_num_nn, nn_method=nn_method
)

anomaly_segmentor = RescaleSegmentor(
    device=device, target_size=input_shape[-2:]
)

# ----------
# sampler
percentage = 0.1

feature_sampler = ApproximateGreedyCoresetSampler(
    percentage=percentage,
    device=device,
    number_of_starting_points=10,
    dimension_to_project_features_to=128
)


##################################################################################################################
# ----------------------------------------------------------------------------------------------------------------
# data loader
# ----------------------------------------------------------------------------------------------------------------

# data set path
data_path = 'C:\\Users\\kosei-wada\\Desktop\\mvtec_ad\\mvtec_ad2'

# mvtec_classname = 'screw'
mvtec_classname = 'tmp'

# batch_size = 32
batch_size = 1
train_val_split = 1.0
seed = 0
# num_workers = 12
num_workers = 1

resize = 256

train_dataset = MVTecDataset(
    data_path,
    classname=mvtec_classname,
    resize=resize,
    train_val_split=train_val_split,
    imagesize=cropsize,
    split=DatasetSplit.TRAIN,
    seed=seed,
    augment=True,
)


train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=False,
    # num_workers=num_workers,
    pin_memory=True,
)


# ----------------------------------------------------------------------------------------------------------------
# eval model
# ----------------------------------------------------------------------------------------------------------------

_ = forward_modules.eval()


# ----------------------------------------------------------------------------------------------------------------
# get one batch from train image
# ----------------------------------------------------------------------------------------------------------------

# get one batch
input_images = next(iter(train_dataloader))
input_images = input_images['image']
print(input_images)
print(input_images.shape)
print(input_images.sum())


# load to device
images = input_images.to(torch.float).to(device)

# (32, 3, 224, 224) = (batchsize, 3, cropsize, cropsize)
print(images.shape)
print(images.sum())


# ----------------------------------------------------------------------------------------------------------------
# feature aggregation
# ----------------------------------------------------------------------------------------------------------------

_ = forward_modules["feature_aggregator"].eval()

with torch.no_grad():
    features = forward_modules["feature_aggregator"](images)


# ----------------------------------------------------------------------------------------------------------------
# extract features by tflite
# ----------------------------------------------------------------------------------------------------------------

features_tflite = extractor_tflite.extract_features(input_data=images)


# ----------------------------------------------------------------------------------------------------------------
# extract features by torchextractor
# ----------------------------------------------------------------------------------------------------------------

import torch
import torchextractor

class Extractor(torchextractor.Extractor):
    def forward(self, *args, **kwargs):
        _ = self.model(*args, **kwargs)
        return torch.cat([torch.flatten(feature, start_dim=1) for feature in self.feature_maps.values()], dim=1)

pytorch_model = timm.create_model('mobilenetv2_100', pretrained=True)

pytorch_model.eval()

extractor = Extractor(pytorch_model, layers_to_extract_from)

extractor.eval()

with torch.no_grad():
    feats_extractor = extractor(images).detach().numpy()


##################################################################################################################
# ----------------------------------------------------------------------------------------------------------------
# compare
# ----------------------------------------------------------------------------------------------------------------

print(features.keys())
print(features_tflite.keys())

print(features[layers_to_extract_from[0]].shape)
print(features_tflite[layers_to_extract_from[0]].shape)

print(features[layers_to_extract_from[1]].shape)
print(features_tflite[layers_to_extract_from[1]].shape)

feats_concat = torch.cat([torch.flatten(feat, start_dim=1) for feat in features.values()], dim=1)
feats_concat_tflite = torch.cat([torch.flatten(feat, start_dim=1) for feat in features_tflite.values()], dim=1)

print(feats_concat.shape)
print(feats_concat_tflite.shape)
print(feats_extractor.shape)

print(feats_concat.sum())
print(feats_concat_tflite.sum())
print(feats_extractor.sum())

print(feats_concat.mean())
print(feats_concat_tflite.mean())
print(feats_extractor.mean())


# ----------
tmp0 = pd.DataFrame(feats_concat.detach().numpy().T)
tmp1 = pd.DataFrame(feats_concat_tflite.T)
tmp2 = pd.DataFrame(feats_extractor.T)

tmp0.hist(bins=100)
tmp1.hist(bins=100)
tmp2.hist(bins=100)
