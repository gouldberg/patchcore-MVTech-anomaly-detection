
import os
import torch
import torch.nn.functional as F
import timm
import torchvision.models as models
import numpy as np


from src import *
from src import _BACKBONES


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
# patch maker
# ----------------------------------------------------------------------------------------------------------------

patchsize = 3
patchstride = 1

patch_maker = PatchMaker(patchsize, stride=patchstride)


# ----------------------------------------------------------------------------------------------------------------
# construct forward modules
# ----------------------------------------------------------------------------------------------------------------

# base
forward_modules = torch.nn.ModuleDict({})


# ----------
# 1. feature aggregator
feature_aggregator = NetworkFeatureAggregator(backbone, layers_to_extract_from, device)

cropsize = 224
input_shape = (3, cropsize, cropsize)
feature_dimensions = feature_aggregator.feature_dimensions(input_shape)

forward_modules["feature_aggregator"] = feature_aggregator


# ----------
# 2. preprocessing
# adaptive_avg_pool1d (MeanMapper) :  to pretrain_embed_dimension
pretrain_embed_dimension = 1024
# pretrain_embed_dimension = 500
preprocessing = Preprocessing(feature_dimensions, pretrain_embed_dimension)

forward_modules["preprocessing"] = preprocessing


# ----------
# 3. preadapt aggregator
# adaptive_avg_pool1d:  batchsize x number_of_layers x input_dim -> batchsize x target_embed_dimension

target_embed_dimension = 1024
# target_embed_dimension = 250
preadapt_aggregator = Aggregator(target_dim=target_embed_dimension)

_ = preadapt_aggregator.to(device)

forward_modules["preadapt_aggregator"] = preadapt_aggregator


# ----------
# THIS IS THE FORWARD MODULES

for (module_name, module) in forward_modules.named_modules():
    #  print(module_name, module)
     print(module_name)


# ----------------------------------------------------------------------------------------------------------------
# construct others
#  - nearest neighbour scorer
#  - segmentor
#  - sampler
# ----------------------------------------------------------------------------------------------------------------

faiss_on_gpu = True
faiss_num_workers = 12
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

data_path = '/media/kswada/MyFiles/dataset/mvtec_ad'
mvtec_classname = 'screw'

batch_size = 32
train_val_split = 1.0
seed = 0
num_workers = 12

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
    num_workers=num_workers,
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


# load to device
images = input_images.to(torch.float).to(device)

# (32, 3, 224, 224) = (batchsize, 3, cropsize, cropsize)
print(images.shape)


# ----------------------------------------------------------------------------------------------------------------
# feature aggregation
# ----------------------------------------------------------------------------------------------------------------

_ = forward_modules["feature_aggregator"].eval()

with torch.no_grad():
    features = forward_modules["feature_aggregator"](images)

print(features.keys())
# (32, 512, 28, 28) <-- WideResNet50 (layer2, layer3)
# (32, 64, 28, 28) <-- MobileNetV2_100 (blocks.2, blocks.3)
print(features[layers_to_extract_from[0]].shape)
# (32, 1024, 14, 14) <-- WideResNet50
# (32, 64, 14, 14) <-- MobileNetV2_100
print(features[layers_to_extract_from[1]].shape)


features = [features[layer] for layer in layers_to_extract_from]
print(len(features))
# (32, 512, 28, 28) <-- WideResNet50
# (32, 64, 28, 28) <-- MobileNetV2_100
print(features[0].shape)
# (32, 1024, 14, 14) <-- WideResNet50
# (32, 64, 14, 14) <-- MobileNetV2_100
print(features[1].shape)


# ----------------------------------------------------------------------------------------------------------------
# patchify
# ----------------------------------------------------------------------------------------------------------------

features = [
    patch_maker.patchify(x, return_spatial_info=True) for x in features
]

print(len(features))
print(len(features[0]))
print(len(features[1]))

# (32, 784, 512, 3, 3) = (batchsize, 28*28, original channel, patchsize, patchsize)
# (32, 784, 32, 3, 3)
print(features[0][0].shape)

# (32, 196, 1024, 3, 3) = (batchsize, 14*14, original channel, patchsize, patchsize)
# (32, 196, 64, 3, 3)
print(features[1][0].shape)

# here patchsize = 3 (3 * 3)
print(features[0][0][0][0][0])


# ----------------------------------------------------------------------------------------------------------------
# get basic info (shapes)
# ----------------------------------------------------------------------------------------------------------------

patch_shapes = [x[1] for x in features]
ref_num_patches = patch_shapes[0]

features = [x[0] for x in features]


# [[28, 28], [14, 14]]
print(patch_shapes)
# [28, 28]
print(ref_num_patches)

# (32, 784, 512, 3, 3)
# (32, 784, 32, 3, 3)
print(features[0].shape)
# (32, 196, 1024, 3, 3)
# (32, 196, 64, 3, 3)
print(features[1].shape)


# ----------------------------------------------------------------------------------------------------------------
# reshape
# ----------------------------------------------------------------------------------------------------------------

for i in range(1, len(features)):
    _features = features[i]
    patch_dims = patch_shapes[i]

    # TODO(pgehler): Add comments
    _features = _features.reshape(
        _features.shape[0], patch_dims[0], patch_dims[1], *_features.shape[2:]
    )
    _features = _features.permute(0, -3, -2, -1, 1, 2)
    perm_base_shape = _features.shape
    _features = _features.reshape(-1, *_features.shape[-2:])
    _features = F.interpolate(
        _features.unsqueeze(1),
        size=(ref_num_patches[0], ref_num_patches[1]),
        mode="bilinear",
        align_corners=False,
    )
    _features = _features.squeeze(1)
    _features = _features.reshape(
        *perm_base_shape[:-2], ref_num_patches[0], ref_num_patches[1]
    )
    _features = _features.permute(0, -2, -1, 1, 2, 3)
    _features = _features.reshape(len(_features), -1, *_features.shape[-3:])
    features[i] = _features

features = [x.reshape(-1, *x.shape[-3:]) for x in features]

print(len(features))
# (25088, 512, 3, 3) = (batchsize * ref_num_patches, original channel, patchsize, patchsize)
# (25088, 32, 3, 3)
print(features[0].shape)
# (25088, 1024, 3, 3) = (batchsize * ref_num_patches, original channel, patchsize, patchsize)
# (25088, 64, 3, 3)
print(features[1].shape)


# ----------------------------------------------------------------------------------------------------------------
# preprocessing to 'pretain_embed_dimension'
# ----------------------------------------------------------------------------------------------------------------

# As different feature backbones & patching provide differently
# sized features, these are brought into the correct form here.
features = forward_modules["preprocessing"](features)

print(len(features))
# (2, 500) = (num of layers, pretrain_embed_dimension)
print(features[0].shape)
# (2, 500) = (num of layers, pretrain_embed_dimension)
print(features[1].shape)


# ----------------------------------------------------------------------------------------------------------------
# preadapt aggretation to 'target_embed_dimension'
# ----------------------------------------------------------------------------------------------------------------

features = forward_modules["preadapt_aggregator"](features)

# 25088
print(len(features))
# (250) = (target_embed_dimension)
print(features[0].shape)
# (250) = (target_embed_dimension)
print(features[1].shape)


# ----------------------------------------------------------------------------------------------------------------
# detach
# ----------------------------------------------------------------------------------------------------------------

batch_features = [x.detach().cpu().numpy() for x in features]

# 25088 = batchsize * ref_num_paches
print(len(batch_features))

# (1024,) = target_embed_dimension
print(batch_features[0].shape)
print(batch_features[1].shape)


# ----------------------------------------------------------------------------------------------------------------
# all features (here only 1 batch)
# ----------------------------------------------------------------------------------------------------------------

all_features = []
all_features.append(batch_features)


# ----------------------------------------------------------------------------------------------------------------
# concatenate all batches  (here only 1 batch)
# ----------------------------------------------------------------------------------------------------------------

all_features = np.concatenate(all_features, axis=0)

print(len(all_features))
print(all_features[0].shape)


# ----------------------------------------------------------------------------------------------------------------
# sampler
# ----------------------------------------------------------------------------------------------------------------

features_embed = feature_sampler.run(all_features)

# 2508 = target_dimension(=1024) * num of layers(=2)
print(len(features_embed))

# (1024,) = target_embed_dimension
print(features_embed[0].shape)


# ----------------------------------------------------------------------------------------------------------------
# anomaly scorer
# ----------------------------------------------------------------------------------------------------------------

# anomaly_scorer.fit(detection_features=[features_embed])
