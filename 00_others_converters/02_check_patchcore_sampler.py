
import os
import torch
import torch.nn.functional as F
import timm
import torchvision.models as models
import numpy as np
import tqdm

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
backbone_names = ['wideresnet50']
layers_to_extract_from = ['layer2', 'layer3']


# MobileNetV2_100
# backbone_names = ['mobilenetv2_100']
# layers_to_extract_from = ['blocks.2', 'blocks.3']
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
# patch maker
# ----------------------------------------------------------------------------------------------------------------

patchsize = 3
# patchsize = 3*3
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

for (module_name, module) in forward_modules.named_modules():
    #  print(module_name, module)
     print(module_name)


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
dimension_to_project_features_to = 128
number_of_starting_points = 10

feature_sampler = ApproximateGreedyCoresetSampler(
    percentage=percentage,
    device=device,
    number_of_starting_points=number_of_starting_points,
    dimension_to_project_features_to=dimension_to_project_features_to
)


##################################################################################################################
# ----------------------------------------------------------------------------------------------------------------
# data loader
# ----------------------------------------------------------------------------------------------------------------

# data_path = '/media/kswada/MyFiles/dataset/mvtec_ad'
data_path = 'C:\\Users\\kosei-wada\\Desktop\\mvtec_ad\\mvtec_ad'
mvtec_classname = 'screw'
# mvtec_classname = 'tmp'

# batch_size = 32
batch_size = 1
# batch_size = 320
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


print(features.keys())
# (32, 512, 28, 28) <-- WideResNet50 (layer2)
# (32, 32, 28, 28) <-- MobileNetV2_100 (blocks.2)
print(features[layers_to_extract_from[0]].shape)
print(features[layers_to_extract_from[0]].sum())
# (32, 1024, 14, 14) <-- WideResNet50 (layer3)
# (32, 64, 14, 14) <-- MobileNetV2_100 (blocks.3)
print(features[layers_to_extract_from[1]].shape)
print(features[layers_to_extract_from[1]].sum())

# print(torch.flatten(features['blocks.2'], start_dim=1).shape)
# print(torch.flatten(features['blocks.3'], start_dim=1).shape)

feats_concat = torch.cat([torch.flatten(feat, start_dim=1) for feat in features.values()], dim=1)
print(feats_concat.shape)
print(feats_concat.sum())
print(features[layers_to_extract_from[1]].sum() + features[layers_to_extract_from[0]].sum())

features = [features[layer] for layer in layers_to_extract_from]
print(len(features))
# (32, 512, 28, 28) <-- WideResNet50
# (32, 32, 28, 28) <-- MobileNetV2_100
print(features[0].shape)
# (32, 1024, 14, 14) <-- WideResNet50
# (32, 64, 14, 14) <-- MobileNetV2_100
print(features[1].shape)


##################################################################################################################
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
# preprocessing to 'pretain_embed_dimension'  --> here is important !!!
# ----------------------------------------------------------------------------------------------------------------

# As different feature backbones & patching provide differently
# sized features, these are brought into the correct form here.
# features = forward_modules["preprocessing"](features)

class MeanMapper(torch.nn.Module):
    def __init__(self, preprocessing_dim):
        super(MeanMapper, self).__init__()
        self.preprocessing_dim = preprocessing_dim

    def forward(self, features):
        features = features.reshape(len(features), 1, -1)
        return F.adaptive_avg_pool1d(features, self.preprocessing_dim).squeeze(1)

# ----------
print(feature_dimensions)
print(pretrain_embed_dimension)
# pretrain_embed_dimension = 1024

preprocessing_modules = torch.nn.ModuleList()
for input_dim in feature_dimensions:
    module = MeanMapper(pretrain_embed_dimension)
    preprocessing_modules.append(module)

print(preprocessing_modules)
print(features[0].shape)
print(features[1].shape)

_features = []
for module, feature in zip(preprocessing_modules, features):
    _features.append(module(feature))

print(len(_features))


##############
# NOW (784,32,3,3) --> (784, 1024)
# NOW (784,64,3,3) --> (784, 1024)
print(len(_features))
print(_features[0].shape)
print(_features[1].shape)

feature = torch.stack(_features, dim=1)

# ----------
print(len(features))
# (2, 1024) = (num of layers, pretrain_embed_dimension)
print(features[0].shape)
# (2, 1024) = (num of layers, pretrain_embed_dimension)
print(features[1].shape)


# ----------------------------------------------------------------------------------------------------------------
# preadapt aggretation to 'target_embed_dimension'  --> here is important !!!
# ----------------------------------------------------------------------------------------------------------------

# features = forward_modules["preadapt_aggregator"](features)

# ----------
# preadapt aggregator by Aggregator
features_after = features.reshape(len(features), 1, -1)
# (784, 1, 2048)
print(features_after.shape)

features_after = F.adaptive_avg_pool1d(features_after, target_embed_dimension)
# (784, 1, 1024)
print(features_after.shape)

features_after = features_after.reshape(len(features_after), -1)
# (784, 1024)
print(features_after.shape)


# ----------
features = features_after

# 25088
print(len(features))
# (1024) = (target_embed_dimension)
print(features[0].shape)
# (1024) = (target_embed_dimension)
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
# print(all_features[25087].shape)


# ----------------------------------------------------------------------------------------------------------------
# sampler
# ----------------------------------------------------------------------------------------------------------------

# sampled_features = feature_sampler.run(all_features)


# ----------
# check by step

# ----------
# _store_type(features)
features_is_numpy = isinstance(all_features, np.ndarray)
if not features_is_numpy:
    all_features_device = all_features.device

if isinstance(all_features, np.ndarray):
    all_features = torch.from_numpy(all_features)

# ----------
# _reduce_features(features)
# if all_features.shape[1] == dimension_to_project_features_to:
#     return all_features

# in=1024, out=128
mapper = torch.nn.Linear(
    all_features.shape[1], dimension_to_project_features_to, bias=False
)

_ = mapper.to(device)
all_features = all_features.to(device)

reduced_features = mapper(all_features)

# (784, 128)
print(reduced_features.shape)


# ----------
# ApproximateGreedyCoresetSampler
# _compute_greedy_coreset_indices(reduced_features)
def _compute_batchwise_differences(matrix_a: torch.Tensor, matrix_b: torch.Tensor) -> torch.Tensor:
    """Computes batchwise Euclidean distances using PyTorch."""
    a_times_a = matrix_a.unsqueeze(1).bmm(matrix_a.unsqueeze(2)).reshape(-1, 1)
    b_times_b = matrix_b.unsqueeze(1).bmm(matrix_b.unsqueeze(2)).reshape(1, -1)
    a_times_b = matrix_a.mm(matrix_b.T)
    return (-2 * a_times_b + a_times_a + b_times_b).clamp(0, None).sqrt()


number_of_starting_points = np.clip(number_of_starting_points, None, len(features))
start_points = np.random.choice(len(features), number_of_starting_points, replace=False).tolist()
print(start_points)

approximate_distance_matrix = _compute_batchwise_differences(reduced_features, reduced_features[start_points])
# (784, 784) --> (784, 10)
print(approximate_distance_matrix.shape)

approximate_coreset_anchor_distances = torch.mean(approximate_distance_matrix, axis=-1).reshape(-1, 1)
# (784, 1)
print(approximate_coreset_anchor_distances.shape)

coreset_indices = []
num_coreset_samples = int(len(reduced_features) * percentage)
# (78)
print(num_coreset_samples)

with torch.no_grad():
    for _ in tqdm.tqdm(range(num_coreset_samples), desc="Subsampling..."):
        select_idx = torch.argmax(approximate_coreset_anchor_distances).item()
        coreset_indices.append(select_idx)
        coreset_select_distance = _compute_batchwise_differences(reduced_features, reduced_features[select_idx: select_idx + 1])
        approximate_coreset_anchor_distances = torch.cat(
            [approximate_coreset_anchor_distances, coreset_select_distance],
            dim=-1,
        )
        approximate_coreset_anchor_distances = torch.min(
            approximate_coreset_anchor_distances, dim=1
        ).values.reshape(-1, 1)

sample_indices = np.array(coreset_indices)
# (78,)
print(sample_indices.shape)
sampled_features = all_features[sample_indices]
# (78, 1024)
print(sampled_features.shape)


# ----------
# _restore_type(features)
sampled_features = sampled_features.cpu().numpy()


# (78,)
print(len(sampled_features))

# (1024,) = target_embed_dimension
print(sampled_features[0].shape)


# ----------------------------------------------------------------------------------------------------------------
# anomaly scorer
# ----------------------------------------------------------------------------------------------------------------

anomaly_scorer.fit(detection_features=[sampled_features])

