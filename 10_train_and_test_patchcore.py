
import os
import numpy as np
import torch

from src import *
from src import _BACKBONES, _CLASSNAMES


########################################################################################################################
# ----------------------------------------------------------------------------------------------------------------------
# base setting
# ----------------------------------------------------------------------------------------------------------------------

# data set path
data_path = '/media/kswada/MyFiles/dataset/mvtec_ad'
name = 'mvtec'


# ----------
# 'cuda' or 'cpu'
device = set_torch_device([0])
# device = 'cuda' if torch.cuda.is_available() else 'cpu'

faiss_on_gpu = True
faiss_num_workers = 12
num_workers = 12


# ----------
# result is saved here
results_path = '/home/kswada/kw/mvtech_ad/patchcore_official/results'
log_project = 'MVTecAD_Results'


# ----------
# backbones
print(_BACKBONES.keys())

# set backbones
# set layers manually

# WideResNet50
# backbone_names = ['wideresnet50']
# layers_to_extract_from = ['layer2', 'layer3']

# MobileNetV2_100
backbone_names = ['mobilenetv2_100']
layers_to_extract_from = ['blocks.2', 'blocks.3']


# ----------
# resize and crop size
resize = 256
cropsize = 224

# embedding dimension
pretrain_embed_dimension = 1024
target_embed_dimension = 1024

# coreset subsampling
percentage = 0.1

# number of nearest neighbours
anomaly_scorer_num_nn = 5

# patchsize
patchsize = 3


# ----------
# log group and create path
# log_group = f'IM{str(cropsize)}_WR50_L2-3_P01_D{str(pretrain_embed_dimension)}-{target_embed_dimension}_PS-{str(patchsize)}_AN-1_S0'
log_group = f'IM{str(cropsize)}_MBNV2100_B2-3_P01_D{str(pretrain_embed_dimension)}-{target_embed_dimension}_PS-{str(patchsize)}'

run_save_path = create_storage_folder(
    results_path, log_project, log_group, mode="iterate"
)

print(run_save_path)


# ----------------------------------------------------------------------------------------------------------------------
# base setting-2:  select category (for MVTec)
# ----------------------------------------------------------------------------------------------------------------------

idx = 9
mvtec_classname = _CLASSNAMES[idx]

print(f'classname: {mvtec_classname}')

dataset_name = f'{name}_{mvtec_classname}'


########################################################################################################################
# ----------------------------------------------------------------------------------------------------------------------
# data loaders
# ----------------------------------------------------------------------------------------------------------------------

batch_size = 32
train_val_split = 1.0
seed = 0

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

test_dataset = MVTecDataset(
    data_path,
    classname=mvtec_classname,
    resize=resize,
    imagesize=cropsize,
    split=DatasetSplit.TEST,
    seed=seed,
)

train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
)

test_dataloader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
)


train_dataloader.name = name


# ----------
torch.cuda.empty_cache()

imagesize = train_dataloader.dataset.imagesize

print(f'image size: {imagesize}')


# ----------------------------------------------------------------------------------------------------------------------
# extract layers
#   - if not ensemble, len(backbone_names) == 1
# ----------------------------------------------------------------------------------------------------------------------

if len(backbone_names) > 1:
    layers_to_extract_from_coll = [[] for _ in range(len(backbone_names))]
    for layer in layers_to_extract_from:
        idx = int(layer.split(".")[0])
        layer = ".".join(layer.split(".")[1:])
        layers_to_extract_from_coll[idx].append(layer)
else:
    layers_to_extract_from_coll = [layers_to_extract_from]

print(f'layers: {layers_to_extract_from_coll}')


# ----------------------------------------------------------------------------------------------------------------------
# set sampler:  ApproximateGreedyCoresetSampler
# ----------------------------------------------------------------------------------------------------------------------

# sampler = IdentitySampler()


# sampler = GreedyCoresetSampler(
#     percentage=percentage,
#     device=device,
#     dimension_to_project_features_to=128
# )


# this is required
sampler = ApproximateGreedyCoresetSampler(
    percentage=percentage,
    device=device,
    number_of_starting_points=10,
    dimension_to_project_features_to=128
)


########################################################################################################################
# ----------------------------------------------------------------------------------------------------------------------
# patchcore instance, loader
# ----------------------------------------------------------------------------------------------------------------------

loaded_patchcores = []

for backbone_name, layers_to_extract_from in zip(
    backbone_names, layers_to_extract_from_coll
):
    backbone_seed = None
    if ".seed-" in backbone_name:
        backbone_name, backbone_seed = backbone_name.split(".seed-")[0], int(
            backbone_name.split("-")[-1]
        )
    backbone = load(backbone_name)
    backbone.name, backbone.seed = backbone_name, backbone_seed

    nn_method = FaissNN(faiss_on_gpu, faiss_num_workers)

    patchcore_instance = PatchCore(device)
    patchcore_instance.load(
        backbone=backbone,
        layers_to_extract_from=layers_to_extract_from,
        device=device,
        input_shape=imagesize,
        pretrain_embed_dimension=pretrain_embed_dimension,
        target_embed_dimension=target_embed_dimension,
        patchsize=patchsize,
        featuresampler=sampler,
        anomaly_scorer_num_nn=anomaly_scorer_num_nn,
        nn_method=nn_method,
    )
    loaded_patchcores.append(patchcore_instance)


# ----------
print(len(loaded_patchcores))
print(loaded_patchcores[0])


########################################################################################################################
# ----------------------------------------------------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------------------------------------------------

for i, PC in enumerate(loaded_patchcores):
    torch.cuda.empty_cache()
    print("Training models ({}/{})".format(i + 1, len(loaded_patchcores)))
    if PC.backbone.seed is not None:
        fix_seeds(PC.backbone.seed, device)
    torch.cuda.empty_cache()
    PC.fit(train_dataloader)


# ----------------------------------------------------------------------------------------------------------------------
# Embedding test data
# ----------------------------------------------------------------------------------------------------------------------

torch.cuda.empty_cache()

aggregator = {"scores": [], "segmentations": []}

for i, PC in enumerate(loaded_patchcores):
    torch.cuda.empty_cache()
    print("Embedding test data with models ({}/{})".format(i + 1, len(loaded_patchcores)))
    scores, segmentations, labels_gt, masks_gt = PC.predict(test_dataloader)
    aggregator["scores"].append(scores)
    aggregator["segmentations"].append(segmentations)


# ----------
# length is equal to number of images for test
print(len(aggregator['scores'][0]))
print(len(aggregator['segmentations'][0]))


# ----------
# 1st image in train data
idx_img = 0
print(aggregator['scores'][0][idx_img])
print(aggregator['segmentations'][0][idx_img].shape)  # (cropsize, cropsize)
print(aggregator['segmentations'][0][idx_img])


########################################################################################################################
# ----------------------------------------------------------------------------------------------------------------------
# score - 1:  anomaly scores  (normalize by min-max scaling)
# ----------------------------------------------------------------------------------------------------------------------

# scores
scores = np.array(aggregator["scores"])

min_scores = scores.min(axis=-1).reshape(-1, 1)
max_scores = scores.max(axis=-1).reshape(-1, 1)
scores = (scores - min_scores) / (max_scores - min_scores)
scores = np.mean(scores, axis=0)

print(f'min score: {min_scores}    max score:  {max_scores}')
print(len(scores))

# normalized anomaly score for 1st image
print(scores[idx_img])


# ----------------------------------------------------------------------------------------------------------------------
# get label  (note that this is ground truth not prediction)
#  - False:  good
#  - True:  defective
# ----------------------------------------------------------------------------------------------------------------------

idx_img = 100
tmp = test_dataloader.dataset.data_to_iterate
print(tmp[idx_img])


# x[0]: category
# x[1]:  'good', 'defective' (for example)
# x[2]:  image path
# x[3]:  ground truth mask image path
anomaly_labels = [
    x[1] != "good" for x in test_dataloader.dataset.data_to_iterate
]

print(anomaly_labels)


# ----------------------------------------------------------------------------------------------------------------------
# score - 2:  segmentation  (normalize by min-max scaling)
# ----------------------------------------------------------------------------------------------------------------------

segmentations = np.array(aggregator["segmentations"])

min_scores = (
    segmentations.reshape(len(segmentations), -1)
    .min(axis=-1)
    .reshape(-1, 1, 1, 1)
)

max_scores = (
    segmentations.reshape(len(segmentations), -1)
    .max(axis=-1)
    .reshape(-1, 1, 1, 1)
)

print(f'min score: {min_scores}    max score:  {max_scores}')


# ----------
segmentations = (segmentations - min_scores) / (max_scores - min_scores)
# (1, # of test images, cropsize, cropsize)
print(segmentations.shape)


# remove axis=0
segmentations = np.mean(segmentations, axis=0)
# now the dimension is (# of test images, cropsize, cropsize)
print(segmentations.shape)


########################################################################################################################
# ----------------------------------------------------------------------------------------------------------------------
# plot and save
#  - input test image, mask (ground truth), segmentation by each image
# ----------------------------------------------------------------------------------------------------------------------

# # x[2]:  image path
# image_paths = [
#     x[2] for x in test_dataloader.dataset.data_to_iterate
# ]
#
# # x[3]:  ground truth mask image path
# mask_paths = [
#     x[3] for x in test_dataloader.dataset.data_to_iterate
# ]
#
# image_save_path = os.path.join(
#     run_save_path, "segmentation_images", dataset_name
# )
#
# os.makedirs(image_save_path, exist_ok=True)
#
#
# def image_transform(image):
#     # reshape to apply each value to each channel
#     in_std = np.array([0.229, 0.224, 0.225]).reshape(-1, 1, 1)
#     in_mean = np.array([0.485, 0.456, 0.406]).reshape(-1, 1, 1)
#     image = test_dataloader.dataset.transform_img(image)
#     return np.clip((image.numpy() * in_std + in_mean) * 255, 0, 255).astype(np.uint8)
#
#
# def mask_transform(mask):
#     return test_dataloader.dataset.transform_mask(mask).numpy()
#
#
# plot_segmentation_images(
#     image_save_path,
#     image_paths,
#     segmentations,
#     scores,
#     mask_paths,
#     image_transform=image_transform,
#     mask_transform=mask_transform,
#     save_depth=4
# )


########################################################################################################################
# ----------------------------------------------------------------------------------------------------------------------
# compute evaluation metrics
# ----------------------------------------------------------------------------------------------------------------------

auroc = compute_imagewise_retrieval_metrics(scores, anomaly_labels)["auroc"]


# ----------
# Compute PRO score & PW Auroc for all images
pixel_scores = compute_pixelwise_retrieval_metrics(segmentations, masks_gt)
full_pixel_auroc = pixel_scores["auroc"]


# ----------
# Compute PRO score & PW Auroc only images with anomalies
sel_idxs = []

for i in range(len(masks_gt)):
    if np.sum(masks_gt[i]) > 0:
        sel_idxs.append(i)

pixel_scores = compute_pixelwise_retrieval_metrics(
    [segmentations[i] for i in sel_idxs],
    [masks_gt[i] for i in sel_idxs],
)

anomaly_pixel_auroc = pixel_scores["auroc"]


# ----------------------------------------------------------------------------------------------------------------------
# arrange metrics and save
#  - note that this part should be run for all category, but here only 1 category
# ----------------------------------------------------------------------------------------------------------------------

result_collect = []

result_collect.append(
    {
        "dataset_name": dataset_name,
        "instance_auroc": auroc,
        "full_pixel_auroc": full_pixel_auroc,
        "anomaly_pixel_auroc": anomaly_pixel_auroc,
    }
)

for key, item in result_collect[-1].items():
    if key != "dataset_name":
        print("{0}: {1:3.3f}".format(key, item))

print(result_collect)


# ---------
# Store all results and mean scores to a csv-file

result_metric_names = list(result_collect[-1].keys())[1:]

result_dataset_names = [results["dataset_name"] for results in result_collect]

result_scores = [list(results.values())[1:] for results in result_collect]

compute_and_store_final_results(
    run_save_path,
    result_scores,
    column_names=result_metric_names,
    row_names=result_dataset_names,
)


########################################################################################################################
# ----------------------------------------------------------------------------------------------------------------------
# Store PatchCore model for later re-use
# ----------------------------------------------------------------------------------------------------------------------

save_path = os.path.join(
    run_save_path, "models", dataset_name
)

os.makedirs(save_path, exist_ok=True)

print(save_path)


for i, PC in enumerate(loaded_patchcores):
    prepend = (
        "Ensemble-{}-{}_".format(i + 1, len(loaded_patchcores))
        if len(loaded_patchcores) > 1
        else ""
    )
    PC.save_to_path(save_path, prepend)



