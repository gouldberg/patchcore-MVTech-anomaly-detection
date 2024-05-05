
import os
import numpy as np
import torch

from src import *
from src import _BACKBONES, _CLASSNAMES

import gc
import contextlib


########################################################################################################################
# ----------------------------------------------------------------------------------------------------------------------
# base setting
# ----------------------------------------------------------------------------------------------------------------------

# data set path
data_path = '/media/kswada/MyFiles/dataset/mvtec_ad_nogt'
name = 'mvtec'


# ----------
# 'cuda' or 'cpu'
device = set_torch_device([0])
# device = 'cuda' if torch.cuda.is_available() else 'cpu'

faiss_on_gpu = True
faiss_num_workers = 12
num_workers = 12


# ----------
# resize and crop size
resize = 256
cropsize = 224


# ----------
# train results
results_path = '/home/kswada/kw/mvtech_ad/patchcore_official/results'
log_project = 'MVTecAD_Results'
log_group = f'IM224_MBNV2100_B2-3_P01_D1024-1024_PS-3'


# ----------------------------------------------------------------------------------------------------------------------
# base setting-2:  select category
# ----------------------------------------------------------------------------------------------------------------------

idx = 9
mvtec_classname = _CLASSNAMES[idx]

print(f'classname: {mvtec_classname}')

dataset_name = f'{name}_{mvtec_classname}'


# ----------
# trained patch core path
patch_core_path = os.path.join(results_path, log_project, log_group, 'models', dataset_name)

# evaluated results save path
results_save_path = os.path.join(
    '/home/kswada/kw/mvtech_ad/patchcore_official/evaluated_results',
    log_group, dataset_name)

print(patch_core_path)
print(results_save_path)

if not os.path.exists(results_save_path):
    os.makedirs(results_save_path)


########################################################################################################################
# ----------------------------------------------------------------------------------------------------------------------
# data loaders
# ----------------------------------------------------------------------------------------------------------------------

batch_size = 32
seed = 0

test_dataset = MVTecDataset(
    data_path,
    classname=mvtec_classname,
    resize=resize,
    imagesize=cropsize,
    split=DatasetSplit.TEST,
    seed=seed,
)

test_dataloader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
)


########################################################################################################################
# ----------------------------------------------------------------------------------------------------------------------
# evaluate test data by created faiss index at training
# ----------------------------------------------------------------------------------------------------------------------

result_collect = []

device_context = (
    torch.cuda.device("cuda:{}".format(device.index))
    if "cuda" in device.type.lower()
    else contextlib.suppress()
)

fix_seeds(seed, device)

with device_context:

    torch.cuda.empty_cache()

    ##################################################################
    loaded_patchcores = []
    gc.collect()
    n_patchcores = len(
        [x for x in os.listdir(patch_core_path) if ".faiss" in x]
    )
    if n_patchcores == 1:
        nn_method = FaissNN(faiss_on_gpu, faiss_num_workers)
        patchcore_instance = PatchCore(device)
        patchcore_instance.load_from_path(
            load_path=patch_core_path, device=device, nn_method=nn_method
        )
        loaded_patchcores.append(patchcore_instance)
    else:
        for i in range(n_patchcores):
            nn_method = FaissNN(faiss_on_gpu, faiss_num_workers)
            patchcore_instance = PatchCore(device)
            patchcore_instance.load_from_path(
                load_path=patch_core_path,
                device=device,
                nn_method=nn_method,
                prepend="Ensemble-{}-{}_".format(i + 1, n_patchcores),
            )
            loaded_patchcores.append(patchcore_instance)
    ##################################################################

    aggregator = {"scores": [], "segmentations": []}
    for i, PC in enumerate(loaded_patchcores):
        torch.cuda.empty_cache()
        print(
            "Embedding test data with models ({}/{})".format(
                i + 1, len(loaded_patchcores)
            )
        )
        scores, segmentations, labels_gt, masks_gt = PC.predict(test_dataloader)
        aggregator["scores"].append(scores)
        aggregator["segmentations"].append(segmentations)

    scores = np.array(aggregator["scores"])
    min_scores = scores.min(axis=-1).reshape(-1, 1)
    max_scores = scores.max(axis=-1).reshape(-1, 1)
    scores = (scores - min_scores) / (max_scores - min_scores)
    scores = np.mean(scores, axis=0)

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
    segmentations = (segmentations - min_scores) / (max_scores - min_scores)
    segmentations = np.mean(segmentations, axis=0)

    anomaly_labels = [
        x[1] != "good" for x in test_dataloader.dataset.data_to_iterate
    ]

    ##################################################################
    # Plot images.
    image_paths = [
        x[2] for x in test_dataloader.dataset.data_to_iterate
    ]
    mask_paths = [
        x[3] for x in test_dataloader.dataset.data_to_iterate
    ]

    def image_transform(image):
        in_std = np.array([0.229, 0.224, 0.225]).reshape(-1, 1, 1)
        in_mean = np.array([0.485, 0.456, 0.406]).reshape(-1, 1, 1)
        image = test_dataloader.dataset.transform_img(image)
        return np.clip((image.numpy() * in_std + in_mean) * 255, 0, 255).astype(np.uint8)

    def mask_transform(mask):
        return test_dataloader.dataset.transform_mask(mask).numpy()

    plot_segmentation_images(
        results_save_path,
        image_paths,
        segmentations,
        scores,
        mask_paths,
        image_transform=image_transform,
        mask_transform=mask_transform,
    )
    ##################################################################

    print("Computing evaluation metrics.")
    # Compute Image-level AUROC scores for all images.
    auroc = compute_imagewise_retrieval_metrics(scores, anomaly_labels)["auroc"]

    result_collect.append(
        {
            "dataset_name": dataset_name,
            "instance_auroc": auroc,
        }
    )

    for key, item in result_collect[-1].items():
        if key != "dataset_name":
            print("{0}: {1:3.3f}".format(key, item))

    del loaded_patchcores
    gc.collect()


# ----------------------------------------------------------------------------------------------------------------------
# arrange metrics and save
# ----------------------------------------------------------------------------------------------------------------------

result_metric_names = list(result_collect[-1].keys())[1:]

result_dataset_names = [results["dataset_name"] for results in result_collect]

result_scores = [list(results.values())[1:] for results in result_collect]

compute_and_store_final_results(
    results_save_path,
    result_scores,
    column_names=result_metric_names,
    row_names=result_dataset_names,
)
