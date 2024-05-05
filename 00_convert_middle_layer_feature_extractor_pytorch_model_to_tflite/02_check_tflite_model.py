
import os

from src2 import *

import torch
import numpy as np
import tensorflow as tf


base_path = '~/mvtec_ad/patchcore'


##################################################################################################################
# ----------------------------------------------------------------------------------------------------------------
# load and prepare for tflite model
# ----------------------------------------------------------------------------------------------------------------

# model_path = os.path.join(base_path, 'model/model_ext_f32.tflite')
# model_path = os.path.join(base_path, 'model/model_f32.tflite')
# model_path = os.path.join(base_path, 'model/model_uint8.tflite')
# model_path = os.path.join(base_path, 'model/model_uint8_screw.tflite')
model_path = os.path.join(base_path, 'model/model_uint8_kwflexfront.tflite')

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
image_means = np.array(IMAGENET_MEAN)[None, None, :]
image_stds = np.array(IMAGENET_STD)[None, None, :]

layer_name_list = ['blocks.2', 'blocks.3']
dim_by_layer = [[1, 32, 28, 28], [1, 64, 14, 14]]

# ----------
dim_flatten_by_layer = []
for d in dim_by_layer:
    dim_flatten_by_layer.append(d[1] * d[2] * d[3])


# ----------
model = tf.lite.Interpreter(model_path=model_path)

model.allocate_tensors()


# ----------
input_details = model.get_input_details()[0]
input_idx = model.get_input_details()[0]['index']
output_idx = model.get_output_details()[0]['index']
mv_quant_params = input_details['quantization_parameters']

print(mv_quant_params)
print(mv_quant_params['scales'])
print(input_details['shape'])

mv_scale, mv_zero_point = None, None

if len(mv_quant_params['scales']) > 0:
    mv_scale, mv_zero_point = mv_quant_params['scales'][0], mv_quant_params['zero_points'][0]

print(mv_scale)
print(mv_zero_point)


# ----------------------------------------------------------------------------------------------------------------
# get one batch
# ----------------------------------------------------------------------------------------------------------------

# data set path
data_path = '~/mvtec_ad/mvtec_ad2'

# mvtec_classname = 'screw'
mvtec_classname = 'tmp'

# batch_size = 32
batch_size = 1
train_val_split = 1.0
seed = 0
# num_workers = 12
num_workers = 1

device = 'cpu'

resize = 256
cropsize = 224

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
# one batch to tflite model
# ----------------------------------------------------------------------------------------------------------------

xs = images.detach().cpu().numpy()
# remove batchsize to (3, cropsize, cropsize)
xs = np.squeeze(xs, 0)
# convert to (cropsize, cropsize, 3)
xs = np.transpose(xs, (1, 2, 0))
# (224, 224, 3)
print(xs.shape)

if len(mv_quant_params['scales']) > 0:
    xs = (xs - image_means) / (image_stds * mv_scale) + mv_zero_point
    xs = np.clip(xs, 0, 255).astype(np.uint8)

xs = xs[None, :]
# (1, 224, 224, 3)
print(xs.shape)


# ----------
model.set_tensor(input_idx, xs)
model.invoke()
_features = model.get_tensor(output_idx)

# (1, 37632), 37632 = sum(dim_flatten_by_layer)
print(_features.shape)

# ----------
dict_features = {}
features = np.zeros((1, sum(dim_flatten_by_layer)))

if len(mv_quant_params['scales']) > 0:
    features[0, :] = _features.astype(np.float32) / 255.0
else:
    features[0, :] = _features.astype(np.float32)

# (1, 37632)
print(features.shape)

# ----------
dim_to = 0
for idx, layer in enumerate(layer_name_list):
    dim_from = dim_to
    dim_to = dim_from + dim_flatten_by_layer[idx]
    d = dim_by_layer[idx]
    np_feature = features[:, dim_from:dim_to].reshape(1, d[2], d[3], d[1])
    np_feature = np.transpose(np_feature, (0, 3, 1, 2))
    dict_features[layer] = torch.from_numpy(np_feature).to(torch.float32)

print(dict_features.keys())
print(dict_features[layer_name_list[0]].shape)
print(dict_features[layer_name_list[1]].shape)
