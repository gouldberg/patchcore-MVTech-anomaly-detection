
import os
import glob
import cv2

# import onnx
# from onnx_tf.backend import prepare

import torch
import torchextractor
import timm
import tensorflow as tf

import numpy as np
import itertools


# ---------------------------------------------------------------------------------------------------------------
# base setting
# ---------------------------------------------------------------------------------------------------------------

base_path = 'C:\\Users\\kosei-wada\\Desktop\\mvtec_ad\\patchcore'


# ---------------------------------------------------------------------------------------------------------------
# extractor
# ---------------------------------------------------------------------------------------------------------------

# class Extractor(torchextractor.Extractor):
#     def forward(self, *args, **kwargs):
#         _ = self.model(*args, **kwargs)
#         return torch.cat([torch.flatten(feature, start_dim=1) for feature in self.feature_maps.values()], dim=1)

class Extractor_for_PatchCore(torchextractor.Extractor):
    def forward(self, *args, **kwargs):
        _ = self.model(*args, **kwargs)
        feature_list = []
        dim_list = []
        for feature in self.feature_maps.values():
            print(feature.shape)
            feature_list.append(feature)
            dim = feature.shape[1] * feature.shape[2] * feature.shape[3]
            dim_list.append(dim)
        layer1_2D = feature_list[0].reshape(1, dim_list[0])
        layer2_2D = feature_list[1].reshape(1, dim_list[1])
        output = torch.cat([layer1_2D, layer2_2D], dim=1)
        return output


#################################################################################################################
# ---------------------------------------------------------------------------------------------------------------
# load pytorch model
# set layers
# ---------------------------------------------------------------------------------------------------------------

pytorch_model = timm.create_model('mobilenetv2_100', pretrained=True)

layers_to_extract_from = ['blocks.2', 'blocks.3']

# pytorch_model.load_state_dict(torch.load(self.pytorch_model_path, map_location='cpu'))

# pytorch_model = models.wide_resnet50_2(pretrained=True)


# ---------------------------------------------------------------------------------------------------------------
# feature extractor
# ---------------------------------------------------------------------------------------------------------------

# original extractor
# extractor = Extractor(pytorch_model, layers_to_extract_from)


# special for PatchCore
extractor = Extractor_for_PatchCore(pytorch_model, layers_to_extract_from)


# ---------------------------------------------------------------------------------------------------------------
# convert pytorch model to onnx original model
# ---------------------------------------------------------------------------------------------------------------

batchsize = 1
image_size = 224

input_shape = (batchsize, 3, image_size, image_size)
print(input_shape)

onnx_ori_path = os.path.join(base_path, 'model\\model_mobilenetv2100_orig.onnx')

torch.onnx.export(model=extractor,
                  args=torch.ones(input_shape),
                  f=onnx_ori_path,
                  input_names=['input'],
                  output_names=['output'])


# ---------------------------------------------------------------------------------------------------------------
# onnx simplify
# ---------------------------------------------------------------------------------------------------------------

onnx_path = os.path.join(base_path, 'model\\model_mobilenetv2100.onnx')

os.system(f'python -m onnxsim {onnx_ori_path} {onnx_path}')


# ---------------------------------------------------------------------------------------------------------------
# convert onnx to vino
# ---------------------------------------------------------------------------------------------------------------

vino_path = os.path.join(base_path, 'model\\model_mobilenetv2100.vino')

# os.system('python3 ${INTEL_OPENVINO_DIR}/deployment_tools/model_optimizer/mo.py ' \
#           f'--input_model {onnx_path} ' \
#           f'--input_shape "[{batchsize}, 3, 224, 224]" ' \
#           f'--output_dir {vino_path} ' \
#           '--data_type FP32')

os.system('python ./venv/Lib/site-packages/openvino/tools/mo/main.py ' \
          f'--input_model {onnx_path} ' \
          f'--input_shape "[{batchsize}, 3, 224, 224]" ' \
          f'--output_dir {vino_path} ' \
          '--data_type FP32')

# ---------------------------------------------------------------------------------------------------------------
# convert openvino to tf
# ---------------------------------------------------------------------------------------------------------------

tf_path = os.path.join(base_path, 'model\\model.tf')

os.system(f'python ./venv/Scripts/openvino2tensorflow --model_path {vino_path}\\model_mobilenetv2100.xml ' \
          f'--model_output_path {tf_path} ' \
          '--output_saved_model')


# ---------------------------------------------------------------------------------------------------------------
# convert onnx to tf
# ---------------------------------------------------------------------------------------------------------------

# tf_path = os.path.join(base_path, 'model\\model.tf')
#
# onnx_model = onnx.load(onnx_path)
#
# onnx.checker.check_model(onnx_model)
#
# prepare(onnx_model).export_graph(tf_path)


# ---------------------------------------------------------------------------------------------------------------
# prepare representative data
# ---------------------------------------------------------------------------------------------------------------

# data_path = os.path.join('C:\\Users\\kosei-wada\\Desktop\\mvtec_ad\\mvtec_ad2')
# data_path = os.path.join('C:\\Users\\kosei-wada\\Desktop\\mvtec_ad\\image_ks')
# data_path = os.path.join('C:\\Users\\kosei-wada\\Desktop\\mvtec_ad\\image_ks\\flexcable')
data_path = os.path.join('C:\\Users\\kosei-wada\\Desktop\\mvtec_ad\\mpdd')

texture_classes = ["carpet", "grid", "leather", "tile", "wood"]
object_classes = ["cable", "capsule", "hazelnut", "metal_nut", "pill", "screw", "toothbrush", "transistor", "zipper"]
others = ["bottle"]
others2 = ["flexcrop"]
others3 = ['ksiflexfront', 'ksiflexright', 'ksiflexleft']
others4 = ['ksiflexall']
others5 = ['flexcrop2']
others6 = ['ksiflexfront2', 'ksiflexright2', 'ksiflexleft2']
others7 = ['ksiflexall2']
others8 = ['cbcase1']
others9 = ['fccase2']
others10 = ['bracket_black', 'bracket_brown', 'bracket_white', 'connector', 'metal_plate', 'tubes']

# mvtec_class = texture_classes + object_classes + others + others2
# mvtec_class = texture_classes + object_classes + others
# mvtec_class = others3
mvtec_class = others10
# mvtec_class = ['hazelnut']


for cls_obj in mvtec_class:
    print(f'processing -- {cls_obj}')

    img_files_list = []

    folder_obj_cls = os.path.join(data_path, cls_obj, 'train', 'good')
    # img_files = glob.glob(os.path.join(folder_obj_cls, '*.png'))
    img_files = glob.glob(os.path.join(folder_obj_cls, '*.jpg'))
    img_files_list.append(img_files)

    img_files_list = list(itertools.chain.from_iterable(img_files_list))
    print(len(img_files_list))


    # ---------------------------------------------------------------------------------------------------------------
    # convert tf to tflite  (uint8)
    # ---------------------------------------------------------------------------------------------------------------

    tflite_path = os.path.join(base_path, f'model\\model_uint8_{cls_obj}.tflite')
    # tflite_path = os.path.join(base_path, f'model\\model_uint8_mvtecall.tflite')

    MEAN = np.array([0.485, 0.456, 0.406])[None, None, None, :]
    STD = np.array([0.229, 0.224, 0.225])[None, None, None, :]

    def representative_dataset():
        for bi in range(len(img_files_list) // batchsize):
            imgs = np.zeros((batchsize, image_size, image_size, 3), dtype=np.float32)
            for ii in range(batchsize):
                img = cv2.imread(img_files_list[bi * batchsize + ii])
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img,
                                 (image_size, image_size),
                                 interpolation=cv2.INTER_AREA)
                imgs[ii] = img
            imgs = (imgs / 255.0 - MEAN) / STD
            yield [imgs.astype(np.float32)]


    converter = tf.lite.TFLiteConverter.from_saved_model(tf_path)
    converter.experimental_new_converter = True

    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset

    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    # converter.inference_input_type = tf.float32
    # converter.inference_output_type = tf.float32

    tflite_model = converter.convert()

    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
