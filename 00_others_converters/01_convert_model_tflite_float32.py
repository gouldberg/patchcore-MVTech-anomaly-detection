
import os
import glob
import cv2

# import onnx
# from onnx_tf.backend import prepare

import torch
import torchextractor
import timm
import tensorflow as tf


# https://pypi.org/project/openvino-dev/
# https://pypi.org/project/openvino2tensorflow/1.34.0/


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
# convert tf to tflite  (float32)
# ---------------------------------------------------------------------------------------------------------------

tflite_f32_path = os.path.join(base_path, 'model\\model_f32.tflite')

converter = tf.lite.TFLiteConverter.from_saved_model(tf_path)

tflite_f32_model = converter.convert()

with open(tflite_f32_path, 'wb') as f:
    f.write(tflite_f32_model)

