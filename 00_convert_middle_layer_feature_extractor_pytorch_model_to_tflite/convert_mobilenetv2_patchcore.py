import argparse
from glob import glob
import json
import os
import time
import cv2

import torchvision.models as models

if 'INTEL_OPENVINO_DIR' not in os.environ:
    import numpy as np
    import onnx
    from onnx_tf.backend import prepare
    from PIL import Image
    import timm
    import tensorflow as tf
    import torch
    from torch.utils.data import DataLoader

    from utils.utils import Extractor, VideoConcatImageData, Extractor_for_PatchCore


class Converter:
    def __init__(self, args):
        self.expid = args.expid
        self.image_size = args.image_size
        self.batchsize = args.batchsize
        self.pytorch_model_path = args.pytorch_model_path
        self.image_path = args.image_path

        ####################################################
        self.layer_names = ['blocks.2', 'blocks.3']
        ####################################################

        self.exp_path = f'static/{self.expid}'
        self.pytorch_path = f'{self.exp_path}/model.pth'
        self.onnx_ori_path = f'{self.exp_path}/model_ext_ori.onnx'
        self.onnx_path = f'{self.exp_path}/model_ext.onnx'
        self.vino_path = f'{self.exp_path}/model_ext.vino'
        self.tf_path = f'{self.exp_path}/model_ext.tf'
        self.tflite_path = f'{self.exp_path}/model_ext.tflite'
        self.tflite_f32_path = f'{self.exp_path}/model_ext_f32.tflite'
        self.edge_path = f'{self.exp_path}/model_ext.edge'

        self.input_shape = (self.batchsize, 3, self.image_size, self.image_size)

    def pytorch_onnx_tf_tflite_edge(self):
        self.load_pytorch_model()
        self.pytorch2onnxori()
        self.onnx_simplify()
        self.onnx2tf()
        self.tf2tflite(self.tf_path, self.tflite_path)
        self.tflite2edge(self.tflite_path, self.edge_path, 'a')

    def pytorch2edge(self):
        if 'INTEL_OPENVINO_DIR' in os.environ:
            self.onnx2vino()

        elif not os.path.exists(self.vino_path):
            self.load_pytorch_model()
            self.pytorch2onnxori()
            self.onnx_simplify()

        else:
            self.vino2tf()
            self.tf2tflitef32()
            self.tf2tflite()
            self.tflite2edge()

    def load_pytorch_model(self):

        ####################################################
        pytorch_model = timm.create_model('mobilenetv2_100', pretrained=True)
        #pytorch_model = timm.create_model('mobilenetv2_100', pretrained=False)
        #pytorch_model.load_state_dict(torch.load(self.pytorch_model_path, map_location='cpu'))
        #pytorch_model = models.wide_resnet50_2(pretrained=True)

        self.pytorch_model = pytorch_model
        ####################################################

        #self.pytorch_model = Extractor(pytorch_model, self.layer_names)
        self.pytorch_model = Extractor_for_PatchCore(pytorch_model, self.layer_names)

    def pytorch2onnxori(self):
        print('Converting pytorch model to onnx original model...')
        start_time = time.perf_counter()
        torch.onnx.export(model=self.pytorch_model,
                          args=torch.ones(self.input_shape),
                          f=self.onnx_ori_path,
                          input_names=['input'],
                          output_names=['output'])
        passed_time = time.perf_counter() - start_time
        print(f'Converted pytorch model to onnx original model using {passed_time} seconds.')

    def onnx_simplify(self):
        os.system(f'python3 -m onnxsim {self.onnx_ori_path} {self.onnx_path}')

    def onnx2tf(self):
        print('Converting onnx model to tf model...')
        start_time = time.perf_counter()
        onnx_model = onnx.load(self.onnx_path)
        onnx.checker.check_model(onnx_model)
        prepare(onnx_model).export_graph(self.tf_path)
        passed_time = time.perf_counter() - start_time
        print(f'Converted onnx model to tf model using {passed_time} seconds.')

    def onnx2vino(self):
        os.system('python3 ${INTEL_OPENVINO_DIR}/deployment_tools/model_optimizer/mo.py '\
                  f'--input_model {self.onnx_path} '\
                  f'--input_shape "[{self.batchsize}, 3, 224, 224]" '\
                  f'--output_dir {self.vino_path} '\
                  '--data_type FP32')

    def vino2tf(self):
        os.system(f'openvino2tensorflow --model_path {self.vino_path}/model_ext.xml '\
                  f'--model_output_path {self.tf_path} '\
                  '--output_saved_model')

    def tf2tflitef32(self):
        print('Converting tf model to tflite f32 model...')
        start_time = time.perf_counter()

        converter = tf.lite.TFLiteConverter.from_saved_model(self.tf_path)
        tflite_f32_model = converter.convert()
        with open(self.tflite_f32_path, 'wb') as f:
            f.write(tflite_f32_model)

    def tf2tflite(self):
        print('Converting tf model to tflite model...')
        start_time = time.perf_counter()

        MEAN = np.array([0.485, 0.456, 0.406])[None, None, None, :]
        STD = np.array([0.229, 0.224, 0.225])[None, None, None, :]
        img_paths = sorted(glob(self.image_path))

        def representative_dataset():
            for bi in range(len(img_paths) // self.batchsize):
                imgs = np.zeros((self.batchsize, 224, 224, 3), dtype=np.float32)
                for ii in range(self.batchsize):
                    img = cv2.imread(img_paths[bi * self.batchsize + ii])
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img,
                                     (self.image_size, self.image_size),
                                     interpolation=cv2.INTER_AREA)
                    imgs[ii] = img
                imgs = (imgs / 255.0 - MEAN) / STD
                yield [imgs.astype(np.float32)]

        converter = tf.lite.TFLiteConverter.from_saved_model(self.tf_path)
        converter.experimental_new_converter = True

        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset

        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8

        tflite_model = converter.convert()
        with open(self.tflite_path, 'wb') as f:
            f.write(tflite_model)

        passed_time = time.perf_counter() - start_time
        print(f'Converted tf model to tflite model using {passed_time} seconds.')

    def tflite2edge(self):
        start_time = time.perf_counter()

        os.makedirs(self.edge_path, exist_ok=True)
        os.system(f'edgetpu_compiler -sa {self.tflite_path} '
                  f'-o {self.edge_path} | tee {self.edge_path}/edge.log')

        passed_time = time.perf_counter() - start_time
        print(f'Compiled using {passed_time} seconds.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--expid', '-i', type=str, default='0000')
    parser.add_argument('--image_size', '-is', type=int, default=224)
    parser.add_argument('--batchsize', '-bs', type=int, default=1)
    parser.add_argument('--pytorch_model_path', type=str, default='./model.pth')
    parser.add_argument('--image_path', type=str, default='./rep_dataset/*png')
    args = parser.parse_args()

    conv = Converter(args)
    conv.pytorch2edge()
