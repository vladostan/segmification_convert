# -*- coding: utf-8 -*-

import onnx

import sys
sys.path.append('../onnx-tensorrt/.')

import onnx_tensorrt.backend as backend
import numpy as np

model = onnx.load("model.onnx")
engine = backend.prepare(model, device='CUDA:0')
input_data = np.random.random(size=(1, 3, 256, 640)).astype(np.float32)
output_data = engine.run(input_data)[0]
print(output_data)
print(output_data.shape)