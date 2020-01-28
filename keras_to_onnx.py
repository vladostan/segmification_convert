# -*- coding: utf-8 -*-

# In[]: Set GPU
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# In[]: Imports
import pickle
import json
from tqdm import tqdm
import matplotlib.pylab as plt
from glob import glob
import numpy as np
from PIL import Image
import segmentation_models as sm
from keras import optimizers
import cv2

# In[]:
import numpy as np
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
import keras2onnx
import onnxruntime

# In[]: Parameters
classification_classes = 1
segmentation_classes = 1

resize = True
input_shape = (256, 640, 3) if resize else (512, 1280, 3)

backbone = 'resnet18'

weights = "2019-09-30 17-32-13"

# In[]:
dataset_dir = "../datasets/supervisely/kisi/"

subdirs = ["2019-05-22"]

obj_class_to_machine_color = dataset_dir + "obj_class_to_machine_color.json"

with open(obj_class_to_machine_color) as json_file:
    object_color = json.load(json_file)

ann_files = []
for subdir in subdirs:
    ann_files += [f for f in glob(dataset_dir + subdir + '/ann/' + '*.json', recursive=True)]
    
print("DATASETS USED: {}".format(subdirs))
print("TOTAL IMAGES COUNT: {}\n".format(len(ann_files)))
    
# In[]:
def get_image(path, label = False, resize = False):
    img = Image.open(path)
    if resize:
        img = img.resize(input_shape[:2][::-1])
    img = np.array(img) 
    if label:
        return img[..., 0]
    return img  

with open(ann_files[0]) as json_file:
    data = json.load(json_file)
    tags = data['tags']
    objects = data['objects']
    
img_path = ann_files[0].replace('/ann/', '/img/').split('.json')[0]
label_path = ann_files[0].replace('/ann/', '/masks_machine/').split('.json')[0]

print("Images dtype: {}".format(get_image(img_path).dtype))
print("Labels dtype: {}\n".format(get_image(label_path, label = True).dtype))
print("Images shape: {}".format(get_image(img_path, resize = True if resize else False).shape))
print("Labels shape: {}\n".format(get_image(label_path, label = True, resize = True if resize else False).shape))

# In[]:
model = sm.Linknet_bottleneck_crop(backbone_name=backbone, input_shape=input_shape, classification_classes=classification_classes, segmentation_classes = segmentation_classes, classification_activation = 'sigmoid', segmentation_activation='sigmoid')
model.load_weights('weights/' + weights + '.hdf5')

# In[]:
from keras.utils import plot_model
plot_model(model, to_file='keras_model.png')

# In[]:
onnx_model = keras2onnx.convert_keras(model, model.name)

# In[]:
import onnx
onnx.save_model(onnx_model, 'model.onnx')

# In[]:
preprocessing_fn = sm.get_preprocessing(backbone)

i = 228
x = get_image(ann_files[i].replace('/ann/', '/img/').split('.json')[0], resize = True if resize else False)
x = preprocessing_fn(x.astype(np.single))
x = np.expand_dims(x, axis=0)

# In[]:
y_pred = model.predict(x)

y1_pred = y_pred[1]
y2_pred = y_pred[0]

plt.imshow(np.squeeze(y2_pred > 0.5))
    
offlane = np.squeeze(y1_pred) > 0.5

print("OFFLANE PREDICT: {}".format(offlane))
# In[]: runtime prediction
content = onnx_model.SerializeToString()
sess = onnxruntime.InferenceSession(content)
x = x if isinstance(x, list) else [x]
feed = dict([(input.name, x[n]) for n, input in enumerate(sess.get_inputs())])
pred_onnx = sess.run(None, feed)

# In[]:
y1_pred_onnx = pred_onnx[1]
y2_pred_onnx = pred_onnx[0]

plt.imshow(np.squeeze(y2_pred_onnx > 0.5))
    
offlane = np.squeeze(y1_pred_onnx) > 0.5

print("OFFLANE PREDICT ONNX: {}".format(offlane))

# In[]:
import time

num = 1000

start_time = time.time()

for i in tqdm(range(num)):
    
    x = np.zeros((1,input_shape[0],input_shape[1],3))
    x = preprocessing_fn(x.astype(np.single))
    
    x = x if isinstance(x, list) else [x]
    feed = dict([(input.name, x[n]) for n, input in enumerate(sess.get_inputs())])
    pred_onnx = sess.run(None, feed)
                    
print("--- {} seconds ---".format(time.time() - start_time))
print("--- {} fps ---".format(num/(time.time() - start_time)))