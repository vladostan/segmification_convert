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

# In[]: Parameters
visualize = True
save_results = True

classification_classes = 1
segmentation_classes = 1

resize = True
input_shape = (256, 640, 3) if resize else (512, 1280, 3)

backbone = 'resnet18'

random_state = 28
batch_size = 1

verbose = 1

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
preprocessing_fn = sm.get_preprocessing(backbone)

# In[]: Bottleneck
model = sm.Linknet_bottleneck_crop(backbone_name=backbone, input_shape=input_shape, classification_classes=classification_classes, segmentation_classes = segmentation_classes, classification_activation = 'sigmoid', segmentation_activation='sigmoid')
model.load_weights('weights/' + weights + '.hdf5')

# In[]: 
#from losses import dice_coef_binary_loss

losses = {
        "classification_output": "binary_crossentropy",
        "segmentation_output": sm.losses.dice_loss
}

loss_weights = {
        "classification_output": 1.0,
        "segmentation_output": 1.0
}

optimizer = optimizers.Adam(lr = 1e-4)
model.compile(optimizer=optimizer, loss=losses, loss_weights=loss_weights, metrics=["accuracy"])

# In[]:
i = 228
x = get_image(ann_files[i].replace('/ann/', '/img/').split('.json')[0], resize = True if resize else False)
x = preprocessing_fn(x)
y_pred = model.predict(np.expand_dims(x,axis=0))

with open(ann_files[i]) as json_file:
    data = json.load(json_file)
    tags = data['tags']

y1_true = 0
if len(tags) > 0:
    for tag in range(len(tags)):
        tag_name = tags[tag]['name']
        if tag_name == 'offlane':
            value = tags[tag]['value']
            if value == '1':
                y1_true = 1
                break

# In[]
y1_pred = y_pred[1]
y2_pred = y_pred[0]

if visualize:
    plt.imshow(np.squeeze(y2_pred > 0.5))
    
offlane = np.squeeze(y1_pred) > 0.5

print("OFFLANE PREDICT: {}".format(offlane))
print("OFFLANE GT: {}".format(bool(y1_true)))