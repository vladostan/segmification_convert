# -*- coding: utf-8 -*-

# In[]: Set GPU
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# In[]: Imports
import matplotlib.pylab as plt
import numpy as np
from PIL import Image
import segmentation_models as sm
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
def get_image(path, label = False, resize = False):
    img = Image.open(path)
    if resize:
        img = img.resize(input_shape[:2][::-1])
    img = np.array(img) 
    if label:
        return img[..., 0]
    return img  

image_path = '2019-05-22-001.png'

# In[]:
preprocessing_fn = sm.get_preprocessing(backbone)

# In[]: Bottleneck
model = sm.Linknet_bottleneck_crop(backbone_name=backbone, input_shape=input_shape, classification_classes=classification_classes, segmentation_classes = segmentation_classes, classification_activation = 'sigmoid', segmentation_activation='sigmoid')
model.load_weights('weights/' + weights + '.hdf5')

# In[]:
i = 228
x = get_image(image_path, resize = True if resize else False)
x = preprocessing_fn(x)
y_pred = model.predict(np.expand_dims(x,axis=0))

y1_pred = y_pred[1]
y2_pred = y_pred[0]

if visualize:
    plt.imshow(np.squeeze(y2_pred > 0.5))
    
offlane = np.squeeze(y1_pred) > 0.5

print("OFFLANE PREDICT: {}".format(offlane))

# In[]:
import sys
sys.path.append("../")
from cyber_py import cyber
from modules.common.util.testdata.simple_pb2 import SimpleMessage




# In[]:





# In[]:





# In[]:





# In[]:





# In[]:





# In[]:





