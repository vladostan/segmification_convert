# -*- coding: utf-8 -*-

# In[1]:
import os
import numpy as np
import time
import segmentation_models as sm
from keras import optimizers
from tqdm import tqdm

# In[]
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 

# In[4]:
classification_classes = 1
segmentation_classes = 1
resize = False
input_shape = (256, 640, 3) if resize else (512, 1280, 3)
backbone = 'resnet18'
weights = "2019-09-30 17-32-13"

# In[]:
preprocessing_fn = sm.get_preprocessing(backbone)
model = sm.Linknet_bottleneck_crop(backbone_name=backbone, input_shape=input_shape, classification_classes=classification_classes, segmentation_classes = segmentation_classes, classification_activation = 'sigmoid', segmentation_activation='sigmoid')
model.load_weights('weights/' + weights + '.hdf5')

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

# In[56]:
model.predict(np.zeros((1,input_shape[0],input_shape[1],3)))

num = 1000

start_time = time.time()

for i in tqdm(range(num)):
    
    x = np.zeros((1,input_shape[0],input_shape[1],3))
    x = preprocessing_fn(x)
    model.predict(x)
                
print("--- {} seconds ---".format(time.time() - start_time))
print("--- {} fps ---".format(num/(time.time() - start_time)))