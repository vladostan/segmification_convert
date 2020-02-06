# -*- coding: utf-8 -*-

from cyber_py3 import cyber
from modules.drivers.proto.sensor_image_pb2 import Image
import numpy as np
import cv2
import segmentation_models as sm
    
def callback(image):
    """
    Reader message callback.
    """    
    rgb_img = cv2.cvtColor(np.resize(np.frombuffer(image.data, dtype=np.uint8), (image.height, image.width, 3)), cv2.COLOR_BGR2RGB)
    x = cv2.resize(rgb_img, input_shape[:2][::-1])
    
    print(x.shape)
    print(image.measurement_time)
    
    y = model.predict(np.expand_dims(x, axis=0))
    
    y1 = y[1]
    y2 = y[0]
    
    offlane = np.squeeze(y1) > 0.5
    mask = np.squeeze(y2 > 0.5)
    
    text = 'Prediction: OFF LANE' if offlane else 'Prediction: IN LANE'
    
    out_img = np.vstack((x, cv2.cvtColor(255*mask.astype(np.uint8), cv2.COLOR_GRAY2RGB)))
    
    (text_width, text_height) = cv2.getTextSize(text, font, fontScale=fontScale, thickness=1)[0]
    box_coords = ((textPosition[0] - 10, textPosition[1] + 10), (textPosition[0] + text_width + 5, textPosition[1] - text_height - 10))
    cv2.rectangle(out_img, box_coords[0], box_coords[1], (0,0,0), cv2.FILLED)
    cv2.putText(out_img, text, textPosition, font, fontScale, fontColor, lineType)
    
    cv2.imwrite(f"data/vis/{image.measurement_time}.png", out_img)

def test_listener_class():
    """
    Reader message.
    """
    print("=" * 120)
    test_node = cyber.Node("listener")
    test_node.create_reader("/apollo/sensor/camera/front_6mm/image", Image, callback)
    test_node.spin()

if __name__ == '__main__':

    resize = True
    input_shape = (256, 640, 3) if resize else (512, 1280, 3)

    classification_classes = 1
    segmentation_classes = 1
    
    classification_activation = 'sigmoid' if classification_classes == 1 else 'softmax'
    segmentation_activation = 'sigmoid' if segmentation_classes == 1 else 'softmax'

    backbone = 'resnet18'

    weights = "2019-09-30 17-32-13"
    
    model = sm.Linknet_bottleneck_crop(backbone_name=backbone, input_shape=(256, 640, 3), classification_classes=classification_classes, segmentation_classes=segmentation_classes, classification_activation=classification_activation, segmentation_activation=segmentation_activation)
    model.load_weights('weights/' + weights + '.hdf5')
    
    preprocessing_fn = sm.get_preprocessing(backbone)
    
    model._make_predict_function() 
    
    print("START ZERO PREDICT")
    model.predict(np.zeros(((1,) + input_shape), dtype=np.uint8))
    print("END ZERO PREDICT")
    
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    fontScale              = 1
    fontColor              = (255,255,255)
    lineType               = 2
    textPosition           = (5, 30)

    cyber.init()
    test_listener_class()
    cyber.shutdown()