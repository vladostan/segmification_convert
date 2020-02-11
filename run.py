# -*- coding: utf-8 -*-

from cyber_py3 import cyber
from modules.drivers.proto.sensor_image_pb2 import Image
from modules.common.util.testdata.offlane_pb2 import OfflaneMessage
import numpy as np
import cv2
import segmentation_models as sm
    
def callback_12mm(image):

    rgb_img = cv2.cvtColor(np.resize(np.frombuffer(image.data, dtype=np.uint8), (image.height, image.width, 3)), cv2.COLOR_BGR2RGB)
    x = cv2.resize(rgb_img, input_shape[:2][::-1])
    
    print(x.shape)
    print(image.measurement_time)
    
    y = model.predict(np.expand_dims(x, axis=0))
    
    y1 = y[1]
    y2 = y[0]
    
    offlane = np.squeeze(y1) > 0.5
    mask = np.squeeze(y2 > 0.5)
    mask = cv2.cvtColor(255*mask.astype(np.uint8), cv2.COLOR_GRAY2RGB)

    mask_msg.frame_id = image.frame_id
    mask_msg.measurement_time = image.measurement_time
    mask_msg.step = image.step
    mask_msg.data = mask.tobytes()
    mask_publisher.write(mask_msg)

    offmsg.frame_id = image.frame_id
    offmsg.measurement_time = image.measurement_time
    offmsg.offlane = offlane

    offlane_tag_publisher.write(offmsg)
    
    if save:
        text = 'Prediction: OFF LANE' if offlane else 'Prediction: IN LANE'
        
        out_img = np.vstack((x, mask))
        
        (text_width, text_height) = cv2.getTextSize(text, font, fontScale=fontScale, thickness=1)[0]
        box_coords = ((textPosition[0] - 10, textPosition[1] + 10), (textPosition[0] + text_width + 5, textPosition[1] - text_height - 10))
        cv2.rectangle(out_img, box_coords[0], box_coords[1], (0,0,0), cv2.FILLED)
        cv2.putText(out_img, text, textPosition, font, fontScale, fontColor, lineType)
        
        # cv2.imwrite(f"test.png", out_img)
        cv2.imwrite(f"test.png", np.resize(np.frombuffer(mask_msg.data, dtype=np.uint8), (256, 640, 3)))


if __name__ == '__main__':

    resize = True
    save = True
    input_shape = (256, 640, 3) if resize else (512, 1280, 3)

    classification_classes = 1
    segmentation_classes = 1
    
    classification_activation = 'sigmoid' if classification_classes == 1 else 'softmax'
    segmentation_activation = 'sigmoid' if segmentation_classes == 1 else 'softmax'

    backbone = 'resnet18'

    weights = "2019-09-30 17-32-13"
    
    model = sm.Linknet_bottleneck_crop(backbone_name=backbone, input_shape=input_shape, classification_classes=classification_classes, segmentation_classes=segmentation_classes, classification_activation=classification_activation, segmentation_activation=segmentation_activation)
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

    # Node
    road_segmentation_node = cyber.Node("road_segmentation")
    
    # Publisher (talker)
    mask_publisher = road_segmentation_node.create_writer("/apollo/modules/perception/road_segmentation/mask", Image)
    offlane_tag_publisher = road_segmentation_node.create_writer("/apollo/modules/perception/road_segmentation/offlane_tag", OfflaneMessage)
    mask_msg = Image()
    mask_msg.height = input_shape[0]
    mask_msg.width = input_shape[1]
    mask_msg.encoding = 'rgb8'

    offmsg = OfflaneMessage()

    # Subscriber (listener)
    road_segmentation_node.create_reader("/apollo/sensor/camera/front_6mm/image", Image, callback_12mm)

    road_segmentation_node.spin()

    cyber.shutdown()