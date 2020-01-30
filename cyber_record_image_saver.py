# -*- coding: utf-8 -*-

import sys

from cyber_py3 import cyber
from cyber_py3 import record
from cyber.proto import record_pb2
from cyber.proto.unit_test_pb2 import ChatterBenchmark
from modules.drivers.proto.sensor_image_pb2 import Image
import numpy as np
import cv2
    
def callback(image):
    """
    Reader message callback.
    """
    print("=" * 80)
    print(f"Image height: {image.height}")
    print(f"Image width: {image.width}")
    print("=" * 80)
    
    rgb_img = cv2.cvtColor(np.resize(np.frombuffer(image.data, dtype=np.uint8), (image.height, image.width, 3)), cv2.COLOR_BGR2RGB)
    
    cv2.imwrite("data/a.png", rgb_img)
    

def test_listener_class():
    """
    Reader message.
    """
    print("=" * 120)
    test_node = cyber.Node("listener")
    test_node.create_reader("/apollo/sensor/camera/front_6mm/image", Image, callback)
    test_node.spin()

if __name__ == '__main__':
    cyber.init()
    test_listener_class()
    cyber.shutdown()