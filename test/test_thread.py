#!/usr/bin/env python
import os
import sys
import cv2
PROJECT_PATH = os.getcwd()
SOURCE_PATH = os.path.join(
    PROJECT_PATH
)
sys.path.append(SOURCE_PATH)
import matplotlib.pyplot as plt

from module.cam_server import CamServer
from module.camera import Camera

from utils import image_process, file
import time    
server = CamServer()
server.run()
while True:
    yolo_result = server.get_detection_result()
    image = yolo_result['image']
    image = image_process.draw_yolo_frame_cv(image,yolo_result)
    cv2.namedWindow('yolo')
    cv2.imshow('yolo', image)
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        cv2.destroyWindow('yolo') 
        server.stop()
        break