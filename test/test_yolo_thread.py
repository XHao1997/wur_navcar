import os
import sys
PROJECT_PATH = os.getcwd()
SOURCE_PATH = os.path.join(
    PROJECT_PATH
)
sys.path.append(SOURCE_PATH)

import time
import cv2
import queue 
from threading import Thread
from module.camera import Camera
from utils import image_process,thread_process
from module.AI_model import AI_model_factory,Yolo
from utils import vision

# 对于图像的处理方法


if __name__ == "__main__":
    creator = AI_model_factory()
    yolo = creator.create_model(Yolo)
    
    frame_queue = queue.LifoQueue()
    result_queue = queue.LifoQueue()
    
    kinect = Camera(frame_queue)
    kinect.run()
    stop_threads = False
    thread_yolo = Thread(target=vision.get_yolo_pred, args=(yolo,frame_queue,result_queue, stop_threads, ))
    thread_yolo.start()
    
    while stop_threads==False:
        yolo_image = result_queue.get()
        cam_image = frame_queue.get()
        cv2.namedWindow("yolo", cv2.WINDOW_AUTOSIZE)    
        cv2.namedWindow("camera", cv2.WINDOW_AUTOSIZE)    
        cv2.imshow("camera", image_process.rgb2bgr(yolo_image))
        cv2.imshow("yolo", image_process.rgb2bgr(yolo_image))
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            cv2.destroyWindow('camera') 
            cv2.destroyWindow('yolo') 
            thread_process.stop_thread_now(thread_yolo)
            break
    kinect.stop()
    print('killed')
