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
import threading
from utils import image_process, file
import time    
# import gc
# gc.set_threshold(100*1024*1024)
# image = image_process.draw_yolo_frame_cv(image,yolo_result)


server = CamServer()
server.run()

print('segment')

server.segment_leaf()
yolo_result = server.yolo_result

for i in range(len(yolo_result)):
    print(next(server.get_leaf_center_by(0)))
    print('*'*40)
    # print(next(server.get_leaf_center_by(1)))
    # print('*'*40)
    # print(next(server.get_leaf_center_by(2)))

    # print(next(generator))
    # time.sleep(10)
# while True:
#     rgb_img = server.get_rgb_img()
#     yolo_img = 
#     cv2.namedWindow('yolo')
#     cv2.imshow('yolo', rgb_img)
#     if cv2.waitKey(1) & 0xFF == ord('q'): 
#         cv2.destroyWindow('yolo') 
#         server.stop()
#         break
# print(server.get_leaf_center_by(1))
# print(server.get_leaf_center_by(1))
    
    
    
    
# sys.setrecursionlimit(2097152)    # adjust numbers
# threading.stack_size(134217728)   # for your needs

# main_thread = threading.Thread(target=main)
# main_thread.start()
# main_thread.join()
# plt.imshow(image)
# plt.show()

# print(server.get_temp_rgb_img())
# while True:
#     rgb_img = server.get_rgb_img()
    
#     cv2.namedWindow('yolo')
#     cv2.imshow('yolo', rgb_img)
#     if cv2.waitKey(1) & 0xFF == ord('q'): 
#         cv2.destroyWindow('yolo') 
#         server.stop()
#         break

# server.select_leaf_by(0)
# server.get_chosen_leaf_roi()
# sam_img = server.get_leaf_center_by(0)
# print(sam_img)
# server.join()

# plt.imshow(sam_img)
# plt.show()


# server.stop()
# yolo_test = Thread(target=test_fps,args=('yolo',server,))
# cam_test = Thread(target=test_fps,args=('cam',server,))

# yolo_test.start()
# cam_test.start()
# # Wait for both threads to finish
# yolo_test.join()
# cam_test.join()
# # camera_thread.join()
# def test_fps(name, server):
#     while True:
#         if name=='yolo':
            
#             server.get_detection_image()
#             print(name)
#         else:
#             server.get_rgb_img()
#             print(name)