import os
import sys
from PIL import Image
PROJECT_PATH = os.getcwd()
SOURCE_PATH = os.path.join(
    PROJECT_PATH
)
sys.path.append(SOURCE_PATH)

from module.camera import Camera
import time
import numpy as np
import cv2
import freenect
if __name__ == "__main__":  
    kinect = Camera()
    
    
    cv2.namedWindow('Video')
    cv2.namedWindow('Depth')



    while True:
        async_result,async_result2 = kinect.capture()
        cv2.imshow('Depth', async_result2.get())
        cv2.imshow('Video', async_result.get())
        ch = cv2.waitKey(25)
        if ch== ord('q'):
            break
        # data.show()
        # print(async_result2.get())
    kinect.close()

    #     depth_img = kinect.capture_depth_img() 
    #     print('depth image captured')
    #     center = kinect.execute_task(rgb_img,depth_img,'block',debug=False)
    #     block_center.append(np.round(center))
    # print(np.median(block_center,axis=0))

    # # for i in range(10):
    # #     time.sleep(.1)

    # #     depth_img = kinect.capture_depth_img() 
    # #     print('depth image captured')
    # #     center = kinect.execute_task(rgb_img,depth_img,'block',debug=False)
    # #     block_center.append(np.round(center))
    # # print(np.median(block_center,axis=0))
    # # for i in range(10):
    # #     time.sleep(.1)
    # #     depth_img = kinect.capture_depth_img() 
    # #     print('depth image captured')
    # #     center = kinect.execute_task(rgb_img,depth_img,'block',debug=False)
    # #     block_center.append(np.round(center))
    # # print(np.median(block_center,axis=0))

    # for i in range(10):
    #     time.sleep(.1)
    #     depth_img = kinect.capture_depth_img() 
    #     print('depth image captured')
    #     center = kinect.execute_task(rgb_img,depth_img,'block',debug=False)
    #     block_center.append(np.round(center))
    # print(np.median(block_center,axis=0))
