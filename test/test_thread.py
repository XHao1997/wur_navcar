import os
import sys
from PIL import Image
import PIL.Image as pimg
import time
PROJECT_PATH = os.getcwd()
SOURCE_PATH = os.path.join(
    PROJECT_PATH
)
sys.path.append(SOURCE_PATH)
import matplotlib.pyplot as plt
PROJECT_PATH = os.getcwd()
SOURCE_PATH = os.path.join(
    PROJECT_PATH
)
from utils.freenect import video_cv
import freenect
import cv2

def capture_rgb_img():
    rgb_img = video_cv(freenect.sync_get_video()[0])[:, :, ::-1]
    return rgb_img
def capture_depth_img():
    depth_img = freenect.sync_get_depth()[0]
    
    return depth_img
from multiprocessing.pool import ThreadPool
cv2.namedWindow('Video')
cv2.namedWindow('Depth')

pool = ThreadPool(processes=3)


while True:
    async_result = pool.apply_async(capture_rgb_img) # tuple of args for foo
    async_result2 = pool.apply_async(capture_depth_img) # tuple of args for foo
    cv2.imshow('Depth', async_result2.get())
    
    cv2.imshow('Video', async_result.get())
    ch = cv2.waitKey(25)
    if ch== ord('q'):
        break
    # data.show()
    # print(async_result2.get())
    
pool.close()
pool.join()