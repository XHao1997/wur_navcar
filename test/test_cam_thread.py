import os
import sys
from PIL import Image
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
from utils import image_process
from utils.file import save_file

""

# 对于图像的处理方法

# Directory where files are saved
directories = {
    'rgb': 'rgb_cali/',
    'depth': 'depth_cali',
    'eye_to_hand': 'eye_to_hand/',
    'joint1_nn': 'joint1_nn/'
}

        
if __name__ == "__main__":
    # 启动 获取摄像头画面的 线程
    cap = Camera()
    rgb_img, depth_img = cap.__capture()
    
    cv2.imwrite('test.jpg', rgb_img)
    save_file(directories, rgb_img, 'rgb')