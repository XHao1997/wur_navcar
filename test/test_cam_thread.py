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
""

# 对于图像的处理方法

        
if __name__ == "__main__":
    # 启动 获取摄像头画面的 线程
    frame_queue = queue.LifoQueue()
    kinect = Camera(frame_queue)
    kinect.run()
    # 启动处理（显示）摄像头画面的线程
    thread_show = Thread(target=image_process.show_frame, args=(frame_queue,))
    thread_show.start()
    thread_show.join()
    kinect.stop()
