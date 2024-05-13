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
import cv2


from multiprocessing.pool import ThreadPool
cv2.namedWindow('Video')
cv2.namedWindow('Depth')

pool = ThreadPool(processes=3)


while True:


