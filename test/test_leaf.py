import os
import sys
from PIL import Image
import matplotlib.pyplot as plt
PROJECT_PATH = os.getcwd()
SOURCE_PATH = os.path.join(
    PROJECT_PATH
)
sys.path.append(SOURCE_PATH)
import time
import numpy as np
import cv2
from module.AI_model import AI_model_factory,Yolo,Mobile_SAM
from utils.leaf import get_leaf_center, draw_circle
from utils.file import save_file

# Directory where files are saved
directories = {
    'rgb': 'rgb_cali/',
    'depth': 'depth_cali',
    'eye2hand': 'eye_to_hand/',
    'joint1': 'joint1_nn/',
    'seg': 'seg/'
}
creator = AI_model_factory()
yolo = creator.create_model(Yolo)
mobile_sam = creator.create_model(Mobile_SAM)

path = 'rgb_cali/rgb_81.png'
image = Image.open(path)
yolo_results = yolo.predict(image)
yolo.visualise_result(image, yolo_results)
sam_results = mobile_sam.predict(image,yolo_results)
# # save_file(directories,sam_results,'seg')
# sam_results = cv2.imread('seg/seg_02.png',cv2.IMREAD_GRAYSCALE )
# plt.imshow(sam_results)
# # plt.show()
mask = np.zeros_like(sam_results)
point = get_leaf_center(sam_results,0)
# print(point)
for p in point:
    mask= draw_circle(p, mask)
plt.imshow(mask)
plt.show()
