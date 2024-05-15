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
import utils.leaf
# Directory where files are saved
directories = {
    'rgb': 'rgb_cali/',
    'depth': 'depth_cali',
    'eye2hand': 'eye_to_hand/',
    'joint1': 'joint1_nn/',
    'seg': 'seg/'
}
path = 'rgb_cali/rgb_81.png'
image = Image.open(path)

# This code snippet is performing the following tasks:
# This code snippet is creating instances of AI models using a factory design pattern.
creator = AI_model_factory()
# `yolo = creator.create_model(Yolo)` and `mobile_sam = creator.create_model(Mobile_SAM)` are creating
# instances of AI models using a factory design pattern.
yolo = creator.create_model(Yolo)
mobile_sam = creator.create_model(Mobile_SAM)

# `yolo_results = yolo.predict(image)` is calling the `predict` method 
# detection model) to make predictions on the input image. 
# The result of this prediction is stored as a list of Detector Classthe
yolo_results = yolo.predict(image)

yolo.visualise_result(image, yolo_results)
chosen_leaf = utils.leaf.choose_leaf(yolo_results, 2)

sam_results = mobile_sam.predict(image, chosen_leaf)
mobile_sam.visualise_result(image,sam_results)

