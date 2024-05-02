import os
import sys
from PIL import Image
PROJECT_PATH = os.getcwd()
SOURCE_PATH = os.path.join(
    PROJECT_PATH
)
sys.path.append(SOURCE_PATH)
import time
import numpy as np
import cv2
from module.AI_model import AI_model_factory,Yolo,Mobile_SAM
creator = AI_model_factory()
yolo = creator.create_model(Yolo)
mobile_sam = creator.create_model(Mobile_SAM)

path = 'rgb_cali/rgb_81.png'
image = Image.open(path)
# image = cv2.imread(path).astype(np.uint8)
start = time.time()
for i in range(100):
    yolo_results = yolo.predict(image)
end = time.time()
print(end-start,'s')
# yolo.visualise_result(image,yolo_results)
# start = time.time()
sam_results = mobile_sam.predict(image,yolo_results)
# end = time.time()
# print(end-start)
mobile_sam.visualise_result(image, sam_results)
# plt.imshow(sam_results)
# import os,sys
# project_dir=os.path.dirname(os.path.abspath(sys.executable))

# print(project_dir)