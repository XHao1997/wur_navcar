import os
from time import time
import joblib
from glob import glob
from functools import partial
import cv2
from kinect_smoothing import HoleFilling_Filter, Denoising_Filter
from kinect_smoothing import Crop_Filter, Smooth_Filter, Motion_Sampler
from kinect_smoothing.utils import plot_image_frame

def standard_pipeline(image_frame):
    hole_filter = HoleFilling_Filter(flag='mode', min_valid_depth=50, min_valid_neighbors=5,radius=0.2)
    image_frame = hole_filter.smooth_image(image_frame)
    hole_filter = HoleFilling_Filter(flag='ns', min_valid_depth=0, max_valid_depth=255., radius=1)
    image_frame = hole_filter.smooth_image(image_frame)
    noise_filter = Denoising_Filter(flag='gaussian')
    image_frame = noise_filter.smooth_image(image_frame)
    return image_frame


def kinect_preprocess(img_path):
    t1 = time()
    image_frame = cv2.imread(img_path)
    image_frame = image_frame[:,:,0]
    # print(image_frame.shape)
    image_frame = standard_pipeline(image_frame)


    print('preprocessed image %s, time-cost %f s' % (img_path, time() - t1))
    while (1):
        cv2.imshow('img', image_frame)
        k = cv2.waitKey(33)
        if k == 27:  # Esc key to stop
            break
        elif k == -1:  # normally -1 returned,so don't print it
            continue
        else:
            print
            k  # else print its value
    # cv2.imshow('image', image_frame)

if __name__ == '__main__':
    kinect_preprocess('data/depth.png')
