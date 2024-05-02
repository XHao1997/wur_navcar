#import the necessary modules
import freenect
import frame_convert2
import cv2
import numpy as np
import time 
#function to get RGB image from kinect
def get_video():
    rgb, _ = freenect.sync_get_video(0, freenect.VIDEO_RGB)
    return rgb.astype(np.uint8)
#function to get depth image from kinect
def get_depth():
    array,_ = freenect.sync_get_depth(0,freenect.FREENECT_DEPTH_11BIT)
    return array.astype(np.uint8)


def get_ir():
    array,_ = freenect.sync_get_video(0, freenect.VIDEO_IR_10BIT)
    return array.astype(np.uint8)
def pretty_depth(depth):
    np.clip(depth, 0, 2**10-1, depth)
    depth  >>=2
    #depth = 0.1236*np.tan(depth/2842.5+1.1863)
    depth=depth.astype(np.uint8)
    return depth

import os

if __name__ == "__main__":
    f=0


    # Directory where files are saved
    directory = 'ir_cali_single/'
    # directory = 'rgb_cali_single/'  
    # Get a list of files in the directory
    files = os.listdir(directory)

    while 1:
        # Filter only files with specific extensions, like jpg, png, etc.
        image_files = [file for file in files if file.endswith('.jpg') or file.endswith('.png')]

        # Find the maximum number in the filenames
        max_num = max([int(file.split('_')[1].split('.')[0]) for file in image_files]) if image_files else 0

        # Save the current file with the next number
        new_filename = f"ir_{f}.jpg"
        # new_filename = f"rgb_{f}.jpg"
        #get a frame from RGB camera
        frame = get_ir()
        # frame = get_video()
        cv2.imwrite(os.path.join(directory, new_filename),frame)
        #get a frame from depth sensor
        #frame = pretty_depth(get_depth())

        #frame = get_video()
        #display RGB image
        cv2.imshow('image', frame)
        ch =  cv2.waitKey(25)
        f = f+1
        print("saved")
        time.sleep(0.02)
        if f ==300:
            break


