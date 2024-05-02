#!/usr/bin/env python
import freenect
import cv2
import frame_convert2
import numpy as np
import time


cv2.namedWindow('Depth')
cv2.namedWindow('Video')
print('Press ESC in window to stop')


def get_depth():
    return frame_convert2.pretty_depth(freenect.sync_get_depth()[0])


def get_ir():
    array,_ = freenect.sync_get_video(0, freenect.VIDEO_IR_10BIT)
    return array.astype(np.uint8)

def get_video():
    return frame_convert2.video_cv(freenect.sync_get_video()[0])[:,:,::-1]
    
def get_depth_raw():
    
    array,_ = freenect.sync_get_depth()
    return array

def get_video_raw():
    return freenect.sync_get_video()[1]

import cv2
import freenect
import os
f = 0
# Directory where files are saved
directory_rgb = 'rgb_cali/'
directory_depth = 'depth_cali'
# Get a list of files in the directory
files = os.listdir(directory_rgb)

# Filter only files with specific extensions, like jpg, png, etc.
image_files = [file for file in files if file.endswith('.jpg') or file.endswith('.png')]

# Find the maximum number in the filenames
max_num = max([int(file.split('_')[1].split('.')[0]) for file in image_files]) if image_files else 0


while 1:
    # Save the current file with the next number
    new_filename_rgb = f"rgb_{max_num + 1+f}.png"
    new_filename_depth = f"depth_{max_num + 1+f}.png"
    depth_img,_= freenect.sync_get_depth()
    # ir_img = get_ir()
    rgb_img,_ = freenect.sync_get_video()

    cv2.imshow('Video', rgb_img)
    # cv2.imshow('Depth', depth_img)

    ch = cv2.waitKey(25)
    if ch == ord('s'):
        cv2.imwrite(os.path.join(directory_rgb, new_filename_rgb),rgb_img)
        cv2.imwrite(os.path.join(directory_depth, new_filename_depth),depth_img)
        f = f+1
        print("saved")
    elif ch == ord('q'):
        break

cv2.destroyAllWindows()

