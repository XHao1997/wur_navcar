#!/usr/bin/env python
import freenect
import cv2
import frame_convert2
import numpy as np
import time


# cv2.namedWindow('Depth')
# cv2.namedWindow('Video')
# print('Press q in window to stop')


def get_video():
    return frame_convert2.video_cv(freenect.sync_get_video()[0])

def get_video_raw():
    return freenect.sync_get_video()[0]

f=0
time.sleep(2)
while 1:
	rgb_img = get_video()
	# cv2.imshow('Video', rgb_img)
	ch =  cv2.waitKey(25)	
	cv2.imwrite('pc_test/rgb_'+str(f)+'.jpg',rgb_img)
	f = f+1
	print("saved")
	time.sleep(0.05)
	if f ==200:
		break
	if ch== ord('q'):
		break
