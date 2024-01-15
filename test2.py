#!/usr/bin/env python
import freenect
import cv2
import frame_convert2

cv2.namedWindow('Depth')
cv2.namedWindow('Video')
print('Press ESC in window to stop')


def get_depth():
    return frame_convert2.pretty_depth_cv(freenect.sync_get_depth()[0])


def get_video():
    return frame_convert2.video_cv(freenect.sync_get_video()[0])


while 1:
	depth_img = get_depth()
	rgb_img = get_video()
	cv2.imshow('Depth', rgb_img)
	cv2.imshow('Video', depth_img)
	ch =  cv2.waitKey(25)
	if ch == ord('s'):
		cv2.imwrite('rgb.png',rgb_img)
		cv2.imwrite('depth.png',depth_img)
		cv.destroyAllWindows()
		break
