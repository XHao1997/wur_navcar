#import the necessary modules
import freenect
import socket
import frame_convert2
import cv2
import numpy as np
import time 
import math
#function to get RGB image from kinect
def get_video():
    rgb, _ = freenect.sync_get_video(0, freenect.VIDEO_RGB)
    return rgb[:,:,::-1].astype(np.uint8)
    
#function to get depth image from kinect
def get_depth():
    array,_ = freenect.sync_get_depth()
    return array


def get_ir():
    array,_ = freenect.sync_get_video(0, freenect.VIDEO_IR_10BIT)
    return array
def pretty_depth(depth):
    
    np.clip(depth, 0, 2**10-1, depth)
    # dist =  1/(-0.0030711016* depth  + 3.3309495161)
    # dist = 123.6* np.tan ( depth/2842.5 + 1.1863 )/1000
    dist = 0.075*585/(1090-depth)*8
    print(dist[315:325,235:245])
    depth>>=2
    depth=depth.astype(np.uint8)
    
    return depth, dist



if __name__ == "__main__":
    while 1:
        #get a frame from RGB camera
        # frame = pretty_depth(get_ir())
        #get a frame from depth sensor
        frame_rgb = get_video()
        # frame, _ = pretty_depth(get_depth())
        
        # print(frame)
        #display RGB image
        # cv2.imshow('image', frame)
        cv2.imshow('Depth image', frame_rgb)
        #display depth image
        #cv2.imshow('Depth image', pretty_depth(frame_ir))
        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            
            #cv2.imwrite('rgb.jpg',pretty_depth(frame_ir))
            cv2.imwrite('cali/ir'+str(time.localtime().tm_min)+str(time.localtime().tm_sec)+'.jpg',frame)
            #print('rgb'+str(time.localtime().tm_min)+str(time.localtime().tm_sec)+'.jpg')
            #cv2.imwrite('cali/rgb'+str(time.localtime().tm_min)+str(time.localtime().tm_sec)+'.jpg',frame)
            break
    cv2.destroyAllWindows()
