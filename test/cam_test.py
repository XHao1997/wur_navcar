from module.camera import Camera
import time
import numpy as np
import freenect
if __name__ == "__main__":  
    
    
    kinect = Camera()
    kinect.initialise_yolo()
    leaf_center = kinect.get_leaf_center()
    











    # block_center = []
    # rgb_img = kinect.capture_rgb_img()
    # print('rgb image captured')

    # for i in range(10):

    #     depth_img = kinect.capture_depth_img() 
    #     print('depth image captured')
    #     center = kinect.execute_task(rgb_img,depth_img,'block',debug=False)
    #     block_center.append(np.round(center))
    # print(np.median(block_center,axis=0))

    # # for i in range(10):
    # #     time.sleep(.1)

    # #     depth_img = kinect.capture_depth_img() 
    # #     print('depth image captured')
    # #     center = kinect.execute_task(rgb_img,depth_img,'block',debug=False)
    # #     block_center.append(np.round(center))
    # # print(np.median(block_center,axis=0))
    # # for i in range(10):
    # #     time.sleep(.1)
    # #     depth_img = kinect.capture_depth_img() 
    # #     print('depth image captured')
    # #     center = kinect.execute_task(rgb_img,depth_img,'block',debug=False)
    # #     block_center.append(np.round(center))
    # # print(np.median(block_center,axis=0))

    # for i in range(10):
    #     time.sleep(.1)
    #     depth_img = kinect.capture_depth_img() 
    #     print('depth image captured')
    #     center = kinect.execute_task(rgb_img,depth_img,'block',debug=False)
    #     block_center.append(np.round(center))
    # print(np.median(block_center,axis=0))
