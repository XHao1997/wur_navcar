from module.camera import Camera
import time
    
if __name__ == "__main__":  
    
    
    kinect = Camera()
    rgb_img = kinect.capture_rgb_img()
    depth_img = kinect.capture_depth_img()
    start = time.time()
    kinect.execute_task(rgb_img,depth_img,'block')
    end = time.time()
    print('time:', end - start)