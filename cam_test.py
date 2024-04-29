from camera import Camera
import time
import cv2
if __name__ == "__main__":

    kinect = Camera()
    for i in range(1):
        # rgb_img = kinect.capture_rgb_img()
        #
        # depth_img = kinect.capture_depth_img()
        rgb_img = cv2.imread('test_img/rgb_80.png')
        depth_img = cv2.imread('test_img/depth_80.png',cv2.IMREAD_UNCHANGED)

        start = time.time()
        print(kinect.execute_task(rgb_img, depth_img, 'block', debug=False))

        # kinect.execute_task(rgb_img, depth_img, 'default', debug=True)
        end = time.time()
        print('time:', end - start)