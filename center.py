#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from openni import openni2
import cv2
from datetime import datetime
import platform
import numpy as np
import array
from PIL import Image
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from matplotlib import cm
from camera import camera
import threading
from queue import Queue
import time
from queue import Queue
from remoted_car import remoted_car
# Initializing a queue
q = Queue(maxsize = 3)
  


# Initialize OpenNI
if platform.system() == "Windows":
    openni2.initialize("C:/Program Files (x86)/OpenNI2/Redist")  # Specify path for Redist
else:
    openni2.initialize()  # can also accept the path of the OpenNI redistribution
# Connect and open device
dev = openni2.Device.open_any()
# Create depth stream
depth_stream = dev.create_depth_stream()
depth_stream.start()
#outfile = open("depth-"+str(datetime.timestamp(datetime.now())) + ".hex","wb")

camera0 =  camera()
car = remoted_car()
def camera_pusher():
    while True:
        dist = camera0.get_center_dist(depth_stream)
        q.put(dist)
        print("push:{:.3f}m".format(dist))
        
def car_listener():
    while True:
        dist = q.get()
        car.get_c(dist)
        car.move()
        # print("listen:{:.3f}m".format(dist))
    
    
thread_camera = threading.Thread(target=camera_pusher)
thread_camera.start()

thread_car = threading.Thread(target=car_listener)
thread_car.start()