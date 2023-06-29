#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from openni import openni2
from datetime import datetime
import platform
import numpy as np
import array
from PIL import Image
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from matplotlib import cm

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

 

while True:
    frame = depth_stream.read_frame()
    frame_data = frame.get_buffer_as_uint16()
    frame_data = frame.get_buffer_as_uint16()
    Z = np.asarray(frame_data).reshape((80, 60)) 
    row = Z.shape[0]
    col = Z.shape[1]
    r = 10
    center_area = Z[row//2-r:row//2+r,col//2-r:col//2+r]
    len = np.argwhere(np.isnan(center_area.astype(int)))
    center_dist = center_area.sum()/(row*col-len.shape[0])

    print("Center pixel distance is {} m".format(center_dist/1000))



depth_stream.stop()
openni2.unload()
