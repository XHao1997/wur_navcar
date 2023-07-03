from openni import openni2
import cv2
from datetime import datetime
import platform
import numpy as np

class camera:
    def __init__(self):
        return

    def get_center_dist(self,depth_stream):
        frame = depth_stream.read_frame()        
        frame_data = frame.get_buffer_as_uint16()
        
        Z = np.asarray(frame_data).reshape((80, 60)) 
        row = Z.shape[0]
        col = Z.shape[1]
        r = 20
        center_area = Z[row//2-r:row//2+r,col//2-r:col//2+r]
        noise = np.argwhere(np.isnan(np.rint(center_area)))
        center_dist = center_area.sum()/((2*r)**2-noise.shape[0])/1000
        return center_dist

    def stop(self):
        self.depth_stream.stop()
        openni2.unload()
        return



