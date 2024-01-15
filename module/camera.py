import numpy as np

class camera:
    def __init__(self):

        self.cx = 0
        self.cy = 0
        self.z = 0
        self.cx = 0
        self.cx = 0
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



