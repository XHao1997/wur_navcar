#!/usr/bin/env python
import os
import sys
import time
PROJECT_PATH = os.getcwd()
SOURCE_PATH = os.path.join(
    PROJECT_PATH
)
sys.path.append(SOURCE_PATH)
from module.camera import Camera
from module.kinect import Kinect 
from module.AI_model import Yolo, Mobile_SAM
from utils import vision, image_process, leaf, file
from threading import Thread
import queue 




# This Python class `CamServer` contains methods for detecting and segmenting leaves using AI models
# and a camera.
class CamServer():
    def __init__(self):
        super().__init__()
        self.frame_queue = queue.Queue()
        self.yolo_queue = queue.LifoQueue()
        """ 
        camera and yolo are the models running in threads, Therefore, the queue 
        is used as an input for initialising the model and send the result out. 
        """
        ####################### yolo and sam init ################################
        self.cap = Camera(self.frame_queue)
        self.kinect = Kinect()
        self.yolo_model = Yolo(self.yolo_queue)
        # sam model used only for one-time prediction, because it takes a long time
        self.sam = Mobile_SAM()
        
        
        # "chosen_leaf_roi" and "chosen_leaf_center" 
        # are two variables sent to local qt server
        ################## temporary storage for picking action ##################
        self.sam_mask = None
        self.yolo_result = None
        self.chosen_leaf_roi = None
        self.chosen_leaf_center = None
        self.chosen_leaf = None
        self.rgb_image = None
        self.depth_image = None        
        ##################### flag for threads ##################
        self.is_running = False
        
    def run(self):
        self.is_running = True
        self.cap.run()     
        self.yolo_model.run(self.get_rgb_img(),self.get_depth_image())
    
            
    def stop(self):
        self.is_running = False
        self.cap.stop()
        self.yolo_model.stop()
        
        
    def segment_leaf(self):
        self.rgb_image =  self.yolo_queue.get()['image']
        result = self.yolo_queue.get()['result']
        self.yolo_result = result
        print(len(self.yolo_result))
        self.depth_image = self.yolo_queue.get()['depth']
        start= time.time()
        self.sam_mask = self.sam.predict(self.rgb_image,result)
        end= time.time()
        print("sam model takes {:.2f} seconds".format(end - start))
        self.yolo_queue.task_done()
        
    def __select_leaf_by(self, id):
        result = self.yolo_result[id]
        print(result)
        self.chosen_leaf_roi = image_process.get_yolo_roi(self.sam_mask, result)
        self.chosen_leaf = [result]
    
    # When this method called, the temporary storage of the server will be updated    
    def get_leaf_center_by(self,id):
        self.stop()
        self.__select_leaf_by(id)
        # self.chosen_leaf_center = leaf.get_leaf_center(roi_img)
        contours = leaf.get_cnts(self.chosen_leaf_roi)
        mask, _ = leaf.get_incircle(self.chosen_leaf_roi, contours)
        picking_point = self.kinect.get_point_xyz(mask, 
                                                self.rgb_image,self.depth_image)

        return picking_point
    
    def get_leaf_corners(self):
        
        pass    
        
        
    def get_leaves_center(self):
        pass
    
    def get_chosen_leaf_roi(self):
        return self.chosen_leaf_roi
        
    def get_detection_result(self):
        image = self.yolo_queue.get()
        return image
    
    def get_depth_image(self):
        image = self.frame_queue.get()['depth']
        return image
    
    def get_rgb_img(self):
        image = self.frame_queue.get()['image']
        return image

    def get_temp_depth_image(self):
        return self.rgb_image
    
    def get_temp_rgb_img(self):
        return self.depth_image
