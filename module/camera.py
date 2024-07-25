#!/usr/bin/env python
import os
import sys
PROJECT_PATH = os.getcwd()
SOURCE_PATH = os.path.join(
    PROJECT_PATH
)
sys.path.append(SOURCE_PATH)
import time
from multiprocessing.pool import ThreadPool
from utils import freenect
from threading import Thread,Lock
import open3d as o3d
import queue

class Camera():
    def __init__(self):
        self.__pool = ThreadPool()
        # The code snippet you provided is to initialize the attriute 
        # for sending data from camera
        self.frame_queue = queue.LifoQueue()  # queue for each fps image
        self.is_running = False  # status
        self.fps = 0.0  # real-time fps
        self.__t_last = time.time() * 1000
        self.__data = {} 
        
    def capture_ir_img(self):
        return freenect.capture_ir_img()
    
    def __capture(self):
        rgb_img = self.__pool.apply_async(freenect.capture_rgb_img) # tuple of args for foo
        depth_img = self.__pool.apply_async(freenect.capture_depth_img) # tuple of args for foo
        
        return rgb_img.get(), depth_img.get()
    
    def get_rgb_images(self):
        rgb_img = self.frame_queue.get()['rgb']
        return rgb_img

    def get_depth_images(self):
        depth_img = self.frame_queue.get()['depth']
        return depth_img
            
    def get_images(self):
        imgs = self.frame_queue.get()
        return imgs  
        
        
        
    def __capture_queue(self):
        # capture image
        self.__t_last = time.time() * 1000
        lock = Lock()
        while self.is_running:
            rgb_img, depth_imag = self.__capture()
            t  = time.time() * 1000
            t_span = t - self.__t_last                
            self.fps = int(1000.0 / t_span)
            self.__data["rgb"] = rgb_img
            self.__data["depth"] = depth_imag
            self.__data["fps"] = self.fps
            self.frame_queue.put(self.__data)
            self.__t_last = t

    def run(self):
        self.is_running = True
        self.thread_capture = Thread(target=self.__capture_queue)
        self.thread_capture.start()

    def stop(self):
        self.is_running = False
        self.__pool.close()
        