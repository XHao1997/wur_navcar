import time
import zmq
import random
import numpy as np
import cv2
from utils import ssh, image_process

class LocalClient(object):
    def __init__(self):
        self.context = zmq.Context()
    def luanch(self):
        ssh.run_remote_stream()
        print('cam_server started')
        # receive work
        consumer_receiver = self.context.socket(zmq.PULL)
        consumer_receiver.connect("tcp://192.168.101.12:5557")
        # send work
        print('local pc started')
        while True:
            image = consumer_receiver.recv_pyobj()
            cv2.imshow('test', image_process.rgb2bgr(image))
            cv2.waitKey(1)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    def send_cmd_to_arm(self, cmd):
        pass

    def send_cmd_to_car(self, cmd):
        pass

    def send_cmd_to_cam(self, cmd):
        pass

    def cal_leaf_center(self):
        pass

    def cal_leaf_size(self):
        pass



client_server= LocalClient()
client_server.luanch()
