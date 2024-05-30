import copy
from typing import Any

import cv2
import numpy as np
import zmq
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtWidgets import QWidget, QGraphicsScene, QApplication
from cv2 import Mat
from numpy import ndarray, dtype, generic

from ui.LeafBot_ui import Ui_LeafBotForm
from module.AI_model import Yolo, MobileSAM
from module.msg import ARMTASK, Msg
from module.kinect import Kinect
from utils import ssh, image_process, leaf
import threading
import time
import requests


def Singleton(cls):  # This is a function that aims to implement a "decorator" for types.
    """
    cls: represents a class name, i.e., the name of the singleton class to be designed.
         Since in Python everything is an object, class names can also be passed as arguments.
    """
    instance = {}

    def singleton(*args, **kargs):
        if cls not in instance:
            instance[cls] = cls(*args, **kargs)  # If the class does not exist in the dictionary, create an instance
            # and save it in the dictionary.
        return instance[cls]

    return singleton


@Singleton
class Communicator:
    def __init__(self):
        self.context = zmq.Context()
        self.pair_cam = self.context.socket(zmq.PULL)
        self.pair_cam.connect("tcp://192.168.101.12:5555")
        self.pair_arm = self.context.socket(zmq.PAIR)
        self.pair_arm.bind("tcp://192.168.101.14:3333")

        car_ip_addr = '192.168.101.19'
        self.url = "http://" + car_ip_addr + "/js?json="

        self.lock = threading.Lock()

    def get_data(self):
        with self.lock:
            return self.pair_cam.recv_pyobj()


class ImageThread(threading.Thread):
    def __init__(self, server):
        super().__init__()
        self.rgb_image = None  # Store the latest image
        self.depth_image = None
        self.lock = threading.Lock()  # Create a lock object
        self.server = server

    def run(self):
        while True:
            frame = self.server.get_data()  # Read image from the camera
            with self.lock:  # Acquire the lock
                self.rgb_image = frame[0]  # Update the latest image
                self.depth_image = frame[1]  # Update the latest image

    def get_latest_image(self):
        with self.lock:  # Acquire the lock
            return self.rgb_image, self.depth_image  # Return the latest image


class ClientServer:
    def __init__(self):
        ssh.run_remote_stream()
        self.server = Communicator()
        self.yolo = Yolo()
        self.sam = MobileSAM()
        self.camera = Kinect()
        self.image_thread = ImageThread(self.server)
        self.image_thread.start()
        self.yolo_result = None

    def rgb_image(self):
        return self.image_thread.get_latest_image()[0]

    def depth_image(self):
        return self.image_thread.get_latest_image()[1]

    def sam_mask(self):
        image = self.image_thread.get_latest_image()[0]
        mask = self.sam.predict(image, self.yolo_result)
        return mask

    def yolo_image(self):
        image = self.image_thread.get_latest_image()[0]
        yolo_results = self.yolo.predict(image)
        self.yolo_result = yolo_results
        yolo_img = image_process.draw_yolo_frame_cv(copy.deepcopy(image), yolo_results)
        return yolo_img

    def send_cmd_arm(self, cmd):
        self.server.pair_arm.send_pyobj(cmd)

    def receive_cmd_arm(self):
        return self.server.pair_arm.recv_pyobj()

    def send_json(self, json_data):
        cmd = self.server.url + json_data
        requests.get(cmd)

    def get_leaves_location(self):
        yolo_result = self.yolo_result
        sam_mask = self.sam_mask()
        rgb_img = self.rgb_image()
        depth_img = self.depth_image()
        picking_point = []
        for id in range(len(yolo_result)):
            result = yolo_result[id]
            chosen_leaf_roi = image_process.get_yolo_roi(sam_mask, result)
            contours = leaf.get_cnts(chosen_leaf_roi)
            mask, _ = leaf.get_incircle(chosen_leaf_roi, contours)
            picking_point.append(self.camera.get_point_xyz(mask, rgb_img, depth_img))
        picking_point = np.array(picking_point)
        picking_point = picking_point[np.isfinite(picking_point)].reshape(-1, 3)
        return picking_point


class LeafBot(QWidget, Ui_LeafBotForm):
    def __init__(self):
        super().__init__()

        """ set up server """
        self.server = ClientServer()
        """ set up UI """
        print("set up ui")
        self.setupUi(self)

        self.LeafBotLogo.setPixmap(QPixmap(u"logo.jpg"))  # fix logo missing

        self.thread_show_rgb = QTimer()
        self.thread_show_rgb.timeout.connect(self.__showRgb)
        self.thread_show_yolo = QTimer()
        self.thread_show_yolo.timeout.connect(self.__show_bbox)
        self.thread_show_mask = QTimer()
        self.thread_show_mask.timeout.connect(self.__show_mask)
        self.show_img_width = self.graphicsView.width()
        self.show_img_height = self.graphicsView.height()
        '''connect button to slot'''
        self.pushButton_J1.clicked.connect(self.send_cmd_to_J1)
        self.pushButton_J2.clicked.connect(self.send_cmd_to_J2)
        self.pushButton_J3.clicked.connect(self.send_cmd_to_J3)
        self.pushButton_J4.clicked.connect(self.send_cmd_to_J4)
        self.pushButton_J5.clicked.connect(self.send_cmd_to_J5)
        self.pushButton_J6.clicked.connect(self.send_cmd_to_J6)
        self.pushButton_detect_leaf.clicked.connect(self.detect_leaf)
        self.pushButton_segment_leaf.clicked.connect(self.segment_leaf)
        self.pushButton_move_all.clicked.connect(self.send_cmd_to_all_joints)
        self.pushButton_zero_position.clicked.connect(self.send_cmd_zero_position)
        self.pushButton_read_servo.clicked.connect(self.send_cmd_to_read_servo)
        self.pushButton_move_forward.clicked.connect(self.send_cmd_to_move_forward)
        self.pushButton_move_backward.clicked.connect(self.send_cmd_to_move_backward)
        self.pushButton_show_origin.clicked.connect(self.__show_origin)
        self.pushButton_pick_leaf.clicked.connect(self.send_leaf_location)

    def __showRgb(self):
        image = self.server.rgb_image()
        scene = QGraphicsScene(self)
        if image is not None:
            image = QImage(image, image.shape[1], image.shape[0],
                           image.strides[0], QImage.Format.Format_RGB888)
            image = QPixmap.fromImage(image)
            image_scaled = QPixmap.scaled(image, self.show_img_width - 5, self.show_img_height - 5,
                                          Qt.AspectRatioMode.IgnoreAspectRatio)
            scene.addPixmap(image_scaled)
            self.graphicsView.setScene(scene)

    def __show_bbox(self):
        yolo_img = self.server.yolo_image()
        scene = QGraphicsScene(self)
        image = QImage(yolo_img, yolo_img.shape[1], yolo_img.shape[0],
                       yolo_img.strides[0], QImage.Format.Format_RGB888)
        image = QPixmap.fromImage(image)
        image_scaled = QPixmap.scaled(image, self.show_img_width - 5, self.show_img_height - 5,
                                      Qt.AspectRatioMode.IgnoreAspectRatio)
        scene.addPixmap(image_scaled)
        self.graphicsView.setScene(scene)

    def __show_mask(self):
        mask = self.server.sam_mask()
        image_rgb = self.server.rgb_image()
        image = cv2.bitwise_and(image_rgb, image_rgb, mask=mask)
        scene = QGraphicsScene(self)
        image = QImage(image, image_rgb.shape[1], image_rgb.shape[0],
                       image_rgb.strides[0], QImage.Format.Format_RGB888)
        image = QPixmap.fromImage(image)
        image_scaled = QPixmap.scaled(image, self.show_img_width - 5, self.show_img_height - 5,
                                      Qt.AspectRatioMode.IgnoreAspectRatio)
        scene.addPixmap(image_scaled)
        self.graphicsView.setScene(scene)

    def __show_origin(self):
        self.thread_show_yolo.stop()
        self.thread_show_rgb.start()

    def detect_leaf(self):
        self.thread_show_rgb.stop()
        self.thread_show_yolo.start(100)

    def segment_leaf(self):
        self.thread_show_yolo.stop()
        self.thread_show_rgb.stop()
        QTimer.setSingleShot(self.thread_show_mask, 1)
        self.thread_show_mask.start()

    def send_cmd_to_J1(self):
        print("send_cmd_to_arm called")
        cmd_dict = {self.pushButton_J1.text(): self.spinBox_J1.value()}
        cmd = Msg()
        cmd.task = ARMTASK.MOVE_SINGLE_JOINT
        cmd.cmd = cmd_dict
        self.server.send_cmd_arm(cmd)

    def send_cmd_to_J2(self):
        cmd_dict = {self.pushButton_J2.text(): self.spinBox_J2.value()}
        cmd = Msg()
        cmd.task = ARMTASK.MOVE_SINGLE_JOINT
        cmd.cmd = cmd_dict
        self.server.send_cmd_arm(cmd)

    def send_cmd_to_J3(self):
        cmd_dict = {self.pushButton_J3.text(): self.spinBox_J3.value()}
        cmd = Msg()
        cmd.task = ARMTASK.MOVE_SINGLE_JOINT
        cmd.cmd = cmd_dict
        self.server.send_cmd_arm(cmd)

    def send_cmd_to_J4(self):
        cmd_dict = {self.pushButton_J4.text(): self.spinBox_J4.value()}
        cmd = Msg()
        cmd.task = ARMTASK.MOVE_SINGLE_JOINT
        cmd.cmd = cmd_dict
        self.server.send_cmd_arm(cmd)

    def send_cmd_to_J5(self):
        cmd_dict = {self.pushButton_J5.text(): self.spinBox_J5.value()}
        cmd = Msg()
        cmd.task = ARMTASK.MOVE_SINGLE_JOINT
        cmd.cmd = cmd_dict
        self.server.send_cmd_arm(cmd)

    def send_cmd_to_J6(self):
        cmd_dict = {self.pushButton_J6.text(): self.spinBox_J6.value()}
        cmd = Msg()
        cmd.task = ARMTASK.MOVE_SINGLE_JOINT
        cmd.cmd = cmd_dict
        self.server.send_cmd_arm(cmd)

    def send_cmd_zero_position(self):
        cmd = Msg(ARMTASK.MOVE_ZERO_POSITION)
        self.server.send_cmd_arm(cmd)

    def send_cmd_to_all_joints(self):
        cmd_dict = [int(self.spinBox_J1.value()), int(self.spinBox_J2.value()), int(self.spinBox_J3.value()),
                    int(self.spinBox_J4.value()), int(self.spinBox_J5.value()), int(self.spinBox_J6.value())]
        cmd = Msg(ARMTASK.MOVE_ALL_JOINT, cmd_dict)
        self.server.send_cmd_arm(cmd)

    def send_cmd_to_read_servo(self):
        cmd = Msg(ARMTASK.READ_SERVO)
        self.server.send_cmd_arm(cmd)
        joint_list = self.server.receive_cmd_arm()
        self.spinBox_J1.setValue(joint_list[0])
        self.spinBox_J2.setValue(joint_list[1])
        self.spinBox_J3.setValue(joint_list[2])
        self.spinBox_J4.setValue(joint_list[3])
        self.spinBox_J5.setValue(joint_list[4])
        self.spinBox_J6.setValue(joint_list[5])

    def cal_leaf_center(self):
        pass

    def cal_leaf_size(self):
        pass

    def send_cmd_to_move_forward(self):
        L_speed = 0.05 * int(self.comboBox_car_speed.currentText())
        R_speed = 0.05 * int(self.comboBox_car_speed.currentText())

        cmd = {"T": 1, "L": L_speed, "R": R_speed}
        self.server.send_json(str(cmd))

    def send_cmd_to_move_backward(self):
        L_speed = -0.05 * int(self.comboBox_car_speed.currentText())
        R_speed = -0.05 * int(self.comboBox_car_speed.currentText())

        cmd = {"T": 1, "L": L_speed, "R": R_speed}
        self.server.send_json(str(cmd))

    def send_leaf_location(self):
        self.thread_show_yolo.stop()
        picking_points = self.server.get_leaves_location()
        cmd = Msg(ARMTASK.PICK_CLOSEST_LEAF, picking_points)
        self.server.send_cmd_arm(cmd)
        print(picking_points)
        self.thread_show_yolo.start()
        self.server.send_cmd_arm(cmd)


if __name__ == '__main__':
    # client = ImagePostProcessor()
    # for i in range(1000):
    #     print(client.rgb_image())
    app = QApplication([])
    window = LeafBot()
    window.show()
    app.exec()
