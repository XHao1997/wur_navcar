import copy
import os
import sys
import time

import cv2
import zmq
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtWidgets import QWidget, QGraphicsScene, QApplication

from ui.LeafBot_ui import Ui_LeafBotForm

from module.AI_model import Yolo, MobileSAM
from utils import ssh, image_process

import threading
import time
import cv2


class Communicator:
    def __init__(self):
        self.context = zmq.Context()
        self.publisher = self.context.socket(zmq.PUB)
        self.consumer_receiver = self.context.socket(zmq.PULL)
        self.publisher.bind("tcp://192.168.101.14:5555")
        self.consumer_receiver.connect("tcp://192.168.101.12:5557")

    def get_data(self):
        return self.consumer_receiver.recv_pyobj()


class ImageThread(threading.Thread):
    def __init__(self, server):
        super().__init__()
        self.latest_image = None  # Store the latest image
        self.lock = threading.Lock()  # Create a lock object
        self.server = server

    def run(self):
        while True:
            frame = self.server.get_data()  # Read image from the camera
            with self.lock:  # Acquire the lock
                self.latest_image = frame  # Update the latest image

    def get_latest_image(self):
        with self.lock:  # Acquire the lock
            return self.latest_image  # Return the latest image


class ClientServer:
    def __init__(self):
        ssh.run_remote_stream()
        self.server = Communicator()
        self.yolo = Yolo()
        self.sam = MobileSAM()
        while self.server.get_data() is None:
            continue
        self.image_thread = ImageThread(self.server)
        self.image_thread.start()

    def rgb_image(self):
        return self.image_thread.get_latest_image()

    def sam_mask(self):
        image = self.image_thread.get_latest_image()
        yolo_results = self.yolo.predict(image)
        mask = self.sam.predict(image, yolo_results)
        return mask

    def yolo_image(self):
        image = copy.deepcopy(self.image_thread.get_latest_image())
        yolo_results = self.yolo.predict(image)
        yolo_img = image_process.draw_yolo_frame_cv(image, yolo_results)
        return yolo_img

    def send_cmd(self, cmd):
        self.server.publisher.send_pyobj(cmd)


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
        self.thread_show_rgb.timeout.connect(self.showRgb)

        self.thread_show_rgb.start()

        self.thread_show_yolo = QTimer()
        self.thread_show_yolo.timeout.connect(self.show_bbox)
        self.thread_show_mask = QTimer()
        self.thread_show_mask.timeout.connect(self.show_mask)
        # self.thread_show_yolo.start()
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

    def showRgb(self):
        image = self.server.rgb_image()
        scene = QGraphicsScene(self)
        image = QImage(image, image.shape[1], image.shape[0],
                       image.strides[0], QImage.Format.Format_RGB888)
        image = QPixmap.fromImage(image)
        image_scaled = QPixmap.scaled(image, self.show_img_width - 5, self.show_img_height - 5,
                                      Qt.AspectRatioMode.IgnoreAspectRatio)
        scene.addPixmap(image_scaled)
        self.graphicsView.setScene(scene)

    def show_bbox(self):
        yolo_img = self.server.yolo_image()
        scene = QGraphicsScene(self)
        image = QImage(yolo_img, yolo_img.shape[1], yolo_img.shape[0],
                       yolo_img.strides[0], QImage.Format.Format_RGB888)
        image = QPixmap.fromImage(image)
        image_scaled = QPixmap.scaled(image, self.show_img_width - 5, self.show_img_height - 5,
                                      Qt.AspectRatioMode.IgnoreAspectRatio)
        scene.addPixmap(image_scaled)
        self.graphicsView.setScene(scene)

    def show_mask(self):
        image = self.server.sam_mask()
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        scene = QGraphicsScene(self)
        image = QImage(image, image.shape[1], image.shape[0],
                       image.strides[0], QImage.Format.Format_RGB888)
        image = QPixmap.fromImage(image)
        image_scaled = QPixmap.scaled(image, self.show_img_width - 5, self.show_img_height - 5,
                                      Qt.AspectRatioMode.IgnoreAspectRatio)
        scene.addPixmap(image_scaled)
        self.graphicsView.setScene(scene)

    def detect_leaf(self):
        self.thread_show_rgb.stop()
        self.thread_show_yolo.start()

    def segment_leaf(self):
        self.thread_show_yolo.stop()
        self.thread_show_rgb.stop()
        QTimer.setSingleShot(self.thread_show_mask, 1)
        self.thread_show_mask.start()

    def send_cmd_to_J1(self):
        print("send_cmd_to_arm called")
        cmd_dict = {self.pushButton_J1.text(): self.spinBox_J1.value()}
        self.server.send_cmd(cmd_dict)

    def send_cmd_to_J2(self):
        cmd_dict = {self.pushButton_J2.text(): self.spinBox_J2.value()}
        self.server.send_cmd(cmd_dict)

    def send_cmd_to_J3(self):
        cmd_dict = {self.pushButton_J3.text(): self.spinBox_J3.value()}
        self.server.send_cmd(cmd_dict)

    def send_cmd_to_J4(self):
        cmd_dict = {self.pushButton_J4.text(): self.spinBox_J4.value()}
        self.server.send_cmd(cmd_dict)

    def send_cmd_to_J5(self):
        cmd_dict = {self.pushButton_J5.text(): self.spinBox_J5.value()}
        self.server.send_cmd(cmd_dict)

    def send_cmd_to_J6(self):
        cmd_dict = {self.pushButton_J6.text(): self.spinBox_J6.value()}
        self.server.send_cmd(cmd_dict)

    def send_cmd_to_car(self, cmd):
        pass

    def send_cmd_to_all_joints(self, cmd):
        cmd = [int(self.pushButton_J1.text()), int(self.pushButton_J2.text()), int(self.pushButton_J3.text()),
               int(self.pushButton_J4.text()), int(self.pushButton_J5.text()), int(self.pushButton_J6.text())]
        self.server.send_cmd(cmd)

    def cal_leaf_center(self):
        pass

    def cal_leaf_size(self):
        pass


if __name__ == '__main__':
    # client = ImagePostProcessor()
    # for i in range(1000):
    #     print(client.rgb_image())
    app = QApplication([])
    window = LeafBot()
    window.show()
    app.exec()
