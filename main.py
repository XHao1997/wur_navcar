import copy
import threading
from typing import Any

import cv2
import numpy as np
import zmq
from PySide6.QtCore import QTimer, Qt, QThread, Signal, Slot, QObject
from PySide6.QtGui import QPixmap, QImage, QAction
from PySide6.QtWidgets import QWidget, QGraphicsScene, QApplication
from ui.LeafBot_ui import Ui_LeafBotForm
from module.AI_model import Yolo, MobileSAM
from module.msg import ARMTASK, Msg
from utils import ssh, image_process, leaf, file
from module.kinect import Kinect
import requests
import time
import queue
import sys


def recv_array(socket, flags=0, copy=True, track=False):
    """Receive a numpy array over a ZeroMQ socket."""

    # Receive the actual data
    msg = socket.recv(flags=flags, copy=copy, track=track)

    # Convert received bytes to a NumPy array
    A = np.frombuffer(msg, dtype=np.uint8)

    # Reshape array to its original shape
    return A.reshape(480, 640, 3)


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
        self.pair_cam = self.context.socket(zmq.SUB)
        self.depth_req = self.context.socket(zmq.REQ)

        self.pair_cam.linger = 1
        self.pair_cam.setsockopt(zmq.SUBSCRIBE, b'')
        self.pair_cam.setsockopt(zmq.CONFLATE, 1)  # last msg only.
        self.pair_cam.setsockopt(zmq.RCVHWM, 1)
        self.pair_cam.connect("tcp://192.168.101.12:5555")

        self.depth_req.connect("tcp://192.168.101.12:5556")
        self.pair_arm = self.context.socket(zmq.PAIR)
        self.pair_arm.bind("tcp://192.168.101.14:3333")

        car_ip_addr = '192.168.101.19'
        self.url = "http://" + car_ip_addr + "/js?json="

    def get_data(self):
        return recv_array(self.pair_cam)

    def get_data_depth(self):
        self.depth_req.send_pyobj(None)
        return self.depth_req.recv_pyobj()

    def send_cmd_arm(self, cmd):
        self.pair_arm.send_pyobj(cmd)

    def receive_cmd_arm(self):
        return self.pair_arm.recv_pyobj()

    def send_json(self, json_data):
        cmd = self.url + json_data
        print(requests.get(cmd).text)

    def receive_imu(self):
        cmd = self.url + str({"T": 126})
        print(requests.get(cmd).text)


class ImagePostProcess:
    def __init__(self):
        self.yolo = Yolo()
        self.sam = MobileSAM()
        self.camera = Kinect()

    def sam_mask(self, rgb_img):
        yolo_results = self.yolo.predict(rgb_img)
        mask = self.sam.predict(rgb_img, yolo_results)
        return mask

    def yolo_image(self, rgb_img):
        yolo_results = self.yolo.predict(rgb_img)
        yolo_img = image_process.draw_yolo_frame_cv(rgb_img.copy(), yolo_results)
        return yolo_img

    def get_leaves_location(self, rgb_img, depth_img):
        yolo_results = self.yolo.predict(rgb_img)
        sam_mask = self.sam.predict(rgb_img, yolo_results)
        picking_point = []
        for i in range(len(yolo_results)):
            result = yolo_results[i]
            chosen_leaf_roi = image_process.get_yolo_roi(sam_mask, result)
            contours = leaf.get_cnts(chosen_leaf_roi)
            mask, _ = leaf.get_incircle(chosen_leaf_roi, contours)
            picking_point.append(self.camera.get_point_xyz(mask, rgb_img, depth_img))
        picking_point = np.array(picking_point)
        picking_point = picking_point[np.isfinite(picking_point)].reshape(-1, 3)
        return picking_point


directories = {
    'rgb': 'data/rgb',
    'depth': 'data/depth',
    'eye_to_hand': 'eye_to_hand/',
    'joint1_nn': 'joint1_nn/',
    'imitation_car': 'data/imitation_car/'
}


def launch_all():
    ssh.run_remote_stream()


class LeafBot(QWidget, Ui_LeafBotForm):
    def __init__(self):
        super().__init__()

        # Directory where files are saved
        self.is_update_yolo = True
        self.sam_img = None
        self.yolo_img = None
        self.rgb_img = None
        """ The slot function for Vision """

        """ set up server """
        self.server = ImagePostProcess()
        self.com = Communicator()
        self.imitation_mode = False
        """ set up UI """
        print("set up ui")
        self.setupUi(self)
        self.LeafBotLogo.setPixmap(QPixmap(u"logo.jpg"))  # fix logo missing

        self.show_img_width = self.label_showimg.width()
        self.show_img_height = self.label_showimg.height()
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
        self.pushButton_car_stop.clicked.connect(self.send_cmd_to_stop_car)
        self.pushButton_luanch_all.clicked.connect(launch_all)
        self.pushButton_move_random.clicked.connect(self.send_cmd_to_move_random)
        self.pushButton_train_eye2hand.clicked.connect(self.train_eye2hand)
        self.pushButton_save_img.clicked.connect(self.save_img)
        self.pushButton_car_turn_left.clicked.connect(self.send_cmd_to_turn_left)
        self.pushButton_car_turn_right.clicked.connect(self.send_cmd_to_turn_right)
        self.checkBox_Imitation.clicked.connect(self.set_imitation_mode)
        self.pushButton_park_to_plant.clicked.connect(self.park_to_plant)
        self.pushButton_set_torque.clicked.connect(self.send_cmd_to_set_torque)
        self.pushButton_off_torque.clicked.connect(self.send_cmd_to_off_torque)
        self.pushButton_record_joint.clicked.connect(self.send_cmd_imitation_pose)


        # Create a QAction with a keyboard shortcut (in this case, Ctrl+B)
        action_up = QAction(self)
        action_up.setShortcut(Qt.Key.Key_Up)
        action_up.triggered.connect(self.pushButton_move_forward.click)
        action_down = QAction(self)
        action_down.setShortcut(Qt.Key.Key_Down)
        action_down.triggered.connect(self.pushButton_move_backward.click)
        action_left = QAction(self)
        action_left.setShortcut(Qt.Key.Key_Left)
        action_left.triggered.connect(self.pushButton_car_turn_left.click)
        action_right = QAction(self)
        action_right.setShortcut(Qt.Key.Key_Right)
        action_right.triggered.connect(self.pushButton_car_turn_right.click)
        # 添加QAction对象到主窗口
        self.addAction(action_up)
        self.addAction(action_down)
        self.addAction(action_left)
        self.addAction(action_right)

        print("callback function all set")
        self.data_thread = UpdateData()
        self.data_thread.update_signal.connect(self.update_img)
        self.data_thread.start()

        self.show_rgb_thread = UpdateUI()
        self.show_rgb_thread.update_signal.connect(self.__showRgb)
        # self.show_rgb_thread.start()

        self.show_yolo_thread = UpdateUI()
        self.show_yolo_thread.update_signal.connect(self.__show_bbox)
        self.show_mask_thread = QTimer()
        self.show_mask_thread.timeout.connect(self.__show_mask)
        self.save_image_thread = SavaImage(200)
        self.save_image_thread.update_signal.connect(self.save_img)
        self.car_cmd_thread = CMD()
        self.car_cmd_thread.update_signal.connect(self.send_cmd_to_move_forward)

    def park_to_plant(self):
        self.show_yolo_thread.stop()
        self.show_rgb_thread.start()
        self.car_cmd_thread.start()

    def update_img(self, img):
        self.rgb_img = img

    def set_imitation_mode(self):
        self.imitation_mode = self.checkBox_Imitation.isChecked()
        print('imitation mode:\t', self.imitation_mode)

    def __paint_img(self, image):
        if image is not None:
            image = QImage(image, image.shape[1], image.shape[0],
                           image.strides[0], QImage.Format.Format_RGB888)
            image = QPixmap.fromImage(image)
            image_scaled = QPixmap.scaled(image, self.show_img_width, self.show_img_height,
                                          Qt.AspectRatioMode.IgnoreAspectRatio)
            self.label_showimg.setPixmap(image_scaled)
            self.label_showimg.repaint()

    def __showRgb(self):
        image = self.rgb_img.copy()
        self.__paint_img(image)

    def __show_bbox(self):
        yolo_img = self.server.yolo_image(self.rgb_img)
        self.__paint_img(yolo_img)

    def __show_mask(self):
        self.__paint_img(self.sam_img)

    def __show_origin(self):
        # stop_thread(self.show_yolo_thread)
        self.show_yolo_thread.stop()
        self.show_rgb_thread.start()

    def detect_leaf(self):
        self.show_rgb_thread.stop()
        self.show_yolo_thread.start()

    def segment_leaf(self):
        self.show_rgb_thread.stop()
        self.show_yolo_thread.stop()
        self.data_thread.stop()
        mask = self.server.sam_mask(self.rgb_img)
        self.sam_img = cv2.bitwise_and(self.rgb_img, self.rgb_img, mask=mask)
        QTimer.setSingleShot(self.show_mask_thread, True)
        self.show_mask_thread.start()
        self.data_thread.start()

    def send_leaf_location(self):
        self.data_thread.stop()
        self.show_yolo_thread.stop()
        self.show_rgb_thread.stop()
        picking_points = self.server.get_leaves_location(self.rgb_img, self.com.get_data_depth())
        cmd = Msg(ARMTASK.PICK_CLOSEST_LEAF, picking_points)
        print(picking_points)
        self.data_thread.start()
        self.show_yolo_thread.start()
        self.com.send_cmd_arm(cmd)

    def cal_leaf_center(self):
        pass

    def cal_leaf_area(self):
        pass

    """ The slot function for controlling the Arm """

    def send_cmd_to_J1(self):
        print("send_cmd_to_arm called")
        cmd_dict = {self.pushButton_J1.text(): self.spinBox_J1.value()}
        cmd = Msg()
        cmd.task = ARMTASK.MOVE_SINGLE_JOINT
        cmd.cmd = cmd_dict
        self.com.send_cmd_arm(cmd)

    def send_cmd_to_J2(self):
        cmd_dict = {self.pushButton_J2.text(): self.spinBox_J2.value()}
        cmd = Msg()
        cmd.task = ARMTASK.MOVE_SINGLE_JOINT
        cmd.cmd = cmd_dict
        self.com.send_cmd_arm(cmd)

    def send_cmd_to_J3(self):
        cmd_dict = {self.pushButton_J3.text(): self.spinBox_J3.value()}
        cmd = Msg()
        cmd.task = ARMTASK.MOVE_SINGLE_JOINT
        cmd.cmd = cmd_dict
        self.com.send_cmd_arm(cmd)

    def send_cmd_to_J4(self):
        cmd_dict = {self.pushButton_J4.text(): self.spinBox_J4.value()}
        cmd = Msg()
        cmd.task = ARMTASK.MOVE_SINGLE_JOINT
        cmd.cmd = cmd_dict
        self.com.send_cmd_arm(cmd)

    def send_cmd_to_J5(self):
        cmd_dict = {self.pushButton_J5.text(): self.spinBox_J5.value()}
        cmd = Msg()
        cmd.task = ARMTASK.MOVE_SINGLE_JOINT
        cmd.cmd = cmd_dict
        self.com.send_cmd_arm(cmd)

    def send_cmd_to_J6(self):
        cmd_dict = {self.pushButton_J6.text(): self.spinBox_J6.value()}
        cmd = Msg()
        cmd.task = ARMTASK.MOVE_SINGLE_JOINT
        cmd.cmd = cmd_dict
        self.com.send_cmd_arm(cmd)

    def send_cmd_zero_position(self):
        cmd = Msg(ARMTASK.MOVE_ZERO_POSITION)
        self.com.send_cmd_arm(cmd)

    def send_cmd_to_all_joints(self):
        cmd_dict = [int(self.spinBox_J1.value()), int(self.spinBox_J2.value()), int(self.spinBox_J3.value()),
                    int(self.spinBox_J4.value()), int(self.spinBox_J5.value()), int(self.spinBox_J6.value())]
        cmd = Msg(ARMTASK.MOVE_ALL_JOINT, cmd_dict)
        self.com.send_cmd_arm(cmd)

    def send_cmd_to_read_servo(self):
        cmd = Msg(ARMTASK.READ_SERVO)
        self.com.send_cmd_arm(cmd)
        joint_list = self.com.receive_cmd_arm()
        self.spinBox_J1.setValue(joint_list[0])
        self.spinBox_J2.setValue(joint_list[1])
        self.spinBox_J3.setValue(joint_list[2])
        self.spinBox_J4.setValue(joint_list[3])
        self.spinBox_J5.setValue(joint_list[4])
        self.spinBox_J6.setValue(joint_list[5])

    def send_cmd_to_move_random(self):
        cmd = Msg(ARMTASK.MOVE_ARM_MOVE_RANDOM)
        self.com.send_cmd_arm(cmd)

    def train_eye2hand(self):
        self.save_image_thread.start()
        train_amount = int(self.spinBox_trainig_amount.value())
        cmd = Msg(ARMTASK.MOVE_ARM_EYE2HAND, train_amount)
        self.com.send_cmd_arm(cmd)

    def send_cmd_to_set_torque(self):
        cmd = Msg(ARMTASK.SET_TORQUE)
        self.com.send_cmd_arm(cmd)

    def send_cmd_to_off_torque(self):
        cmd = Msg(ARMTASK.OFF_TORQUE)
        self.com.send_cmd_arm(cmd)

    def send_cmd_imitation_pose(self):
        cmd = Msg(ARMTASK.IMITATION_MOVE)
        self.com.send_cmd_arm(cmd)

    def record_imitation_data(self):
        pass

    def record_imitation_done(self):
        pass
    """ The slot function for controlling the Car """

    def send_cmd_to_move_forward(self):
        L_speed = -0.1 * int(self.comboBox_car_speed.currentText())
        R_speed = -0.1 * int(self.comboBox_car_speed.currentText())
        cmd = {"T": 1, "L": L_speed, "R": R_speed}
        if self.imitation_mode:
            self.save_imitation([self.rgb_img, 1])
        self.com.send_json(str(cmd))
        # self.com.receive_imu()

    def send_cmd_to_move_backward(self):
        L_speed = 0.1 * int(self.comboBox_car_speed.currentText())
        R_speed = 0.1 * int(self.comboBox_car_speed.currentText())
        if self.imitation_mode:
            self.save_imitation([self.rgb_img, -1])
        cmd = {"T": 1, "L": L_speed, "R": R_speed}
        self.com.send_json(str(cmd))
        # self.com.receive_imu()

    def send_cmd_to_turn_left(self):
        L_speed = 0.2 * int(self.comboBox_car_speed.currentText())
        R_speed = -0.01 * int(self.comboBox_car_speed.currentText())

        cmd = {"T": 1, "L": L_speed, "R": R_speed}
        self.com.send_json(str(cmd))
        # self.com.receive_imu()

    def send_cmd_to_turn_right(self):
        L_speed = -0.01 * int(self.comboBox_car_speed.currentText())
        R_speed = 0.2 * int(self.comboBox_car_speed.currentText())

        cmd = {"T": 1, "L": L_speed, "R": R_speed}
        self.com.send_json(str(cmd))

    def send_cmd_to_stop_car(self):
        L_speed = 0
        R_speed = 0
        cmd = {"T": 1, "L": L_speed, "R": R_speed}
        if self.imitation_mode:
            self.save_imitation([self.rgb_img, 0])
        self.com.send_json(str(cmd))

    def save_img(self):
        file.save_file(directories, self.rgb_img, 'rgb')
        file.save_file(directories, self.com.get_data_depth(), 'depth')

    def save_rgb_img(self):
        file.save_file(directories, self.rgb_img, 'rgb')

    def save_imitation(self, data):
        file.save_file(directories, data, 'imitation_car')

    def closeEvent(self, event):
        self.show_yolo_thread.stop()
        self.show_rgb_thread.stop()
        self.data_thread.stop()

        super().closeEvent(event)


class UpdateData(QThread):
    update_signal = Signal(np.ndarray)
    com = Communicator()

    def __init__(self):
        super().__init__()
        self.timer = None

    def run(self):
        self.timer = QTimer()
        self.timer.timeout.connect(self.emitUpdateSignal)
        self.timer.moveToThread(self)
        self.timer.start(125)  # Trigger 10FPS
        self.exec()  # Start the event loop to keep the QTimer running

    def emitUpdateSignal(self):
        data = self.com.get_data()
        self.update_signal.emit(data)

    def stop(self):
        self.timer.stop()
        self.quit()
        self.wait()


class UpdateUI(QThread):
    update_signal = Signal()

    def __init__(self):
        super().__init__()

    def run(self):
        timer = QTimer()
        timer.moveToThread(self)
        timer.timeout.connect(self.emitUpdateSignal)
        timer.start(200)  # Trigger 8FPS
        # Start the event loop to keep the QTimer running
        self.exec()

    def emitUpdateSignal(self):
        self.update_signal.emit()

    def stop(self):
        self.finished.emit()
        self.quit()
        self.wait()


class SavaImage(QThread):
    update_signal = Signal()
    com = Communicator()

    def __init__(self, training_amount):
        super().__init__()
        self._is_running = True
        self._count = training_amount

    def run(self):
        for i in range(self._count):
            self.com.receive_cmd_arm()
            self.update_signal.emit()
        self.finished.emit()
        self.quit()
        self.wait()

    def stop(self):
        self._is_running = False
        self.finished.emit()
        self.quit()
        self.wait()



class CMD(QThread):
    update_signal = Signal()

    def __init__(self):
        super().__init__()

    def run(self):
        timer = QTimer()
        timer.moveToThread(self)
        timer.timeout.connect(self.emitUpdateSignal)
        timer.start(200)  # Trigger every 4FPS
        self.exec()

    def stop(self):
        self.finished.emit()
        self.quit()
        self.wait()

    def emitUpdateSignal(self):
        self.update_signal.emit()


if __name__ == '__main__':
    ssh.run_remote_stream()
    app = QApplication(sys.argv)
    window = LeafBot()
    window.show()
    app.exec_()
