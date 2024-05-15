from __future__ import annotations
import os
import sys
PROJECT_PATH = os.getcwd()
SOURCE_PATH = os.path.join(
    PROJECT_PATH
)
sys.path.append(SOURCE_PATH)

import warnings
warnings.simplefilter('ignore')
from mobile_sam import sam_model_registry, SamPredictor
import cv2
import numpy as np
from utils.image_process import convert_to_xyxy, remove_small_cnt 
from pathlib import Path
from onnxruntime import InferenceSession
from yolonnx.services import Detector
from yolonnx.to_tensor_strategies import PillowToTensorContainStrategy
import time
from threading import Thread, Lock
from PIL import Image
import copy
from ultralytics import YOLO

class Yolo():
    """
    YOLO model for object detection.
    """
    def __init__(self, frame_queue):
        """
        Initialize the YOLO model by onnx format.
        """
        model_det_path = Path("weights/best.onnx")
        model_seg_path = Path("weights/best_seg.pt")
        
        session = InferenceSession(
                                    model_det_path.as_posix(),
                                    providers=[
                                                "CUDAExecutionProvider",
                                                "CPUExecutionProvider",
                                                ],
                                    )
        predictor = Detector(session, PillowToTensorContainStrategy(), 0.6, 0.4)
        ############################# load detect and segment model #########################
        self.model_det = predictor.run
        self.model_seg = YOLO(model_seg_path)
        # for sending data from camera
        self.frame_queue = frame_queue  # queue for each fps prediction
        self.is_running = False  # status
        self.fps = 0.0  # real-time fps
        self.__t_last = time.time() * 1000
        self.__data = {} 
        
    def predict(self, image: np.ndarray, task='detect') -> list:
        """
        Make predictions using the YOLO model.

        Args:
            image (np.ndarray): The input image.

        Returns:
            list: The prediction result containing Detector class.
        """
        print('yolo_start')
        image_yolo = Image.fromarray(image)
        results = self.model_det(image_yolo)
        print('yolo_end')
        
        return results

    def __capture_queue(self, image, depth_img):
        # capture image
        self.__t_last = time.time() * 1000
        lock = Lock()
        while self.is_running:
            with lock:
                result = self.predict(image)
                t  = time.time() * 1000
                t_span = t - self.__t_last                
                self.fps = 1000.0 / t_span
                self.__data['result'] = result
                self.__data["fps"] = self.fps
                self.__data["image"] = image
                self.__data["depth"] = depth_img
                self.frame_queue.put(self.__data)
                self.__t_last = t

    def run(self, image, depth_img):
        self.is_running = True
        self.thread_capture = Thread(target=self.__capture_queue,args=(image,depth_img))
        self.thread_capture.start()

    def stop(self):
        self.is_running = False
        self.thread_capture.join()


class Mobile_SAM():
    """
    Mobile SAM model for image segmentation.
    """
    def __init__(self):
        """
        Initialize the Mobile SAM model.
        """
        model_type = "vit_t"
        sam_checkpoint = "weights/mobile_sam1.pt"
        mobile_sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        self.model = SamPredictor(mobile_sam)

    def predict(self, image: np.ndarray, yolo_results: np.ndarray) -> np.ndarray:
        """
        Make predictions using the Mobile SAM model.

        Args:
            image (np.ndarray): The input image.
            yolo_results (np.ndarray): Results from YOLO model.

        Returns:
            np.ndarray: The prediction result.
        """
        # Store xyxy bounding boxes in a list
        print("segment anything!!!\n")
        image = np.array(image)
        self.model.set_image(image)
        bbox_list = np.asarray([convert_to_xyxy(result) for result in yolo_results])
        centers = np.zeros((bbox_list.shape[0], 2))
        for i, box in enumerate(bbox_list):
            center_x, center_y = box[0] / 2 + box[2] / 2, box[1] / 2 + box[3] / 2
            centers[i, :] = np.array([center_x, center_y])
        for i, center in enumerate(centers):
            masks, scores, logits = self.model.predict(
                                point_coords=center.reshape(1, 2),
                                box=bbox_list[i],
                                point_labels=[1],
                                multimask_output=False)
            masks = (np.moveaxis(masks, 0, -1)).astype(np.uint8)
            best_mask = masks[:, :, np.argmax(scores)]
            if i == 0:
                masks_final = best_mask
            else:
                masks_final += best_mask
        
        contours_final = remove_small_cnt(masks_final)
        result = np.zeros_like(masks_final)
        result = cv2.drawContours(result, contours_final, -1, (255, 255, 255), 
                                    thickness=cv2.FILLED).astype(np.uint8)
        return result

