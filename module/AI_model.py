from __future__ import annotations
import os
import sys
PROJECT_PATH = os.getcwd()
SOURCE_PATH = os.path.join(
    PROJECT_PATH
)
sys.path.append(SOURCE_PATH)

from abc import ABC, abstractmethod
import warnings
warnings.simplefilter('ignore')
from mobile_sam import sam_model_registry, SamPredictor
import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils.image_processing import convert_to_xyxy, remove_small_cnt, draw_rect
from pathlib import Path
from onnxruntime import InferenceSession
from yolonnx.services import Detector
from yolonnx.to_tensor_strategies import PillowToTensorContainStrategy

class AI_model_factory():
    """
    Factory class for creating AI models.
    """
    def create_model(self, AI_model: type[AI_model]) -> AI_model:
        """
        Create an instance of an AI model.

        Args:
            AI_model (type[AI_model]): The type of AI model to create.

        Returns:
            AI_model: An instance of the specified AI model.
        """
        model = AI_model()
        return model

class AI_model(ABC):
    """
    Abstract base class for AI models.
    """
    @abstractmethod
    def __init__(self):
        self.model = None

    @abstractmethod
    def predict(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """
        Make predictions using the AI model.

        Args:
            image (np.ndarray): The input image.
            **kwargs: Additional keyword arguments specific to the model.

        Returns:
            np.ndarray: The prediction result.
        """
        pass

    @abstractmethod
    def visualise_result(self, image: np.ndarray, result: np.ndarray) -> None:
        """
        Visualise the prediction result.

        Args:
            image (np.ndarray): The input image.
            result (np.ndarray): The prediction result.
        """
        pass

class Yolo(AI_model):
    """
    YOLO model for object detection.
    """
    def __init__(self):
        """
        Initialize the YOLO model.
        """
        
        model = Path("weights/best.onnx")
        session = InferenceSession(
                                    model.as_posix(),
                                    providers=[
                                                "CUDAExecutionProvider",
                                                "CPUExecutionProvider",
                                                ],
                                    )
        predictor = Detector(session, PillowToTensorContainStrategy())
        self.model = predictor.run

    def predict(self, image: np.ndarray) -> list:
        """
        Make predictions using the YOLO model.

        Args:
            image (np.ndarray): The input image.

        Returns:
            np.ndarray: The prediction result.
        """
        results = self.model(image)
        return results

    def visualise_result(self, image: np.ndarray, results: np.ndarray) -> None:
        """
        Visualise the prediction result.

        Args:
            image (np.ndarray): The input image.
            results (np.ndarray): The prediction result.
        """
        # Draw rectangles for each detection result
        # Create figure and axis
        fig, ax = plt.subplots()
        image = np.array(image)    
        # Display the image
        ax.imshow(image)
        for detection in results:
            draw_rect(ax, detection)
        # Show the plot
        plt.show()

class Mobile_SAM(AI_model):
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
        image = np.array(image)
        bbox_list = np.asarray([convert_to_xyxy(result) for result in yolo_results])

        centers = np.zeros((bbox_list.shape[0], 2))

        for i, box in enumerate(bbox_list):
            center_x, center_y = box[0] / 2 + box[2] / 2, box[1] / 2 + box[3] / 2
            centers[i, :] = np.array([center_x, center_y])

        self.model.set_image(image)
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
        result = masks_final
        return result

    def visualise_result(self,image: np.ndarray, result: np.ndarray) -> None:
        """
        Visualise the prediction result.

        Args:
            image (np.ndarray): The input image.
            result (np.ndarray): The prediction result.
        """
        image = np.array(image)
        contours_final = remove_small_cnt(result)
        new_mask = np.zeros_like(result)
        masks = cv2.drawContours(new_mask, contours_final, -1, (255, 255, 255), thickness=cv2.FILLED).astype(np.uint8)
        cv2.drawContours(masks, contours_final, -1, (255, 255, 255), cv2.FILLED)
        cv2.drawContours(image=image, contours=contours_final, contourIdx=-1, color=(0, 255, 0), thickness=2,
                        lineType=cv2.LINE_AA)
        cv2.bitwise_and(image, image, mask=masks)
        plt.imshow(image)
        plt.show()
