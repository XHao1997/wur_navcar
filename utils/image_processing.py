import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def rgb2bgr(img):
    # Convert RGB image to BGR
    bgr_image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return bgr_image

# Convert DetectorResult objects to xyxy format
def convert_to_xyxy(result):
    x1 = result.x
    y1 = result.y
    x2 = result.x + result.width
    y2 = result.y + result.height
    return (x1, y1, x2, y2)


# Define a function to draw rectangles on the image
def draw_rect(ax, result, color='r'):
    rect = patches.Rectangle((result.x, result.y), result.width, result.height, linewidth=2, edgecolor=color, facecolor='none')
    ax.add_patch(rect)
    ax.text(result.x, result.y - 5, result.name, fontsize=8, color=color)
    return 

def remove_small_cnt(masks_final):
    contours, hierarchy = cv2.findContours(masks_final, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Find the largest contour
    bigger = max(contours, key=lambda item: cv2.contourArea(item))
    # Filter small contours
    contours_final = []
    for i in range(0, len(contours)):
        area = cv2.contourArea(contours[i])
        if area > cv2.contourArea(bigger) / 10:
            contours_final.append(contours[i])
    return contours_final