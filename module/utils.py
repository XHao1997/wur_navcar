import os
import sys
PROJECT_PATH = os.getcwd()
SOURCE_PATH = os.path.join(
    PROJECT_PATH
)
sys.path.append(SOURCE_PATH)
import cv2
import numpy as np
from pathlib import Path
import open3d as o3d
import matplotlib.pyplot as plt
import matplotlib.patches as patches

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



def filter_roi_in_pcd(rgb_img, rgb_pixel, points_3d):
    roi_index = []
    points_3d_color = []
    for i, v in enumerate(points_3d):
        u, v = rgb_pixel[i][:2]
        if u >= 640 or v >= 480:
            pass
        elif u < 0 or v < 0:
            pass
        else:
            pc = rgb_img[v, u]
            if not np.array_equal(pc, np.array([0, 0, 0])):
                roi_index.append(i)
                points_3d_color.append(pc)
    roi_3d = points_3d[roi_index]
    points_3d_color = np.asarray(points_3d_color)
    pcd = o3d.geometry.PointCloud()
    # Assigning the colors
    pcd.colors = o3d.utility.Vector3dVector(points_3d_color[:, [2, 1, 0]] / 255.0)
    pcd.points = o3d.utility.Vector3dVector(roi_3d[:, :3])
    cl, ind = pcd.remove_radius_outlier(nb_points=20, radius=100)
    pcd = cl.select_by_index(ind)
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=10,
                                            std_ratio=0.2)
    pcd = cl.select_by_index(ind)
    return pcd

def create_point3d_from_xyz(x, y, z):
    # Flatten the matrices
    X_flat = x.flatten()
    Y_flat = y.flatten()
    Z_flat = z.flatten()
    # Combine into a single array of 3D points
    points_3d = np.vstack((X_flat, Y_flat, Z_flat)).T
    # Remove rows where Z < 0
    points_3d = points_3d[points_3d[:, 2] >= 0]
    return points_3d