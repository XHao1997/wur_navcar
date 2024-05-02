import cv2
import numpy as np
from pathlib import Path
import open3d as o3d
import matplotlib.pyplot as plt
import matplotlib.patches as patches

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
    # Visualizing the point cloud
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