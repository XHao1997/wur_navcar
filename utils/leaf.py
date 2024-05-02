import numpy as np
import cv2
import copy

def draw_incircle(mask,cnts):
    mask = np.zeros_like(mask)
    for cnt in cnts:
        M = cv2.moments(cnt)
        if M['m00'] != 0:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            diff = (cnt - [cx,cy]).reshape(-1,2)
            # Calculate Euclidean distance between each point in the contour and center1
            distances = np.linalg.norm(diff, axis=1)
            max_dist = min(distances)
            cv2.circle(mask, (cx, cy), int(0.9*max_dist), (255, 255, 255), thickness=-1)
    return mask

def get_cnts(mask):    
    mask = cv2.resize(mask, (640,480), interpolation = cv2.INTER_LINEAR )
    mask = mask[:,:,0]
    mask_copy = copy.deepcopy(mask)
    contours, _ = cv2.findContours(mask_copy, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)   
    return contours

def get_optimal_picking_points(contours,corner_list, leaf_index=-1):
    optimal_points=[]
    for cnt, corner in zip(contours, corner_list):
        cnt = cnt.reshape(-1,2)
        corner = np.asarray(corner).reshape(-1,2)
        corners_left = corner[np.argmin(corner[:,0])]
        corners_down = corner[np.argmax(corner[:,1])]
        optimal_points.append(cnt[np.argmin(np.linalg.norm(cnt-corners_left,axis=1))])
        optimal_points.append(cnt[np.argmin(np.linalg.norm(cnt-corners_down,axis=1))]) 
    optimal_points = np.asarray(optimal_points).reshape(-1,2)    
    if leaf_index != -1:
        optimal_points = optimal_points[2*leaf_index:2*(leaf_index+1)]
    return optimal_points

def get_incircle(mask,cnts):
    mask = np.zeros_like(mask)
    for cnt in cnts:
        M = cv2.moments(cnt)
        if M['m00'] != 0:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            diff = (cnt - [cx,cy]).reshape(-1,2)
            # Calculate Euclidean distance between each point in the contour and center1
            distances = np.linalg.norm(diff, axis=1)
            max_dist = min(distances)
            cv2.circle(mask, (cx, cy), int(max_dist*0.9), (255, 255, 255), -1)
    return mask, (cx,cy)

def find_furthest_points(cnt):
    max_dist_prev = 0
    for i, points in enumerate(cnt):
        max_dist = np.max(np.linalg.norm(cnt-points,axis=1))
        if max_dist>max_dist_prev:
            max_index = [i, np.argmax(np.linalg.norm(cnt-points,axis=1))]
        max_dist_prev = max_dist
    print(max_index)
    return max_index
def find_midpoint(point1, point2):
    return np.array([(point1[0] + point2[0]) / 2, (point1[1] + point2[1]) /2])  

def find_y_intercept(mid_point, slope):
    return mid_point[1] - slope * mid_point[0]


def calculate_slope(point1, point2):
    slope = (point2[1] - point1[1]) / (point2[0] - point1[0]) if point2[0] != point1[0] else float('inf')
    return -1/slope

def distance_to_line(x0, y0, a, b):
    return abs(a * x0 - y0 + b) / np.sqrt(a**2 + 1)

def get_leaf_corner(leaf_contour):
    point1,point2 = leaf_contour[find_furthest_points(leaf_contour)]
    midpoint = find_midpoint(point1, point2)
    slope = calculate_slope(point1,point2)
    intercept = find_y_intercept(midpoint.astype(int),slope)
    dist_list = np.zeros(leaf_contour.shape[0])
    for index,p in enumerate(leaf_contour):
        dist_list[index] = distance_to_line(p[0],p[1],slope, intercept)
    result = dist_list
    smallest_index = np.argpartition(abs(result), 20)[:20]
    point3 = leaf_contour[smallest_index[0]]
    point4_index = np.argmin(np.linalg.norm((leaf_contour[smallest_index]+point3-2*midpoint),axis=1))
    point4 = leaf_contour[smallest_index[point4_index]]
    return (point1, point2, point3, point4, midpoint)

def draw_leaf_corner(corners, mask):
    mask1 = np.zeros_like(copy.deepcopy(mask))
    for point in corners:
        cv2.circle(mask1, point.astype(int), 5, (255, 0, 255), -1)  # Red point
    return mask1

def get_leaf_center(leaf_contour):
    leaf_center = get_leaf_corner(leaf_contour)[-1]
    return leaf_center
    
def draw_circle(point, image, size=5):
    cv2.circle(image, point, int(size), (255, 255, 255), thickness=-1)
    return image
    