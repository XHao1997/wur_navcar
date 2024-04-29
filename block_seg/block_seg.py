import cv2
import numpy as np

# 滑动条的回调函数，获取滑动条位置处的值
def empty(a):
    h_min = cv2.getTrackbarPos("Hue Min", "TrackBars")
    h_max = cv2.getTrackbarPos("Hue Max", "TrackBars")
    s_min = cv2.getTrackbarPos("Sat Min", "TrackBars")
    s_max = cv2.getTrackbarPos("Sat Max", "TrackBars")
    v_min = cv2.getTrackbarPos("Val Min", "TrackBars")
    v_max = cv2.getTrackbarPos("Val Max", "TrackBars")
    r_min = cv2.getTrackbarPos("Red Min", "TrackBars")
    r_max = cv2.getTrackbarPos("Red Max", "TrackBars")
    g_min = cv2.getTrackbarPos("Green Min", "TrackBars")
    g_max = cv2.getTrackbarPos("Green Max", "TrackBars")
    b_min = cv2.getTrackbarPos("Blue Min", "TrackBars")
    b_max = cv2.getTrackbarPos("Blue Max", "TrackBars")
    return h_min, h_max, s_min, s_max, v_min, v_max, r_min, r_max, g_min, g_max, b_min, b_max
# 定义回调函数，此程序无需回调，所以Pass即可
def callback(object):  #注意这里createTrackbar会向其传入参数即滑动条地址（几乎用不到），所以必须写一个参数
    pass

path = 'rgb_67.png'
    # 创建一个窗口，放置6个滑动条
def create_trackers():
    cv2.namedWindow("TrackBars")
    cv2.resizeWindow("TrackBars", 640, 640)
    cv2.createTrackbar("Hue Min", "TrackBars", 0, 179, callback)
    cv2.createTrackbar("Hue Max", "TrackBars", 19, 179, callback)
    cv2.createTrackbar("Sat Min", "TrackBars", 110, 255, callback)
    cv2.createTrackbar("Sat Max", "TrackBars", 240, 255, callback)
    cv2.createTrackbar("Val Min", "TrackBars", 153, 255, callback)
    cv2.createTrackbar("Val Max", "TrackBars", 255, 255, callback)
    cv2.createTrackbar("Red Min", "TrackBars", 0, 179, callback)
    cv2.createTrackbar("Red Max", "TrackBars", 19, 179, callback)
    cv2.createTrackbar("Blue Min", "TrackBars", 110, 255, callback)
    cv2.createTrackbar("Blue Max", "TrackBars", 240, 255, callback)
    cv2.createTrackbar("Green Min", "TrackBars", 153, 255, callback)
    cv2.createTrackbar("Green Max", "TrackBars", 255, 255, callback)

while True:
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # 调用回调函数，获取滑动条的值
    lower_hsv = np.array([0, 220, 102])
    upper_hsv = np.array([179, 255, 255])
    # 获得指定颜色范围内的掩码
    mask_hsv = cv2.inRange(imgHSV, lower_hsv, upper_hsv)
    contours, hierarchy = cv2.findContours(mask_hsv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Find the largest contour (largest connected component)
    max_contour = max(contours, key=cv2.contourArea)
    # Create an empty image to draw the ellipse
    ellipse_img = np.zeros_like(mask_hsv)
    # Fit an ellipse to the largest contour
    ellipse = cv2.fitEllipse(max_contour)
    center = (int(ellipse[0][0]), int(ellipse[0][1]))
    cv2.circle(ellipse_img, center, 2, 255, thickness=cv2.FILLED)
    cv2.imshow("Mask ellipse", ellipse_img)
    # # Apply the mask to the original image
    imgResult_hsv = cv2.bitwise_and(img, img, mask=ellipse_img).astype(np.uint8)
    cv2.imshow("Mask HSV", imgResult_hsv)
    cv2.waitKey(1)
    cv2.waitKey(1)

