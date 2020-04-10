from imutils.video import VideoStream
from matplotlib import pyplot as plt

from socket import *
from select import *

import pyrealsense2 as rs
import numpy as np
import imutils
import cv2
import time
import argparse
import math

width = 71*4*3
height = 56*4*3

#Lower = [0,73,72]
#Upper = [255,255,255]
Lower = [70,0,0]
Upper = [255,166,87]
#Lower = [0,0,83]
#Upper = [255,71,204]
_check = 1

point = [[452,126],[215,826],[1436,123],[1706,814]]
def nothing(x):
        pass

cv2.namedWindow("Tracking")
cv2.createTrackbar("LB", "Tracking", Lower[0], 255, nothing)
cv2.createTrackbar("LG", "Tracking", Lower[1], 255, nothing)
cv2.createTrackbar("LR", "Tracking", Lower[2], 255, nothing)
cv2.createTrackbar("UB", "Tracking", Upper[0], 255, nothing)
cv2.createTrackbar("UG", "Tracking", Upper[1], 255, nothing)
cv2.createTrackbar("UR", "Tracking", Upper[2], 255, nothing)
#cv2.createTrackbar("LB", "Tracking", 104, 255, nothing)
#cv2.createTrackbar("LG", "Tracking", 81, 255, nothing)
#cv2.createTrackbar("LR", "Tracking", 13, 255, nothing)
#cv2.createTrackbar("UB", "Tracking", 163, 255, nothing)
#cv2.createTrackbar("UG", "Tracking", 165, 255, nothing)
#cv2.createTrackbar("UR", "Tracking", 78, 255, nothing)

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color,0,0,rs.format.bgr8,0)
#config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)

# Start streaming
pipeline.start(config)

try :
    while True :

        frames = pipeline.wait_for_frames()
        frame_color = frames.get_color_frame()

        if not frame_color :
            continue

        frame = np.asanyarray(frame_color.get_data())

        if frame is None :
            break

        pts1 = np.float32(point)
        pts2 = np.float32([[1,1],[1,height-1],[width-1,1],[width-1,height-1]])

        cv2.line(frame, tuple(point[0]),tuple(point[1]), (255,255,255), 3)
        cv2.line(frame, tuple(point[1]),tuple(point[3]), (255,255,255), 3)
        cv2.line(frame, tuple(point[2]),tuple(point[3]), (255,255,255),3)
        cv2.line(frame, tuple(point[0]),tuple(point[2]), (255,255,255), 3)

        M = cv2.getPerspectiveTransform(pts1, pts2)
        frame = cv2.warpPerspective(frame, M, (width,height))
        blurred = cv2.GaussianBlur(frame,(11,11),0)
#        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV) ## picture to hsv values

        l_h = cv2.getTrackbarPos("LB", "Tracking")
        l_s = cv2.getTrackbarPos("LG", "Tracking")
        l_v = cv2.getTrackbarPos("LR", "Tracking")

        u_h = cv2.getTrackbarPos("UB", "Tracking")
        u_s = cv2.getTrackbarPos("UG", "Tracking")
        u_v = cv2.getTrackbarPos("UR", "Tracking")

        l_b = np.array([l_h, l_s, l_v]) ## lower limit blue color
        u_b = np.array([u_h, u_s, u_v]) ## upper limit blue color


#        mask = cv2.inRange(hsv, l_b, u_b) ## extract blue value from the picture
        mask = cv2.inRange(blurred, l_b,u_b) ## extract blue value from the picture
        if _check == 1 :
            mask = cv2.erode(mask,None, iterations=2)
            mask = cv2.dilate(mask,None,iterations=2)

        if _check == 0 :
            res = cv2.bitwise_not(mask)
            cv2.imshow("bitwise_not", res)

        cv2.imshow("frame", frame)
        cv2.imshow("mask", mask)
        '''
        YR_Lower = [(110,0,0),(0,0,90)]
        YR_Upper = [(255,191,100),(93,75,204)]
        mask2 = cv2.inRange(blurred,YR_Lower[0],YR_Upper[0])
        mask2 = cv2.erode(mask2,None, iterations=2)
        mask2 = cv2.dilate(mask2,None,iterations=2)
        mask3 = cv2.inRange(blurred,YR_Lower[1],YR_Upper[1])
        mask3 = cv2.erode(mask3,None, iterations=2)
        mask3 = cv2.dilate(mask3,None,iterations=2)
        mask4= mask+mask2+mask3
        '''
#        cv2.imshow("mask4",mask4)

        key = cv2.waitKey(1) & 0xFF

        # if the 'q' key is pressed, stop the loop
        if key == ord("q") :
            break

finally :
    pipeline.stop()

cv2.destroyAllWindows()
