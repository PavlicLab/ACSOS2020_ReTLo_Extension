## refer to : https://www.pyimagesearch.com/2015/09/14/ball-tracking-with-opencv/
##            https://www.geeksforgeeks.org/detection-specific-colorblue-using-opencv-python/

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

def nothing(x) :
    pass

cv2.namedWindow("Tracking")
cv2.createTrackbar("LU_X", "Tracking", 452, 1920, nothing)
cv2.createTrackbar("LU_Y", "Tracking", 126, 1280, nothing)
cv2.createTrackbar("LD_X", "Tracking", 215, 1920, nothing)
cv2.createTrackbar("LD_Y", "Tracking", 826, 1280, nothing)
cv2.createTrackbar("RU_X", "Tracking", 1436, 1920, nothing)
cv2.createTrackbar("RU_Y", "Tracking", 123, 1280, nothing)
cv2.createTrackbar("RD_X", "Tracking", 1706, 1920, nothing)
cv2.createTrackbar("RD_Y", "Tracking", 814, 1280, nothing)


LU = [0,0]
LD = [0,0]
RU = [0,0]
RD = [0,0]


# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color,0,0,rs.format.bgr8,0)
#config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)

# Start streaming
pipeline.start(config)

try :
    # This drives the program into an infinite loop.
    while(1) :

        frames = pipeline.wait_for_frames()
        frame_color = frames.get_color_frame()

        if not frame_color :
            continue

        frame = np.asanyarray(frame_color.get_data())

        ################ Perspective_Transformation ###########
        LU[0] = cv2.getTrackbarPos("LU_X","Tracking")
        LU[1] = cv2.getTrackbarPos("LU_Y","Tracking")
        LD[0] = cv2.getTrackbarPos("LD_X","Tracking")
        LD[1] = cv2.getTrackbarPos("LD_Y","Tracking")
        RU[0] = cv2.getTrackbarPos("RU_X","Tracking")
        RU[1] = cv2.getTrackbarPos("RU_Y","Tracking")
        RD[0] = cv2.getTrackbarPos("RD_X","Tracking")
        RD[1] = cv2.getTrackbarPos("RD_Y","Tracking")
        # Set pts1 & pts2
        pts1 = np.float32([LU,LD,RU,RD])
        pts2 = np.float32([[1,1],[1,504],[960,1],[960,504]])

        # marking the pts1 in the picture

        cv2.line(frame, (LU[0],LU[1]), (LD[0],LD[1]), (255,255,255), 2)
        cv2.line(frame, (LU[0],LU[1]), (RU[0],RU[1]), (255,255,255), 2)
        cv2.line(frame, (LD[0],LD[1]), (RD[0],RD[1]), (255,255,255), 2)
        cv2.line(frame, (RU[0],RU[1]), (RD[0],RD[1]), (255,255,255), 2)

        cv2.circle(frame, tuple(LU), 5, (255,255,255), -1)
        cv2.circle(frame, tuple(LD), 5, (255,255,255), -1)
        cv2.circle(frame, tuple(RU), 5, (255,255,255), -1)
        cv2.circle(frame, tuple(RD), 5, (255,255,255), -1)

        M = cv2.getPerspectiveTransform(pts1, pts2)
        frame2 = cv2.warpPerspective(frame, M, (961,505))

        cv2.imshow("Original",frame)
        cv2.imshow("Perspective",frame2)
        key = cv2.waitKey(1) & 0xFF

        # if the 'q' key is pressed, stop the loop
        if key == ord("q") :
            break
finally :
    pipeline.stop()

cv2.destroyAllWindows()
