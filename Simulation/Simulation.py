from imutils.video import VideoStream
from matplotlib import pyplot as pyplot

from socket import *
from select import *
import pyrealsense2 as rs
import pandas as pd
import numpy as np
import imutils
import cv2
import time
import argparse
import math
import datetime
from Utils.utils import Calculate_distance, Calculate_Orientation, Calculate_X_Y_O_from_Triangle
from Utils.utils import draw_Triangle, QR_Detection, perspective_image
from Utils.utils import find_robots, head_goal, Calculate_V_and_O, set_Follower_target,calculate_next_goal
import copy

## 1. Connect the robots (Socket Communication)
## 2. Setting the frame (Frame cut and Video setting)
## 3. Find robots and calculate the position and the orientation
#### 3.1 Find triangles in the frame
#### 3.2 Zoom in the area which has the triangle
#### 3.3 Identify the robot using the circles
#### 3.4 Calculate V and O
## 4. Set Target
## 5. Calculate next V and O
#### 5.1 Deliever it to the thymio
## 6. Save Poisition and Orientation to CSV file
## 7. Extract Video and Data

#####################################################################
## Variables
n_robots = 7
name_robot = ['H']
for i in range(0,n_robots-2) :
    name_robot.append('F{}'.format(i+1))
name_robot.append('T')

magnification = 3
width = 71*4*magnification # The width of the arena * 3 (3 pixel per 1 cm)
height = 56*4*magnification # The height of the arena * 3 (3 pixel per 1 cm)

point = [[452,126],[215,826],[1436,123],[1706,814]] # Pesrpective transformation points
frame_number = 0

Lower = (0,73,72) # Lower bound of the triangle
Upper = (255,255,255) # Upper bound of the trianle

# Used by color detection
## Blue (1 point), Red (x points) , Yellow (y points)//
YR_Lower = [(80,0,0),(0,0,83)]
YR_Upper = [(255,166,87),(255,71,204)]
#YR_Lower = [(90,76,13)]
#YR_Upper = [(162,165,80)]

_points = list()
for i in range(0,n_robots) :
    _points.append([[0,0],0])

# used by setting Target point
random_point = [0,0]
prev_point = [-1,-1]
m = 6
n = 4
blank = 40
stop = 0

bound_x = list()
for i in range(0,m+1) :
    bound_x.append(int((int(width-blank*2)/m) * i + blank))

bound_y = list()
for i in range(0,n+1) :
    bound_y.append(int((int(height-blank*2)/n)* i + blank))

Record = {'Frame':[],'Target_x':[],'Target_y':[]}
for name in name_robot :
    for classification in ['x','y','o'] :
        Record['{}_{}'.format(name,classification)] = []


## The path of CSV file
path = './Results/'
suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
csv_filename = "_".join(["Robot7",suffix])+".csv"
filename = "_".join([path,suffix])+".avi"

####################################################################
## 1.
## set video pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color,0,0,rs.format.bgr8,0)
pipeline.start(config)
time.sleep(1)
#config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)

### Set Socket communication ###
target = [0,0]
previous_target_location = 0
target_location = 0
HOST = ''
PORT = 10000
BUFSIZE = 1024
ADDR = (HOST,PORT)
# make socket
serverSocket = socket(AF_INET, SOCK_STREAM)
# allocate socket address
serverSocket.bind(ADDR)
print('bind')

# Status of waiting connection
# Accept the connection
clientSocket = [0]
for i in range(0,n_robots-1) :
    clientSocket.append(0)

for i in range(0,n_robots) :
    serverSocket.listen(100)
    print('listen'+str(i))
    clientSocket[i], addr_info = serverSocket.accept()
    print('accept'+str(i))

print('--client information--')
for i in range(0,n_robots) :
    print(clientSocket[i])

time.sleep(1)

####################################################################
## 2.
### set save video ##
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(filename,fourcc,12.0,(width,height))

Start_Time = time.time()
##### *** Start Loop **** ######
try :
    while(1) :
        frames = pipeline.wait_for_frames()
        frame_color = frames.get_color_frame()

        if not frame_color :
            continue

        frame = np.asanyarray(frame_color.get_data())
        frame_number += 1

        if frame is None :
            print('break')
            break

        #### perspective - 컷팅 : def perspective_image(frame,width,height,point)
        frame2 = perspective_image(frame,width,height,point)

        ####################################################################
        ## 3.
        frame2, info_robots,stop,_points = find_robots(frame2, Lower, Upper, n_robots,stop,YR_Lower,YR_Upper,_points)

        ####################################################################
        ## 4.
        stop,random_point,prev_point = head_goal(random_point, prev_point, info_robots, width, height, m,n,blank, bound_x,bound_y)

        ####################################################################
        ## 5.
        next_goals = calculate_next_goal(info_robots,random_point,magnification)

        ####################################################################
        #### 5.1
        if frame_number % 3 == 0 :
                for i in range(0,n_robots) :
                    if len(next_goals) != n_robots or stop == 1:
                        clientSocket[i].send('0 0'.encode())
                    else :
                        clientSocket[i].send(next_goals[i].encode())

        ####################################################################
        ## 6.
        Record['Frame'].append(frame_number)
        Record['Target_x'].append(random_point[0])
        Record['Target_y'].append(random_point[1])

        for i,name in enumerate(name_robot) :
            for j,classification in enumerate(['x','y']) :
                Record['{}_{}'.format(name,classification)].append(info_robots[i][0][j])
            Record['{}_o'.format(name)].append(Calculate_Orientation(info_robots[i][1],[1,0]))

        key = cv2.waitKey(1) & 0xFF
        cv2.circle(frame2,(random_point[0],random_point[1]),2,(220,220,220),-1)
        frame3 = copy.copy(frame2)
        for i in range(0,len(bound_x)) :
            cv2.line(frame3,(bound_x[i],0),(bound_x[i],(frame2.shape)[0]),(255,255,255),1,8)
        for i in range(0,len(bound_y)) :
            cv2.line(frame3,(0,bound_y[i]),((frame2.shape)[1],bound_y[i]),(255,255,255),1,8)
        cv2.imshow("Result",frame3)
        out.write(frame2)
        if key == ord("q") or frame_number >= 3720 :
            print(frame_number/(60*12))
            break
finally :
    pipeline.stop()
    for i in range(0,n_robots) :
        clientSocket[i].send('0 0'.encode())
    save_data = pd.DataFrame(Record)
    save_data.to_csv('{}{}'.format(path,csv_filename))

    serverSocket.close()
    out.release()
    cv2.destroyAllWindows()

for i in range(0,n_robots) :
    clientSocket[i].send('0 0'.encode())


###################################################################
## 7. 

save_data = pd.DataFrame(Record)
save_data.to_csv('{}{}'.format(path,csv_filename))

serverSocket.close()
out.release()
cv2.destroyAllWindows()
