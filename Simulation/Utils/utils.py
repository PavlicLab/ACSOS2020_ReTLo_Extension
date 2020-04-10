from matplotlib import pyplot as pyplot
import pyrealsense2 as rs
import pandas as pd
import numpy as np
import imutils
import cv2
import time
import argparse
import math
import datetime
import qrcode
import copy
import pyzbar.pyzbar as pyzbar

## Indentification of the robot
def Robot_num(image,YR_Lower,YR_Upper) :
    _num = 0
    for i in range(0,len(YR_Lower)) :
#        blurred = cv2.GaussianBlur(image,(11,11),0)
#        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(image.copy(), YR_Lower[i], YR_Upper[i]) ## extract blue value from the picture
        mask = cv2.erode(mask,None, iterations=2)
        mask = cv2.dilate(mask,None,iterations=2)
        cnts = cv2.findContours(mask.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        for j in range(len(cnts)) :
            c = cnts[j]
            circle = cv2.minEnclosingCircle(c)
            if circle[1] >= 5.5 :
                _num += i*3+1
    if _num == 0 :
        _num = 7
    return _num

## Caculate Ecuclidean distance
def Calculate_distance(pts1,pts2) :
    dist_x = pts1[0]-pts2[0]
    dist_y = pts1[1]-pts2[1]
    dist = (dist_x**2 + dist_y**2)**0.5
    return dist

## Calculate Orientation
def Calculate_Orientation(Vector1, Vector2) :
    ab = Vector1[0]*Vector2[0] + Vector1[1]*Vector2[1]
    a = math.sqrt(Vector1[0] **2 + Vector1[1] ** 2)
    b = math.sqrt(Vector2[0] **2 + Vector2[1] ** 2)
    if a*b != 0 :
        theta = math.acos(ab/(a*b))
    else :
        theta = 0
    theta = int(theta*(180/3.14))
    if Vector1[1] > 0 :
        theta = theta * -1
    return theta

## Calculate Positions
def Calculate_X_Y_O_from_Triangle(image,pts1,pts2,pts3) :

    temp_vertex = [0,0]
    tmp_dist = (pts1-pts2)**2
    dist_12 = tmp_dist[0]+tmp_dist[1]
    tmp_dist = (pts2-pts3)**2
    dist_23 = tmp_dist[0]+tmp_dist[1]
    tmp_dist = (pts1-pts3)**2
    dist_13 = tmp_dist[0]+tmp_dist[1]
    dist = [dist_23,dist_13,dist_12]
    middle_pts = [(pts2+pts3)/2,(pts1+pts3)/2,(pts1+pts2)/2]

    if dist[0] == min(dist) :
#        cv2.line(image, tuple(pts1),tuple(np.int0(middle_pts[0])),(0,0,0),2)
        center = np.int0(middle_pts[0])
        O = pts1-center
        temp_vertex = pts1
        center = [int((pts1[0]+2*center[0])/3), int((pts1[1]+2*center[1])/3)]
        #        cv2.line(image, tuple(pts1),tuple(np.int0(middle_pts[0])),(0,0,0),2)

#        cv2.circle(image,tuple(center),3,(0,255,255),-1)

    elif dist[1] == min(dist) :
#        cv2.line(image, tuple(pts2),tuple(np.int0(middle_pts[1])),(0,0,0),2)
        center = np.int0(middle_pts[1])
        O = pts2-center
        temp_vertex = pts2
        center = [int((pts2[0]+2*center[0])/3), int((pts2[1]+2*center[1])/3)]
#        cv2.circle(image,tuple(center),3,(0,255,255),-1)

    else :
#        cv2.line(image, tuple(pts3),tuple(np.int0(middle_pts[2])),(0,0,0),2)
        center = np.int0(middle_pts[2])
        O = pts3-center
        temp_vertex = pts3
        center = [int((pts3[0]+2*center[0])/3), int((pts3[1]+2*center[1])/3)]
#        cv2.circle(image,tuple(center),3,(0,255,255),-1)

    return center,O,temp_vertex

## Draw traingle in the frame
def draw_Triangle(frame,points) :
    for i in range(0,len(points)) :
        cv2.line(frame, tuple(points[i][0]),tuple(points[i][1]),(255,0,0),2)
        cv2.line(frame, tuple(points[i][1]),tuple(points[i][2]),(255,0,0),2)
        cv2.line(frame, tuple(points[i][2]),tuple(points[i][0]),(255,0,0),2)
    return frame

## QR code detection
def QR_Detection(image) :
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _d = pyzbar.decode(gray)
    data = _d[0].data.decode('utf-8')
    return data

## Perspective transformation
def perspective_image(frame,width,height,point) :
    pts1 = np.float32(point)
    pts2 = np.float32([[1,1],[1,height-1],[width-1,1],[width-1,height-1]])
    # marking the pts1 in the picture.

    cv2.line(frame, tuple(point[0]),tuple(point[1]), (255,255,255), 3)
    cv2.line(frame, tuple(point[1]),tuple(point[3]), (255,255,255), 3)
    cv2.line(frame, tuple(point[2]),tuple(point[3]), (255,255,255),3)
    cv2.line(frame, tuple(point[0]),tuple(point[2]), (255,255,255), 3)

    M = cv2.getPerspectiveTransform(pts1, pts2)
    frame2 = cv2.warpPerspective(frame, M, (width,height))
    return frame2

## Find the robots in the frame
def find_robots(frame, Lower,Upper,n_robots,stop,YR_Lower,YR_Upper,points ) :
    blurred = cv2.GaussianBlur(frame,(11,11),0)
#    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    _mask = cv2.inRange(frame,Lower,Upper)
    '''
    for i in range(0,len(YR_Lower)) :
        _mask = cv2.inRange(frame,YR_Lower[i],YR_Upper[i])
        _mask = cv2.erode(_mask,None, iterations=2)
        _mask = cv2.dilate(_mask,None,iterations=2)
        mask+=_mask
    '''
    mask = cv2.bitwise_not(_mask)
    # remove any small blobs that may e left on the mask
#    mask = cv2.erode(mask,None, iterations=2)
#    mask = cv2.dilate(mask,None,iterations=2)

    cnts = cv2.findContours(mask.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    dots = list()
    _tmp = list()
    for i in range(len(cnts)) :
        c = cnts[i]
        tri = cv2.minEnclosingTriangle(c)
        point = list()
        if tri[0] >= 700 :
            for j in range(0,3) :
                point.append([np.int(tri[1][j][0][0]),np.int(tri[1][j][0][1])])
            dots.append(point)
            center,o,vertex = Calculate_X_Y_O_from_Triangle(frame, np.array(point[0]),np.array(point[1]),np.array(point[2]))
            _x = [min(point[0][0],point[1][0],point[2][0]),max(point[0][0],point[1][0],point[2][0])]
            _y = [min(point[0][1],point[1][1],point[2][1]),max(point[0][1],point[1][1],point[2][1])]
            if _x[0] <= 0 :
                _x[0] = 1
            if _y[0] <= 0 :
                _y[0] = 1
            image = frame.copy()
            image = image[_y[0]:_y[1],_x[0]:_x[1]]
            h,w,c = image.shape
            l_image = cv2.pyrUp(image,dstsize=(w*2,h*2),borderType = cv2.BORDER_DEFAULT)
            data = Robot_num(l_image,YR_Lower,YR_Upper)
            _tmp.append(data)
            if data == 'None' or int(data)-1 >= len(points):
                print('Wrong Detection')
                stop = 1
            else :
                points[int(data)-1] = [center,o]
            cv2.circle(frame,tuple(center),4,(0,255,0),-1)
    print(_tmp)
    print('------')
    frame = draw_Triangle(frame,dots)
    return frame, points, stop, points

## Set target point
def head_goal(random_point, prev_point,info_robots,width,height,m,n,blank,bound_x,bound_y) :
    stop = 0
    robots_now = list()
    for i in range(0,len(info_robots)) :
        _x = int((info_robots[i][0][0]-blank)/int((width-blank*2)/m))
        _y = int((info_robots[i][0][1]-blank)/int((height-blank*2)/n))
        if _x >= m :
            _x = m-1
        if _y >= n :
            _y = n-1
        robots_now.append([_x,_y])

    _x_r = int((random_point[0]-blank)/int((width-blank*2)/m))
    _y_r = int((random_point[1]-blank)/int((height-blank*2)/n))
    if _x_r >= m :
        _x_r = m-1
    if _y_r >= n :
        _y_r = n-1
    _random_point = [_x_r,_y_r]

    if prev_point == [-1,-1] or Calculate_distance(info_robots[0][0],random_point) <= 50 :
        prev = prev_point
        if prev == [-1,-1] :
            for i in range(1,len(info_robots)) :
                if robots_now[i] != robots_now[0] :
                    prev = robots_now[i]
                    break
            _random_point = robots_now[0]
#        prev_point = robots_now[0]
        prev_point = _random_point

        ## 다음 목표 정하기 (section)
        direction = np.array(prev) -np.array(_random_point)
        binary = [-1,0,1]
        next_candidates = list()
        bound = 0

        ## 2, 4, 5, 7
        if direction[0] == 0 :
            for i in binary :
                if (_random_point[0] +i) >= 0 and  (_random_point[1] + direction[1]*-1) >= 0 and (_random_point[0] +i) < m and  (_random_point[1] + direction[1]*-1) < n :
                    next_candidates.append([_random_point[0]+i,_random_point[1] + direction[1]*-1])
            if len(next_candidates) == 0 :
                bound = 1
                if _random_point[0]+1 < m :
                    next_candidates.append([_random_point[0]+1,_random_point[1]])
                if _random_point[0]-1 >= 0 :
                    next_candidates.append([_random_point[0]-1,_random_point[1]])
        elif direction[1] == 0 :
            for i in binary :
                if (_random_point[1] +i) >= 0 and  (_random_point[0] + direction[0]*-1) >= 0 and (_random_point[1] +i) < n and  (_random_point[0] + direction[0]*-1) < m :
                    next_candidates.append([_random_point[0] + direction[0]*-1, _random_point[1] + i])
            if len(next_candidates) == 0 :
                bound = 1
                if _random_point[1]+1 < n :
                    next_candidates.append([_random_point[0],_random_point[1]+1])
                if _random_point[1]-1 >= 0 :
                    next_candidates.append([_random_point[0],_random_point[1]-1])
        ## 1, 3, 6, 8
        else :
            if _random_point[0] + direction[0]*-1 >= 0 and  _random_point[0] + direction[0]*-1 < m:
                next_candidates.append([_random_point[0]+direction[0]*-1,_random_point[1]])
            if _random_point[1] + direction[1]*-1 >= 0 and _random_point[1] + direction[1]*-1 < n :
                next_candidates.append([_random_point[0],_random_point[1]+direction[1]*-1])
            if _random_point[0] + direction[0]*-1 >= 0 and _random_point[1] + direction[1]*-1 >= 0  and  _random_point[0] + direction[0]*-1 < m and _random_point[1] + direction[1]*-1 < n :
                next_candidates.append([_random_point[0]+direction[0]*-1,_random_point[1]+direction[1]*-1])

        if len(next_candidates) == 0 :
            if _random_point == [0,0] :
                bound = 1
                next_candidates.append([0,1])
                next_candidates.append([1,0])
            if _random_point == [0,n-1] :
                bound = 1
                next_candidates.append([0,n-2])
                next_candidates.append([1,n-1])
            if _random_point == [m-1,0] :
                bound = 1
                next_candidates.append([m-2,0])
                next_candidates.append([m-1,1])
            if _random_point == [m-1,n-1] :
                bound = 1
                next_candidates.append([m-1,n-2])
                next_candidates.append([m-2,n-1])

        for i in range(1,len(robots_now)) :
            for j in range(0,len(next_candidates)) :
                if j < len(next_candidates) :
                    if next_candidates[j] == robots_now[i] :
                        next_candidates.remove(next_candidates[j])
        '''
        ## 2, 4, 5, 7
        if direction[0] == 0 :
            for i in binary :
                if (robots_now[0][0] +i) >= 0 and  (robots_now[0][1] + direction[1]*-1) >= 0 and (robots_now[0][0] +i) < m and  (robots_now[0][1] + direction[1]*-1) < n :
                    next_candidates.append([robots_now[0][0]+i,robots_now[0][1] + direction[1]*-1])
            if len(next_candidates) == 0 :
                bound = 1
                if robots_now[0][0]+1 < m :
                    next_candidates.append([robots_now[0][0]+1,robots_now[0][1]])
                if robots_now[0][0]-1 >= 0 :
                    next_candidates.append([robots_now[0][0]-1,robots_now[0][1]])
        elif direction[1] == 0 :
            for i in binary :
                if (robots_now[0][1] +i) >= 0 and  (robots_now[0][0] + direction[0]*-1) >= 0 and (robots_now[0][1] +i) < n and  (robots_now[0][0] + direction[0]*-1) < m :
                    next_candidates.append([robots_now[0][0] + direction[0]*-1, robots_now[0][1] + i])
            if len(next_candidates) == 0 :
                bound = 1
                if robots_now[0][1]+1 < n :
                    next_candidates.append([robots_now[0][0],robots_now[0][1]+1])
                if robots_now[0][1]-1 >= 0 :
                    next_candidates.append([robots_now[0][0],robots_now[0][1]-1])
        ## 1, 3, 6, 8
        else :
            if robots_now[0][0] + direction[0]*-1 >= 0 and  robots_now[0][0] + direction[0]*-1 < m:
                next_candidates.append([robots_now[0][0]+direction[0]*-1,robots_now[0][1]])
            if robots_now[0][1] + direction[1]*-1 >= 0 and robots_now[0][1] + direction[1]*-1 < n :
                next_candidates.append([robots_now[0][0],robots_now[0][1]+direction[1]*-1])
            if robots_now[0][0] + direction[0]*-1 >= 0 and robots_now[0][1] + direction[1]*-1 >= 0  and  robots_now[0][0] + direction[0]*-1 < m and robots_now[0][1] + direction[1]*-1 < n :
                next_candidates.append([robots_now[0][0]+direction[0]*-1,robots_now[0][1]+direction[1]*-1])

        if len(next_candidates) == 0 :
            if robots_now[0] == [0,0] :
                bound = 1
                next_candidates.append([0,1])
                next_candidates.append([1,0])
            if robots_now[0] == [0,n-1] :
                bound = 1
                next_candidates.append([0,n-2])
                next_candidates.append([1,n-1])
            if robots_now[0] == [m-1,0] :
                bound = 1
                next_candidates.append([m-2,0])
                next_candidates.append([m-1,1])
            if robots_now[0] == [m-1,n-1] :
                bound = 1
                next_candidates.append([m-1,n-2])
                next_candidates.append([m-2,n-1])

        for i in range(1,len(robots_now)) :
            for j in range(0,len(next_candidates)) :
                if j < len(next_candidates) :
                    if next_candidates[j] == robots_now[i] :
                        next_candidates.remove(next_candidates[j
        '''

        if len(next_candidates) == 0 :
            stop = 1
        else :
            select_next = np.random.randint(0,len(next_candidates))
            if bound == 1 :
                if next_candidates[select_next][0] == 0 :
                    _x_rand = np.random.randint(25,blank)
                elif next_candidates[select_next][0] == m-1 :
                    _x_rand = np.random.randint(width-blank,width-25)
                else :
                    _x_rand = np.random.randint(bound_x[next_candidates[select_next][0]],bound_x[next_candidates[select_next][0]+1])

                if next_candidates[select_next][1] == 0 :
                    _y_rand = np.random.randint(25,blank)
                elif next_candidates[select_next][1] == n-1 :
                    _y_rand = np.random.randint(height-blank,height-25)
                else :
                    _y_rand = np.random.randint(bound_y[next_candidates[select_next][1]],bound_y[next_candidates[select_next][1]+1])

            else :
                # 4이면 4-5사이, 5면 5-6사
                _x_rand = np.random.randint(bound_x[next_candidates[select_next][0]],bound_x[next_candidates[select_next][0]+1])
                _y_rand = np.random.randint(bound_y[next_candidates[select_next][1]],bound_y[next_candidates[select_next][1]+1])
            random_point = [_x_rand,_y_rand]

    return stop,random_point,prev_point

## Calculate V and O
def Calculate_V_and_O(targets,myposition,myorientation,magnification) :

    dist = Calculate_distance(myposition,targets)* (1/magnification)
    v = dist
    delta = 0.000000001

    vector1 = [targets[0]-myposition[0],targets[1]-myposition[1]]
#    print(myorientation)
    vector2 = [myorientation[0],myorientation[1]]
    size_vector1 = (vector1[0]**2 + vector1[1]**2)**0.5
    size_vector2 = (vector2[0]**2 + vector2[1]**2)**0.5
    v1_v2 = vector1[0]*vector2[1]-vector1[1]-vector2[0]
    o = np.arccos((vector1[0]*vector2[0]+vector1[1]*vector2[1])/((size_vector1*size_vector2)+delta))

    ## 이걸로 하면 방향 바로 구할 수 있음.
    # o = np.arcsin((vector1[0]*vector2[1]-vector1[1]*vector2[0])/((size_vector1*size_vector2)+delta))

    ## Left, Right
    temp_gradient = vector2[1]/(vector2[0]+delta)
    temp_intercept_y = vector2[1]
    temp_target = vector1[1] - (temp_gradient*vector1[0] + temp_intercept_y)
    if temp_target * vector2[0] > 0 :
       o = o * -1

    if v <= 5 :
        o = o * 0.1
#    if o >= 1 :
#        o = 1
#    if o <= -1 :
#        o = -1
##    o1 = np.arcsin(v1_v2/((size_vector1*size_vector2)+delta))
##    if o1 <= 0 :
##        o = o*-1

    return v,o

## Set desired position following by the motion rule
def set_Follower_target(back_coordination,forward_coordination, my_coordination) :
#    if distinct == 0 :
    f_goal_dst = 90
    b_goal_dst = 140

    forward_coordination, back_coordination, my_coordination = np.array(forward_coordination), np.array(back_coordination), np.array(my_coordination)
    forward_dst = np.sqrt(np.sum((forward_coordination - my_coordination) ** 2))
    back_dst = np.sqrt(np.sum((back_coordination - my_coordination) ** 2))

    forward_v = (forward_coordination - my_coordination)
    back_v = (back_coordination - my_coordination)

    target_coordination = my_coordination + forward_v * np.maximum((forward_dst - f_goal_dst), 0) + back_v * np.maximum((back_dst - b_goal_dst), 0)
    target_coordination = [int(target_coordination[0]), int(target_coordination[1])]

#    print('{}: f_dst = {}, f_f = {}, f_v = {}, b_dst = {}, b_f = {}, b_v = {}'.format(distinct, forward_dst, np.maximum((forward_dst - f_goal_dst), 0),
#                                            forward_v, back_dst, np.maximum((back_dst - b_goal_dst), 0), back_v))

    return target_coordination

## Set next random target
def calculate_next_goal(info_robots,random_point,magnification) :
    next_target = list()
    message = list()
    for i in range(0,len(info_robots)) :
        if i == 0 :
            _target = random_point
#            _target = set_Follower_target(info_robots[i+1][0],random_point,info_robots[i][0])
        elif i != len(info_robots)-1 :
            _target = set_Follower_target(info_robots[i+1][0],info_robots[i-1][0],info_robots[i][0])
        else :
            _target = set_Follower_target(info_robots[i-1][0],info_robots[i-1][0],info_robots[i][0])
        next_target.append(_target)

    for i in range(0,len(info_robots)) :
        next_v,next_o = Calculate_V_and_O(next_target[i],info_robots[i][0],info_robots[i][1],magnification)
        if i == 0 and next_v >= 10 :
            next_v = 10
        if i != 0 and next_v >= 10 :
            next_v = 10
        _v = int(next_v)
        _o = int(next_o * (180/3.14))
        message.append('{} {}'.format(_v,_o))
    return message
