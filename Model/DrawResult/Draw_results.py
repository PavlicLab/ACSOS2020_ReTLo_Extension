import numpy as np
import pandas as pd
import argparse
import cv2
from matplotlib import pyplot as plt
import copy

def draw_line(type = 0,frame = [],orientation=[],pts1=[],color1=(0,0,0),color3=(0,0,0),color2=(0,0,0),_size=7) :
    if type == 0 :
        for i in range(len(pts1)) :
            cv2.circle(frame,(pts1[i][0],pts1[i][1]),_size,(255,84,0),2)
            cv2.circle(frame,(pts1[i][0],pts1[i][1]),3,(204,183,61),2)
    if type == 1 :
        for i in range(len(pts1)-1) :
            cv2.line(frame, (pts1[i][0],pts1[i][1]),(pts1[i+1][0],pts1[i+1][1]),color2,2,cv2.LINE_8)
            cv2.circle(frame,(pts1[i][0],pts1[i][1]),_size,color3,-1)
            cv2.circle(frame,(pts1[i][0],pts1[i][1]),_size,color1,2)
            cv2.circle(frame,(pts1[i][0],pts1[i][1]),4,color1,-1)

            rad_orientation = np.deg2rad(orientation[i])
            arrow_end_x = pts1[i][0]+int(np.cos(rad_orientation)*_size)
            arrow_end_y = pts1[i][1]+int((-1)*np.sin(rad_orientation)*_size)
            cv2.arrowedLine(frame, (pts1[i][0],pts1[i][1]),tuple([arrow_end_x,arrow_end_y]),(0,0,0),thickness=2, tipLength=0.3)

        cv2.circle(frame,(pts1[len(pts1)-1][0],pts1[len(pts1)-1][1]),_size,color3,-1)
        cv2.circle(frame,(pts1[len(pts1)-1][0],pts1[len(pts1)-1][1]),_size,color1,2)
        cv2.circle(frame,(pts1[len(pts1)-1][0],pts1[len(pts1)-1][1]),4,color1,-1)
        rad_orientation = np.deg2rad(orientation[len(pts1)-1])
        arrow_end_x = pts1[len(pts1)-1][0]+int(np.cos(rad_orientation)*_size)
        arrow_end_y = pts1[len(pts1)-1][1]+int((-1)*np.sin(rad_orientation)*_size)
        cv2.arrowedLine(frame, (pts1[len(pts1)-1][0],pts1[len(pts1)-1][1]),tuple([arrow_end_x,arrow_end_y]),(0,0,0),thickness=2, tipLength=0.3)

    return frame

def set_pts(t=0,data=[],std_data=[],std_coord =0,start=0,instance=0,robots = ['H','F1','F2','F3','F4','F5']) :
    pts = list()
    ori = list()
    std_x = '{}_T_x'.format(start)
    std_y = '{}_T_y'.format(start)
    if t==0 :
        for robot in robots :
            target_x = '{}_{}_x'.format(start,robot)
            target_y = '{}_{}_y'.format(start,robot)
            pts.append([int(data[target_x][instance]*std_coord+std_data[std_x][instance]),int(data[target_y][instance]*std_coord+std_data[std_y][instance])])
            ori.append(int(data['{}_{}_o'.format(start,robot)][instance]*180))
        pts.append([std_data[std_x][instance],std_data[std_y][instance]])
        ori.append(int(data['{}_T_o'.format(start)][instance]*180))
    else :
        for robot in robots :
            target_x = '{}_{}_x'.format(start,robot)
            target_y = '{}_{}_y'.format(start,robot)
            pts.append([std_data[target_x][instance],std_data[target_y][instance]])
        pts.append([std_data[std_x][instance],std_data[std_y][instance]])
    return pts,ori

csv_path = './Results/'
answer_path = './Data/Instances/'
video_path = './Data/video/'

std_coordi = [260,310,400]

color_1 = [(22,219,29),(0,187,255),(0,0,255)]
color_2 = [(22,219,29),(0,187,255),(0,0,255)]
#color_2 = [(0,242,171),(0,228,255),(95,95,241)]
#color_3 = [(201,251,206),(197,236,250),(216,216,255)]
color_3 = [(0,242,171),(0,228,255),(95,95,241)]

_comparison_type = 3
_save_n = 0
_size = 18
#_save_file = [7,10,13]
#_save_file_num = [2,12,11]
#_save_file = [8,12,14]
#_save_file_num = [2,9,5]
_save_file = [7,10,11]
_save_file_num = [4,0,8]

for j,i in enumerate(_save_file) :
    if _comparison_type == 1 :
        m1 = pd.read_csv(csv_path+'M1_{}.csv'.format(i))
        m2 = pd.read_csv(csv_path+'M2_{}.csv'.format(i))
        m3 = pd.read_csv(csv_path+'M3_{}.csv'.format(i))
    elif _comparison_type == 2 :
        m3 = pd.read_csv(csv_path+'M3_{}.csv'.format(i))
        m3_2 = pd.read_csv(csv_path+'M3-2_{}.csv'.format(i))
        m3_3 = pd.read_csv(csv_path+'M3-3_{}.csv'.format(i))
    else :
        m1 = pd.read_csv(csv_path+'M1_{}.csv'.format(i))
        h6 = pd.read_csv(csv_path+'H6_{}.csv'.format(i))
        h2 = pd.read_csv(csv_path+'H2_{}.csv'.format(i))

    ans = pd.read_csv(answer_path+'{}.csv'.format(i))
    cap = cv2.VideoCapture(video_path+'{}.avi'.format(i))
    _n = 0
    frame_cnt = -1
    tmp = 0
    save_or_not = 0
    while(_n < len(ans)) :
        ret,frame = cap.read()
        frame_cnt += 1
        if ans['Frame'][_n] == frame_cnt :
            save_or_not = 1

        if frame_cnt == ans['Frame'][_n]+6*tmp and save_or_not == 1:
            pts_std,ori_std = set_pts(t=1,std_data=ans,std_coord=std_coordi[0],start=tmp,instance=_n)
            if tmp >= 10 :
                if _comparison_type == 1 :
                    pts1,ori1 = set_pts(data=m3,std_data=ans,std_coord=std_coordi[2],start=tmp,instance=_n)
                    pts2,ori2 = set_pts(data=m2,std_data=ans,std_coord=std_coordi[1],start=tmp,instance=_n)
                    pts3,ori3 = set_pts(data=m1,std_data=ans,std_coord=std_coordi[0],start=tmp,instance=_n)
                elif _comparison_type == 2 :
                    pts1,ori1 = set_pts(data=m3_3,std_data=ans,std_coord=std_coordi[2],start=tmp,instance=_n)
                    pts2,ori2 = set_pts(data=m3,std_data=ans,std_coord=std_coordi[2],start=tmp,instance=_n)
                    pts3,ori3 = set_pts(data=m3_2,std_data=ans,std_coord=std_coordi[2],start=tmp,instance=_n)
                else :
                    pts1,ori1 = set_pts(data=m1,std_data=ans,std_coord=std_coordi[0],start=tmp,instance=_n)
                    pts2,ori2 = set_pts(data=h6,std_data=ans,std_coord=std_coordi[0],start=tmp,instance=_n)
                    pts3,ori3 = set_pts(data=h2,std_data=ans,std_coord=std_coordi[0],start=tmp,instance=_n)
#            frame = draw_line(frame=frame, pts1=pts_std,color1=(100,100,0))
#            frame1 = draw_line(frame=copy.deepcopy(frame), pts1=pts3,color1=color_1[0],color2=color_2[0])
#            frame2 = draw_line(frame=copy.deepcopy(frame), pts1=pts2,color1=color_1[1],color2=color_2[1])
#            frame3 = draw_line(frame=copy.deepcopy(frame), pts1=pts1,color1=color_1[2],color2=color_2[2])
#            save_path='./Image/{}_{}_{}.jpg'.format(i,_n,tmp)
#            frame_save = frame1*0.5+frame2*0.5+frame3*0.5-frame*0.5

            frame_tmp = copy.deepcopy(frame)
            frame = draw_line(type = 0, frame=frame, pts1 = pts_std)
            if tmp >= 10 :
                frame = draw_line(type = 1,frame=frame, pts1=pts3,orientation=ori3,color1=color_1[2],color2=color_2[2],color3=color_3[2],_size=_size)
                frame = draw_line(type = 1,frame=frame, pts1=pts2,orientation=ori2,color1=color_1[1],color2=color_2[1],color3=color_3[1],_size=_size)
                frame = draw_line(type = 1,frame=frame, pts1=pts1,orientation=ori1,color1=color_1[0],color2=color_2[0],color3=color_3[0],_size=_size)
            save_path='./Image/Comparison{}/{}_{}_{}.jpg'.format(_comparison_type,i,_n,tmp)
#            frame_save = frame1*0.5+frame2*0.5+frame3*0.5-frame*0.5
            if _n == _save_file_num[j] :
                frame_save = frame_tmp*0.5+frame*0.5
                cv2.imwrite(save_path,frame_save)
            tmp+=1
        if tmp == 21 :
            tmp = 0
            _n += 1
            save_or_not = 0
