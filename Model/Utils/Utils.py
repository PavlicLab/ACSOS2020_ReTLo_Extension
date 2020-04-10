import numpy as np
import pandas as pd
import time
import os
import pathlib
import copy

import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras import backend as K
from tensorflow.keras import losses, optimizers
from tensorflow.keras.models import Sequential, load_model
import tensorflow.keras.layers as layers
from tensorflow.keras.layers import Input, Activation, Dense, TimeDistributed,Conv2D,MaxPooling2D, Lambda,UpSampling2D
from tensorflow.keras.layers import add, dot, concatenate, LSTM, Bidirectional,Reshape,Flatten
import tensorflow.keras as keras
from tensorflow.compat.v1.keras.layers import CuDNNLSTM, CuDNNGRU
from Utils.PlotLosses import PlotLosses

def dist_euclidean(data = '', answer='',robot='',step='',std = 500) :
    _x = '{}_{}_x'.format(step,robot)
    _y = '{}_{}_y'.format(step,robot)
    ans_x = answer[_x]
    data_x = data[_x]
    dist_x = (ans_x - data_x) 

    ans_y = answer[_y]
    data_y = data[_y]
    dist_y = (ans_y - data_y) 

    dist = np.sqrt(np.square(dist_x)+np.square(dist_y))*std
    average = np.average(dist)
    std = np.std(dist)
    return dist, average, std

def cal_accuracy_tmp(data='',model_n = 1,data_test='',std_coordinates=500, _n_samples=12) :
    add_robots_list = ['H','F1','F2','F3','F4','F5','T']
    test_data = s_relative_and_std(data = data_test, std_coordinates = std_coordinates,n_time_in_instance=_n_samples,std_robot ='T',other_robots=['H','F1','F3','F2','F4','F5'])
    _result = copy.deepcopy(data)
    
    _robots = ['F1','F2','F3','F4','F5']
    for i in range(0,model_n-1) :
        _robots.remove(_robots[0])

    for i in range(0,len(add_robots_list)-2) :
        if i >= model_n :
            _robots.remove(_robots[0])
        _std_robot = add_robots_list[i]
#        print(_robots)
#        print(_std_robot)
        _result = re_coordinates(data=_result,std_robot=_std_robot,other_robots=_robots,n_samples=26)

    _error_list = list()
    robots_list = ['F4','F3','F2','F1','H']
    for i in range(0,model_n-1) :
        robots_list.remove(robots_list[0])
       
    for i,pred_robot in enumerate(robots_list) : 
        _error = list()
        for step in range(10,26-i) :
            _x = '{}_{}_x'.format(step,pred_robot)
            _y = '{}_{}_x'.format(step,pred_robot)
            _error.append(dist_coordinates(std=std_coordinates,data1 = [_result[_x],_result[_y]],data2=[test_data[_x],test_data[_y]]))
        _error_list.append(_error)

    return _error_list,_result

def get_reculsive_error_m123(_Model='',_n_model = 1,start=0,std_robot='T',_obs_list=[],pred_list=[],robots=['H','F1','F2','F3','F4','F5','T'],data_test = '',_n_samples=12,_n_hist=10,std_coordinates=500) :
    other_robots = copy.deepcopy(robots)
 #   print(other_robots)
#    print(std_robot)
    other_robots.remove(std_robot)
    pred_result = pd.DataFrame()
    error_model1=list()
        
    test_data = s_relative_and_std(data = data_test, std_coordinates = std_coordinates,n_time_in_instance=_n_samples,std_robot =std_robot,other_robots=other_robots)
    _answer = s_relative_and_std(data = data_test, std_coordinates = std_coordinates,n_time_in_instance=_n_samples,std_robot =std_robot,other_robots=other_robots)

    for i in range(0,len(_obs_list)) :
        _errors = list()
        pred_result = pd.DataFrame()

        for step in range(0,_n_samples-_n_hist-1-i) :
#        for step in range(0,_n_samples-_n_hist-1-i-2) :
            ## Set input
            x_test = set_input(_obs_list = _obs_list[i], start=start,pred_list= pred_list[i],step = step,  _hist_len = _n_hist,_n_samples = _n_samples, data1 = test_data,data2 = test_data)    
#            x_test = set_input(_obs_list = _obs_list[i], pred_list= pred_list[i],timedifference = [2,2,0,1,1,0,0],step = step,  _hist_len = _n_hist,_n_samples = _n_samples, data1 = test_data,data2 = test_data)    
            ## Predict
            _result = _Model.predict(x_test)

            ## Insert pred_result to next input
            if _n_model == 1 :
                _label_list = [['{}_{}_x'.format(step+_n_hist,robots[4-i]),'{}_{}_y'.format(step+_n_hist,robots[4-i])],
                        ['{}_{}_o'.format(step+_n_hist,robots[5-i]),'{}_{}_o'.format(step+_n_hist,robots[4-i])]]
            elif _n_model == 2 :
                _label_list = [['{}_{}_x'.format(step+_n_hist,robots[3-i]),'{}_{}_y'.format(step+_n_hist,robots[3-i])],
                               ['{}_{}_o'.format(step+_n_hist,robots[3-i]),'{}_{}_o'.format(step+_n_hist,robots[4-i]),'{}_{}_o'.format(step+_n_hist,robots[5-i])]]
            else :
                _label_list = [['{}_{}_x'.format(step+_n_hist,robots[2-i]),'{}_{}_y'.format(step+_n_hist,robots[2-i])],
                               ['{}_{}_o'.format(step+_n_hist,robots[2-i]),'{}_{}_o'.format(step+_n_hist,robots[3-i]),'{}_{}_o'.format(step+_n_hist,robots[4-i]),'{}_{}_o'.format(step+_n_hist,robots[5-i])]]              
#                               ['{}_{}_o'.format(step+_n_hist,robots[2-i]),'{}_{}_o'.format(step+_n_hist,robots[3-i])]]              
            print('Label : {}'.format(_label_list))

#            print(_label_list)
#            print(pred_result.columns())
            for j in range(len(_label_list)) :
                for k in range(len(_label_list[j])) :
                    pred_result[_label_list[j][k]] = _result[j][:,k]
            
            _error = dist_coordinates(std=std_coordinates,
                                      data1=[pred_result['{}_{}_x'.format(_n_hist+step,robots[5-_n_model-i])],pred_result['{}_{}_y'.format(_n_hist+step,robots[5-_n_model-i])]], 
                                      data2 = [_answer['{}_{}_x'.format(_n_hist+step,robots[5-_n_model-i])],_answer['{}_{}_y'.format(_n_hist+step,robots[5-_n_model-i])]])
            _errors.append(_error)
            
            ## concatenates pred_result and original data
            for columns in pred_result.columns :
                test_data[columns] = pred_result[columns]

        ## Change std_robot and change relative coordinates
        _std_robot = robots[5-i]
        other_robots.remove(_std_robot)
#        print(_std_robot)
#        print(other_robots)
        test_data = change_relative_coordinates(std_robot=_std_robot,other_robots=other_robots,data = test_data,n_samples=_n_samples)
        _answer = change_relative_coordinates(std_robot=_std_robot,other_robots=other_robots,data = _answer,n_samples=_n_samples)
       
        error_model1.append(_errors)
    return error_model1,test_data

def get_reculsive_error_m3_only_nearest(_Model='',_n_model = 1,std_robot='T',_obs_list=[],pred_list=[],robots=['H','F1','F2','F3','F4','F5','T'],data_test = '',_n_samples=12,_n_hist=10,std_coordinates=500) :
    other_robots = copy.deepcopy(robots)
 #   print(other_robots)
#    print(std_robot)
    other_robots.remove(std_robot)
    pred_result = pd.DataFrame()
    error_model1=list()
        
    test_data = s_relative_and_std(data = data_test, std_coordinates = std_coordinates,n_time_in_instance=_n_samples,std_robot =std_robot,other_robots=other_robots)
    _answer = s_relative_and_std(data = data_test, std_coordinates = std_coordinates,n_time_in_instance=_n_samples,std_robot =std_robot,other_robots=other_robots)

    for i in range(0,len(_obs_list)) :
        _errors = list()
        pred_result = pd.DataFrame()

        for step in range(0,_n_samples-_n_hist-1-i) :
#        for step in range(0,_n_samples-_n_hist-1-i-2) :
            ## Set input
            x_test = set_input(_obs_list = _obs_list[i], pred_list= pred_list[i],step = step,  _hist_len = _n_hist,_n_samples = _n_samples, data1 = test_data,data2 = test_data)    
#            x_test = set_input(_obs_list = _obs_list[i], pred_list= pred_list[i],timedifference = [2,2,0,1,1,0,0],step = step,  _hist_len = _n_hist,_n_samples = _n_samples, data1 = test_data,data2 = test_data)    
            ## Predict
            _result = _Model.predict(x_test)

            ## Insert pred_result to next input
            if _n_model == 1 :
                _label_list = [['{}_{}_x'.format(step+_n_hist,robots[4-i]),'{}_{}_y'.format(step+_n_hist,robots[4-i])],
                        ['{}_{}_o'.format(step+_n_hist,robots[5-i]),'{}_{}_o'.format(step+_n_hist,robots[4-i])]]
            elif _n_model == 2 :
                _label_list = [['{}_{}_x'.format(step+_n_hist,robots[3-i]),'{}_{}_y'.format(step+_n_hist,robots[3-i])],
                               ['{}_{}_o'.format(step+_n_hist,robots[3-i]),'{}_{}_o'.format(step+_n_hist,robots[4-i]),'{}_{}_o'.format(step+_n_hist,robots[5-i])]]
            else :
                _label_list = [['{}_{}_x'.format(step+_n_hist,robots[2-i]),'{}_{}_y'.format(step+_n_hist,robots[2-i])],
#                               ['{}_{}_o'.format(step+_n_hist,robots[2-i]),'{}_{}_o'.format(step+_n_hist,robots[3-i]),'{}_{}_o'.format(step+_n_hist,robots[4-i]),'{}_{}_o'.format(step+_n_hist,robots[5-i])]]              
                               ['{}_{}_o'.format(step+_n_hist,robots[2-i]),'{}_{}_o'.format(step+_n_hist,robots[3-i])]]              
            print('Label : {}'.format(_label_list))

#            print(_label_list)
#            print(pred_result.columns())
            for j in range(len(_label_list)) :
                for k in range(len(_label_list[j])) :
                    pred_result[_label_list[j][k]] = _result[j][:,k]
            
            _error = dist_coordinates(std=std_coordinates,
                                      data1=[pred_result['{}_{}_x'.format(_n_hist+step,robots[5-_n_model-i])],pred_result['{}_{}_y'.format(_n_hist+step,robots[5-_n_model-i])]], 
                                      data2 = [_answer['{}_{}_x'.format(_n_hist+step,robots[5-_n_model-i])],_answer['{}_{}_y'.format(_n_hist+step,robots[5-_n_model-i])]])
            _errors.append(_error)
            
            ## concatenates pred_result and original data
            for columns in pred_result.columns :
                test_data[columns] = pred_result[columns]

        ## Change std_robot and change relative coordinates
        _std_robot = robots[5-i]
        other_robots.remove(_std_robot)
#        print(_std_robot)
#        print(other_robots)
        test_data = change_relative_coordinates(std_robot=_std_robot,other_robots=other_robots,data = test_data,n_samples=_n_samples)
        _answer = change_relative_coordinates(std_robot=_std_robot,other_robots=other_robots,data = _answer,n_samples=_n_samples)
       
        error_model1.append(_errors)
    return error_model1,test_data

def get_reculsive_error_m3_diff_timestep(_Model='',_n_model = 1,std_robot='T',_obs_list=[],pred_list=[],robots=['H','F1','F2','F3','F4','F5','T'],data_test = '',_n_samples=12,_n_hist=10,std_coordinates=500) :
    other_robots = copy.deepcopy(robots)
 #   print(other_robots)
#    print(std_robot)
    other_robots.remove(std_robot)
    pred_result = pd.DataFrame()
    error_model1=list()
        
    test_data = s_relative_and_std(data = data_test, std_coordinates = std_coordinates,n_time_in_instance=_n_samples,std_robot =std_robot,other_robots=other_robots)
    _answer = s_relative_and_std(data = data_test, std_coordinates = std_coordinates,n_time_in_instance=_n_samples,std_robot =std_robot,other_robots=other_robots)

    for i in range(0,len(_obs_list)) :
        _errors = list()
        pred_result = pd.DataFrame()

#        for step in range(0,_n_samples-_n_hist-1-i) :
        for step in range(0,_n_samples-_n_hist-1-i-2) :
            ## Set input
#            x_test = set_input(_obs_list = _obs_list[i], pred_list= pred_list[i],step = step,  _hist_len = _n_hist,_n_samples = _n_samples, data1 = test_data,data2 = test_data)    
            x_test = set_input(_obs_list = _obs_list[i], pred_list= pred_list[i],timedifference = [2,2,0,1,1,0,0],step = step,  _hist_len = _n_hist,_n_samples = _n_samples, data1 = test_data,data2 = test_data)    
            ## Predict
            _result = _Model.predict(x_test)

            ## Insert pred_result to next input
            if _n_model == 1 :
                _label_list = [['{}_{}_x'.format(step+_n_hist,robots[4-i]),'{}_{}_y'.format(step+_n_hist,robots[4-i])],
                        ['{}_{}_o'.format(step+_n_hist,robots[5-i]),'{}_{}_o'.format(step+_n_hist,robots[4-i])]]
            elif _n_model == 2 :
                _label_list = [['{}_{}_x'.format(step+_n_hist,robots[3-i]),'{}_{}_y'.format(step+_n_hist,robots[3-i])],
                               ['{}_{}_o'.format(step+_n_hist,robots[3-i]),'{}_{}_o'.format(step+_n_hist,robots[4-i]),'{}_{}_o'.format(step+_n_hist,robots[5-i])]]
            else :
                _label_list = [['{}_{}_x'.format(step+_n_hist,robots[2-i]),'{}_{}_y'.format(step+_n_hist,robots[2-i])],
                               ['{}_{}_o'.format(step+_n_hist,robots[2-i]),'{}_{}_o'.format(step+_n_hist,robots[3-i]),'{}_{}_o'.format(step+_n_hist,robots[4-i]),'{}_{}_o'.format(step+_n_hist,robots[5-i])]]              
#                               ['{}_{}_o'.format(step+_n_hist,robots[2-i]),'{}_{}_o'.format(step+_n_hist,robots[3-i])]]              
            print('Label : {}'.format(_label_list))

#            print(_label_list)
#            print(pred_result.columns())
            for j in range(len(_label_list)) :
                for k in range(len(_label_list[j])) :
                    pred_result[_label_list[j][k]] = _result[j][:,k]
            
            _error = dist_coordinates(std=std_coordinates,
                                      data1=[pred_result['{}_{}_x'.format(_n_hist+step,robots[5-_n_model-i])],pred_result['{}_{}_y'.format(_n_hist+step,robots[5-_n_model-i])]], 
                                      data2 = [_answer['{}_{}_x'.format(_n_hist+step,robots[5-_n_model-i])],_answer['{}_{}_y'.format(_n_hist+step,robots[5-_n_model-i])]])
            _errors.append(_error)
            
            ## concatenates pred_result and original data
            for columns in pred_result.columns :
                test_data[columns] = pred_result[columns]

        ## Change std_robot and change relative coordinates
        _std_robot = robots[5-i]
        other_robots.remove(_std_robot)
#        print(_std_robot)
#        print(other_robots)
        test_data = change_relative_coordinates(std_robot=_std_robot,other_robots=other_robots,data = test_data,n_samples=_n_samples)
        _answer = change_relative_coordinates(std_robot=_std_robot,other_robots=other_robots,data = _answer,n_samples=_n_samples)
       
        error_model1.append(_errors)
    return error_model1,test_data

def cal_accuracy(data='',model_n = 1,data_test='',std_coordinates=500, _n_samples=12) :
    add_robots_list = ['H','F1','F2','F3','T']
    test_data = s_relative_and_std(data = data_test, std_coordinates = std_coordinates,n_time_in_instance=_n_samples,std_robot ='T',other_robots=['H','F1','F3','F2'])
    _result = copy.deepcopy(data)
    
    _robots = ['F1','F2','F3']
    for i in range(0,model_n-1) :
        _robots.remove(_robots[0])

    for i in range(0,len(add_robots_list)-2) :
        if i >= model_n :
            _robots.remove(_robots[0])
        _std_robot = add_robots_list[i]
        _result = re_coordinates(data=_result,std_robot=_std_robot,other_robots=_robots,n_samples=12)

    _error_list = list()
    robots_list = ['F2','F1','H']
    for i in range(0,model_n-1) :
        robots_list.remove(robots_list[0])
       
    for i,pred_robot in enumerate(robots_list) : 
        _error = list()
        for step in range(5,11-i) :
            _x = '{}_{}_x'.format(step,pred_robot)
            _y = '{}_{}_x'.format(step,pred_robot)
            _error.append(dist_coordinates(std=std_coordinates,data1 = [_result[_x],_result[_y]],data2=[test_data[_x],test_data[_y]]))
        _error_list.append(_error)

    return _error_list,_result

def re_coordinates(data='',std_robot='',other_robots=[],n_samples=12) :   
    for name in other_robots : 
        for n in range(0,n_samples) :
            for c in ['x','y'] :                    
                std_coordinates = '{}_{}_{}'.format(n,std_robot,c) 
                other_coordinates = '{}_{}_{}'.format(n,name,c)
                data[std_coordinates] += data[other_coordinates]
    return data           

def get_reculsive_error(_Model='',timedifference=[],timeshift=0,_n_model = 1,start=0,std_robot='T',_obs_list=[],pred_list=[],robots=['H','F1','F2','F3','F4','F5','T'],data_test = '',_n_samples=12,_n_hist=10,std_coordinates=500) :
    other_robots = copy.deepcopy(robots)
 #   print(other_robots)
#    print(std_robot)
    other_robots.remove(std_robot)
    pred_result = pd.DataFrame()
    error_model1=list()
        
    test_data = s_relative_and_std(data = data_test, std_coordinates = std_coordinates,n_time_in_instance=_n_samples,std_robot =std_robot,other_robots=other_robots)
    _answer = s_relative_and_std(data = data_test, std_coordinates = std_coordinates,n_time_in_instance=_n_samples,std_robot =std_robot,other_robots=other_robots)

    for i in range(0,len(_obs_list)) :
        _errors = list()
        pred_result = pd.DataFrame()

        for step in range(0,_n_samples-_n_hist-1-i-timeshift) :
            
            ##Set input
            if timeshift == 0 :
                x_test = set_input(_obs_list = _obs_list[i],
                                   start=start,pred_list= pred_list[i],
                                   step = step, _hist_len = _n_hist,
                                   _n_samples = _n_samples, 
                                   data1 = test_data,data2 = test_data)    
            else :
                x_test = set_input(_obs_list = _obs_list[i], 
                                   start=start, pred_list= pred_list[i],
                                   timedifference = timedifference,
                                   step = step,  _hist_len = _n_hist,
                                   _n_samples = _n_samples, data1 = test_data,data2 = test_data)    
            
            ## Predict
            _result = _Model.predict(x_test)

            ## Insert pred_result to next input
            if _n_model == 1 :
                _label_list = [['{}_{}_x'.format(step+_n_hist,robots[4-i]),'{}_{}_y'.format(step+_n_hist,robots[4-i])],
                        ['{}_{}_o'.format(step+_n_hist,robots[5-i]),'{}_{}_o'.format(step+_n_hist,robots[4-i]),'{}_{}_o'.format(step+_n_hist+1,robots[4-i])]]
            elif _n_model == 2 :
                _label_list = [['{}_{}_x'.format(step+_n_hist,robots[3-i]),'{}_{}_y'.format(step+_n_hist,robots[3-i])],
                               ['{}_{}_o'.format(step+_n_hist,robots[3-i]),'{}_{}_o'.format(step+_n_hist,robots[4-i]),'{}_{}_o'.format(step+_n_hist,robots[5-i]),'{}_{}_o'.format(step+_n_hist+1,robots[5-i])]]
            else :
                _label_list = [['{}_{}_x'.format(step+_n_hist,robots[2-i]),'{}_{}_y'.format(step+_n_hist,robots[2-i])],
                               ['{}_{}_o'.format(step+_n_hist,robots[2-i]),'{}_{}_o'.format(step+_n_hist,robots[3-i]),'{}_{}_o'.format(step+_n_hist,robots[4-i]),'{}_{}_o'.format(step+_n_hist,robots[5-i]),'{}_{}_o'.format(step+_n_hist+1,robots[5-i])]]              
#                               ['{}_{}_o'.format(step+_n_hist,robots[2-i]),'{}_{}_o'.format(step+_n_hist,robots[3-i])]]              
            print('Label : {}'.format(_label_list))

#            print(_label_list)
#            print(pred_result.columns())
            for j in range(len(_label_list)) :
                for k in range(len(_label_list[j])) :
                    pred_result[_label_list[j][k]] = _result[j][:,k]
                   
            ## concatenates pred_result and original data
            for columns in pred_result.columns :
                test_data[columns] = pred_result[columns]

        ## Change std_robot and change relative coordinates
        _std_robot = robots[5-i]
        other_robots.remove(_std_robot)
#        print(_std_robot)
#        print(other_robots)
        test_data = change_relative_coordinates(std_robot=_std_robot,other_robots=other_robots,data = test_data,n_samples=_n_samples)
        _answer = change_relative_coordinates(std_robot=_std_robot,other_robots=other_robots,data = _answer,n_samples=_n_samples)
       
    return test_data


def change_relative_coordinates(direction = 1,std_robot=[],other_robots=[],data = '',n_samples=12) :
    for name in other_robots : 
        for n in range(0,n_samples) :
            for c in ['x','y'] :                    
                std_coordinates = '{}_{}_{}'.format(n,std_robot,c) 
                other_coordinates = '{}_{}_{}'.format(n,name,c)
                data[other_coordinates] = (-data[std_coordinates]+data[other_coordinates])*direction
    return data           

def s_labels(_label_list=[],data='') :
    y_label = list()
    for i in range(0,len(_label_list)) :
        y_data = s_DataFrame(index = _label_list[i],data = data)
        y_label.append(y_data)
    return y_label


def set_input(_obs_list=[],step=0,start =0,data1='',data2='',pred_list=[],_hist_len=0,_n_samples=0,timedifference=[0,0,0,0,0,0,0]) :
    ## step : 초기 여부 0이면 초기 / 1~ : 1step 후의 값

    _data_hist = pd.DataFrame()
    for i in range(step+start,_hist_len+step) :
        _index_hist = list() 
        _index_tmp = list()
        _index_tmp_2 = list()
        for name in _obs_list :
            _index_hist.append('{}{}'.format(i,name))
        for name in pred_list :
            if i < _hist_len :
                _index_tmp.append('{}{}'.format(i,name))
            else :
                _index_tmp_2.append('{}{}'.format(i,name))                   
        _data_hist_1 = s_DataFrame(index=_index_hist,data = data1)
        _data_hist_2 = s_DataFrame(index=_index_tmp,data=data1)
        _data_hist_3 = s_DataFrame(index=_index_tmp_2,data=data2)
        _data_hist_tmp = pd.concat([_data_hist_1,_data_hist_2,_data_hist_3],axis = 1)
        _data_hist = pd.concat([_data_hist,_data_hist_tmp],axis = 1)
        print('History : {}'.format([_index_hist,_index_tmp,_index_tmp_2]))
    
    _data_hist = np.array(_data_hist).reshape((-1,_hist_len-start,len(pred_list)+len(_obs_list)))

    _index_obs = list()
    for i in range(_hist_len+step, _hist_len+step+2) :
        for j,name in enumerate(_obs_list) :
            _index_obs.append('{}{}'.format(i+timedifference[j],name))
    _data_obs = np.array(s_DataFrame(index=_index_obs,data=data1))
    print('Observed : {}'.format(_index_obs))
    
    return [_data_hist,_data_obs]       

def euclidean_distance_loss(y_true, y_pred):
    return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))        

def s_relative_and_std(data='',std_coordinates = 500,std_orientation = 180, std_robot ='T',other_robots=['H','F1','F3','F2'],n_time_in_instance = 2) :
    new_data = pd.DataFrame()
    std = std_coordinates
    for name in other_robots : 
        for n in range(0,n_time_in_instance) :
            for c in ['x','y'] :                    
                std_coordinates = '{}_{}_{}'.format(n,std_robot,c) 
                other_coordinates = '{}_{}_{}'.format(n,name,c)
#                new = (data[std_coordinates] - data[other_coordinates])/std_coordinates
                new = (-data[std_coordinates] + data[other_coordinates])
                new_data[other_coordinates] = new / std
#                new_data.append(data[std_coordinates]-other_coordinates)
            orientation = '{}_{}_o'.format(n,name)
            new_data[orientation] = data[orientation] / std_orientation
    for n in range(n_time_in_instance) :
        orientation = '{}_{}_o'.format(n,std_robot)
        new_data[orientation] = data[orientation]/std_orientation
    return new_data 

def s_DataFrame(data,index) :    
    x_train = pd.DataFrame()
    for name in index :
        x_train[name] = data[name]
    return x_train    

def TrainModel(_model='', save_path='', lr=0.01, epochs=100, opt_name='Adam', train_x=[], train_y=[],val_x=[],val_y=[],losses=['categorical_crossentropy','categorical_crossentropy','mean_squared_error']
) :

    #save_path ='./Saved_Parameters/{}'.format('Shape_model1.h')
    if opt_name == 'Adam' :   
        opt = keras.optimizers.Adam(lr=lr)
    elif opt_name == 'SGD' :
        opt = keras.optimizers.SGD(lr=lr, clipnorm=1.)
    elif opt_name == 'RMSprop' :
        opt = keras.optimizers.RMSprop(lr=lr, rho=0.9, epsilon=None, decay=0.0)
    elif opt_name == 'Adagrad' :
        opt = keras.optimizers.Adagrad(lr=lr, epsilon=None, decay=0.0)
    elif opt_name == 'Adadelta' :
        opt = keras.optimizers.Adadelta(lr=lr, rho=0.95, epsilon=None, decay=0.0)
        
    plot_losses = PlotLosses(targets=['loss'])
    mcp = keras.callbacks.ModelCheckpoint(save_path, monitor='val_loss', 
                                          verbose=1, save_best_only=True,
                                         save_weights_only=False, mode='auto', period=1)

    #losses=['binary_crossentropy','binary_crossentropy','mean_squared_error']
#    losses=['mean_squared_error','mean_squared_error','mean_squared_error']
#    losses=['mean_squared_error','mean_squared_error','mean_squared_error']
#    losses=['categorical_crossentropy','categorical_crossentropy']
    _model.compile(optimizer=opt,loss=losses)
    _history = _model.fit(train_x,train_y,epochs = epochs,
                        validation_data = [val_x,val_y],
                        callbacks=[plot_losses,mcp])
    return _model, _history


def dist_coordinates(std=1,data1=[],data2=[]) :
    x = data1[0]-data2[0]
    y = data1[1]-data2[1]
    dist = np.sqrt(np.square(x)+np.square(y))*std
    return np.average(dist)