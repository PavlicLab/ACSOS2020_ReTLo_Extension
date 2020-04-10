import numpy as np
import pandas as pd
import time
import os
import pathlib


def data_trim(csv_path='', save_file_name ='',frames_per_second = 13, choose_per_second = 2, cut_forward = 2, cut_back = 5,blacklist=['Time','Frame','Unnamed: 0'],Record_Frame = {}) :
    
    pathlib.Path(os.path.dirname(save_file_name)).mkdir(parents=True, exist_ok=True) 

    df = pd.read_csv(csv_path)

    cols = [c for c in df.columns if c not in blacklist]
    df.drop(blacklist,axis=1,inplace=True)

    devide_tmp = int(frames_per_second/choose_per_second)
    start_frame = cut_forward*frames_per_second
    end_frame = len(df) - cut_back*frames_per_second
    total_seconds = int((end_frame-start_frame-1)/frames_per_second)

    for i in range(0, total_seconds) :
        for j in range(0,choose_per_second) :
            for col in cols :
                Record_Frame[col].append(df[col][start_frame+i*frames_per_second+j*devide_tmp])

    save = pd.DataFrame(Record_Frame)
    save.to_csv(save_file_name)
    

def data_merge(csv_list =[''], save_path ='', verbose = 0):
    
    pathlib.Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True) 
    
    n_instances = 0
    df = pd.read_csv(csv_list[0])
    cols = [c for c in df.columns]
    save_data = []
    
    for i in range(0, len(cols)) :
        save_data.append(cols[i])

    for i in range(0,len(csv_list)) :
        df = pd.read_csv(csv_list[i])
        n_instances += len(df)
        
        for j in range(0, len(df)) :
            for k in range(0, len(cols)) :
                save_data.append(df[cols[k]][j])

    save_data = (np.array(save_data)).reshape((-1,len(cols)))
    _df = pd.DataFrame(save_data)
    _df.to_csv(save_path, index=False, header = False)
    
    if verbose == 1 :
        print('Total the number of instances is {}'.format(n_instances))


def convert_data_to_instances(csv_path = '', save_path='', robots=['H','F1','F2'],sample_len = 40, skip_len = 20) :
    
    pathlib.Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True) 
    
    df = pd.read_csv(csv_path)
    _detail = ['x','y','o']

    ## set data labels -- ex) 0_H_x, 0_H_y ..... 39_T2_x
    save_data = []
    for i in range(0, sample_len) :
        for j in range(0,len(robots)) :
            for k in range(0,len(_detail)) :
                _title = str(i)+'_'+str(robots[j])+'_'+str(_detail[k])
                save_data.append(_title)

    # make instances
    n_instance = int(len(df) / (sample_len+skip_len))
    print(n_instance)

    for i in range(0,n_instance) :
        for j in range(0,sample_len) :
            for k in range(0,len(robots)) :
                for m in range(0, len(_detail)) :
                    _name = str(robots[k])+'_'+str(_detail[m])
                    _data = df[_name][i*(sample_len+skip_len)+j]
                    save_data.append(_data)

    save_data = (np.array(save_data)).reshape((-1,sample_len*len(_detail)*len(robots)))
    _df = pd.DataFrame(save_data)
    _df.to_csv(save_path, index=False, header = False)
    
    
def data_devide(csv_path='',save_path='',devide_rate=[0.7,0.1]) :
    
    assert np.sum(np.array(devide_rate)) < 1, 'devide_rate must be less than 1'
    
    pathlib.Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True) 
    
    df = pd.read_csv(csv_path)
#     _tmp = [0]*len(df)
    _train = []
    _val = []
    _test = []
    
    df = df.sample(frac=1) # shuffle 
    n = len(df) 
    n_train, n_val = int(n*devide_rate[0]), int(n*devide_rate[1])
    
    data_train = df.iloc[:n_train]  
    data_val = df.iloc[n_train:n_train+n_val]
    data_test = df.iloc[n_train+n_val:]

#     np.random.seed(int(time.time()))

#     _n = 0
#     while(1) :
#         n_tmp = np.random.randint(0,len(df))
#         if _tmp[n_tmp] == 0 :
#             _n = _n +1
#             _tmp[n_tmp] = 1
#             _test.append(n_tmp)
#         if _n >= len(df)*devide_rate[2] :
#             break

#     _n = 0
#     while(1) :
#         n_tmp = np.random.randint(0,len(df))
#         if _tmp[n_tmp] == 0 :
#             _n = _n +1
#             _tmp[n_tmp] = 1
#             _val.append(n_tmp)
#         if _n >= len(df)*devide_rate[1] :
#             break

#     for i in range(0,len(df)) :
#         if _tmp[i] == 0 :
#             _train.append(i)

#     data_train = df.iloc[_train]
#     data_val = df.iloc[_val]
#     data_test = df.iloc[_test]

    data_train.to_csv(save_path+'/Train.csv',index=False)
    data_val.to_csv(save_path+'/Val.csv',index=False)
    data_test.to_csv(save_path+'/Test.csv',index=False)
