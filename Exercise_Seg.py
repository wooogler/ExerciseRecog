import Labeling2
import numpy as np

def segmentation(modeldir_noex,rawdir):

    start_time_=[]
    start_time=[]
    seg_val_st = 100

    end_time_ = []
    end_time = []
    end_time_pair=[]
    seg_val_et = 100
    seg_time = 2000

    y_sliding=Labeling2.labeling2(modeldir_noex,rawdir)
    y_sliding=np.append(y_sliding,np.zeros(seg_val_et))

    for i in range(1,len(y_sliding)):
        if y_sliding[i]==1 and y_sliding[i-1]==0:
            start_time_.append(i)

    for i in start_time_:
        if np.mean(y_sliding[i:i+seg_val_st]) == 1:
            start_time.append(i)

    for i in range(1,len(y_sliding)):
        if y_sliding[i]==0 and y_sliding[i-1]==1:
            end_time_.append(i)

    for i in end_time_:
        if np.mean(y_sliding[i:i+seg_val_et]) == 0:
            end_time.append(i)

    for i in start_time:
        for j in range(seg_time):
            if i+j in end_time:
                if(len(end_time_pair)<=len(start_time)):
                    end_time_pair.append(i+j)

    return start_time, end_time_pair




