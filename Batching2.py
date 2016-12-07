import numpy as np
import random as rd
import Butterworth as bw

def batching2(SENSORS, WINDOW, CLASSES,dir,no_ex,ex):
    x_batch = []
    y_batch = []
    iden = np.identity(CLASSES)
    for i in range(CLASSES):
        if(i==0):
            x_raw = (np.loadtxt(dir+no_ex[rd.randrange(0,len(no_ex))])+32768)/65536
        else:
            x_raw = (np.loadtxt(dir+ex[rd.randrange(0,len(ex))])+32768)/65536
        x_raw_ax=x_raw[:,0]
        x_pre=bw.butterworth(6,40,0.8,x_raw_ax)
        startindex=rd.randrange(40,len(x_raw)-WINDOW-100)
        x_part = x_pre[startindex:startindex+WINDOW]
        y_part = iden[i]
        x_batch.append(np.reshape(x_part, (SENSORS, WINDOW, 1)))
        y_batch.append(np.reshape(y_part, (CLASSES)))

    return x_batch, y_batch


