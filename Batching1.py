import numpy as np
import random as rd

def batching1(SENSORS, WINDOW, CLASSES,dir,data_set,ex):
    x_batch = []
    y_batch = []
    iden = np.identity(CLASSES)
    for data in data_set:
        for i in range(CLASSES):
            x_raw = (np.loadtxt(dir+data+"/"+ex[i])+32768)/65536
            startindex=rd.randrange(40,len(x_raw)-WINDOW-100)
            x_part = x_raw[startindex:startindex+WINDOW]
            y_part = iden[i]
            x_batch.append(np.reshape(x_part, (SENSORS, WINDOW, 1)))
            y_batch.append(np.reshape(y_part, (CLASSES)))

    return x_batch, y_batch


