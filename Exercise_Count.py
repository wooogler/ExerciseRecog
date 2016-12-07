import Labeling1
import Exercise_Seg
import Butterworth as bw
import numpy as np
from scipy import signal

modeldir_ex="CNN1_Models/Model_Ex.ckpt"
modeldir_noex="CNN2_Models/Model_NoEx.ckpt"
rawdir="raw_data/raw1.txt"

start_time,end_time_pair = Exercise_Seg.segmentation(modeldir_noex,rawdir)
labeled_data = Labeling1.labeling1(modeldir_ex,rawdir,start_time,end_time_pair)

exercise = []
exercise_data = []
exercise_count = []

for arr in labeled_data:
    exercise.append(np.bincount(arr).argmax())

raw_data=(np.loadtxt(rawdir) + 32768) / 65536
raw_data_x=bw.butterworth(6,40,0.8,raw_data[:,0])

for i in range(len(start_time)):
    exercise_data.append(raw_data_x[start_time[i]-40,end_time_pair[i]+40])

for i in range(len(start_time)):
    exercise_count.append(len(signal.find_peaks_cwt(exercise_data[i], np.arange(5,40), min_length=40)))

print(start_time)
print(end_time_pair)
print(exercise)
print(exercise_count)