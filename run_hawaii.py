import Stacker
import Hawaii_Stacker
import sys
from obspy import read
import pickle
from itertools import zip_longest

import numpy as np

# Reading in

file =sys.argv[1]
print("Reading...")
with open(file, 'rb') as f:
    seis = pickle.load(f)

# Formatting relevant data

seis_data = np.array(seis['T_decon'])
locs = np.array(seis['ScS_bounce']).astype(float)
coords = np.array([(l[1], l[0]) for l in locs])
times = np.array(seis['Time'])

max_time = 7.5
min_time = -7.5
cut_1 = np.where(min_time<times)[0][0]
cut_2 = np.where(0<times)[0][0]
cut_3 = np.where(max_time<times)[0][0]
seis_data1 = np.array([seis[cut_1:cut_2] for seis in seis_data])
seis_data2 = np.array([seis[cut_2:cut_3] for seis in seis_data])
times = times[cut_2:cut_3]

# Flip-reverse stacking

seis_data1 = -1*np.flip(seis_data1, axis=1)
seis_data = np.array([[j+k for j, k in zip_longest(seis_data1[i], seis_data2[i], fillvalue=0)] for i in range(len(seis_data1))])
stacks = seis_data

s1 = Hawaii_Stacker.Hawaii_Stacker(times, coords, seis_data, 'hawaii_default')
s2 = Hawaii_Stacker.Hawaii_Stacker(times, coords, seis_data, 'hawaii_geo')
s1.adaptive_stack()
s1.plot()
s2.adaptive_stack(geographical=True)
s2.plot()

