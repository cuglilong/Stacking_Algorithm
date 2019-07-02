import Stacker
import Hawaii_Stacker
import sys
from obspy import read
import pickle

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

max_time = 7
min_time = -7
cut_1 = np.where(min_time<times)[0][0]
cut_2 = np.where(max_time<times)[0][0]
seis_data = np.array([seis[cut_1:cut_2] for seis in seis_data])
times = times[cut_1:cut_2]
stacks = seis_data

s1 = Hawaii_Stacker.Hawaii_Stacker(times, coords, seis_data, 'hawaii_default')
s2 = Hawaii_Stacker.Hawaii_Stacker(times, coords, seis_data, 'hawaii_geo')
s1.adaptive_stack()
s1.plot()
s2.adaptive_stack()
s2.plot()

