import Stacker
import Hawaii_Stacker
import sys
from obspy import read
import pickle

import numpy as np

# Reading in

file1 =sys.argv[1]
print("Reading...")
seis = read(file1,format='PICKLE')

# Formatting relevant data

seis_data1 = np.array([tr.data for tr in seis])
locs = np.array([t.stats.piercepoints['P410s']['410'] for t in seis]).astype(float)
coords1 = np.array([(l[1], l[2]) for l in locs])
depths = np.array(seis[0].stats.depth)

# Reading in

file2 =sys.argv[2]
print("Reading...")
with open(file2, 'rb') as f:
    seis = pickle.load(f)

# Formatting relevant data

seis_data2 = np.array(seis['T_decon'])
locs = np.array(seis['ScS_bounce']).astype(float)
coords2 = np.array([(l[1], l[0]) for l in locs])
times = np.array(seis['Time'])

max_time = 7.5
min_time = -7.5
cut_1 = np.where(min_time<times)[0][0]
cut_2 = np.where(max_time<times)[0][0]
seis_data2 = seis_data2[:][cut_1:cut_2]
times = times[cut_1:cut_2]
coords2 = coords2[:][cut_1:cut_2]

stacks1 = seis_data1
stacks2 = seis_data2
s1 = Stacker.Stacker(depths, coords1, seis_data1, 'test1')
s2 = Hawaii_Stacker.Hawaii_Stacker(times, coords2, seis_data2, 'test2')
s1.adaptive_stack()
s1.plot()
print(s1.average_cluster_variance())
s2.adaptive_stack()
s2.plot()
print(s2.average_cluster_variance())
