import Stacker
import sys
from obspy import read
import pickle

import numpy as np

# Reading in

file1 =sys.argv[1]
filename =sys.argv[2]
print("Reading " + file + "...")
seis = read(file,format='PICKLE')

# Formatting relevant data

seis_data1 = np.array([tr.data for tr in seis])
locs = np.array([t.stats.piercepoints['P410s']['410'] for t in seis]).astype(float)
coords1 = np.array([(l[1], l[2]) for l in locs])
depths = np.array(seis[0].stats.depth)

file2 =sys.argv[1]
print("Reading " + input_file + "...")
with open(file2, 'rb') as f:
    seis = pickle.load(f)

# Formatting relevant data

seis_data2 = np.array(seis['T_decon'])
locs = np.array(seis['ScS_bounce']).astype(float)
coords2 = np.array([(l[1], l[0]) for l in locs])
times = np.array(seis['Time'])

stacks1 = seis_data1
stacks2 = seis_data2
cluster = range(1, len(seis_data)+1)

s1 = Stacker.Stacker(depths, coords1, seis_data1, 'test1')
s2 = Stacker.Stacker(times, coords2, seis_data2, 'test2')
s1.adaptive_stack()
s2.adaptive_stack()
s1.plot()
s2.plot()
