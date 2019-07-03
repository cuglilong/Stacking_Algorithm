import Stacker
import Hawaii_Stacker
import sys
from obspy import read
import pickle
from numpy import random
import numpy as np

# Reading in

file1 =sys.argv[1]
print("Reading " + file1 + "...")
seis = read(file1,format='PICKLE')

# Formatting relevant data

seis_data1 = np.array([tr.data for tr in seis])
locs = np.array([t.stats.piercepoints['P410s']['410'] for t in seis]).astype(float)
coords1 = np.array([(l[1], l[2]) for l in locs])
depths1 = np.array(seis[0].stats.depth)

# Reading in

file2 =sys.argv[2]
print("Reading " + file2 + "...")
seis = read(file2,format='PICKLE')

# Formatting relevant data

seis_data2 = np.array([tr.data for tr in seis])
locs = np.array([t.stats.piercepoints['P410s']['410'] for t in seis]).astype(float)
coords2 = np.array([(l[1], l[0]) for l in locs])
depths2 = np.array(seis[0].stats.depth)

# Choosing points to remove

rand_remove = random.choice(range(len(seis_data1)), 1000, replace=False)
seis_data3 = np.delete(seis_data1, rand_remove, axis=0)
coords3 = np.delete(coords1, rand_remove, axis=0)

s1 = Stacker.Stacker(depths1, coords1, seis_data1, 'default')
s2 = Stacker.Stacker(depths1, coords1, seis_data1, 'geographical')
s3 = Stacker.Stacker(depths2, coords2, seis_data2, 'prem')
s4 = Stacker.Stacker(depths1, coords3, seis_data3, 'stability')
s5 = Stacker.Stacker(depths1, coords3, seis_data3, 'stability_geo')

s1.adaptive_stack()
s1.plot()
print(s1.average_cluster_variance())
s2.adaptive_stack(geographical = True)
s2.plot()
print(s2.average_cluster_variance())
#s3.adaptive_stack()
#s3.plot()
#print(s3.average_cluster_variance())
s4.adaptive_stack()
s4.plot()
print(s4.average_cluster_variance())
s5.adaptive_stack()
s5.plot()
print(s5.average_cluster_variance())
