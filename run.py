import Stacker
import Hawaii_Stacker
import sys
from obspy import read
import pickle
from numpy import random
import numpy as np

# Reading in

file =sys.argv[1]
print("Reading " + file + "...")
seis = read(file,format='PICKLE')

# Formatting relevant data

seis_data1 = np.array([tr.dataSL2014 for tr in seis])
print(len(seis_data1))
print(seis_data1.shape)
locs = np.array([t.stats.piercepoints['P410s']['410'] for t in seis]).astype(float)
coords = np.array([(l[1], l[2]) for l in locs])
depths = np.array(seis[0].stats.depth)

seis_data2 = np.array([tr.dataPREM for tr in seis])

s1 = Stacker.Stacker(depths, coords, seis_data1, 'default')
s2 = Stacker.Stacker(depths, coords, seis_data1, 'geographical')
s3 = Stacker.Stacker(depths, coords, seis_data2, 'prem')

s1.adaptive_stack()
s1.plot()
print(s1.average_cluster_variance())
s2.adaptive_stack(geographical = True)
s2.plot()
print(s2.average_cluster_variance())
s3.adaptive_stack()
s3.plot()
print(s3.average_cluster_variance())
s1.compare_cluster_similarity(s1.coords, s1.cluster, s2.coords, s2.cluster)
s1.compare_cluster_similarity(s1.coords, s1.cluster, s4.coords, s4.cluster)
s1.compare_cluster_similarity(s2.coords, s2.cluster, s4.coords, s4.cluster)
