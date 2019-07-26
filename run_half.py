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

seis_data = np.array([tr.dataSL2014 for tr in seis])
locs = np.array([t.stats.piercepoints['P410s']['410'] for t in seis]).astype(float)
coords = np.array([(l[1], l[2]) for l in locs])
depths = np.array(seis[0].stats.depth)

# Choosing random half

rand_half = random.choice(range(len(seis_data)), round(len(seis_data)/2), replace=False)
seis_data1 = seis_data[rand_half]
seis_data2 = np.delete(seis_data, rand_half, axis=0)
coords1 = coords[rand_half]
coords2 = np.delete(coords, rand_half, axis=0)

s1 = Stacker.Stacker(depths, coords1, seis_data1, 'half1')
s2 = Stacker.Stacker(depths, coords2, seis_data2, 'half2')

#s1.adaptive_stack()
#s1.plot()
s2.adaptive_stack()
s2.plot()
