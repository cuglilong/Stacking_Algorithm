import Stacker
import Hawaii_Stacker
import sys
from obspy import read
import pickle
from numpy import random
import numpy as np
import clustering_scripts as cs
import plotting_scripts as ps

# Reading in

file =sys.argv[1]
print("Reading " + file + "...")
seis = read(file,format='PICKLE')

# Formatting relevant data

seis_data = np.array([tr.dataSL2014 for tr in seis])
locs = np.array([t.stats.piercepoints['P410s']['410'] for t in seis]).astype(float)
coords = np.array([(l[1], l[2]) for l in locs])
depths = np.array(seis[0].stats.depth)

#s1 = Stacker.Stacker(depths, coords, seis_data, 'default_stable2')
s2 = Stacker.Stacker(depths, coords, seis_data, 'default_stable_vote2')

cs.stability_test(s2, 19)
