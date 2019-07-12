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


d1 = 200
d2 = 500
d3 = 780
cut_1 = np.where(d1<depths)[0][0]
cut_2 = np.where(d2<depths)[0][0]
cut_3 = np.where(d3<depths)[0][0]
seis_data1 = np.array([seis[cut_1:cut_2] for seis in seis_data])
seis_data2 = np.array([seis[cut_2:cut_3] for seis in seis_data])
depths1 = depths[cut_1:cut_2]
depths2 = depths[cut_2:cut_3]
shallow = Stacker.Stacker(depths1, coords, seis_data1, 'shallow')
deep = Stacker.Stacker(depths2, coords, seis_data2, 'deep')

shallow.adaptive_stack()
shallow.plot()

deep.adaptive_stack()
deep.plot()
