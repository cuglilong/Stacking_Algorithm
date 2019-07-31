import Stacker
import Hawaii_Stacker
import Stacker_Test
import sys
from obspy import read
import pickle
from numpy import random
import numpy as np

sys.path.append('./CCP_stacks/Plotting_Scripts')

#import plot_CCP

# Reading in

file =sys.argv[1]
print("Reading " + file + "...")
seis = read(file,format='PICKLE')
# Formatting relevant data

seis_data1 = np.array([tr.dataSL2014 for tr in seis])
locs = np.array([t.stats.piercepoints['P410s']['410'] for t in seis]).astype(float)
coords = np.array([(l[1], l[2]) for l in locs])
depths = np.array(seis[0].stats.depth)

seis_data2 = np.array([tr.dataPREM for tr in seis])

s1 = Stacker_Test.Stacker_Test(depths, coords, seis_data1, 'test')
#s2 = Stacker.Stacker(depths, coords, seis_data1, 'geographical1')
#s3 = Stacker.Stacker(depths, coords, seis_data2, 'prem1')


s1.adaptive_stack()
s1.plot()
print(s1.average_cluster_variance())
#s2.adaptive_stack(geographical = True)
#s2.plot()
#print(s2.average_cluster_variance())
#s3.adaptive_stack()
#s3.plot()
#print(s3.average_cluster_variance())

