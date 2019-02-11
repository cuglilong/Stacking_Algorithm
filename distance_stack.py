# -*- coding: utf-8 -*-

from obspy import read
import matplotlib.pyplot as plt
import sys
import numpy as np
import math

file =sys.argv[1]
print(file)

seis = read(file,format='PICKLE')
seis = seis.sort(keys=['dist'])
data_length = len(seis[0].data)
seis_depth = [tr.depth for tr in seis]
seis_data = [tr.data for tr in seis]

dists = [t.stats.dist for t in seis]
dist_diffs = np.diff(dists)
dist_sum = np.cumsum(dist_diffs)
split_range = np.arange(math.ceil(np.amax(dist_sum)/10.0))
split_indices = [np.argmax(dist_sum>i*10) for i in split_range]
split_indices = split_indices[1:]
stack_datas = np.split(seis_data, split_indices)
stack_depths = np.split(seis_depth, split_indices)

stacked_data = [[np.sum([st[i] for st in stack]) for i in range(data_length)] for stack in stack_datas]
stacked_depth = [[np.average([st[i] for st in stack]) for i in range(data_length)] for stack in stack_depths]

plt.xlabel('depth (km)')
plt.ylabel('amplitude relative to main P wave')
for i in range(len(stacked_data)):
	plt.plot(stacked_depth[i], stacked_data[i])

plt.show()
plt.savefig("epicentral_distance_stack")
