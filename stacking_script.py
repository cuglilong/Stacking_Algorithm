import hierarchical_stack as hs
from obspy import read
from random import randint
import matplotlib.pyplot as plt
import sys
from math import sin, cos, sqrt, atan2, pi
import numpy as np
import scipy as sp
import scipy.cluster.hierarchy as cluster
import mpl_toolkits
import mpl_toolkits.basemap
from mpl_toolkits.basemap import Basemap
import csv

def print_out(cluster, stacks):
	file = open("stack_data.csv", 'w')
	for c in cluster:
		file.write(str(c) + ', ')
	file.write('\n')
	for stack in stacks:
		for s in stack:
			file.write(str(s) + ', ')
		file.write('\n')

def read_in(no_stacks):
	with open('stack_data.csv') as csv_file:
		stacks = np.zeros((no_stacks,len(hs.seis_data[0])))
		csv_reader = csv.reader(csv_file, delimiter=',')
		line_count = 0
		for row in csv_reader:
			if line_count == 0:
				cluster = row
				cluster = np.array(cluster[:-1]).astype(float)
				line_count += 1
			else:
				stacks[line_count-1] = np.array(row[:-1]).astype(np.float)
				line_count += 1
	return cluster, stacks

print("First stack...")
cluster = np.arange(1,len(hs.seis_data)+1)
cluster, stacks = hs.second_cluster(cluster, hs.coordinates, hs.seis_data, 200)
#print_out(cluster, stacks)
#cluster, stacks = read_in(100)
hs.plot(stacks, hs.depths, cluster, hs.coordinates, "test")
