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
from collections import Counter
import mpl_toolkits.basemap
from mpl_toolkits.basemap import Basemap
import csv

def print_out(cluster, stacks, filename):
	file = open(filename, 'w')
	for c in cluster:
		file.write(str(c) + ', ')
	file.write('\n')
	for stack in stacks:
		for s in stack:
			file.write(str(s) + ', ')
		file.write('\n')

def read_in(no_stacks, filename):
	with open(filename) as csv_file:
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

def remove_anoms(cluster, stacks, coords, size):
	to_remove = np.array([])
	remove_cluster = np.array([])
	for c in np.arange(np.max(cluster)):
		a = np.where(cluster == c+1)[0]
		if (len(a) < size):
			to_remove = np.append(to_remove, [c], axis=0)
			remove_cluster = np.append(remove_cluster, a, axis=0)
	stacks = np.delete(stacks, to_remove.astype(int), axis=0)
	cluster = np.delete(cluster, remove_cluster.astype(int))
	cluster_set = set(cluster)
	count = 1
	for c in cluster_set:
		cluster[np.where(cluster==c)[0]] = count
		count += 1
	coords = np.delete(coords, remove_cluster.astype(int), axis=0)
	return cluster, stacks, coords

print("Stacking...")
cluster, stacks = read_in(1750, 'stack_data.csv')
print(np.max(cluster))
cluster, stacks, coords = remove_anoms(cluster, stacks, hs.coordinates, 10)
print(np.max(cluster))
cluster, stacks = hs.second_cluster(cluster, hs.stack_coords(cluster,coords), stacks, threshold=300, crit='maxclust', dist = True, corr = True)
print(np.max(cluster))
cluster, stacks, coords = remove_anoms(cluster, stacks, coords, 20)
print(np.max(cluster))
cluster, stacks = hs.second_cluster(cluster, hs.stack_coords(cluster,coords), stacks, threshold = 1, crit='inconsistent', dist = False, corr = True)
print(np.max(cluster))
cluster, stacks, coords = remove_anoms(cluster, stacks, coords, 100)
print(np.max(cluster))
hs.plot(stacks, hs.depths, cluster, coords, "dist_and_corr", anomal = False, plot_individual = True)
