import plotting_scripts as ps
import clustering_scripts as cs
from obspy import read
import random
import matplotlib.pyplot as plt
import sys
import os

sys.path.append('./CCP_stacks/Plotting_Scripts')

from math import sin, cos, sqrt, atan2, pi
import numpy as np
import scipy as sp
import scipy.cluster.hierarchy as cluster
import mpl_toolkits
from collections import Counter
import mpl_toolkits.basemap
from mpl_toolkits.basemap import Basemap
import csv
import rand_score
import plot_CCP

# Write out cluster and stacks to a csv

def print_out(print_file):
	
	# Print out stacks and cluster
	
	file = open(print_file, 'w')
	for c in cluster:
		file.write(str(c) + ', ')
	file.write('\n')
	count = 1
	for stack in stacks:
		for s in stack:
			file.write(str(s) + ', ')
		count += 1
		if (count != len(stacks)+1):
			file.write('\n')
	file.close()
	
	# Print out coordinates
	
	file1 = open(print_file+'_coords', 'w')
	count = 1
	for coord in coords:
		file1.write(str(coord[0])+', '+ str(coord[1]))
		count +=1
		if (count != len(coords)+1):
			file1.write('\n')
	file1.close()

	return

# Read in cluster and stacks from a csv - input parameters are
# no_stacks = number of stacks to read in
# no_coords = number of coordinates to read in
# len_data = number of data points in a stack - eg, number of depth values

def read_in(no_stacks, no_coords, len_data, read_file):
	
	global cluster, stacks, coords
	
	# Reading in cluster and stack data
	
	with open(read_file) as csv_file:
		stacks = np.zeros((no_stacks, len_data))
		csv_reader = csv.reader(csv_file, delimiter=',')
		line_count = 0
		for row in csv_reader:
			if line_count == 0:
				cluster = row
				cluster = np.array(cluster[:-1]).astype(float)
				line_count+=1
			else:
				stacks[line_count-1] = np.array(row[:-1]).astype(np.float)
				line_count += 1
	
	# Reading in coordinate data
	
	with open(read_file+'_coords') as csv_file_1:
		coords = np.zeros((no_coords, 2))
		csv_reader_1 = csv.reader(csv_file_1, delimiter=',')
		line_count = 0
		for row in csv_reader_1:
			coords[line_count] = np.array(row).astype(float)
			line_count += 1

	return

# Remove all stacks containing fewer data points than a given size from the clusterer

def remove_anoms(size):
	
	global stacks, cluster, coords, seis_data
	
	# Making list of those to remove

	to_remove = np.array([])
	remove_cluster = np.array([])
	for c in np.arange(1, np.max(cluster)+1):
		a = np.where(cluster == c)[0]
		if (len(a) < size):
			to_remove = np.append(to_remove, [c-1], axis=0)
			remove_cluster = np.append(remove_cluster, a, axis=0)
	
	# Removing all anomalies
	
	stacks = np.delete(stacks, to_remove.astype(int), axis=0)
	cluster = np.delete(cluster, remove_cluster.astype(int))
	coords = np.delete(coords, remove_cluster.astype(int), axis=0)
	seis_data = np.delete(seis_data, remove_cluster.astype(int), axis=0)
	
	# Rebuilding cluster
	
	cluster_set = np.unique(cluster)
	count = 1
	for c in cluster_set:
		cluster[np.where(cluster==c)[0]] = count
		count += 1
	
	return

# Compare results of cluster stacking to results of CCP stacking,
# by taking the average coordinate and doing CCP stack from that point.

def compare_methods(figname):
	
	global cluster, stacks, coords, seis_data
	
	# Getting avg coords for each stack
	
	avg_coords = np.zeros((np.max(cluster), 2))
	for i in range(1, np.max(cluster)+1):
		cluster_inds = np.where(cluster == i)[0]
		avg_coords[i-1][0] = np.average([coords[j][1] for j in cluster_inds])
		avg_coords[i-1][1] = np.average([coords[j][0] for j in cluster_inds])
	
	#Generating random colours
	
	colours = []
	for i in range(1, np.max(cluster).astype(int) +1):
		a = '%06X' % randint(0, 0xFFFFFF)
		b = '#' + a
		colours.append(b)
	
	# Plotting
	
	plt.figure(1)
	fig, axes = plt.subplots(1,2)
	d, profiles = plot_CCP.retrieve_profiles(avg_coords, np.min(depths), np.max(depths))
	diffs = np.zeros(len(avg_coords))
	count = 0
	for avg_coord in avg_coords:
		axes[0].plot(depths, stacks[count], color = colours[count])
		axes[0].set_title('Cluster Stack')
		axes[0].set_xlabel('Depth (km)')
		axes[0].set_ylabel('Amplitude relative to main P wave')
		axes[0].set_ylim(-0.1, 0.1)
		axes[1].plot(d, profiles[count], color = colours[count])
		axes[1].set_title('CCP Stack')
		axes[1].set_xlabel('Depth (km)')
		axes[1].set_ylabel('Amplitude relative to main P wave')
		axes[1].set_ylim(-0.1, 0.1)
		s = stacks[count]
		comp_stack = [np.average([s[i],s[i+1],s[i+2],s[i+3]]) for i in range(0, len(s), 100)]
		diffs[count] = cs.correlation(comp_stack, profiles[count])
		count += 1
	
	fig.savefig(figname)
	
	return

def compare_cluster_similarity(coords1, cluster1, coords2, cluster2):
	
	both_coords = np.append(coords1, coords2, axis=0)
	both_coords = np.unique(both_coords, axis=0)
	comp = both_coords
	isin1 = np.isin(comp, coords1)
	isin2 = np.isin(comp, coords2)
	compare1 = np.zeros(len(both_coords))
	compare2 = np.zeros(len(both_coords))
	for x in range(0, len(both_coords)-1):
		if isin1[x].all():
			compare1[x] = cluster1[np.where(coords1 == both_coords[x])[0][0]]
		if isin2[x].all():
			compare2[x] = cluster2[np.where(coords2 == both_coords[x])[0][0]]
	rand_index = rand_score.rand_index_score(compare1.astype(int), compare2.astype(int))
	print(rand_index)
	return rand_index

# Returns average size of stack in current cluster

def average_stack_size():
	
	global cluster
	avg = np.average([len(np.where(cluster == i)[0]) for i in np.unique(cluster)])
	
	return avg

# Stacks current input data using an adaptive stacking routine

def adaptive_stack():
	
	global cluster, stacks, coords, seis_data, filename
	print("Stacking...")
	
	cut_length = round(len(seis_data)/10)
	cluster, stacks = cs.second_cluster(cluster, coords, stacks, threshold=cut_length, crit='maxclust', dist=True, corr=False)
	#print_out('prem_first_stack')
	#read_in(1929, 19292, 1000, 'adaptive_first_stack')
	remove_anoms(5)
	while len(stacks) > 30:
		cluster, stacks = cs.second_cluster(cluster, cs.stack_coords(cluster, coords), stacks, threshold=1, crit='inconsistent', dist=True, corr=True)
		cluster, stacks = cs.second_cluster(cluster, cs.stack_coords(cluster, coords), stacks, threshold=1, crit='inconsistent', dist=True, corr=False)
		remove_anoms(round(average_stack_size()/4))
	print_out(filename +'_final')
	print(len(stacks))
	print(len(coords))

	return cluster, stacks, coords, seis_data

# Plots current stacks and other graphsand saves in directory of name filename
# Indiv = boolean variable, if true then plot individual stacks, otherise don't

def plot(indiv):
	
	global cluster, coords, stacks, seis_data, filename
	print("Plotting...")
	
	os.mkdir(filename)
	os.chdir(filename)
	ps.plot(stacks, depths, cluster, coords, seis_data, filename, anomal = True, plot_individual = indiv)
	#ps.depth_plot(cluster, stacks, coords, depths, 630, 700, filename+'_depths')
	#ps.temp_plot(cluster, stacks, coords, depths, filename+'_temps')
	ps.MTZ_plot(cluster, stacks, coords, depths, filename + '_MTZ')
	#compare_methods(cluster, stacks, coords, seis_data, 'default_compare')

	return

# Start program - reading in

file =sys.argv[1]
filename =sys.argv[2]
min_depth =float(sys.argv[3])
max_depth =float(sys.argv[4])
print("Reading " + file + "...")
seis = read(file,format='PICKLE')
print(len(seis))

# Formatting relevant data

seis_data = np.array([tr.data for tr in seis])
locs = np.array([t.stats.piercepoints['P410s']['410'] for t in seis]).astype(float)
coords = np.array([(l[1], l[2]) for l in locs])
depths = np.array(seis[0].stats.depth)

stacks = seis_data
cut_index_1 = np.where(depths>min_depth)[0][0]
cut_index_2 = np.where(depths>max_depth)[0][0]
depths = np.array(depths[cut_index_1:cut_index_2])
seis_data = np.array([seis[cut_index_1:cut_index_2] for seis in seis_data])
stacks = np.array([stack[cut_index_1:cut_index_2] for stack in stacks])
cluster = range(1, len(seis_data)+1)

adaptive_stack()
plot(True)

#cluster1, stacks1, coords1 = read_in(12, 10723, 1000, 'test')
#cluster2, stacks2, coords2 = read_in(8, 8082, 1000, 'adapt_remove_3000')
#print(compare_cluster_similarity(coords1, cluster1, coords2, cluster2))

