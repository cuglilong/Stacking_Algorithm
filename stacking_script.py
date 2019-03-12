import hierarchical_stack as hs
from obspy import read
from random import randint
import matplotlib.pyplot as plt
import sys

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
import plot_CCP

# Write out cluster and stacks to a csv

def print_out(cluster, stacks, filename):

	file = open(filename, 'w')
	for c in cluster:
		file.write(str(c) + ', ')
	file.write('\n')
	for stack in stacks:
		for s in stack:
			file.write(str(s) + ', ')
		file.write('\n')

	return

# Read in cluster and stacks from a csv

def read_in(no_stacks, filename):
	with open(filename) as csv_file:
		stacks = np.zeros((no_stacks,len(seis_data[0])))
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

# Remove all stacks containing fewer data points than a given size from the clusterer

def remove_anoms(cluster, stacks, coords, seis_data, size):
	
	# Making list of those to remove
	
	to_remove = np.array([])
	remove_cluster = np.array([])
	for c in np.arange(np.max(cluster)):
		a = np.where(cluster == c+1)[0]
		if (len(a) < size):
			to_remove = np.append(to_remove, [c], axis=0)
			remove_cluster = np.append(remove_cluster, a, axis=0)
	
	# Removing all anomalies
	
	stacks = np.delete(stacks, to_remove.astype(int), axis=0)
	cluster = np.delete(cluster, remove_cluster.astype(int))
	coords = np.delete(coords, remove_cluster.astype(int), axis=0)
	seis_data = np.delete(seis_data, remove_cluster.astype(int), axis=0)
	
	# Rebuilding cluster
	
	cluster_set = set(cluster)
	count = 1
	for c in cluster_set:
		cluster[np.where(cluster==c)[0]] = count
		count += 1
	
	return cluster, stacks, coords, seis_data

def compare_methods(cluster, stacks, coords, seis_data, figname):
	avg_coords = np.zeros((np.max(cluster), 2))
	for i in range(1, np.max(cluster)+1):
		cluster_inds = np.where(cluster == i)[0]
		avg_coords[i-1][0] = np.average([coords[j][1] for j in cluster_inds])
		avg_coords[i-1][1] = np.average([coords[j][0] for j in cluster_inds])
	colours = []
	for i in range(1, np.max(cluster).astype(int) +1):
		a = '%06X' % randint(0, 0xFFFFFF)
		b = '#' + a
		colours.append(b)
	plt.figure(1)
	fig, axes = plt.subplots(1,2)
	d, profiles = plot_CCP.retrieve_profiles(avg_coords, np.min(depths), np.max(depths))
	diffs = np.zeros(len(avg_coords))
	count = 0
	for avg_coord in avg_coords:
		axes[0].plot(depths, stacks[count], color = colours[count])
		axes[0].set_title('Cluster Stack')
		axes[0].set_ylim(-0.1, 0.1)
		axes[1].plot(d, profiles[count], color = colours[count])
		axes[1].set_title('CCP Stack')
		axes[1].set_ylim(-0.1, 0.1)
		s = stacks[count]
		comp_stack = [np.average([s[i],s[i+1],s[i+2],s[i+3]]) for i in range(0, len(s), 100)]
		diffs[count] = hs.correlation(comp_stack, profiles[count])
		count += 1
	fig.savefig(figname + '_all')
	sizes = [len(np.where(cluster==cl)[0]) for cl in range(1, np.max(cluster)+1)]
	to_examine = np.where(diffs>0.00)[0]
	fig1, axes1 = plt.subplots(1, 2)
	ax1 = axes1[0]
	ax2 = axes1[1]

	for cl in to_examine:
		
		lon410 = np.array([co[1] for co in coords])
		lat410 = np.array([co[0] for co in coords])
		m = Basemap(llcrnrlon=np.min(lon410)-5.,llcrnrlat=np.min(lat410)-5.,urcrnrlon=np.max(lon410)+5.,urcrnrlat=np.max(lat410)+5.,
		resolution='i',projection='merc',lon_0=np.mean(lon410),lat_0=np.mean(lat410))
		m.shadedrelief()
		m.drawparallels(np.arange(-40,80.,5.),color='gray')
		m.drawmeridians(np.arange(-30.,80.,5.),color='gray')
		
		ax1.set_xlabel('depth (km)')
		ax1.set_ylabel('amplitude relative to main P wave')
		inds = np.where(cluster == cl+1)[0]
		ax1.plot(depths, stacks[cl], label='Cluster Stack')
		ax1.plot(d, profiles[cl], label='CCP Stack')
		ax1.legend()
		for i in inds:
			lon, lat = m(coords[i][1], coords[i][0])
			m.plot(lon, lat, marker='x', markeredgecolor='black')
		fig1.savefig(figname + str(cl+1))
		ax1.clear()
		ax2.clear()
	return

min_depth = 280
max_depth = 750
seis_data, coordinates, depths = hs.set_up()
cut_index_1 = np.where(depths>min_depth)[0][0]
cut_index_2 = np.where(depths>max_depth)[0][0]
cluster, stacks = read_in(1750, 'stack_data_whole_range.csv')
depths = np.array(depths[cut_index_1:cut_index_2])
seis_data = np.array([seis[cut_index_1:cut_index_2] for seis in seis_data])
stacks = np.array([stack[cut_index_1:cut_index_2] for stack in stacks])

print("Stacking...")

cluster, stacks, coords, seis_data = remove_anoms(cluster, stacks, coordinates, seis_data, 10)
cluster, stacks = hs.second_cluster(cluster, hs.stack_coords(cluster,coords), stacks, threshold=200, crit='maxclust', dist = True, corr = True) #true, true
cluster, stacks, coords, seis_data = remove_anoms(cluster, stacks, coords, seis_data, 20)
cluster, stacks = hs.second_cluster(cluster, hs.stack_coords(cluster,coords), stacks, threshold = 1, crit='inconsistent', dist = True, corr = False) #true, false
cluster, stacks = hs.second_cluster(cluster, hs.stack_coords(cluster,coords), stacks, threshold = 1, crit = 'inconsistent', dist = True, corr = True) #true, true
cluster, stacks, coords, seis_data = remove_anoms(cluster, stacks, coords, seis_data, 100)

print("Plotting...")

hs.plot(stacks, depths, cluster, coords, seis_data, "whole_range", anomal = False, plot_individual = True)
hs.MTZ_plot(cluster, stacks, coords, depths, 'MTZ')
#hs.depth_plot(cluster, stacks, coords, depths, 350, 450, 'corr_depths')
hs.var_plot(cluster, stacks, coords, seis_data, 'whole_range_vars')
compare_methods(cluster, stacks, coords, seis_data, 'whole_range_compare')
