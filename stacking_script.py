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

def remove_anoms(cluster, stacks, coords, seis_data, size):
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
	seis_data = np.delete(seis_data, remove_cluster.astype(int), axis=0)
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
	d, profiles = plot_CCP.retrieve_profiles(avg_coords, 280, 480)
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
	print(sizes)
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
	fig2 = plt.figure()
	plt.xlabel('size of cluster')
	plt.ylabel('diff between methods')
	plt.plot(sizes, diffs, linestyle='None', marker='x')
	fig2.savefig(figname + '_size')
	return

print("Assign...")
coords = hs.coordinates
seis_data = hs.seis_data
depths = hs.depths
print("Stacking...")
cluster, stacks = read_in(1750, 'stack_data_1.csv')
cluster, stacks, coords, seis_data = remove_anoms(cluster, stacks, hs.coordinates, seis_data, 10)
cluster, stacks, short_cluster = hs.second_cluster(cluster, hs.stack_coords(cluster,coords), stacks, threshold=200, crit='maxclust', dist = True, corr = True)
cluster, stacks, coords, seis_data = remove_anoms(cluster, stacks, coords, seis_data, 20)
cluster, stacks, short_cluster = hs.second_cluster(cluster, hs.stack_coords(cluster,coords), stacks, threshold = 1, crit='inconsistent', dist = True, corr = False)
cluster, stacks, short_cluster = hs.second_cluster(cluster, hs.stack_coords(cluster,coords), stacks, threshold = 1, crit = 'inconsistent', dist = True, corr = True)
cluster, stacks, coords, seis_data = remove_anoms(cluster, stacks, coords, seis_data, 100)
print("Plotting...")
#hs.plot(stacks, depths, cluster, coords, seis_data, "default", anomal = False, plot_individual = True)
#hs.depth_plot(cluster, stacks, coords, depths, 'default_depths', fourten=True)
#hs.var_plot(cluster, stacks, coords, seis_data, 'default_vars')
#compare_methods(cluster, stacks, coords, seis_data, 'compare')
