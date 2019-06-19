import hierarchical_stack as hs
from obspy import read
import random
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
import rand_score
import plot_CCP

# Write out cluster and stacks to a csv

def print_out(cluster, stacks, coords, filename):

	file = open(filename, 'w')
	for c in cluster:
		file.write(str(c) + ', ')
	file.write('\n')
	for stack in stacks:
		for s in stack:
			file.write(str(s) + ', ')
		file.write('\n')
	file.close()
	file1 = open(filename+'_coords', 'w')
	for coord in coords:
		file1.write(str(coord[0])+', '+ str(coord[1]))
		file1.write('\n')
	file1.close()

	return

# Read in cluster and stacks from a csv

def read_in(no_stacks, no_coords, filename):
	with open(filename) as csv_file:
		stacks = np.zeros((no_stacks,len(seis_data[0])))
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
	with open(filename+'_coords') as csv_file_1:
		coords = np.zeros((no_coords, 2))
		csv_reader_1 = csv.reader(csv_file_1, delimiter=',')
		line_count = 0
		for row in csv_reader_1:
			coords[line_count] = np.array(row[:-1]).astype(float)
			line_count += 1
	return cluster, stacks, coords

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

	return rand_index

def average_stack_size(cluster):
	
	avg = np.average([len(np.where(cluster == i)[0]) for i in np.unique(cluster)])
	
	return avg

min_depth = 280
max_depth = 780
seis_data, coords, depths = hs.set_up()
stacks = seis_data
cut_index_1 = np.where(depths>min_depth)[0][0]
cut_index_2 = np.where(depths>max_depth)[0][0]
depths = np.array(depths[cut_index_1:cut_index_2])
seis_data = np.array([seis[cut_index_1:cut_index_2] for seis in seis_data])
stacks = np.array([stack[cut_index_1:cut_index_2] for stack in stacks])
cluster = range(1, len(seis_data)+1)

print("Stacking...")
cut_length = round(len(seis_data)/10)
cluster, stacks = hs.second_cluster(cluster, coords, stacks, threshold=cut_length, crit='maxclust', dist=True, corr=False)
cluster, stacks, coords, seis_data = remove_anoms(cluster, stacks, coords, seis_data, 10)
while len(stacks) > 25:
	print('hi')
	cluster, stacks = hs.second_cluster(cluster, hs.stack_coords(cluster, coords), stacks, threshold=1, crit='inconsistent', dist=True, corr=True)
	cluster, stacks, coords, seis_data = remove_anoms(cluster, stacks, coords, seis_data, round(average_stack_size(cluster)/2))
print(cluster)
print(average_stack_size(cluster))

print("Plotting...")

hs.plot(stacks, depths, cluster, coords, seis_data, "test", anomal = True, plot_individual = False)
print_out(cluster, stacks, coords, 'test')
#hs.temp_plot(cluster, stacks, coords, depths, 'temps')
#hs.MTZ_plot(cluster, stacks, coords, depths, 'test_MTZ')
#hs.var_plot(cluster, stacks, coords, depths, 'test_var')
#hs.var_plot(cluster, stacks, coords, seis_data, 'SL2014_sixsixty_vars')
#compare_methods(cluster, stacks, coords, seis_data, 'default_compare')

