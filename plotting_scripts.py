from obspy import read
from random import randint
import matplotlib.pyplot as plt
import sys
from math import sin, cos, sqrt, atan2, pi
import random
import numpy as np
import scipy as sp
import cluster_variation as c_v
import scipy.cluster.hierarchy as cluster
import mpl_toolkits
import mpl_toolkits.basemap
from mpl_toolkits.basemap import Basemap
from collections import Counter
from itertools import combinations
import colorsys

# Create map of area to overlay location points on

def create_map(lon, lat):
	m = Basemap(llcrnrlon=np.min(lon)-5.,llcrnrlat=np.min(lat)-5.,urcrnrlon=np.max(lon)+5.,urcrnrlat=np.max(lat)+5.,
        resolution='i',projection='merc',lon_0=np.mean(lon),lat_0=np.mean(lat))
	m.shadedrelief()
	m.drawparallels(np.arange(-40,80.,5.),color='gray')
	m.drawmeridians(np.arange(-30.,80.,5.),color='gray')
	
	return m

# Plots final stacks with and cluster locations.
# Input: a stacker_object and figname
# plot_indiviudal - if true, plots every individual stack in its own figure

def plot(s_o, figname, plot_individual = False, vote_map=[]):
	
	# Creating a random set of colours to distinguish between different clusters
	
	colours = []
	for i in set(s_o.cluster.astype(int)):
		hex = '%06X' % randint(0, 0xFFFFFF)
		rgb = tuple(int(hex[i:i+2], 16)/255 for i in (0, 2, 4))
		colours.append(rgb)
	colour_clusters = [colours[(s_o.cluster[j]-1)] for j in range(len(s_o.cluster))]

	# If a number of tests have been run, using vote map to determine color luminance for each point
	
	if vote_map != []:
		
		# Rescale vote_map to go between 0 and 1

		colour_clusters = [list(i) for i in colour_clusters]
		max = np.max(vote_map)
		min = np.min(vote_map)
		grad = (max-min)
		sat_map = [(i-min)/grad for i in vote_map]
		print(len(colour_clusters))
		for j in range(len(colour_clusters)):
			c_c = colour_clusters[j]
			hsv = list(colorsys.rgb_to_hsv(c_c[0],c_c[1],c_c[2]))
			hsv[1] = sat_map[j]
			colour_clusters[j] = colorsys.hsv_to_rgb(hsv[0],hsv[1],hsv[2])
			
	# Initialising figure
	
	plt.figure(1)
	fig, axes = plt.subplots(1,2)
	ax1 = axes[0]
	ax2 = axes[1]
	lons = np.array([co[1] for co in s_o.coords])
	lats = np.array([co[0] for co in s_o.coords])
	
	# Plotting individual stacks, if plot_individual is true
	
	if plot_individual == True:
		count = 0
		for stack in s_o.stacks:
			ax1.set_xlabel('depth (km)')
			ax1.set_ylabel('amplitude relative to main P wave')
			ax1.plot(s_o.x_var, stack, color = colours[count])
			
			ax2.set_xlabel('longitude')
			ax2.set_ylabel('latitude')
			m = create_map(lons, lats)
			inds = np.where(s_o.cluster == count+1)[0]
			avg_lon = np.average([s_o.coords[i][1] for i in inds])
			avg_lat = np.average([s_o.coords[i][0] for i in inds])
			avg_lon1, avg_lat1 = m(avg_lon, avg_lat)
			for i in inds:
				lon, lat = m(s_o.coords[i][1], s_o.coords[i][0])
				m.plot(lon, lat, marker='x', markeredgecolor = colour_clusters[i])
			m.plot(avg_lon1, avg_lat1, marker='x', markersize=7, markeredgecolor='black', LineStyle='None', label=str(avg_lat) + ', ' + str(avg_lon))
			count+=1
			ax2.legend()
			fig.savefig(figname + str(count))
			ax1.clear()
			ax2.clear()
	
	# Initialising final figure
	
	ax1.set_xlabel('depth (km)')
	ax1.set_ylabel('amplitude relative to main P wave')
	ax2.set_xlabel('longitude')
	ax2.set_ylabel('latitude')
	m = create_map(lons, lats)

	# Plotting final figure
	
	count = 0
	for stack in s_o.stacks:
		ax1.plot(s_o.x_var, stack, color = colours[count])
		count+=1
	for i in range(len(s_o.coords)):
		lon, lat = m(s_o.coords[i][1], s_o.coords[i][0])
		m.plot(lon, lat, marker='x', markeredgecolor=colour_clusters[i])
	fig.savefig(figname)
	fig.clear()
	
	return
	
# Plots depth of peak as a colourmap for all coordinates

def depth_plot(s_o, min, max, figname):
	
	# Find peaks

	depths = s_o.x_var
	cut_index_1 = np.argmax(depths>min)
	cut_index_2 = np.argmax(depths>max)
	cropped_depths = depths[cut_index_1:cut_index_2]
	cropped_stacks = [stack[cut_index_1:cut_index_2] for stack in s_o.stacks]
	signal_depths = [cropped_depths[np.argmax(stack)] for stack in cropped_stacks]
	coord_depths = [signal_depths[s_o.cluster[i]-1] for i in range(len(s_o.coords))]
	
	# Plot figure
	
	plot_heatmap(coord_depths, s_o.coords, figname)
	
	return

# Plots thickness of MTZ as colormap

def MTZ_plot(s_o, figname):
	
	# Find peaks and therefore MTZ thickness
	
	depths = s_o.x_var
	cut_index_1 = np.argmax(depths>380)
	cut_index_2 = np.argmax(depths>460)
	cut_index_3 = np.argmax(depths>590)
	cut_index_4 = np.argmax(depths>680)
	cropped_depths_1 = depths[cut_index_1:cut_index_2]
	cropped_stacks_1 = [stack[cut_index_1:cut_index_2] for stack in s_o.stacks]
	cropped_depths_2 = depths[cut_index_3:cut_index_4]
	cropped_stacks_2 = [stack[cut_index_3:cut_index_4] for stack in s_o.stacks]
	signal_depths_1 = [cropped_depths_1[np.argmax(stack)] for stack in cropped_stacks_1]
	signal_depths_2 = [cropped_depths_2[np.argmax(stack)] for stack in  cropped_stacks_2]
	MTZ_widths = [j - i for i, j in np.stack((signal_depths_1, signal_depths_2), axis = -1)]
	coord_widths = [MTZ_widths[s_o.cluster[i]-1] for i in range(len(s_o.coords))]
	
        # Plot figure
	
	plot_heatmap(coord_widths, s_o.coords, figname)
	
	return

# Plots variance within each cluster as a colourmap

def var_plot(s_o, figname):
	
	# Find cluster variances
	
	vars = np.zeros(len(s_o.stacks))
	count = 1
	for stack in s_o.stacks:
		orig = s_o.seis_data[np.where(s_o.cluster==count)[0]]
		vars[count-1] = c_v(s_o.cluster, count, s_o.seis_data)
		count += 1
	coord_vars = [vars[s_o.cluster[i]-1] for i in range(len(s_o.coords))]
	
	# Plot figure
	
	plot_heatmap(coord_vars, s_o.coords, figname)
	
	return

# Plots the signed magnitude of the largest peak after 0 ScS as a colourmap

def mag_plot(s_o, figname):
	
	# Find (signed) magnitude of largest peak after ScS
	
	mags = np.zeros(len(s_o.stacks))
	cut1 = np.where(0<s_o.x_var)[0][0]
	cut2 = np.where(5.5<s_o.x_var)[0][0]
	stacks = np.array([stack[cut1:cut2] for stack in s_o.stacks])
	count = 0
	for stack in stacks:
		mags[count] = stack[np.argmax(np.abs(stack))]
		count += 1
	coord_mags = [mags[s_o.cluster[i]-1] for i in range(len(s_o.coords))]
	
	# Plot figure
	
	plot_heatmap(coord_mags, s_o.coords, figname)
	 
	return

# Plots distance of largest peak along the the x-variable as a colour map

def peak_dist_plot(s_o, figname):
	
	# Find peak distances
	
	peak_dist = np.zeros(len(s_o.stacks))
	count = 0
	for stack in s_o.stacks:
		peak_dist[count] = s_o.x_var[np.argmax(np.abs(stack))]
		count += 1
	coord_peak_dist = [peak_dist[s_o.cluster[i]-1] for i in range(len(s_o.coords))]
	
	# Plot figure
	
	plot_heatmap(coord_peak_dist, s_o.coords, figname)
	
	return

# Generates an array demonstrating how much different test runs agree that points are in the same cluster
# Takes a series of several test clusters for the same data set, ie with a small number of points removed to introduce randomness

def cluster_vote_map(base_cluster, tests):
	
	# Going through all points to find the degree of agreement between different test runs
	
	coords = base_cluster.coords
	cluster = base_cluster.cluster_keep
	data_range = np.array(range(len(base_cluster.seis_data)))
	vote_map = np.zeros(len(data_range))
	for test in tests:
		for i in data_range:
			same = set(np.where(cluster==cluster[i])[0])
			diff = set(np.where(cluster!=cluster[i])[0])
			same_test = set(np.where(test.cluster_keep==test.cluster_keep[i])[0])
			diff_test = set(np.where(test.cluster_keep!=test.cluster_keep[i])[0])
			intersect = len(same.intersection(same_test))
			intersect += len(diff.intersection(diff_test))
			vote_map[i] += intersect

	return vote_map

def plot_heatmap(coord_heats, coords, figname):
	
        fig = plt.figure(1)
        axes = plt.gca()
        axes.set_xlabel('longitude')
        axes.set_ylabel('latitude')
        lon = [coords[i][1] for i in range(len(coords))]
        lat = [coords[i][0] for i in range(len(coords))]
        cb = axes.scatter(lon, lat, c=coord_heats, marker='x', cmap=plt.cm.get_cmap('cool'))
        fig.colorbar(cb, ax=axes)
        fig.savefig(figname)
        fig.clear()

        return
