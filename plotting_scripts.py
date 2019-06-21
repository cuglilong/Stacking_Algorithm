from obspy import read
from random import randint
import matplotlib.pyplot as plt
import sys
from math import sin, cos, sqrt, atan2, pi
import random
import numpy as np
import scipy as sp
import scipy.cluster.hierarchy as cluster
import mpl_toolkits
import mpl_toolkits.basemap
from mpl_toolkits.basemap import Basemap
from collections import Counter

# Create map of western US to overlay location points on

def create_map(lon, lat):
	m = Basemap(llcrnrlon=np.min(lon)-5.,llcrnrlat=np.min(lat)-5.,urcrnrlon=np.max(lon)+5.,urcrnrlat=np.max(lat)+5.,
        resolution='i',projection='merc',lon_0=np.mean(lon),lat_0=np.mean(lat))
	m.shadedrelief()
	m.drawparallels(np.arange(-40,80.,5.),color='gray')
	m.drawmeridians(np.arange(-30.,80.,5.),color='gray')
	
	return m

# Plots final stacks with and cluster locations.
# anomal - if false, exclude all stacks with less than 400 data points for clarity
# plot_indiviudal - if true, plots every individual stack in its own figure

def plot(stacks, depths, cluster, coords, seis_data, figname, anomal=True, plot_individual = False):
	
	# Creating a random set of colours to distinguish between different clusters
	
	colours = []
	for i in range(1, np.max(cluster).astype(int) +1):
		a = '%06X' % randint(0, 0xFFFFFF)
		b = '#' + a
		colours.append(b)
	colour_clusters = [colours[(cluster[j]-1)] for j in range(len(cluster))]
	
	# Initialising figure
	
	plt.figure(1)
	fig, axes = plt.subplots(1,2)
	ax1 = axes[0]
	ax2 = axes[1]
	lons = np.array([co[1] for co in coords])
	lats = np.array([co[0] for co in coords])
	# Plotting individual stacks, if plot_individual is true
	
	if plot_individual == True:
		count = 0
		for stack in stacks:
			if (anomal == False and len(np.where(cluster == count+1)[0])<400):
				count+=1
			else:
				
				ax1.set_xlabel('depth (km)')
				ax1.set_ylabel('amplitude relative to main P wave')
				ax1.plot(depths, stack, color = colours[count])
				
				ax2.set_xlabel('longitude')
				ax2.set_ylabel('latitude')
				m = create_map(lons, lats)
				inds = np.where(cluster == count+1)[0]
				avg_lon = np.average([coords[i][1] for i in inds])
				avg_lat = np.average([coords[i][0] for i in inds])
				avg_lon1, avg_lat1 = m(avg_lon, avg_lat)
				for i in inds:
					lon, lat = m(coords[i][1], coords[i][0])
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
	for stack in stacks:
		if (anomal == False and len(np.where(cluster == count+1)[0])<400):
			count+=1
		else:
			ax1.plot(depths, stack, color = colours[count])
			count+=1
	for i in range(len(coords)):
		lon, lat = m(coords[i][1], coords[i][0])
		m.plot(lon, lat, marker='x', markeredgecolor=colour_clusters[i])
	fig.savefig(figname)
	
# Plots depth of peak at a certain lon, lat as a colourmap

def depth_plot(cluster, stacks, coords, depths, min_depth, max_depth, figname):
	
	# Find peaks
	
	cut_index_1 = np.argmax(depths>min_depth)
	cut_index_2 = np.argmax(depths>max_depth)
	cropped_depths = depths[cut_index_1:cut_index_2]
	cropped_stacks = [stack[cut_index_1:cut_index_2] for stack in stacks]
	signal_depths = [cropped_depths[np.argmax(stack)] for stack in cropped_stacks]
	coord_depths = [signal_depths[cluster[i]-1] for i in range(len(coords))]
	
	# Plot figure
	
	fig = plt.figure(1)
	axes = plt.gca()
	axes.set_xlabel('longitude')
	axes.set_ylabel('latitude')
	lon = [coords[i][1] for i in range(len(coords))]
	lat = [coords[i][0] for i in range(len(coords))]
	cb = axes.scatter(lon, lat, c=coord_depths, marker='x', cmap=plt.cm.get_cmap('cool'))
	fig.colorbar(cb, ax=axes)
	fig.savefig(figname)

# Plots thickness of MTZ as colormap

def MTZ_plot(cluster, stacks, coords, depths, figname):
	
	# Find peaks and therefore MTZ thickness
	
	cut_index_1 = np.argmax(depths>380)
	cut_index_2 = np.argmax(depths>460)
	cut_index_3 = np.argmax(depths>590)
	cut_index_4 = np.argmax(depths>680)
	cropped_depths_1 = depths[cut_index_1:cut_index_2]
	cropped_stacks_1 = [stack[cut_index_1:cut_index_2] for stack in stacks]
	cropped_depths_2 = depths[cut_index_3:cut_index_4]
	cropped_stacks_2 = [stack[cut_index_3:cut_index_4] for stack in stacks]
	signal_depths_1 = [cropped_depths_1[np.argmax(stack)] for stack in cropped_stacks_1]
	signal_depths_2 = [cropped_depths_2[np.argmax(stack)] for stack in  cropped_stacks_2]
	MTZ_widths = [j - i for i, j in np.stack((signal_depths_1, signal_depths_2), axis = -1)]
	coord_widths = [MTZ_widths[cluster[i]-1] for i in range(len(coords))]
	
        # Plot figure
	
	fig = plt.figure(1)
	axes = plt.gca()
	axes.set_xlabel('longitude')
	axes.set_ylabel('latitude')
	lon = [coords[i][1] for i in range(len(coords))]
	lat = [coords[i][0] for i in range(len(coords))]
	cb = axes.scatter(lon, lat, c=coord_widths, marker='x', cmap=plt.cm.get_cmap('cool'))
	fig.colorbar(cb, ax=axes)
	plt.grid()
	fig.savefig(figname)

	return

def temp_plot(cluster, stacks, coords, depths, figname):

        # Find peaks and therefore MTZ thickness

	cut_index_1 = np.argmax(depths>390)
	cut_index_2 = np.argmax(depths>470)
	cut_index_3 = np.argmax(depths>620)
	cut_index_4 = np.argmax(depths>710)
	cropped_depths_1 = depths[cut_index_1:cut_index_2]
	cropped_stacks_1 = [stack[cut_index_1:cut_index_2] for stack in stacks]
	cropped_depths_2 = depths[cut_index_3:cut_index_4]
	cropped_stacks_2 = [stack[cut_index_3:cut_index_4] for stack in stacks]
	signal_depths_1 = [cropped_depths_1[np.argmax(stack)] for stack in cropped_stacks_1]
	signal_depths_2 = [cropped_depths_2[np.argmax(stack)] for stack in  cropped_stacks_2]
	MTZ_widths = [j - i for i, j in np.stack((signal_depths_1, signal_depths_2), axis = -1)]
	print(MTZ_widths)
	temps = [3600-8*depth for depth in MTZ_widths]
	coord_temps = [temps[cluster[i]-1] for i in range(len(coords))]

        # Plot figure

	fig = plt.figure(1)
	axes = plt.gca()
	axes.set_xlabel('longitude')
	axes.set_ylabel('latitude')
	lon = [coords[i][1] for i in range(len(coords))]
	lat = [coords[i][0] for i in range(len(coords))]
	cb = axes.scatter(lon, lat, c=coord_temps, marker='x', cmap=plt.cm.get_cmap('cool'))
	fig.colorbar(cb, ax=axes, label="Temperature (K)")
	fig.savefig(figname)

	return


def var_plot(cluster, stacks, coords, seis_data, figname):
	
	# Find cluster variances
	
	vars = np.zeros(len(stacks))
	count = 1
	for stack in stacks:
		orig = seis_data[np.where(cluster==count)[0]]
		vars[count-1] = group_corr(orig, stack)
		count += 1
	coord_vars = [vars[cluster[i]-1] for i in range(len(coords))]
	
	# Plot figure
	
	fig = plt.figure(1)
	axes = plt.gca()
	axes.set_xlabel('longitude')
	axes.set_ylabel('latitude')
	lon = [coords[i][1] for i in range(len(coords))]
	lat = [coords[i][0] for i in range(len(coords))]
	cb = axes.scatter(lon, lat, c=coord_vars, marker='x', cmap=plt.cm.get_cmap('cool'))
	fig.colorbar(cb, ax=axes)
	fig.savefig(figname)

	return
