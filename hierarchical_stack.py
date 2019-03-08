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
from collections import Counter

# Clustering and stacking data
# Clusters using data from b
# Then stacks the data in a
# Returns stacked data and list of cluster indices

# Returns L2 correlation between traces tr1 and tr2

def correlation(tr1, tr2, n=2):
	return np.absolute(np.sum([(tr1[i] - tr2[i])**n for i in range(len(tr1))]))

# Returns average L2 correlation between all traces in a stack and their mean

def group_corr(traces, stack):
	return np.average([correlation(trace, stack) for trace in traces])

# Haversine distance formula

def haversine(coord1, coord2):

	lat1, long1 = coord1*(pi/180)
	lat2, long2 = coord2*(pi/180)
	del_lat = lat2 - lat1
	del_long = long2 - long1
	R = 6371
	a = sin(del_lat/2)**2 + cos(lat1)*cos(lat2)*(sin(del_long/2)**2)
	c = 2*atan2(sqrt(a), sqrt(1-a))

	return R*c

# 'Distance' metric to use for clustering - can weight by distance, correlation, or both

def combined_metric(a, b, include_dist = True, include_corr = False):
	if (include_corr == False):
		return haversine(a[-2:], b[-2:])
	elif (include_dist == False):
		return correlation(a[:-2],b[:-2])
	else:
		corr = correlation(a[:-2],b[:-2])
		dist = haversine(a[-2:], b[-2:])
		return corr * dist

# Returns an array of the average coordinate in a cluster - this means we are automatically making the assumption
# that the average of a cluster is the 'typical' point, rather than the centroid, or another more complicated metric 

def stack_coords(cluster, coords):

	clusters = np.arange(1,np.max(cluster).astype(int)+1)
	coords = np.array([[np.average([coords[i][j] for i in np.where(cluster == cl)[0]]) for j in [0, 1]] for cl in clusters])
	coords = np.nan_to_num(coords)
	print(len(coords))
	print(len(clusters))

	return coords

def plot(stacks, depths, cluster, coords, seis_data, figname, anomal=True, plot_individual = False):

	colours = []
	for i in range(1, np.max(cluster).astype(int) +1):
		a = '%06X' % randint(0, 0xFFFFFF)
		b = '#' + a
		colours.append(b)
	colour_clusters = [colours[(cluster[j]-1).astype(np.int)] for j in range(len(cluster))]
	plt.figure(1)
	fig, axes = plt.subplots(1,2)
	ax1 = axes[0]
	ax2 = axes[1]
	lon410 = np.array([co[1] for co in coordinates])
	lat410 = np.array([co[0] for co in coordinates])

	if plot_individual == True:
		count = 0
		for stack in stacks:
			if (anomal == False and len(np.where(cluster == count+1)[0])<400):
				count+=1
			else:
				ax2.set_xlabel('longitude')
				ax2.set_ylabel('latitude')
				lon410 = np.array([co[1] for co in coordinates])
				lat410 = np.array([co[0] for co in coordinates])
				m = Basemap(llcrnrlon=np.min(lon410)-5.,llcrnrlat=np.min(lat410)-5.,urcrnrlon=np.max(lon410)+5.,urcrnrlat=np.max(lat410)+5.,
				resolution='i',projection='merc',lon_0=np.mean(lon410),lat_0=np.mean(lat410))
				m.shadedrelief()
				m.drawparallels(np.arange(-40,80.,5.),color='gray')
				m.drawmeridians(np.arange(-30.,80.,5.),color='gray')
				
				ax1.set_xlabel('depth (km)')
				ax1.set_ylabel('amplitude relative to main P wave')
				inds = np.where(cluster == count+1)[0]
				ax1.plot(depths, stack, color = colours[count])

				for i in inds:
					lon, lat = m(coords[i][1], coords[i][0])
					m.plot(lon, lat, marker='x', markeredgecolor = colour_clusters[i])
				count+=1
				fig.savefig(figname + str(count))
				ax1.clear()
				ax2.clear()
	
	ax1.set_xlabel('depth (km)')
	ax1.set_ylabel('amplitude relative to main P wave')
	ax2.set_xlabel('longitude')
	ax2.set_ylabel('latitude')
	m = Basemap(llcrnrlon=np.min(lon410)-5.,llcrnrlat=np.min(lat410)-5.,urcrnrlon=np.max(lon410)+5.,urcrnrlat=np.max(lat410)+5.,
	resolution='i',projection='merc',lon_0=np.mean(lon410),lat_0=np.mean(lat410))
	m.shadedrelief()
	m.drawparallels(np.arange(-40,80.,5.),color='gray')
	m.drawmeridians(np.arange(-30.,80.,5.),color='gray')
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
	fig.show()
	
def depth_plot(cluster, stacks, coords, depths, figname, fourten=True):
	
	fig = plt.figure(1)
	axes = plt.gca()
	axes.set_xlabel('longitude')
	axes.set_ylabel('latitude')
	lon = [coords[i][1] for i in range(len(coords))]
	lat = [coords[i][0] for i in range(len(coords))]
	if fourten==True:
		cut_index = np.argmax(depths>390)
	else:
		cut_index = np.argmax(depths>630)

	cropped_depths = depths[cut_index:]
	cropped_stacks = [stack[cut_index:] for stack in stacks]
	sig_depths = [cropped_depths[np.argmax(stack)] for stack in cropped_stacks]
	coord_depths = [sig_depths[cluster[i]-1] for i in range(len(coords))]
	cb = axes.scatter(lon, lat, c=coord_depths, marker='x', cmap=plt.cm.get_cmap('cool'))
	fig.colorbar(cb, ax=axes)
	fig.savefig(figname)

def var_plot(cluster, stacks, coords, seis_data, figname):
	
	fig = plt.figure(1)
	axes = plt.gca()
	axes.set_xlabel('longitude')
	axes.set_ylabel('latitude')
	lon = [coords[i][1] for i in range(len(coords))]
	lat = [coords[i][0] for i in range(len(coords))]
	
	vars = np.zeros(len(stacks))
	count = 1
	for stack in stacks:
		orig = seis_data[np.where(cluster==count)[0]]
		vars[count-1] = group_corr(orig, stack)
		count += 1
	coord_vars = [vars[cluster[i]-1] for i in range(len(coords))]

	cb = axes.scatter(lon, lat, c=coord_vars, marker='x', cmap=plt.cm.get_cmap('cool'))
	fig.colorbar(cb, ax=axes)
	fig.savefig(figname)

def cluster_data(a, b, metric, threshold, crit):
	cluster = sp.cluster.hierarchy.fclusterdata(b, t=threshold, criterion=crit, metric=metric, method='average')
	stacks = np.array([[np.sum([a[i][j] for i in np.where(cluster == cl+1)[0]]) for j in range(len(a[0]))] for cl in range(np.max(cluster))])
	for cl in range(1, np.max(cluster+1)):
		length = len(np.where(cluster == cl)[0])
		stacks[cl-1] = stacks[cl-1]/length
	stacks = np.nan_to_num(stacks)
	return cluster, stacks

def second_cluster(cluster, coordinates, stacks, threshold, crit, dist = True, corr = False):
	
	# Formatting data
	comb_metric = (lambda a, b: combined_metric(a, b, include_dist=dist, include_corr=corr))
	new_data = np.append(stacks, coordinates, axis=1)
	# Clustering data
	cluster_2, stacks = cluster_data(new_data, new_data, comb_metric, threshold, crit)
	# Running through old cluster and re-formatting data
	new_cluster = np.zeros(len(cluster))
	for cl in np.arange(1, np.max(cluster).astype(int)+1):
        	new_cluster[np.where(cluster == cl)] = cluster_2[cl-1].astype(int)
	stacks = stacks[:, :-2]
	cluster = new_cluster.astype(int)

	return cluster, stacks, cluster_2

file =sys.argv[1]
print("Reading " + file + "...")
seis = read(file,format='PICKLE')

# Formatting relevant data
length = len(seis[0].data)
seis_data = np.array([tr.data for tr in seis])
loc410 = np.array([t.stats.piercepoints['P410s']['410'] for t in seis]).astype(float)
coordinates = np.array([(l410[1], l410[2]) for l410 in loc410])
depths = np.array(seis[0].depth)
coordinates = coordinates[:, :]
seis_data = seis_data[:, :]
