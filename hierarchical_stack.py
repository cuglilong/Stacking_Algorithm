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

def correlation(tr1, tr2):
	return np.sum([(tr1[i] - tr2[i])**2 for i in range(len(tr1))])

def haversine(coord1, coord2):

	lat1, long1 = coord1*(pi/180)
	lat2, long2 = coord2*(pi/180)
	del_lat = lat2 - lat1
	del_long = long2 - long1
	R = 6371
	a = sin(del_lat/2)**2 + cos(lat1)*cos(lat2)*(sin(del_long/2)**2)
	c = 2*atan2(sqrt(a), sqrt(1-a))

	return R*c

def combined_metric(a, b, include_dist = True, include_corr = False):
	if (include_corr == False):
		return haversine(a[-2:], b[-2:])
	elif (include_dist == False):
		return correlation(a[:-2],b[:-2])
	else:
		corr = correlation(a[:-2],b[:-2])
		dist = haversine(a[-2:], b[-2:])
		return corr * dist

def stack_coords(cluster, coords):

	clusters = np.arange(1,np.max(cluster).astype(int)+1)
	coords = np.array([[np.average([coords[i][j] for i in np.where(cluster == cl)[0]]) for j in [0, 1]] for cl in clusters])
	coords = np.nan_to_num(coords)

	return coords

def plot(stacks, depths, cluster, coords, figname, anomal=True):

	colours = []
	for i in range(1, np.max(cluster).astype(int) +1):
		a = '%06X' % randint(0, 0xFFFFFF)
		b = '#' + a
		colours.append(b)
	colour_clusters = [colours[(cluster[j]-1).astype(np.int)] for j in range(len(cluster))]
	plt.figure(1)
	
	plt.subplot(121)
	plt.xlabel('depth (km)')
	plt.ylabel('amplitude relative to main P wave')
	count = 0
	for stack in stacks:
		if (anomal == False and len(np.where(cluster == count+1)[0])<15):
			count+=1
			continue
		else:
			plt.plot(depths, stack, color = colours[count])
			count+=1
	
	plt.subplot(122)
	lon410 = np.array([co[1] for co in coordinates])
	lat410 = np.array([co[0] for co in coordinates])
	m = Basemap(llcrnrlon=np.min(lon410)-5.,llcrnrlat=np.min(lat410)-5.,urcrnrlon=np.max(lon410)+5.,urcrnrlat=np.max(lat410)+5.,
            resolution='i',projection='merc',lon_0=np.mean(lon410),lat_0=np.mean(lat410))
	m.shadedrelief()
	m.drawparallels(np.arange(-40,80.,5.),color='gray')
	m.drawmeridians(np.arange(-30.,80.,5.),color='gray')
	
	for i in range(len(coords)):
		lon, lat = m(coords[i][1], coords[i][0])
		m.plot(lon, lat, marker='x', markeredgecolor=colour_clusters[i])
	
	plt.show()
	plt.savefig(figname)

def cluster_data(a, b, metric, threshold, crit):
        cluster = sp.cluster.hierarchy.fclusterdata(b, t=threshold, criterion=crit, metric=metric, method='average')
        stacks = np.array([[np.sum([a[i][j] for i in np.where(cluster == cl)[0]]) for j in range(len(a[0]))] for cl in range(np.max(cluster))])
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

	return cluster, stacks

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
