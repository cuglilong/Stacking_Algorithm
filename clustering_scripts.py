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
	
	cls = np.arange(1,np.max(cluster).astype(int)+1)
	coords = np.array([[np.average([coords[i][j] for i in np.where(cluster == cl)[0]]) for j in [0, 1]] for cl in cls])
	coords = np.nan_to_num(coords)
	
	return coords

# Clustering and stacking data
# Clusters using distances between data in b (b is combined trace data and coordinates)
# Then stacks the data in a according to resulting clusters (a is just trace data)
# Threshold is clustering threshold, crit is clustering criterion
# Returns stacked data and list of cluster indices

def cluster_data(a, b, metric, threshold, crit):
	
	# Clustering and stacking
	
	cluster = sp.cluster.hierarchy.fclusterdata(b, t=threshold, criterion=crit, metric=metric, method='average')
	stacks = np.array([[np.sum([a[i][j] for i in np.where(cluster == cl+1)[0]]) for j in range(len(a[0]))] for cl in range(np.max(cluster))])
	
	# Renomalising stacks
	
	for cl in range(1, np.max(cluster+1)):
		length = len(np.where(cluster == cl)[0])
		stacks[cl-1] = stacks[cl-1]/length
	stacks = np.nan_to_num(stacks)

	return cluster, stacks

# Wrapper function for the clusterer
# Returns cluster data extended to correspond to each individual data point

def second_cluster(cluster, coords, stacks, threshold, crit, dist = True, corr = False):
	
	# Formatting data - adding coordinates on end of trace data and creating 'distance' metric
	comb_metric = (lambda a, b: combined_metric(a, b, include_dist=dist, include_corr=corr))
	new_data = np.append(stacks, coords, axis=1)
	
	# Clustering data
	
	cluster_2, stacks = cluster_data(new_data, new_data, comb_metric, threshold, crit)
	
	# Reformatting cluster to correspond to original data points
	
	new_cluster = np.zeros(len(cluster))
	for cl in np.arange(1, np.max(cluster).astype(int)+1):
        	new_cluster[np.where(cluster == cl)[0]] = cluster_2[cl-1].astype(int)
	cluster = new_cluster.astype(int)
	
	# Cutting coordinates off the stacks
	
	stacks = stacks[:, :-2]
	
	return cluster, stacks
