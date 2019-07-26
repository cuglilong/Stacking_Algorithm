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
import Stacker
import plotting_scripts as ps
from copy import deepcopy

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
	
	cls = set(cluster.astype(int))
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
	for cl in set((cluster).astype(int)):
        	new_cluster[np.where(cluster == cl)[0]] = cluster_2[cl-1].astype(int)
	cluster = new_cluster.astype(int)
	
	# Cutting coordinates off the stacks
	
	stacks = stacks[:, :-2]
	
	return cluster, stacks

def stability_test(s_o, no_trials):
	
	s = np.array([])
	base_class = s_o.__class__
	
	for i in np.arange(no_trials):
		rand_remove = np.random.choice(range(len(s_o.seis_data)), round(len(s_o.seis_data)/50), replace=False)
		temp_data = np.delete(s_o.seis_data, rand_remove, axis=0)
		temp_coords = np.delete(s_o.coords, rand_remove, axis=0)
		temp_cluster_keep = np.arange(1, len(s_o.seis_data)+1)
		temp_cluster_keep[rand_remove] = 0
		ss = base_class(s_o.x_var, temp_coords, temp_data, 'test'+str(i))
		ss.cluster_keep = temp_cluster_keep
		s = np.append(s, ss)
	
	s_o.adaptive_stack()
	s_o.plot(indiv=False)
	for test in s:
		test.adaptive_stack()
	
	vote_map = ps.cluster_vote_map(s_o, s)
	ps.plot(s_o, s_o.filename, vote_map=vote_map, plot_individual = False)
	
	return
