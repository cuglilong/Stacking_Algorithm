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

class Stacker:
	
	# Write out cluster and stacks to a csv

	def print_out(self, print_file):
		
		# Print out stacks and cluster
		
		file = open(print_file, 'w')
		for c in self.cluster:
			file.write(str(c) + ', ')
		file.write('\n')
		count = 1
		for stack in self.stacks:
			for s in stack:
				file.write(str(s) + ', ')
			count += 1
			if (count != len(self.stacks)+1):
				file.write('\n')
		file.close()
		
		# Print out coordinates
		
		file1 = open(print_file+'_coords', 'w')
		count = 1
		for coord in self.coords:
			file1.write(str(coord[0])+', '+ str(coord[1]))
			count +=1
			if (count != len(self.coords)+1):
				file1.write('\n')
		file1.close()

		return

	# Read in cluster and stacks from a csv - input parameters are
	# no_stacks = number of stacks to read in
	# no_coords = number of coordinates to read in
	# len_data = number of data points in a stack - eg, number of depth values

	def read_in(self, no_stacks, no_coords, len_data, read_file):
		
		# Reading in cluster and stack data
		
		with open(read_file) as csv_file:
			self.stacks = np.zeros((no_stacks, len_data))
			csv_reader = csv.reader(csv_file, delimiter=',')
			line_count = 0
			for row in csv_reader:
				if line_count == 0:
					self.cluster = row
					self.cluster = np.array(self.cluster[:-1]).astype(float)
					line_count+=1
				else:
					self.stacks[line_count-1] = np.array(row[:-1]).astype(np.float)
					line_count += 1
		
		# Reading in coordinate data
		
		with open(read_file+'_coords') as csv_file_1:
			self.coords = np.zeros((no_coords, 2))
			csv_reader_1 = csv.reader(csv_file_1, delimiter=',')
			line_count = 0
			for row in csv_reader_1:
				self.coords[line_count] = np.array(row).astype(float)
				line_count += 1
		
		return

	# Remove all stacks containing fewer data points than a given size from the clusterer

	def remove_anoms(self, stack_size):
		
		# Making list of those to remove
		
		to_remove = np.array([])
		remove_cluster = np.array([])
		for c in np.arange(1, np.max(self.cluster)+1):
			a = np.where(self.cluster == c)[0]
			
			if (len(a) < stack_size):
				to_remove = np.append(to_remove, [c-1], axis=0)
				remove_cluster = np.append(remove_cluster, a, axis=0)
		
		# Removing all anomalies
		
		self.stacks = np.delete(self.stacks, to_remove.astype(int), axis=0)
		self.cluster = np.delete(self.cluster, remove_cluster.astype(int))
		self.coords = np.delete(self.coords, remove_cluster.astype(int), axis=0)
		self.seis_data = np.delete(self.seis_data, remove_cluster.astype(int), axis=0)
		
		# Rebuilding cluster
		
		cluster_set = np.unique(self.cluster)
		count = 1
		for c in cluster_set:
			self.cluster[np.where(self.cluster==c)[0]] = count
			count += 1
		
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

	# Returns average size of stack in current cluster conformation

	def average_stack_size(self):
		
		avg = np.average([len(np.where(self.cluster == i)[0]) for i in np.unique(self.cluster)])
		
		return avg
	
	# Returns average cluster variance of a stack in current cluster conformation
	
	def average_cluster_variance(self):
		
		for cl in range(1, np.max(self.cluster)):
			b = c_v(self.cluster, cl, self.seis_data)
			a+=b
		
		return(a/np.max(cluster))
	
	# Stacks current input data using an adaptive stacking routine
	
	def adaptive_stack(self):
		
		print("Stacking...")
		
		cut_length = round(len(self.seis_data)/10)
		self.cluster, self.stacks = cs.second_cluster(self.cluster, self.coords, self.stacks, threshold=cut_length, crit='maxclust', dist=True, corr=False)
		self.remove_anoms(5)
		while len(self.stacks) > 30:
			self.cluster, self.stacks = cs.second_cluster(self.cluster, cs.stack_coords(self.cluster, self.coords), self.stacks, threshold=1, crit='inconsistent', dist=True, corr=True)
			self.remove_anoms(round(self.average_stack_size()/4))
		print(len(self.stacks))
		print(len(self.coords))
	
		return 

	# Plots current stacks and other graphsand saves in directory of name filename
	# Indiv = boolean variable, if true then plot individual stacks, otherise don't
	
	def plot(self, indiv=True, anomalies = True):
	
		print("Plotting...")
		
		os.mkdir(self.filename)
		os.chdir(self.filename)
		ps.plot(self.stacks, self.x_var, self.cluster, self.coords, self.seis_data, self.filename, anomal = anomalies, plot_individual = indiv)
		return

	cluster = np.array([])
	seis_data = np.array([])
	stacks = np.array([])
	coords = np.array([])
	x_var = np.array([])

	# Class constructor

	def __init__(self, x_var, coords, seis_data, filename):

		self.x_var = x_var
		self.coords = coords
		self.seis_data = seis_data
		self.filename = filename
		self.stacks = seis_data
		self.cluster = range(1, len(seis_data)+1)

