import plotting_scripts as ps
import clustering_scripts as cs
from obspy import read
import random
import matplotlib.pyplot as plt
import sys
import os
import Stacker

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
from cluster_variation import c_v
#import plot_CCP

class Stacker_Test:
	
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

	def remove_anoms(self, anom_threshold, variance=False):
		
		# Making list of those to remove
		
		remove_cluster = np.array([])

		if variance == False:
			for c in set(self.cluster):
				a = np.where(self.cluster == c)[0]
				if (len(a) < anom_threshold):
					remove_cluster = np.append(remove_cluster, a, axis=0)
		else:
			for c in range(len(self.seis_data)):
				stack = self.stacks[self.cluster[c]-1]
				if cs.correlation(self.seis_data[c], stack) > anom_threshold:
					remove_cluster = np.append(remove_cluster, c)
		
		# Removing all anomalies
		
		self.cluster = np.delete(self.cluster, remove_cluster.astype(int))
		self.coords = np.delete(self.coords, remove_cluster.astype(int), axis=0)
		self.seis_data = np.delete(self.seis_data, remove_cluster.astype(int), axis=0)
		
		# Rebuilding clusters
		
		cluster_set = np.unique(self.cluster)
		count = 1
		for c in cluster_set:
			self.cluster[np.where(self.cluster==c)[0]] = count
			count += 1
		
		# Rebuilding keep cluster to use as comparison to other clusters at the end
		
		i = 0
		j = 0
		for c in np.arange(len(self.cluster_keep)):
			if (self.cluster_keep[c] == 0):
				pass
			elif(np.isin(i, remove_cluster)):
				self.cluster_keep[c]=0
				i+=1
			else:
				self.cluster_keep[c]=self.cluster[j]
				j+=1
				i+=1
		
		# Restacking
		
		stacks = np.array([[np.sum([self.seis_data[i][j] for i in np.where(self.cluster == cl+1)[0]]) for j in range(len(self.seis_data[0]))] for cl in range(np.max(self.cluster))])
		for cl in set(self.cluster):
			length = len(np.where(self.cluster == cl)[0])
			stacks[cl-1] = stacks[cl-1]/length
		self.stacks = np.nan_to_num(stacks)
		
		return

	# Returns average size of stack in current cluster conformation

	def average_stack_size(self):
		
		avg = np.average([len(np.where(self.cluster == i)[0]) for i in np.unique(self.cluster)])
		
		return avg
	
	# Returns average cluster variance of a stack in current cluster conformation
	
	def average_cluster_variance(self):
		
		a = 0
		for cl in set(self.cluster.astype(int)):
			b = c_v(self.cluster, cl, self.seis_data)
			a+=b
		
		return(a/np.max(self.cluster))
	
	# Stacks current input data using an adaptive stacking routine
	
	def adaptive_stack(self, geographical = False):
		
		print("Stacking...")
		
		cut_length = round(len(self.seis_data)/10)
		self.cluster, self.stacks = cs.second_cluster(self.cluster, self.coords, self.stacks, threshold=cut_length, crit='maxclust', dist=True, corr=False)
		#self.read_in(3847, 38468, 1040, 'large_data_stack')
		#self.print_out('large_data_stack')
		self.remove_anoms(5)
		while len(self.stacks) > 35:
			if geographical == True:
				self.cluster, self.stacks = cs.second_cluster(self.cluster, cs.stack_coords(self.cluster, self.coords), self.stacks, threshold=1, crit='inconsistent', dist=True, corr=False)
			else:
				self.cluster, self.stacks = cs.second_cluster(self.cluster, cs.stack_coords(self.cluster, self.coords), self.stacks, threshold=1, crit='inconsistent', dist=True, corr=True)
			self.remove_anoms(self.average_cluster_variance()*2, variance=True)
			self.remove_anoms(round(self.average_stack_size()/3))
			
		return

	# Plots current stacks and other graphsand saves in directory of name filename
	# Indiv = boolean variable, if true then plot individual stacks, otherise don't
	
	def plot(self, indiv=True):
	
		print("Plotting...")
		
		os.mkdir(self.filename)
		os.chdir(self.filename)
		ps.plot(self, self.filename, plot_individual=indiv)
		ps.MTZ_plot(self, self.filename+'_MTZ')
		os.chdir('..')

		return
	
	cluster = np.array([])
	seis_data = np.array([])
	stacks = np.array([])
	coords = np.array([])
	x_var = np.array([])
	cluster_keep = np.array([])

	# Class constructor

	def __init__(self, x_var, coords, seis_data, filename):

		self.x_var = x_var
		self.coords = coords
		self.seis_data = seis_data
		self.filename = filename
		self.stacks = seis_data
		self.cluster = np.arange(1, len(seis_data)+1)
		self.cluster_keep = np.arange(1, len(seis_data)+1)
