from Stacker import Stacker
import clustering_scripts as cs
import numpy as np
import os
import plotting_scripts as ps

class Depth_Stacker(Stacker):

	def plot(self, indiv=True, anomalies = True):
		
		print("Plotting...")
		
		os.mkdir(self.filename)
		os.chdir(self.filename)
		ps.plot(self.stacks, self.x_var, self.cluster, self.coords, self.seis_data, self.filename, anomal = anomalies, plot_individual = indiv)
		ps.depth_plot(self.cluster, self.stacks, self.coords, self.x_var, self.min+10, self.max-10, self.filename+'_depths')
		os.chdir('..')

		return

	def __init__(self, x_var, coords, seis_data, filename, min, max):
		
		self.x_var = x_var
		self.coords = coords
		self.seis_data = seis_data
		self.filename = filename
		self.stacks = seis_data
		self.cluster = np.arange(1, len(seis_data)+1)
		self.min = min
		self.max = max

