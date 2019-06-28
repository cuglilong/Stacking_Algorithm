# Cluster Stacking

A series of scripts methods designed to cluster seismology data via hierarchical clustering methods, and then stack the resulting data.

# Manual

- Format your data correctly. For the code to run, you need the following data:
1. An array of all seismic traces (designated `seis_data`), in an NxM numpy array, where N is the number of observations and M is the data points per observation. 
2. An array of grographical coordinates (`coords`) to which each trace corresponds, in a Nx2 numpy array. The first coordinate should correspond to the first trace, and so on. The coordinates should go lat, long rather than long, lat.
3. An array of values that go along the x-axis (`x_var`), eg differential time as measured by a seismometer, or depth, if a velocity model has been used to do a depth conversion. This should be an Mx1 numpy array, where M is the number of data points per observation.

- After this, create a stacker class and pass these arguments along with an output filename to the constructor.

- Execute an adaptive stacking protocol using .adaptive_stack()

- Plot results using .plot(). The default arguments will result in a set of individual plots and an overall plot of all clusters.

- Other notes: The `cluster` variable is formatted in accordance with scipy's method, eg it is a Nx1 array, with each entry corresponding to which cluster every data point from `seis_data` is placed in.
The clusters are numbered from 1. For example, diving six points into three clusters might look like [1, 2, 1, 3, 2, 3]. A final cluster where each point was in their own individual cluster would look like [1, 2, 3, 4, 5, 6]. 
