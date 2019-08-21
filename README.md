## Cluster Stacking

A series of scripts methods designed to cluster receiver functions via hierarchical clustering methods, and then stack the resulting data. There are no special dependencies apart from basic python libraries and matplotlib for plotting.

## Manual For Use

- Start by formatting your data correctly. For the code to run, you need the following data:
1. An array of all seismic traces/RFs (designated `seis_data`), in an NxM numpy array, where N is the number of observations and M is the data points per observation. 
2. An array of grographical coordinates (`coords`) to which each trace corresponds, in a Nx2 numpy array. The first coordinate should correspond to the first trace, and so on. The coordinates should go lat, long rather than long, lat.
3. An array of values that go along the x-axis (`x_var`), eg differential time as measured by a seismometer, or depth, if a velocity model has been used to do a depth conversion. This should be an Mx1 numpy array, where M is the number of data points per observation.

- If you want, subclass the existing Stacker class [eg `class Adapted_Stacker(Stacker):`] to add new variables or information. This also allows you to override the `.plot()` function to specify exactly what plots you want.
Available plots include temperature based on MTZ thickness, depth of specified peak, etc.

- After this, create an instance of the Stacker class and pass arguments 1, 2 and 3 and an output filename to the constructor.

- Execute an adaptive stacking protocol using `.adaptive_stack()`

- Plot results using `.plot()`. The default arguments will result in a set of individual plots and an overall plot of all clusters, along with any other plots you have chosen, if you subclass `Stacker.py`.

- Make any other plots. Plotting scripts usually take the stacking object as a first argument (`s_o`).

## Variables in the Stacker object

- coords: array of all geographical coordinates of the data

- cluster: array assigning every coordinate to a cluster. Starts with all of them in separate clusters (eg, [1, 2, 3, 4,....])

- stacks: array containing all the stacked data corresponding to each cluster. Array is the length of the number of clusters, so gets smaller as the number of clusters go down.

- seis_data: array containing the receiver function corresponding to every coodinate

- x_var: array of the x variable corresponding to each receiver function (eg, depth)

- cluster_keep: This is a variable that keeps track of all data deleted in the 'anomaly removing' stages of the stacking. This variable is updated in each stacking step and can be used to compare different clustering results to each other at the end of the procedure, even though they might have different data deleted from them. This variable should always maintain the same length as at the start, and all the deleted data has a zero placed to correspond to it.

## Available routines

- .adaptive_stack() runs a basic adaptive stack on a Stacker object

- .stability_test() runs a bootstrapping procedure to test the robustness of the clustering and produce a cluster vote map to illustrate this

- If you want to compare two different clustering results numerically, there is a separate script for calculating the `rand score`, that takes the `cluster_keep` variables of two stacker objects as its arguments

## Available Plots

### Default Plot

- The default .plot function available in plotting scripts plots all the final stacked waveforms to the left and their corresponding clusters on a map to the right, in corresponding colours

- While it gives a good overview of how the data ended up being clustered, it's not necessarily good for gleaning information for the resulting stacked traces

### Heat maps

- There is a .plot_heatmap() function that generates a geographical heatmap given an array of coordinates (ie `coords`) and any chosen value corresponding to those coordinates.

- In the script at the moment, there are functions to find mantle transition zone width, the magnitude of the highest peak on the stack, etc, and plot them as heat maps.

### Cluster Vote Map

- This is a visualisation method for how 'robust' a certain set of clusters are to small changes.

- The basic idea is to introduce some randomness into the clustering (eg, by removing a small number of random data points) and then see which areas are most stable by allowing each run to ‘vote’.

- Unlike a conventional vote map, there are no objective definitions of clusters, so it’s hard to compare on a point by point basis

- Start with a 'base' set of clusters, and compare all others to them

- The final implementation is based on the ‘rand index’ - you add sum of all points in the same cluster across runs, plus all points in different clusters across runs

- Then normalise and visualise how 'robust' the clustering of each point was using colour saturation
 
### Interpolation

- This is a method intended to 'smooth' out sharp edges in the clusters and get an overall look at the topographical variation

- It uses the SciPy 2-D interpolation function to interpolate over a regular grid and plots the results

## Other notes

- The `cluster` variable is formatted in accordance with scipy's method, eg it is a Nx1 array, with each entry corresponding to which cluster every data point from `seis_data` is placed in.
The clusters are numbered from 1. For example, diving six points into three clusters might look like [1, 2, 1, 3, 2, 3]. A final cluster where each point was in their own individual cluster would look like [1, 2, 3, 4, 5, 6]. 

- The `remove_anoms` method is intended to reduce noise and increase stability.

- The `stability_test` function in `clustering_scripts` is essentially a cluster 'vote map'. It removes a small amount of the data and runs the program a given number of times. It then uses a metric similar to 'rand score' to calculate the reliability with which you can say a point belongs in a given cluster. It then plots the clusters with colour saturation corresponding to level of confidence. The function requires a 'base' set of clusters to do this calculation.
