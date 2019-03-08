# common conversion point stacking

import CCP_plottingroutines as CCP_plot
import glob
import matplotlib.pyplot as plt
import numpy as np
import sys
import matplotlib.pylab as pylab


params = {'legend.fontsize': 'x-large',
          'figure.figsize': (15, 5),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)

name = 'CCP_WUS'
rffilter = 'jgf1'
conversion = 'prem'
factor = 2.0


CCP = CCP_plot.ccp_volume()
CCP.load_latest(
    name=name,
     filter=rffilter,
     conversion=conversion,
     factor=factor)

print('done loading')

# retrieve and plot profiles
#locs = [[-110., 34.], [-115.,40.],[-110.,44.]]
def retrieve_profiles(locs, min_depth, max_depth):
	d, p = CCP.retrieve_profile(locs[0][0], locs[0][1], min_depth, max_depth)
	profiles = np.zeros((len(locs), len(p)))
	count = 0
	for loc in locs:
		d1, p1 = CCP.retrieve_profile(loc[0], loc[1], min_depth, max_depth)
		profiles[count] = p1
		count +=1
	return d, profiles



### Below are various commands you can uncomment to create plots
## plot data coverage at the 660
#CCP.plot_datacoverage(660,name=name,filter=rffilter, conversion=conversion, factor=factor)

## Plot a north-south ('NS') or east-west ('EW') cross section as a specific latitude or longitdue
#CCP.plot_crosssection('NS',-155,amplify=1.,name=name,filter=rffilter, conversion=conversion, factor=factor,zoom=False, mincoverage=2.)

## Plot a random oriented cross section (slower than the regular one above)
#CCP.plot_crosssection_any(lon1=-157,lon2=-140,lat1=69,lat2=57,numpoints=42,amplify=0.4,mincoverage=40.,conversion=conversion, zoom=True)

## Plot topography on the 410, picked between 380 and 470
#CCP.plot_topography(380, 470,name=name,filter=rffilter,conversion=conversion,factor=factor,mincoverage=40., amplitude = False, blobs = False)

## Plot width of the mtz (660-410)
#CCP.plot_mtzwidth(filter=rffilter, conversion=conversion, factor=factor, mincoverage=40.)


#plt.show()
