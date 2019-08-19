import numpy as np
import clustering_scripts as cs

def c_v(cluster, cl, seis_data):

	waveforms = seis_data[np.where(cluster==cl)]
	average = np.array([np.average([waveforms[i][j] for i in range(len(waveforms))]) for j in range(len(waveforms[0]))])
	a = 0
	for wave in waveforms:
		a+=cs.correlation(wave, average)

	return a/(len(waveforms))
