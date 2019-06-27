import numpy as np
import clustering_scripts as cs

def c_v(cluster, cl, seis_data):

	waveforms = seis_data[np.where(cluster==cl)]
	average = [np.average([i for i in waveforms[j]) for j in range(len(wavforms))]
	a = 0
	for wave in waveforms:
		a+=cs.correlation(wave, average)

	return a/len(waveforms)
