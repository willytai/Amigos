import numpy as np
import util as utl
import sys, math

import matplotlib.pyplot as plt
from biosppy.signals import tools as st
from scipy.signal import welch

filepath = '../database/'
filename = filepath+sys.argv[1]+'_'+sys.argv[2]+'.csv'
data_eeg = utl.load(filename)[:, 0:14]


######################################################
# calculates logarithm of the PSDs of the five bands #
# theta, slow_alpha, alpha, beta, gamma              #
# window size = 1.0s (128 samples)                   #
######################################################
signal      = data_eeg
rate        = 128.
window_size = 1.0
overlap     = .5


# high pass filter
b, a = st.get_filter(ftype='butter',
					 band='highpass',
					 order=8,
					 frequency=2,
					 sampling_rate=rate)

aux, _ = st._filter_signal(b, a, signal=signal, check_phase=True, axis=0)

# low pass filter
b, a = st.get_filter(ftype='butter',
					 band='lowpass',
					 order=16,
					 frequency=47,
					 sampling_rate=rate)

filtered, _ = st._filter_signal(b, a, signal=aux, check_phase=True, axis=0)

plt.plot(signal[:, 0])
plt.plot(filtered[:, 0])
plt.show(); sys.exit()


# returns the index of a specific band
def band2idx(freq, cutoff_low, cutoff_high):
	index = []
	for i in range(len(freq)):
		if freq[i] <= cutoff_high and freq[i] >= cutoff_low:
			index.append(i)
	return np.array(index)


features = []

for i in range(7):

	# this returns the spectral power
	freq_sp, SP_l = welch(filtered[:, i   ].reshape(-1), rate, nperseg=128, noverlap=64, scaling='spectrum')
	_,       SP_h = welch(filtered[:, 13-i].reshape(-1), rate, nperseg=128, noverlap=64, scaling='spectrum')

	######################################
	# window size = 128, 50% overlapping
	######################################

	# this returns the power spectral density
	freq_psd, PSD = welch(filtered[:, i].reshape(-1), rate, nperseg=128, noverlap=64, scaling='density')


	theta      = math.log10(np.average(PSD[band2idx(freq_psd, 3 , 7 )]))
	slow_alpha = math.log10(np.average(PSD[band2idx(freq_psd, 8 , 10)]))
	alpha      = math.log10(np.average(PSD[band2idx(freq_psd, 8 , 13)]))
	beta       = math.log10(np.average(PSD[band2idx(freq_psd, 14, 29)]))
	gamma      = math.log10(np.average(PSD[band2idx(freq_psd, 30, 47)]))

	features   += [theta, slow_alpha, alpha, beta, gamma]

	# this returns the power spectral density
	_,    PSD  = welch(filtered[:, 13-i].reshape(-1), rate, nperseg=128, noverlap=64, scaling='density')


	theta      = math.log10(np.average(PSD[band2idx(freq_psd, 3 , 7 )]))
	slow_alpha = math.log10(np.average(PSD[band2idx(freq_psd, 8 , 10)]))
	alpha      = math.log10(np.average(PSD[band2idx(freq_psd, 8 , 13)]))
	beta       = math.log10(np.average(PSD[band2idx(freq_psd, 14, 29)]))
	gamma      = math.log10(np.average(PSD[band2idx(freq_psd, 30, 47)]))

	features   += [theta, slow_alpha, alpha, beta, gamma]


	# calculate asymmetry
	high       = np.sum(SP_h[band2idx(freq_sp, 3 , 7 )])
	low        = np.sum(SP_l[band2idx(freq_sp, 3 , 7 )])
	theta      = (high - low) / (high + low)

	high       = np.sum(SP_h[band2idx(freq_sp, 8 , 10)])
	low        = np.sum(SP_l[band2idx(freq_sp, 8 , 10)])
	slow_alpha = (high - low) / (high + low)

	high       = np.sum(SP_h[band2idx(freq_sp, 8 , 13)])
	low        = np.sum(SP_l[band2idx(freq_sp, 8 , 13)])
	alpha      = (high - low) / (high + low)

	high       = np.sum(SP_h[band2idx(freq_sp, 14, 29)])
	low        = np.sum(SP_l[band2idx(freq_sp, 14, 29)])
	beta       = (high - low) / (high + low)

	high       = np.sum(SP_h[band2idx(freq_sp, 30, 47)])
	low        = np.sum(SP_l[band2idx(freq_sp, 30, 47)])
	gamma      = (high - low) / (high + low)

	features   += [theta, slow_alpha, alpha, beta, gamma]


label = np.genfromtxt('label.csv', delimiter=',')[:, :2]
participant = int(sys.argv[1])
video = int(sys.argv[2])
label = label[video+16*(participant-1)-1, :]

features = np.array(features)

file = open('EEG.csv', 'a+')
file.write(sys.argv[1]+'_'+sys.argv[2])
for i in range(len(features)):
	file.write(',{}'.format(features[i]))
file.write(',{},{}'.format(label[0], label[1]))
file.write('\n')
file.close()
