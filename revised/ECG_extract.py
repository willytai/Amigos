import numpy as np
import util as utl
import matplotlib.pyplot as plt
import sys

from biosppy.signals.ecg import engzee_segmenter, correct_rpeaks
from biosppy.signals import tools as st
from scipy.signal import filtfilt, butter, lfilter, detrend, welch, periodogram
from scipy.stats import skew, kurtosis


# two rows of ECG data
filename = sys.argv[1]
data_ecg = utl.load(filename)[:, 14:16].T

###################
## low pass filter
###################
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

####################
## root mean square
####################
def RMS(data):
	result = 0
	for i in range(len(data)):
		result += data[i]**2
	result /= len(data)
	result = np.sqrt(result)
	return result

#################################################
## filter out values out of n standard deviation
#################################################
def correct(data, n):
	std  = data.std()
	mean = data.mean()
	min  = mean - n*std
	max  = mean + n*std
	# assert min > data.min(), 'n is too large'
	correct = np.array(list(filter(lambda x : min < x and x < max, data)))
	# print ('range: ({},{})'.format(min,max))
	# print ('{} value removed'.format(len(data)-len(correct)))
	return correct

#########################################
## returns the index of a specific band
#########################################
def band2idx(freq, cutoff_low, cutoff_high):
	index = []
	for i in range(len(freq)):
		if freq[i] < cutoff_high and freq[i] >= cutoff_low:
			index.append(i)
	return np.array(index)

#################
## interpolation
#################
def interpolate(data):
	newPoints = []
	interpol  = []
	interpol.append(data[0])
	for i in range(1, len(data)):
		newPoints.append((data[i] + data[i-1])/2)
	for i in range(len(newPoints)):
		interpol.append(newPoints[i])
		interpol.append(data[i])
	return interpol

###########################
## parameter specification
###########################
feautres_left  = []
feautres_right = []
sampling_rate  = 128.
signals        = data_ecg
order          = 5
cutoff         = 17
rpeak_tol      = 0.36


# two channels of ECG signals
for i, signal in enumerate(signals):
	if i == 0:
		target = feautres_right
	else:
		target = feautres_left

	# low pass filter
	filtered = butter_lowpass_filter(signal, cutoff, sampling_rate, order)

	# detrend with a low-pass
	order = 20
	filtered -= filtfilt([1] * order, [order], filtered)  # this is a very simple moving average filter

	# Rpeaks
	Rpeaks = engzee_segmenter(signal=filtered, sampling_rate=sampling_rate, threshold=0)['rpeaks']
	Rpeaks = correct_rpeaks(signal=filtered, rpeaks=Rpeaks, sampling_rate=sampling_rate, tol=rpeak_tol)['rpeaks']

	# peak to peak intervals
	IBI    = []
	for i in range(1, len(Rpeaks)):
		IBI.append(Rpeaks[i] - Rpeaks[i-1])
	IBI = np.array(IBI) / sampling_rate
	IBI = correct(IBI, 3)

	# feautres for IBI
	target.append(RMS(IBI))
	target.append(IBI.mean())
	target.append(IBI.std())
	target.append(skew(IBI))
	target.append(kurtosis(IBI))
	target.append(utl.above_below_mean_std(IBI))

	# heart rate variability time series
	HR = 1 / IBI
	target.append(HR.mean())
	target.append(HR.std())
	target.append(HR.var())
	target.append(skew(HR))
	target.append(kurtosis(HR))
	target.append(utl.above_below_mean_std(HR))

	# spectral power for HRV low, medium, high, LF/HF
	# samples of HR is not enough, add more data by interpolation
	# sampling freq times twice
	HR       = interpolate(HR)
	freq, sp = welch(HR, 2/IBI.mean(), nperseg=len(HR), scaling='spectrum')
	sp_low   = np.sum(sp[band2idx(freq, 0.01, 0.08)])
	sp_mid   = np.sum(sp[band2idx(freq, 0.08, 0.15)])
	sp_high  = np.sum(sp[band2idx(freq, 0.15, 0.50)])
	target.append(sp_low)
	target.append(sp_mid)
	target.append(sp_high)
	target.append(sp_low/sp_high)

	# 60 pectral power of filtered singal
	# not sure of this
	freq, sp = periodogram(filtered, sampling_rate, window='hann', scaling='spectrum')
	for i in range(60):
		band = sp[band2idx(freq, 0.1*i, 1*(i+1))]
		target.append(band.sum())



# check_left = feautres_left[-60:]
# check_right = feautres_right[-60:]
# plt.plot(utl.norm(check_right))
# plt.plot(check_left)
# plt.show()
# file = open('ECG.csv', 'a+')
# file.write(',{}\n'.format(",".join(list(map(str, feautres_left)))))
# file.write(',{}\n'.format(",".join(list(map(str, feautres_right)))))
# file.write(',{}\n'.format(",".join(list(map(str, abs(np.array(feautres_right) - np.array(feautres_left)))))))