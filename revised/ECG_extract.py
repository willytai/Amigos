import numpy as np
import util as utl
import matplotlib.pyplot as plt
import sys

from biosppy.signals.ecg import engzee_segmenter, correct_rpeaks
from biosppy.signals import tools as st
from scipy.signal import filtfilt, butter, lfilter, detrend
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

###########################
## parameter specification
###########################
features      = []
sampling_rate = 128.
signal        = data_ecg[1]
order         = 5
cutoff        = 17
rpeak_tol     = 0.36

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

# append features for IBI
features.append(RMS(IBI))
features.append(IBI.mean())
features.append(IBI.std())
features.append(skew(IBI))
features.append(kurtosis(IBI))

# heart rate series
HR = 1 / IBI
features.append(HR.mean())
features.append(HR.std())
features.append(skew(HR))
features.append(kurtosis(HR))


