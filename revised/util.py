import numpy as np
import matplotlib.pyplot as plt
import sys, math
# import peakutils
from scipy.signal import butter, lfilter
from scipy.stats import skew, kurtosis

def load(filename):
	print ("loading from {}......".format(filename), end='')
	data = np.genfromtxt(filename, delimiter=',')
	print ("done")
	return data

def DB(arr):
	for i in range(len(arr)):
		arr[i] = math.log10(arr[i])*20
	return arr

def peak(signal):
	# index = peakutils.indexes(signal, thres=0.0, min_dist=50)
	print (index); sys.exit()
	return index

def onsets(signal):
	# index = peakutils.indexes(-1*signal, thres=0.0, min_dist=50)
	print (index)
	return index

def amp(onsets, peaks, signal):
	return None

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def butter_bandpass(fh, fl, fs, order=5):
    nyq = 0.5 * fs
    low = fl / nyq
    high = fh / nyq
    b, a = butter(order, [low, high], btype='band', analog=False)
    return b, a

def butter_bandpass_filter(data, fh, fl, fs, order=5):
    b, a = butter_bandpass(fh, fl, fs, order=order)
    y = lfilter(b, a, data)
    return y

def crossing(signal, ref):
	count = 0
	test = signal - ref
	for i in range(1, len(signal)):
		if test[i] * test[i-1] < 0:
			count += 1
	return count / len(signal)

def above_below_mean_std(signal):
	std        = np.std(signal)
	mean       = np.mean(signal)
	bound_high = mean + std
	bound_low  = mean - std
	count      = 0
	for i in range(len(signal)):
		if bound_low <= signal[i] and signal[i] <= bound_high:
			count += 1
	return (1 - count / len(signal)) * 100

def stat(data):
	mean = np.mean(data)
	std  = np.std(data)
	Skew = skew(data)
	kurt = kurtosis(data)
	last = above_below_mean_std(data)
	return [mean, std, Skew, kurt, last]

def norm(data):
	data = np.array(data)
	std = data.std()
	mean = data.mean()
	for i in range(len(data)):
		data[i] = (data[i] - mean) / std
	return data