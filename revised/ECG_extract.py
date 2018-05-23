import numpy as np
import util as utl
import matplotlib.pyplot as plt
import sys

from biosppy.signals.ecg import christov_segmenter
from biosppy.signals import tools as st
from scipy.signal import filtfilt, butter, lfilter, detrend


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

###########################
## parameter specification
###########################
sampling_rate = 128.
signal        = data_ecg[1]
order         = 5
cutoff        = 17

# low pass filter
filtered = butter_lowpass_filter(signal, cutoff, sampling_rate, order)

# detrend with a low-pass
order = 20
filtered -= filtfilt([1] * order, [order], filtered)  # this is a very simple moving average filter

