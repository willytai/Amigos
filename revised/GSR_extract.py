import sys, math
import util as utl
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, detrend, welch, periodogram
from peakutils import indexes

participant = sys.argv[1]
video       = sys.argv[2]
filename    = '../database/'+participant+'_'+video+'.csv'
print (filename)

######################
## zero crossing rate
######################
def ZeroCR(waveData, frameSize, overLap):
    wlen = len(waveData)
    step = frameSize - overLap
    frameNum = math.ceil(wlen/step)
    zcr = np.zeros((frameNum,1))
    for i in range(frameNum):
        curFrame = waveData[np.arange(i*step,min(i*step+frameSize,wlen))]
        zcr[i] = sum(curFrame[0:-1]*curFrame[1::]<=0)
    return zcr


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

#################################################
## filter out values out of n standard deviation
#################################################
def correct(data, n):
	std  = data.std()
	mean = data.mean()
	minn = mean - n*std
	maxx = mean + n*std
	# assert min > data.min(), 'n is too large'
	correct = np.array(list(filter(lambda x : minn < x and x < maxx, data)))
	# print ('range: ({},{})'.format(minn, maxx))
	# print ('{} value removed'.format(len(data)-len(correct)))
	return correct

##################
## normalization
##################
def norm(data):
	std  = data.std()
	mean = data.mean()
	return (data - mean) / std

#########################################
## returns the index of a specific band
#########################################
def band2idx(freq, cutoff_low, cutoff_high):
	index = []
	for i in range(len(freq)):
		if freq[i] < cutoff_high and freq[i] >= cutoff_low:
			index.append(i)
	return np.array(index)

###########################
## parameter specification
###########################
features      = []
# header        = []
data_gsr      = utl.load(filename)[:, -1]
sampling_rate = 128.
signal        = data_gsr[:]


# remove noise
signal = butter_lowpass_filter(signal, 3, sampling_rate)

# remove weird data
signal = correct(signal, 2.5)

# mean SR signal
features.append(signal.mean())
# header.append('mean_GSR')

# derivative
diff = np.diff(signal)

# mean of first and second derivative
features.append(diff.mean())
features.append(np.diff(diff).mean())
# header.append('mean_diff1')
# header.append('mean_diff2')

# mean diff for neg value, proportion of neg
count = 0
mean_dif_neg = 0
for i in diff:
	if i < 0:
		count += 1
		mean_dif_neg += i
mean_dif_neg /= count

features.append(abs(mean_dif_neg))
features.append(count/len(diff))
# header.append('mean_diff_neg')
# header.append('pro_neg')

# local minima, ave rise
signal_ex = detrend(signal)
test_min  = []
test_max  = []
local_max = indexes( signal_ex, thres=0.23, min_dist=300)
local_min = indexes(-signal_ex, thres=0.23, min_dist=300)

for minn, min_id in enumerate(local_min):
	if len(local_max) == 0:
		break;

	while local_max[0]-300 < min_id:
		local_max = local_max[1:]
		if len(local_max) == 0:
			break
	if len(local_max) == 0:
		break

	if minn < len(local_min)-1 and local_max[0] > local_min[minn+1]:
		continue

	# local max < local min, skip
	if signal_ex[local_max[0]] - signal_ex[min_id] < 0:
		continue

	test_min.append(min_id)
	test_max.append(local_max[0])
	local_max = local_max[1:]

test_min = np.array(test_min)
test_max = np.array(test_max)
ave_rise = test_max - test_min
ave_rise = ave_rise.mean()
features.append(len(test_max))
features.append(ave_rise)
# header.append('local_min')
# header.append('ave_rise')

# spectral power 0-2.4
signal_p = butter_lowpass_filter(signal, 2.4, sampling_rate, order=5)
signal_p = norm(signal_p)
signal_p = detrend(signal_p)
freq, sp = periodogram(signal_p, sampling_rate, window='hann', scaling='spectrum')

for band in range(12):
	features.append(np.sum(sp[band2idx(freq, band*0.12, band*0.12+0.12)]))
	# header.append('sp 0-2.4')

# skin conductance, 
SC = 1 / signal * 100000
features.append(SC.mean())
features.append(SC.std())
# header.append('mean_SC')
# header.append('std_SC')

# two lowpassed signals
SC_lowpassed  = butter_lowpass_filter(SC, 0.2,  sampling_rate, order=5)
SC_vlowpassed = butter_lowpass_filter(SC, 0.08, sampling_rate, order=5)

# stats for SCSR
features.append(SC_lowpassed.mean())
features.append(SC_lowpassed.std())
# header.append('mean_SCSR')
# header.append('std_SCSR')

# diff1, diff2 of SCSR
diff1 = np.diff(SC_lowpassed)
diff2 = np.diff(diff1)
features.append(diff1.mean())
features.append(diff2.mean())
# header.append('SCSR_diff1')
# header.append('SCSR_diff2')

# stats for SCVSR
features.append(SC_vlowpassed.mean())
features.append(SC_vlowpassed.std())
# header.append('mean_SCVSR')
# header.append('std_SCVSR')

# diff1, diff2 of SCVSR
diff1 = np.diff(SC_vlowpassed)
diff2 = np.diff(diff1)
features.append(diff1.mean())
features.append(diff2.mean())
# header.append('SCVSR_diff1')
# header.append('SCVSR_diff2')

# zero crossing rate for SCSR and SCVSR
SCSR_detrend  = detrend(SC_lowpassed)
SCVSR_detrend = detrend(SC_vlowpassed)
ZC_SCSR       = ZeroCR(SCSR_detrend,  1000, 0).reshape(-1).mean()
ZC_SCVSR      = ZeroCR(SCVSR_detrend, 1000, 0).reshape(-1).mean()

# magnitude of SCSR and SCVSR peak
peak_scsr  = indexes(SC_lowpassed,  thres=0.7, min_dist=500)
peak_scvsr = indexes(SC_vlowpassed, thres=0.7, min_dist=500)
features.append(SC_lowpassed[peak_scsr].mean())
features.append(SC_vlowpassed[peak_scvsr].mean())
# header.append('ZC_SCSR')
# header.append('ZC_SCVSR')

# get the label
label = np.genfromtxt('label.csv', delimiter=',')[:, :2]
participant = int(sys.argv[1])
video = int(sys.argv[2])
label = label[video+16*(participant-1)-1, :]
participant = str(participant)
video = str(video)

# write to file
file = open('GSR.csv', 'a+')
file.write('{},{},{},{}\n'.format(participant+'_'+video, ",".join(list(map(str, features))), label[0], label[1]))