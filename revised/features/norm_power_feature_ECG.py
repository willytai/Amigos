import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

###################################
# this file creates the normalized 
# trainig data.
# preprocessing isn't needed if 
# this data is used as the training
# data
###################################


def load(filename):
	data    = pd.read_csv(filename)
	right   = np.arange(1, 1057, 2) # right channel
	left    = np.arange(0, 1056, 2) # left channel
	label   = data.iloc[right, 77:]
	train_r = data.iloc[right, 1:77]
	train_l = data.iloc[left,  1:77]
	return np.array(train_r), np.array(train_l), np.array(label), list(data)

right, left, label, header = load('ECG.csv')


def norm(data):
	mean = data.mean()
	std  = data.std()
	return (data - mean) / std

for i in range(right.shape[0]):
	right[i][17:77] = norm(right[i][17:77])

for i in range(left.shape[0]):
	left[i][17:77] = norm(left[i][17:77])

result = right + left
result/= 2


col = np.arange(0, 17, 1)
fuck = result[:, col]
mean = fuck.mean(axis=0)
maxx = fuck.max(axis=0)
minn = fuck.min(axis=0)
for i in range(fuck.shape[0]):
	fuck[i] = (fuck[i] - mean) / (maxx - minn)
result[:, col] = fuck

for i in range(result.shape[0]):
	result[i] = list(map(str, result[i]))

video       = np.arange(1, 17, 1)
participant = [1,2,3,4,5,6,7,8,10,11,13,14,15,16,17,18,19,20,25,26,27,28,29,30,31,32,34,35,36,37,38,39,40]

file = open('ECG_norm_power.csv', 'w+')
file.write(",{}\n".format(",".join(header[1:])))

for i, par in enumerate(participant):
	for j, v in enumerate(video):
		tmp = list(result[i*16+j])
		tmp = list(map(str, tmp))
		file.write("{}_{},{},{},{}\n".format(par, v, ",".join(tmp), str(label[i*16+j][0]), str(label[i*16+j][1])))