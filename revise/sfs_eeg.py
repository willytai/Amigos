import numpy as np
import pandas as pd
import sys, pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

def load(filename):
	data  = pd.read_csv(filename)

	label = data.iloc[:,106:]
	train = data.iloc[:, 1:106]
	return np.array(train), np.array(label)

def norm(data):
	mean = data.mean(axis=0)
	max  = data.max(axis=0)
	min  = data.min(axis=0)
	for i in range(data.shape[0]):
		data[i] = (data[i] - mean ) / (max - min)
	return data

def transform(data, type=None, ref=0):
	if type == 'mean':
		mid_ref = np.mean(data, axis=0) # mean
	elif type == 'median':
		mid_ref = np.median(data, axis=0) # median
	elif type == '5':
		mid_ref = np.array([5, 5])
	else:
		mid_ref = ref
	
	for i in range(data.shape[0]):
		for col in range(data.shape[1]):
			if data[i][col] >= mid_ref[col]:
				data[i][col] = 1
			else:
				data[i][col] = 0
	return data, mid_ref

def main():
	options      = ['5', 'mean', 'median']
	targets      = [0, 1]
	selections   = ['00', '01', '10', '11']
	filename     = 'EEG.csv'
	train, label = load(filename)
	train        = norm(train)

	############### define classifier ##################
	clf = SVC(C=0.25, kernel='linear')


	for option in options:
		label, _ = transform(label, type=option)
		for selection in selections:
			for target in targets:
				forward  = False
				floating = False
				if selection[0] == '1':
					forward = True
				if selection[1] == '1':
					floating = True

				print ('')
				if forward:
					print ('forward ', end='')
				else:
					print ('backward ', end='')
				if floating:
					print ('floating ', end='')
				print ('selection --- target:', end='')
				if target == 0:
					print (' arousal')
				elif target == 1:
					print (' valence')
				else:
					print ('target error ({})'.format(target))
					sys.exit()
				print ('')

				############### target the label  ##################
				###################
				# 0 : arousal     #
				# 1 : valence     #
				###################
				train_y = label[:, target].reshape(-1)

				sfs = SFS(clf,
						  k_features='best',
						  forward=forward,
						  floating=floating,
						  scoring='accuracy',
						  cv=4,        
						  n_jobs=-1,
						  verbose=1)

				sfs.fit(train, train_y)

				# save model
				pickle_on = open('sequence/eeg_{}_{}_{}'.format(option, selection, target), "wb")
				pickle.dump(sfs.k_feature_idx_, pickle_on)
				pickle_on.close()

if __name__ == '__main__':
	main()
