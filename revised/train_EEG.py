import numpy as np
import pandas as pd
import sys, pickle
from sklearn import preprocessing
from sklearn.externals import joblib
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score

if len(sys.argv) != 4:
	print ('Usage : python3 train_EEG.py <target> <C> <option>')
	print ('option: \'mean\', \'median\', \'5\'')
	print ('\nC is the parameter for SVM')
	print ('target ranges from 0 ~ 9')
	
	###################
	print ('0 : arousal    ')
	print ('1 : valence    ')
	###################
	sys.exit()

def load(filename):
	data  = pd.read_csv(filename)

	label   = data.iloc[:,106:]
	train   = data.iloc[:, 1:106]
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
	filename     = 'EEG.csv'
	train, label = load(filename)


	###################
	# 0 : arousal     #
	# 1 : valence     #
	###################

	target = int(sys.argv[1])  # the target to do classification

	sfs    = pickle.load(open('sequence/eeg_{}_11_{}'.format(sys.argv[3], target), 'rb'))
	

	############# without selection ###############
	# sfs = np.arange(0, 105, 1)

	print ('Number of features: ', len(sfs))

	####### feature selection ########
	train = train[:, sfs]

	############## normaliation ##############
	train = norm(train)


	############## leave-one-participant out for testing ##############
	# test over all of the participants
	kf = KFold(34)
	leave_one_out = kf.split(train, label)

	##### train #####
	ave_test = 0
	ave_val  = 0
	ave_f1   = 0

	for train_id, test_id in leave_one_out:

		# 10-fold cross validation
		acc       = 0
		val_acc   = 0
		cv        = KFold(10)
		cross_val = cv.split(train_id)

		train_      = train[train_id]
		label_, ref = transform(label[train_id], type=sys.argv[3])
		test        = train[test_id]
		test_lbl, _ = transform(label[test_id], ref=ref)

		for trainID, valID in cross_val:
			##################### split train, val ######################
			train_data  = train_[trainID]
			val_data    = train_[valID]

			train_label = label_[trainID, target].reshape((len(trainID), 1))
			val_label   = label_[valID, target].reshape((len(valID), 1))

			########### train ###########
			clf = SVC(C=float(sys.argv[2]), kernel='linear')
			clf.fit(train_data, train_label.reshape((len(train_data, ))))
			score    = clf.score(val_data, val_label)
			val_acc += score

			############# save best model #############
			if acc < score:
				acc = score
				joblib.dump(clf, 'models/model_{}.pkl'.format(target))

		########## perform testing on testing set ############
		Clf       = joblib.load('models/model_{}.pkl'.format(target))
		score     = Clf.score(test, test_lbl[:, target])
		val_acc  /= 10
		ave_test += score
		ave_val  += val_acc
		f1        = f1_score(val_label, Clf.predict(val_data), average='macro')
		ave_f1   += f1
		print ('Validation Accuracy : %f --- ' % (val_acc * 100), end='')
		print ('Testing Accuracy : %f --- ' % (score * 100), end='')
		print ('F1_score : %f' % (f1))

	########### Average testing accuracy over 39 participants ############
	print ('\nAverage')
	print ('Validation Accuracy : %f --- ' % (ave_val / 34 * 100), end='')
	print ('Testing Accuracy : %f' % (ave_test / 34 * 100))
	print ('f1_score : ', ave_f1 / 34)


if __name__ == '__main__':
	main()
