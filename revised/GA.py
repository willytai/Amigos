import numpy as np
import pandas as pd
import sys, random, time


from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score

if len(sys.argv) != 2:
	print ('Usage:  python3 GA.py <target>')
	print ('target: 0 for Arousal')
	print ('        1 for Valence')
	sys.exit()

target = int(sys.argv[1])

def norm(data):
	mean = data.mean(axis=0)
	maxx = data.max(axis=0)
	minn = data.min(axis=0)
	for i in range(data.shape[0]):
		data[i] = (data[i] - mean ) / (maxx - minn)
	return data

def load(filename):
	data  = pd.read_csv(filename)
	train = data.iloc[:, 1:]
	label = pd.read_csv('GSR.csv')
	label = label.iloc[:, -2:]

	return norm(np.array(train)), np.array(label)

def random_0_1(size):
	return np.random.randint(2, size=size)

def initGene(population, gene_length):
	'''
	individuals are stored as list
	each individual is represneted as a list
	[gene, accuracy, fitness]
	'''
	individuals = []
	for i in range(population):
		individuals.append([random_0_1(gene_length), 0, 0])
	return individuals

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

def SVM(train, label):
	############## leave-one-participant out for testing ##############
	# test over all of the participants
	kf = KFold(33)
	leave_one_out = kf.split(train, label)

	##### train #####
	ave_test = 0
	ave_val  = 0
	ave_f1   = 0

	for train_id, test_id in leave_one_out:

		# 10-fold cross validation
		acc       = 0
		val_acc   = 0
		cv        = KFold(10, shuffle=True)
		cross_val = cv.split(train_id)

		train_      = train[train_id]
		label_, ref = transform(label[train_id], type='mean')
		test        = train[test_id]
		test_lbl, _ = transform(label[test_id], ref=ref)

		for trainID, valID in cross_val:
			##################### split train, val ######################
			train_data  = train_[trainID]
			val_data    = train_[valID]

			train_label = label_[trainID, target].reshape((len(trainID), 1))
			val_label   = label_[valID, target].reshape((len(valID), 1))

			########### train ###########
			clf = SVC(C=0.25, kernel='linear')
			clf.fit(train_data, train_label.reshape((len(train_data, ))))
			score    = clf.score(val_data, val_label)
			val_acc += score

			############# save best model #############
			if acc < score:
				acc = score
				Clf = clf

		########## perform testing on testing set ############
		score     = Clf.score(test, test_lbl[:, target])
		val_acc  /= 10
		ave_test += score
		ave_val  += val_acc
		f1        = f1_score(val_label, Clf.predict(val_data), average='macro')
		ave_f1   += f1
		# print ('Validation Accuracy : %f --- ' % (val_acc * 100), end='')
		# print ('Testing Accuracy : %f --- ' % (score * 100), end='')
		# print ('F1_score : %f' % (f1))

	########### Average testing accuracy over 39 participants ############
	# print ('\nAverage')
	# print ('Validation Accuracy : %f --- ' % (ave_val / 33 * 100), end='')
	# print ('Testing Accuracy : %f' % (ave_test / 33 * 100))
	# print ('f1_score : ', ave_f1 / 33)

	return ave_f1 / 33


def Train(individual, train, label):
	# get the gene
	gene = individual[0]

	# get the training features according to the gene
	features = []
	for i, g in enumerate(gene):
		if g == 1:
			features.append(i)
	features = np.array(features)
	train_id = train[:, features]

	# train
	acc = SVM(train_id, label)

	return [gene, acc, 0]


def fit(individuals, train, label, selective_pressure):
	for i in range(len(individuals)):
		print ('\rFitting individual {}/{}'.format(i+1, len(individuals)), end='', flush=True)
		individuals[i] = Train(individuals[i], train, label)
	print ('')

	# reArrange according to its f1 score
	individuals = sorted(individuals, key=lambda x : x[1])

	# calculate rank
	for i in range(len(individuals)):
		individuals[i][2] = selective_pressure*(1+i)

	return individuals

def select(individuals, size):
	selected    = [individuals[-1]]
	check_exist = [individuals[-1][2]]

	# increase randomness
	random.shuffle(individuals)

	while len(selected) < size:
		Max  = sum([ind[2] for ind in individuals])
		pick = random.uniform(0, Max)
		current = 0
		for ind in individuals:
			current += ind[2]
			if current > pick:
				if ind[2] not in check_exist:
					selected.append(ind)
					check_exist.append(ind[2])
				break

	return selected

def mate(gene1, gene2):
	new = []
	select = np.random.randint(2, size=gene1.shape[0])
	for i, g in enumerate(select):
		if g == 0:
			new.append(gene1[i])
		else:
			new.append(gene2[i])
	return [np.array(new), 0, 0]

def crossover(selected):
	parent_num = len(selected)
	offspring  = []

	# print ('===== space =====')
	# for i in selected:
	# 	print(i)
	# print ('')

	while parent_num > 1:
		pick1 = np.random.randint(parent_num)
		pick2 = np.random.randint(parent_num)
		while pick2 == pick1:
			pick2 = np.random.randint(parent_num)

		parent1 = selected[pick1]
		parent2 = selected[pick2]

		# print ('parent1    ', parent1)
		# print ('parent2    ', parent2)

		for i in range(4):
			offspring.append(mate(parent1[0], parent2[0]))
			# print ('offspring',i,offspring[-1][0])

		# print ('===== before =====')
		# for i in selected:
		# 	print(i)
		# print ('')
		
		selected[pick1], selected[parent_num-1] = selected[parent_num-1], selected[pick1]
		selected[pick2], selected[parent_num-2] = selected[parent_num-2], selected[pick2]
		parent_num -= 2

		# print ('===== after =====')
		# for i in selected:
		# 	print(i)
		# print ('')

	return offspring

def mutation(offspring, mutation_rate):
	target_feature = np.random.randint(offspring[0][0].shape[0], size=len(offspring))
	
	for i in range(len(offspring)):
		mutate = random.uniform(0, 1)
		if mutate < mutation_rate:
			offspring[i][0][target_feature[i]] ^= 1

	return offspring

def main():
	filename     = 'f_GSR.csv'
	train, label = load(filename)
	
	###############
	## parameters
	###############
	history            = []
	generation         = 40
	gene_length        = train.shape[1]
	population         = 10*gene_length
	selective_pressure = 1.5 # usually between 1 to 2
	selection_size     = int(population/2)
	mutation_rate      = 1/gene_length
	start_time         = time.time()

	# initialize gene for all individuals
	individuals  = initGene(population, gene_length)

	for i in range(generation):
		print ('\n##################')
		print ('## %d Generation ##' % (i+1))
		print ('##################')

		# fitness assignment
		individuals = fit(individuals, train, label, selective_pressure)

		# save the results
		print ('Saving Individuals and Evaluating Generation...')
		gene = []
		f1   = []
		for ind in individuals:
			gene.append(ind[0])
			f1.append(ind[1])
		gene = np.array(gene)
		np.save('gene_gsr_{}.npy'.format(target), gene)
		history.append(np.array(f1).mean())
		np.save('history_gsr_{}.npy'.format(target), np.array(history))


		# selection
		selected = select(individuals, selection_size)

		# crossover
		offspring = crossover(selected)

		# mutation
		individuals = mutation(offspring, mutation_rate)

		# print time
		print ('Time Elapsed: %2fs' % (time.time() - start_time))


if __name__ == '__main__':
	main()