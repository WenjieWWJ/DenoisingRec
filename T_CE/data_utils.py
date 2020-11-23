import numpy as np 
import pandas as pd 
import scipy.sparse as sp
from copy import deepcopy
import random
import torch.utils.data as data


def load_all(dataset, data_path):

	train_rating = data_path + '{}.train.rating'.format(dataset)
	valid_rating = data_path + '{}.valid.rating'.format(dataset)
	test_negative = data_path + '{}.test.negative'.format(dataset)

	################# load training data #################	
	train_data = pd.read_csv(
		train_rating, 
		sep='\t', header=None, names=['user', 'item', 'noisy'], 
		usecols=[0, 1, 2], dtype={0: np.int32, 1: np.int32, 2: np.int32})

	if dataset == "adressa":
		user_num = 212231
		item_num = 6596
	else:
		user_num = train_data['user'].max() + 1
		item_num = train_data['item'].max() + 1
	print("user, item num")
	print(user_num, item_num)
	train_data = train_data.values.tolist()

	# load ratings as a dok matrix
	train_mat = sp.dok_matrix((user_num, item_num), dtype=np.float32)
	train_data_list = []
	train_data_noisy = []
	for x in train_data:
		train_mat[x[0], x[1]] = 1.0
		train_data_list.append([x[0], x[1]])
		train_data_noisy.append(x[2])

	################# load validation data #################
	valid_data = pd.read_csv(
		valid_rating, 
		sep='\t', header=None, names=['user', 'item', 'noisy'], 
		usecols=[0, 1, 2], dtype={0: np.int32, 1: np.int32, 2: np.int32})
	valid_data = valid_data.values.tolist()
	valid_data_list = []
	for x in valid_data:
		valid_data_list.append([x[0], x[1]])
	
	user_pos = {}
	for x in train_data_list:
		if x[0] in user_pos:
			user_pos[x[0]].append(x[1])
		else:
			user_pos[x[0]] = [x[1]]
	for x in valid_data_list:
		if x[0] in user_pos:
			user_pos[x[0]].append(x[1])
		else:
			user_pos[x[0]] = [x[1]]


	################# load testing data #################
	test_mat = sp.dok_matrix((user_num, item_num), dtype=np.float32)

	test_data_pos = {}
	with open(test_negative, 'r') as fd:
		line = fd.readline()
		while line != None and line != '':
			arr = line.split('\t')
			if dataset == "adressa":
				u = eval(arr[0])[0]
				i = eval(arr[0])[1]
			else:
				u = int(arr[0])
				i = int(arr[1])
			if u in test_data_pos:
				test_data_pos[u].append(i)
			else:
				test_data_pos[u] = [i]
			test_mat[u, i] = 1.0
			line = fd.readline()


	return train_data_list, valid_data_list, test_data_pos, user_pos, user_num, item_num, train_mat, train_data_noisy


class NCFData(data.Dataset):
	def __init__(self, features,
				num_item, train_mat=None, num_ng=0, is_training=0, noisy_or_not=None):
		super(NCFData, self).__init__()
		""" Note that the labels are only useful when training, we thus 
			add them in the ng_sample() function.
		"""
		self.features_ps = features
		if is_training == 0:
			self.noisy_or_not = noisy_or_not
		else:
			self.noisy_or_not = [0 for _ in range(len(features))]
		self.num_item = num_item
		self.train_mat = train_mat
		self.num_ng = num_ng
		self.is_training = is_training
		self.labels = [0 for _ in range(len(features))]

	def ng_sample(self):
		assert self.is_training  != 2, 'no need to sampling when testing'

		self.features_ng = []
		for x in self.features_ps:
			u = x[0]
			for t in range(self.num_ng):
				j = np.random.randint(self.num_item)
				while (u, j) in self.train_mat:
					j = np.random.randint(self.num_item)
				self.features_ng.append([u, j])

		labels_ps = [1 for _ in range(len(self.features_ps))]
		labels_ng = [0 for _ in range(len(self.features_ng))]
		self.noisy_or_not_fill = self.noisy_or_not + [1 for _ in range(len(self.features_ng))]
		self.features_fill = self.features_ps + self.features_ng
		assert len(self.noisy_or_not_fill) == len(self.features_fill)
		self.labels_fill = labels_ps + labels_ng

	def __len__(self):
		return (self.num_ng + 1) * len(self.labels)

	def __getitem__(self, idx):
		features = self.features_fill if self.is_training != 2 \
					else self.features_ps
		labels = self.labels_fill if self.is_training != 2 \
					else self.labels
		noisy_or_not = self.noisy_or_not_fill if self.is_training != 2 \
					else self.noisy_or_not

		user = features[idx][0]
		item = features[idx][1]
		label = labels[idx]
		noisy_label = noisy_or_not[idx]

		return user, item, label, noisy_label
		
