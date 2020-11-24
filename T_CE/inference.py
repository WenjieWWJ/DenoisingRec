import os
import time
import argparse
import numpy as np
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.backends.cudnn as cudnn
# from tensorboardX import SummaryWriter

import model
import evaluate
import data_utils

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', 
	type = str,
	help = 'dataset used for training, options: amazon_book, yelp, adressa',
	default = 'amazon_book')
parser.add_argument('--model', 
	type = str,
	help = 'model used for training. options: GMF, NeuMF-end',
	default = 'GMF')
parser.add_argument('--drop_rate', 
	type = float,
	help = 'drop rate',
	default = 0.2)
parser.add_argument('--num_gradual', 
	type = int, 
	default = 30000,
	help='how many epochs for linear drop rate {5, 10, 15}')
parser.add_argument("--top_k", 
	type=list, 
	default=[50, 100],
	help="compute metrics@top_k")
parser.add_argument("--gpu", 
	type=str,
	default="1",  
	help="gpu card ID")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
cudnn.benchmark = True

data_path = '../data/{}/'.format(args.dataset)
model_path = './models/{}/'.format(args.dataset)
print("arguments: %s " %(args))
print("config model", args.model)
print("config data path", data_path)
print("config model path", model_path)

############################## PREPARE DATASET ##########################
train_data, valid_data, test_data_pos, user_pos, user_num ,item_num, train_mat, train_data_noisy = data_utils.load_all(args.dataset, data_path)

########################### CREATE MODEL #################################
test_model = torch.load('{}{}_{}-{}.pth'.format(model_path, args.model, args.drop_rate, args.num_gradual))
test_model.cuda()

def test(model, test_data_pos, user_pos):
	top_k = args.top_k
	model.eval()
	_, recall, NDCG, _ = evaluate.test_all_users(model, 4096, item_num, test_data_pos, user_pos, top_k)

	print("################### TEST ######################")
	print("Recall {:.4f}-{:.4f}".format(recall[0], recall[1]))
	print("NDCG {:.4f}-{:.4f}".format(NDCG[0], NDCG[1]))

test(test_model, test_data_pos, user_pos)

