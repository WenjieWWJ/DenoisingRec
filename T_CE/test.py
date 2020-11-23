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
import config
import evaluate
import data_utils

parser = argparse.ArgumentParser()

parser.add_argument("--epochs", 
	type=int,
	default=20,  
	help="training epoches")
parser.add_argument("--top_k", 
	type=int, 
	default=10, 
	help="compute metrics@top_k")

parser.add_argument("--test_num_ng", 
	type=int,
	default=999, 
	help="sample part of negative items for testing")


parser.add_argument("--gpu", 
	type=str,
	default="1",  
	help="gpu card ID")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
cudnn.benchmark = True

print("arguments: %s " %(args))
print("config model", config.model)
print("config path", config.main_path)
print("config dataset", config.dataset)

############################## PREPARE DATASET ##########################
train_data, test_data, user_num ,item_num, train_mat, user_neg = data_utils.load_all()

# construct the train and test datasets
# train_dataset = data_utils.NCFData(
# 		train_data, item_num, train_mat, user_neg, args.train_neg, args.num_ng, True)
test_dataset = data_utils.NCFData(
		test_data, item_num, train_mat, user_neg, False, 0, False)
# train_loader = data.DataLoader(train_dataset,
# 		batch_size=args.batch_size, shuffle=True, num_workers=4)
test_loader = data.DataLoader(test_dataset,
		batch_size=args.test_num_ng+1, shuffle=False, num_workers=0)
print("data loaded! user_num:{}, item_num:{} test_data_len:{}".format(user_num, item_num, len(test_data)//(args.test_num_ng+1)))

########################### CREATE MODEL #################################

model = torch.load('{}{}.pth'.format(config.model_path, config.model))
model.cuda()

model.eval()
HR_3, NDCG_3 = evaluate.metrics(model, test_loader, 3)
HR_5, NDCG_5 = evaluate.metrics(model, test_loader, 5)
HR_10, NDCG_10 = evaluate.metrics(model, test_loader, 10)
print("HR_3: {:.3f} NDCG_3: {:.3f}\tHR_5: {:.3f} NDCG_5: {:.3f}\tHR_10: {:.3f} NDCG_10: {:.3f}"\
	.format(np.mean(HR_3), np.mean(NDCG_3), np.mean(HR_5), np.mean(NDCG_5), np.mean(HR_10), np.mean(NDCG_10)))