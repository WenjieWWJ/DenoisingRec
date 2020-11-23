import os
import numpy as np
import random

type_str = "train"
clean_train = open("../data-4-valid/data_clean_train_neg/Adressa-1w.{}.rating".format(type_str), "r").readlines()

all_train = open("../data-4-valid/data_clean_neg/Adressa-1w.{}.rating".format(type_str), "r").readlines()

clean_train_list = {}
for line in all_train:
	clean_train_list[line.strip()]=0
for line in clean_train:
	clean_train_list[line.strip()]=1

result = []
for line in clean_train_list:
	if clean_train_list[line] == 1:
		result.append(line.strip() + '\t1')
	else:
		result.append(line.strip() + '\t0')
file = open("../data-4-valid/data_coteaching/Adressa-1w.{}.rating".format(type_str), "w")
file.write("\n".join(result))
file.close()
