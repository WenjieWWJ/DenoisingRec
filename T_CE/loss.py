import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


def loss_coteaching(y, t, forget_rate, flip_rate, noise_or_not):
    loss = F.binary_cross_entropy_with_logits(y, t, reduce = False)

    loss_mul = loss * t
    ind_sorted = np.argsort(loss_mul.cpu().data).cuda()
    loss_sorted = loss[ind_sorted]

    remember_rate = 1 - forget_rate
    num_remember = int(remember_rate * len(loss_sorted))
    num_flip = int(flip_rate * len(loss_sorted))
    pure_ratio_select = np.sum(noise_or_not[ind_sorted[:num_remember]].numpy())/float(num_remember)

    ind_select = ind_sorted[:num_remember]
    if num_flip == 0:
        pure_ratio_flip = 0
        ind_update = ind_select
    else:
        pure_ratio_flip = 1 - np.sum(noise_or_not[ind_sorted[-num_flip:]].numpy())/float(num_flip)
        ind_flip = ind_sorted[-num_flip:]
        t[ind_flip] = 0
        ind_update = torch.cat([ind_select, ind_flip], 0)

    loss_update = F.binary_cross_entropy_with_logits(y[ind_update], t[ind_update])

    return loss_update, pure_ratio_select, pure_ratio_flip
