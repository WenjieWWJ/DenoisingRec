import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


def loss_function(y, t, alpha):

    loss = F.binary_cross_entropy_with_logits(y, t, reduce = False)
    y_ = torch.sigmoid(y).detach()
    weight = torch.pow(y_, alpha) * t + torch.pow((1-y_), alpha) * (1-t)
    loss_ = loss * weight
    loss_ = torch.mean(loss_)
    return loss_