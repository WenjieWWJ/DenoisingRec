import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


def lossFunction(y, t, alpha):

    # y = torch.sigmoid(y)

    # # loss = - torch.pow(y, alpha) * t * torch.log(y) - torch.pow((1-y), alpha) * (1-t) * torch.log(1-y)
    # loss = - t * torch.log(y) - (1-t) * torch.log(1-y)
    loss = F.binary_cross_entropy_with_logits(y, t, reduce = False)
    # print("loss 1", loss)
    # print("y", torch.sigmoid(y))
    # print("t", t)
    y_ = torch.sigmoid(y).detach()
    # y_ = torch.sigmoid(y).detach()
    # print("mask", (torch.pow(y_, alpha) * t + torch.pow((1-y_), alpha) * (1-t)))
    weight = torch.pow(y_, alpha) * t + torch.pow((1-y_), alpha) * (1-t)
    loss_ = loss * weight
    if torch.sum(torch.isnan(loss_)) > 0:
        print("loss 1", loss)
        print("loss 2", loss_)
        print("y", y)
        print("y_", y_)
        print("t", t)
        print("weight", weight)

    # print("loss 2", loss)
    loss_ = torch.mean(loss_)
    # print(loss_)
    return loss_