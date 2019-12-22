# -*- coding: utf-8 -*-
# @Time    : 2019-11-05 22:14
# @Author  : xls56i

import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss
import torch.nn.functional as func
import numpy as np

class TVLoss(nn.Module):
    def __init__(self):
        super(TVLoss, self).__init__()

    #dimension = 2
    def forward(self, image):
        h_variation = image[:,1:] - image[:,:-1]
        v_variation = image[1:,:] - image[:-1,:]
        loss = torch.norm(h_variation)**2 + torch.norm(v_variation)**2
        return loss

class sum_squared_error(_Loss):  # PyTorch 0.4.1
    """
    Definition: sum_squared_error = 1/2 * nn.MSELoss(reduction = 'sum')
    The backward is defined as: input-target
    """
    def __init__(self, size_average=None, reduce=None, reduction='sum'):
        super(sum_squared_error, self).__init__(size_average, reduce, reduction)

    def forward(self, input, target):
        # return torch.sum(torch.pow(input-target,2), (0,1,2,3)).div_(2)
        return torch.nn.functional.mse_loss(input, target, size_average=None, reduce=None, reduction='sum').div_(2)

class NormalNLLLoss(nn.Module):
    """
    Calculate the negative log likelihood
    of normal distribution.
    This needs to be minimised.

    Treating Q(cj | x) as a factored Gaussian.
    """
    def __init__(self):
        super(NormalNLLLoss, self).__init__()

    def forward(self, x, mu, var):
        logli = -0.5 * (var.mul(2 * np.pi) + 1e-6).log() - (x - mu).pow(2).div(var.mul(2.0) + 1e-6)
        nll = -(logli.sum().mean())
        return nll

# if __name__ == '__main__':
#     criterion = TVLoss()
#     input = torch.Tensor([[1,2,3],[1,2,3]])
#     print(criterion(input))

