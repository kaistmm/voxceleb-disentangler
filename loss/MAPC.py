#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import time, pdb, numpy
from audtorch.metrics.functional import pearsonr

class LossFunction(nn.Module):
	def __init__(self, **kwargs):
	    super(LossFunction, self).__init__()

	    self.test_normalize = True


	    print('Initialised Softmax Loss')

	def forward(self, x, y):
		vx = x - torch.mean(x, dim=1, keepdim=True) # batch-wise centred, (B, D)
		vy = y - torch.mean(y, dim=1, keepdim=True) # (B, D)

		cov = torch.sum(vx * vy, dim=1, keepdim=True) # (B, 1)
		abs_pearson = torch.abs(cov) / (torch.sqrt(torch.sum(vx ** 2, dim=1, keepdim=True)) * torch.sqrt(torch.sum(vy ** 2, dim=1, keepdim=True)))

		nloss = torch.mean(abs_pearson) # mean absolute pearson loss
		return nloss