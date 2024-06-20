#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import time, pdb, numpy
from utils import accuracy

class LossFunction(nn.Module):
	def __init__(self, nOut, nClasses, **kwargs):
	    super(LossFunction, self).__init__()

	    self.test_normalize = True
	    
	    self.criterion  = torch.nn.CrossEntropyLoss()
	    self.fc 		= nn.Linear(nOut,nClasses)

	    print('Initialised Softmax Loss')

	def forward(self, x, label=None):
		if len(x.size()) > 2: # x : [B,S,D]
			if len(label.size()) < 2:
				label = label.repeat_interleave(x.size()[1]) #[B] -> [B,S] - (repeat S)

			x = x.reshape(-1,x.size()[-1])
			label = label.reshape(-1) # [B,S] -> [B*S]


		x 		= self.fc(x)
		nloss   = self.criterion(x, label)
		prec1	= accuracy(x.detach(), label.detach(), topk=(1,))[0]

		return nloss, prec1