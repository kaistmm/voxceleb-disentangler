#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
import loss.softmax as softmax
import loss.angleproto as angleproto

class LossFunction(nn.Module):

    def __init__(self, **kwargs):
        super(LossFunction, self).__init__()

        self.test_normalize = True

        self.softmax = softmax.LossFunction(**kwargs)
        self.angleproto = angleproto.LossFunction(**kwargs)

        print('Initialised SoftmaxPrototypical Loss')

    def forward(self, x, label=None):
        nlossS, prec1   = self.softmax(x, label)
        nlossP, _       = self.angleproto(x,None)

        return nlossS+nlossP, prec1