# -*- coding: utf-8 -*-
"""
Created on Fri Nov 09 18:14:31 2018

@author: techietrader
"""


import torch


# Define the model
def model(X, weights, bias):
    return torch.mm(X,weights.t()) + bias

