# -*- coding: utf-8 -*-
"""
Created on Fri Nov 09 18:20:37 2018

@author: techietrader
"""
import torch


# MSE loss
def mean_squared_error(pred, target):
    
    diff = pred - target
    return torch.sum(diff * diff) / len(diff)