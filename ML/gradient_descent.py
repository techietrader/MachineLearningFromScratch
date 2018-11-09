# -*- coding: utf-8 -*-
"""
Created on Fri Nov 09 18:21:04 2018

@author: techietrader
"""
# Import Numpy & PyTorch

import torch

from ML.linear_model import model
from ML.loss_functions import mean_squared_error


EPOCHS = 1000
LR = 1e-2
CALLBACK_VALUE = 0.1

def compute_gradient( X, y , weights, bias, lr = LR, callback_value=CALLBACK_VALUE, epochs= EPOCHS):
    import time
    previous_epoch_loss = 100000000000.0
    for i in range(epochs):
        preds = model(X, weights, bias)
        loss = mean_squared_error(preds, y)
        if previous_epoch_loss - loss > callback_value:
            previous_epoch_loss = loss
          
        else:
            print ('Loss didnot change with significant margin. Loss is {}'.format(loss))
            break
            
        print ('Loss at epoch {} is {}'.format(i, loss))
        loss.backward()
        with torch.no_grad():
            weights -= weights.grad * lr
            bias -= bias.grad * lr
            weights.grad.zero_()
            bias.grad.zero_()
        time.sleep(0.5)
    return loss, weights, bias
