# -*- coding: utf-8 -*-
"""
Created on Fri Nov 09 18:24:24 2018

@author: techietrader
"""
import numpy as np

def normalizing_values(list):
    normalized_list = list / np.linalg.norm(list)
    return normalized_list


