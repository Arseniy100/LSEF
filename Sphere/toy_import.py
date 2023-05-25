# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 15:31:34 2022

@author: user410
"""

import numpy as np

def generate(seed=1):
    np.random.seed(None)
    return np.random.randint(10000)