# -*- coding: utf-8 -*-
"""
Last modified on Mar 2022
@author: Arseniy Sotskiy
"""

import torch


from torch import nn


"""https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#creating-a-custom-dataset-for-your-files"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

#% %


class SquareActivation(nn.Module):
    def __init__(self, params=None):
        super().__init__()
        self.params = params

    def forward(self, input):
        return input**2


class AbsActivation(nn.Module):
    def __init__(self, params=None):
        super().__init__()
        self.params = params

    def forward(self, input):
        return torch.abs(input)


class ExpActivation(nn.Module):
    def __init__(self, params=None):
        super().__init__()
        self.params = params

    def forward(self, input):
        return torch.exp(input)
    

class IdActivation(nn.Module):
    def __init__(self, params=None):
        super().__init__()
        self.params = params

    def forward(self, input):
        return input
    
    
exp_activ_dict = {'activation_name': 'ExpActivation', 
                  'activation_function': ExpActivation()}
abs_activ_dict = {'activation_name': 'AbsActivation', 
                  'activation_function': AbsActivation()}
square_activ_dict = {'activation_name': 'SquareActivation', 
                  'activation_function': SquareActivation()}
id_activ_dict = {'activation_name': 'IdActivation', 
                  'activation_function': IdActivation()}

activations_dict = {
    'ExpActivation': exp_activ_dict,
    'AbsActivation': abs_activ_dict,
    'SquareActivation': square_activ_dict,
    'IdActivation': id_activ_dict
    }


