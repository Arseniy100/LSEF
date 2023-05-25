# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 12:23:02 2019
Last modified: March 2020

@author: Арсений HP-15
"""

from NeuralNetworks.LossFunctions import (
    PriorSmoothLoss, PriorSmoothLogLoss, AnlsLoss, RandomLoss,
    L2VarLoss
    )


from NeuralNetworks.Activations import (
    SquareActivation, AbsActivation, ExpActivation,
    activations_dict
    )

from NeuralNetworks.DataPreparation import (
    CustomDataset, prepare_data
    )


from NeuralNetworks.Networks import (
    NeuralNetSpherical,
    NeuralNetwork_first_try, NeuralNetwork_deep, 
    NeuralNetwork_small, NeuralNetwork_controlled
    )

