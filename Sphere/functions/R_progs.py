# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 15:35:17 2021
Last modified: Jan 2021

Makes python function CreateExpqBands from R functions

@author: Arseniy Sotskiy
"""

import numpy as np
import rpy2
print(rpy2.__version__)

# from rpy2.rinterface import R_VERSION_BUILD
# print(R_VERSION_BUILD)


import rpy2.robjects as robjects

from rpy2.robjects.packages import importr
# import R's "base" package
base = importr('base')

# import R's "utils" package
utils = importr('utils')

import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()



R_files = [
    'R from MD/SHAPE_S2_2021-03-05/LinIntpl.R',
    'R from MD/SHAPE_S2_2021-03-05/evalFuScaledArg.R',
    'R from MD/SHAPE_S2_2021-03-05/fitScaleMagn.R',
    'R from MD/SHAPE_S2_2021-03-05/V2E_shape_S2.R',

    # 'R from MD/SHAPE_S2/LinIntpl.R',
    # 'R from MD/SHAPE_S2/evalFuScaledArg.R',
    # 'R from MD/SHAPE_S2/fitScaleMagn.R',
    # 'R from MD/SHAPE_S2/V2b_shape_S2.R',

    'R from MD/CreateExpqBands_Double/tranfuBandpassExpq.R',
    'R from MD/CreateExpqBands_Double/bisection.R',
    'R from MD/CreateExpqBands_Double/CreateExpqBands.R',
    ]




for prog in R_files:
    with open(prog, 'r') as file:
        robjects.r(file.read())



LinIntrpl = robjects.globalenv['LinIntpl']
evalFuScaledArg = robjects.globalenv['evalFuScaledArg']
fitScaleMagn = robjects.globalenv['fitScaleMagn']
fit_shape_S2 = robjects.globalenv['V2E_shape_S2']
# V2b_shape_S2 = robjects.globalenv['V2b_shape_S2']


CreateExpqBands_R = robjects.globalenv['CreateExpqBands']

def CreateExpqBands(n_max, nband, halfwidth_min, nc2, halfwidth_max,
                    q_tranfu=3, rectang = False):
    bands = CreateExpqBands_R(n_max+1, nband, halfwidth_min, nc2, halfwidth_max,
                              q_tranfu, rectang)
    tranfu = np.array(bands[0])[:n_max+1,:]
    hhwidth = bands[2]
    band_centers_n = bands[3]
    return tranfu, hhwidth, band_centers_n



