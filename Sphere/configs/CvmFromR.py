# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 14:33:50 2019

@author: Arseniy Sotskiy
"""
# import pandas as pd
from numpy import array as np_array
import rpy2.robjects as robjects
robjects.r['load']("fields_covs_119123_mode1.RData")
filter_py = robjects.r['filter']


filter_py = dict(zip(filter_py.names, list(filter_py)))
BBx = filter_py['BBx']
cov_matrices = np_array(BBx)
