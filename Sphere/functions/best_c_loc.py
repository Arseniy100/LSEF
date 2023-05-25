# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 18:13:54 2021

Last modified: Dec 2021

@author: Arseniy Sotskiy
"""


import numpy as np
import matplotlib.pyplot as plt
import typing as tp

from functions.tools import find_sample_covariance_matrix
from functions.lcz import construct_lcz_matrix
from functions.DLSM_functions import make_analysis

from Grids import LatLonSphericalGrid


c_loc_array = list(map(int, np.linspace(1000, 3000, 9)))
c_loc_array = np.linspace(1000, 3000, 9, dtype=int)
c_loc_array

def find_best_c_loc(ensemble: np.array, grid: LatLonSphericalGrid,
                    c_loc_array: np.array=c_loc_array,
                    *analysis_args: tp.Any, **analysis_kwargs: tp.Any):
    '''Finds best c_loc to construct localization matrix;
    then loc matrix is multiplied (element-wise) by B_sample 
    to get B_sample_loc.
    Best means best in terms of analysis.
    Iterates over possible values from the given list.

    Parameters
    ----------
    ensemble : np.array
        ensemble from which B_sample is constructer.
    grid : LatLonSphericalGrid
        grid for points on the sphere.
    c_loc_array : np.array, optional
        array with possible values of c_loc. 
        The default is np.linspace(1000, 3000, 9, dtype=int).
    *analysis_args : tp.Any
        args to pass into make_analysis function.
    **analysis_kwargs : tp.Any
        kwargs to pass into make_analysis function.

    Returns
    -------
    best_c : int
        c_loc with the best analysis.

    '''

    s_array = []

    for c_loc in c_loc_array:
        print(f'c_loc: {c_loc}')
        B_sample = find_sample_covariance_matrix(ensemble.T)
        lcz_mx = construct_lcz_matrix(grid, c_loc)

        B_sample_loc = np.multiply(B_sample, lcz_mx)

        s = make_analysis(B_sample_loc, *analysis_args, **analysis_kwargs)
        print(f's: {s}')
        s_array.append(s)


    best_c = c_loc_array[np.argmin(s_array)]
    # best_s = s_array[np.argmin(s_array)]
    plt.figure()
    plt.plot(c_loc_array, s_array)
    plt.grid()
    plt.title(f'best c_loc={best_c}')
    return best_c


#%%
# if __name__ == '__main__':
#     find_best_c_loc(ensemble, grid,
#                     c_loc_array,
#                     true_field, grid, n_obs=n_obs,
#                     obs_std_err=obs_std_err, n_seeds=n_seeds, seed=0,
#                     draw=False)
