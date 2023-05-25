# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 21:33:42 2022
Last modified: April 2022

@author: Arseniy Sotskiy
"""

import sys
if '/RHM-Lustre3.2/software/general/plang/Anaconda3-2019.10' in sys.path:
    is_Linux = True
else:
    is_Linux = False
    # sys.path.append(r'/RHM-Lustre3.2/users/wg-da/asotskiy/packages')
sys.path

if is_Linux:
    print('working on super-computer')
    # n_max = 59
else:
    print('working on personal computer')
    # n_max = 47

#%% All imports

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 10000
import numpy as np
import pandas as pd
import os

from time import process_time, strftime

from pyshtools import SHGrid

COLORMAP = plt.cm.seismic

from configs import (
    std_xi_mult, std_xi_add, lamb_mult, lamb_add, 
    gamma_mult, gamma_add, mu_NSL, n_training_spheres, w_smoo,
    n_max,k_coarse, kappa_std_xi, kappa_lamb, kappa_gamma, threshold_coeff,
    is_best_c_loc_needed, c_loc, e_size, n_obs, obs_std_err, n_seeds,
    angular_distance_grid_size, dx,
    expq_bands_params
    )

from functions.tools import (
    draw_1D, draw_2D, find_sample_covariance_matrix, 
     integrate_gridded_function_on_S2, _time, 
    convert_variance_spectrum_to_modal, make_matrix_sparse,
    mkdir
    )
from functions.FLT import (
    Fourier_Legendre_transform_backward
    )
from functions.process_functions import (
    predict_spectrum_from_bands,
    get_variance_spectrum_from_modal
    )
from functions.DLSM_functions import (
    compute_u_from_sigma,
    compute_W_from_u, make_analysis, get_band_variances,
    draw_MAE_and_bias_of_spectrum, get_band_varinaces_dataset_for_nn,
    compute_B_from_spectrum, generate_ensemble, draw_bias_MSE_band_variances
    )
from functions.lcz import (
    construct_lcz_matrix, find_best_c_loc
    )
from functions.R_progs import (
    fit_shape_S2, CreateExpqBands #, V2b_shape_S2
    )
  
from Band import (
    Band_for_sphere
    )

from RandomProcesses import (
    get_modal_spectrum_from_DLSM_params
    )

from Grids import (
    coarsen_grid, flatten_field, make_2_dim_field, 
    interpolate_grid, LatLonSphericalGrid
    )

# from neural_network import (
#     NeuralNetSpherical, SquareActivation,
#     AbsActivation, ExpActivation
#     )
    
from Predictors import (
    SVDPredictor, SVShapePredictor, 
    SampleCovLocPredictor,
    compare_predicted_variance_spectra, 
    draw_errors_in_spatial_covariances,
    draw_cvf, draw_covariances_at_points
    )



#%%
start = process_time()
n_max: int = 47  # 47   # 59
folder_descr = f" c_loc_search big grid n_max {n_max}"

path_to_save = mkdir(folder_descr, is_Linux)


global n_png
n_png = 0

seed = 0 # my_seed_mult * (iteration + 1)
np.random.seed(seed)

draw = True




#%%

transfer_functions, _, __ = CreateExpqBands(**expq_bands_params)

n_bands = expq_bands_params['nband']

bands = [Band_for_sphere(transfer_functions[:,i]) for i in range(n_bands)]
print(transfer_functions.shape)
# n_bands = int(n_max / 5) # !!!!!!!!!!

band_centers = [band.center for band in bands]

n_ensemble_iterations = 10

#%% Get modal spectrum b_n(x)

# After the processes V (x), lambda(x), and gamma(x) are computed
# at each analysis grid point, Eq.(38) is used to find c(x).
# With c(x), lambda(x), and gamma(x) in hand, we finally compute
# the "true" modal spectrum bn(x) using Eq.(39)

c_loc_array = list(map(int, np.linspace(1, 5000, 50)))
gamma_med_arr = np.array([2, 2.5, 3, 3.5, 4, 4.5])
# gamma_med_arr = np.array([3, 3.5])
# lamb_med_arr = np.array([3, 4, 5, 6,]) * dx
lamb_med_arr = np.array([2, 3, 4, 5, 6, 7]) * dx
# gamma_med_arr = np.array([2])
# lamb_med_arr = np.array([3, 4]) * dx
c_loc_arr = np.zeros((len(gamma_med_arr), 
                     len(lamb_med_arr)))
# for here_can_be_any_cycle in [True]:
for i_gamma, gamma_med in enumerate(gamma_med_arr): 
    gamma_mult = gamma_med - gamma_add
    for i_lamb, lamb_med in enumerate(lamb_med_arr):
        lamb_mult = lamb_med - lamb_add
        
        modal_spectrum, lambx, gammax, Vx, cx = \
            get_modal_spectrum_from_DLSM_params(
                1, 1, 1,
                std_xi_mult, std_xi_add,
                lamb_mult, lamb_add,
                gamma_mult, gamma_add,
                mu_NSL, n_max, angular_distance_grid_size,
                draw=draw,
                seed=seed)
            
        variance_spectrum = get_variance_spectrum_from_modal(modal_spectrum, n_max)
            
        nlat, nlon = coarsen_grid(variance_spectrum[0,:,:], k_coarse).shape
        grid = LatLonSphericalGrid(nlat)    
        
        
        #%%
        
        modal_spectrum_med, lambx_med, gammax_med, Vx_med, cx_med = \
                    get_modal_spectrum_from_DLSM_params(
                        kappa_std_xi=1, kappa_lamb=1, kappa_gamma=1,
                        std_xi_mult=std_xi_mult, std_xi_add=std_xi_add,
                        lamb_mult=lamb_mult, lamb_add=lamb_add,
                        gamma_mult=gamma_mult, gamma_add=gamma_add,
                        mu_NSL=mu_NSL, n_max=n_max, 
                        angular_distance_grid_size=angular_distance_grid_size,
                        draw=False)
                    
        variance_spectrum_med = get_variance_spectrum_from_modal(modal_spectrum_med, 
                                                                  n_max)
        
        
        
        #%%
            
        W_true, B_true = compute_B_from_spectrum(variance_spectrum, n_max,
                                                  grid=grid,
                                                  k_coarse=k_coarse, info='true')
        
        W_med, B_med = compute_B_from_spectrum(variance_spectrum_med, n_max,
                                                  grid=grid,
                                                  k_coarse=k_coarse, info='med')
        
        best_c_loc_poss_vals = []
        for ens_iter in range(n_ensemble_iterations):
        
            true_field_interp, ensemble_fields_interp, ensemble, true_field =\
                generate_ensemble(
                    W_true, e_size, grid, draw=draw, info='', 
                    path_to_save=path_to_save,
                    k_coarse=k_coarse
                )
            
            
            # c_loc_array = [2000]
            sample_cvm_loc_predictor = SampleCovLocPredictor(
                bands, n_max, grid, ensemble,
                true_field, n_obs, obs_std_err, 
                c_loc_array = c_loc_array,
                )
            
            print(
                f'Lambda / dx = {(lamb_mult + lamb_add) / dx} '
                f'c_loc = {sample_cvm_loc_predictor.c_loc}'
                )
            best_c_loc_poss_vals.append(sample_cvm_loc_predictor.c_loc)
        print('poss c_loc: ', best_c_loc_poss_vals)
        c_loc_arr[i_gamma, i_lamb] = np.mean(best_c_loc_poss_vals)
        with open(path_to_save + f'c_loc_arr.txt', 'w') as file:
            c_loc_arr_str = np.array2string(c_loc_arr, separator=', ')
            file.write(c_loc_arr_str)
    
    print(f'lamb_med_arr : {lamb_med_arr / dx} * dx')
    print(f'c_loc array: {c_loc_arr}')
    
    plt.figure()
    plt.plot(lamb_med_arr / dx, c_loc_arr[i_gamma, :])
    plt.scatter(lamb_med_arr / dx, c_loc_arr[i_gamma, :])
    plt.xlabel('lamb_med, dx')
    plt.ylabel('c_loc')
    plt.grid()
    title = f'best c_loc, gamma_med={gamma_med}'
    plt.title(title)
    plt.savefig(path_to_save + f'{title}.png')
    plt.show()
    
print(c_loc_arr)



# assert False, 'stop'