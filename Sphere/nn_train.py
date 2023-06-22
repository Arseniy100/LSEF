# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 14:03:48 2022
Last modified: Nov 2022

@author: Arseniy Sotskiy
"""

import os
# import sys
#must set these before loading numpy:
os.environ["OMP_NUM_THREADS"] = '1' # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = '1' # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = '8' # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = '4' # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = '4' # export NUMEXPR_NUM_THREADS=6

#%%
import sys
if '/RHM-Lustre3.2/software/general/plang/Anaconda3-2019.10' in sys.path:
    is_Linux = True
else:
    is_Linux = False
    # sys.path.append(r'/RHM-Lustre3.2/users/wg-da/asotskiy/packages')
# sys.path

if is_Linux:
    print('working on super-computer')
    # n_max = 59
else:
    print('working on personal computer')
    # n_max = 47

#%%
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 10000
import numpy as np
import os
from torch import nn
from tqdm import trange

from collections import namedtuple

from time import process_time

from pyshtools import SHGrid

COLORMAP = plt.cm.seismic

from configs import (
    SDxi_med, SDxi_add, lambda_med, lambda_add, gamma_med, gamma_add,
    mu_NSL, n_training_spheres, w_smoo,
    n_max,k_coarse, kappa_SDxi, kappa_lamb, kappa_gamma,
    c_loc, e_size, n_obs, obs_std_err, n_seeds,
    angular_distance_grid_size, nn_i_multiplier,
    batch_size, lr, momentum, activation_name, non_linearity, n_epochs,
    n_training_spheres_B, is_nn_trained
    )


from functions.tools import (
    draw_1D, find_sample_covariance_matrix, 
    integrate_gridded_function_on_S2, _time,
    convert_variance_spectrum_to_modal,
    convert_modal_spectrum_to_variance
    )
from functions.FLT import (
    Fourier_Legendre_transform_backward
    )

from functions.process_functions import (
    get_variance_spectrum_from_modal
    )
from functions.DLSM_functions import (
    make_analysis, 
    get_band_varinaces_dataset_for_nn,
    compute_B_from_spectrum
    )
from functions.lcz import (
    construct_lcz_matrix
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

from NeuralNetworks import (
    NeuralNetSpherical, 
    activations_dict, SquareActivation,
    AbsActivation, ExpActivation, 
    NeuralNetwork_first_try, 
    NeuralNetwork_small, NeuralNetwork_deep,
    NeuralNetwork_controlled,
    PriorSmoothLogLoss, PriorSmoothLoss, AnlsLoss, RandomLoss,
    L2VarLoss
    )
    
from Predictors import (
    SVDPredictor, SVShapePredictor, 
    compare_predicted_variance_spectra, 
    draw_errors_in_spatial_covariances
    )

#%%

NetParameters = namedtuple('NetParameters', 
                           ['init_params', 'fit_params'])

folder_descr = f" net_train {n_training_spheres} spheres"

path_to_save = 'images\ ' + _time() + \
    folder_descr + r'\ '
if is_Linux:
    path_to_save = path_to_save.replace(r'\ ', r'/')
    trange = range
path_to_save = path_to_save.replace(':', '_')
print(path_to_save)
os.mkdir(path_to_save)
# os.rmdir(r'images\2021_12_15_15_19_07 20 training spheres, w_smoo 0.0001, ')
print(f'made directory {path_to_save}')

# =============================================================================
# 
# =============================================================================
seed = 0
draw = False
# =============================================================================
# 
# =============================================================================


expq_bands_params = {'n_max': n_max, 'nband': 6, 'halfwidth_min': 1.5  * n_max/30,
          'nc2': 1.5 * n_max / 22, 'halfwidth_max': 1.2 * n_max / 4,
          'q_tranfu': 3, 'rectang': False}
transfer_functions, _, __ = CreateExpqBands(**expq_bands_params)

n_bands = expq_bands_params['nband']


bands = [Band_for_sphere(transfer_functions[:, i]) for i in range(n_bands)]
print(transfer_functions.shape)

modal_spectrum, lambx, gammax, Vx, cx = \
    get_modal_spectrum_from_DLSM_params(
        kappa_SDxi, kappa_lamb, kappa_gamma,
        SDxi_med, SDxi_add,
        lambda_med, lambda_add,
        gamma_med, gamma_add,
        mu_NSL, n_max, angular_distance_grid_size,
        draw=draw,
        seed=seed)
    
variance_spectrum = get_variance_spectrum_from_modal(modal_spectrum, n_max)
    
nlat, nlon = coarsen_grid(variance_spectrum[0,:,:], k_coarse).shape
grid = LatLonSphericalGrid(nlat)   



band_true_variances_train_dataset,\
            band_estim_variances_train_dataset, \
            modal_spectrum_train_dataset = \
                get_band_varinaces_dataset_for_nn(
                    n_training_spheres, grid, n_max, k_coarse, 
                    e_size, bands, seed, 
                    DLSM_params = [
                        kappa_SDxi, kappa_lamb, kappa_gamma,
                        SDxi_med, SDxi_add,
                        lambda_med, lambda_add,
                        gamma_med, gamma_add,
                        mu_NSL, n_max, angular_distance_grid_size
                        ],
                    # force_generation=True
                    )

activ_dict = activations_dict[activation_name]
exp_activ_dict = activations_dict['ExpActivation']
square_activ_dict = activations_dict['SquareActivation']
id_activ_dict = activations_dict['IdActivation']

init_default = {'n_input': n_bands, 
                'n_output': n_max,
                'device': 'cpu'}
band_centers = [band.center for band in bands]
fit_default = {'path_to_save': path_to_save,
               'draw': draw,
               'band_centers': band_centers}

# l_obs = min(n_max, int(np.sqrt(n_obs * np.pi / 4)))
l_obs = n_max
anls_loss_kwargs = {'obs_std_err': obs_std_err,
                    'l_obs': l_obs,}



multipliers_grid = (
    [6,12],
    [6,15,12],
    [6,12,20,12],
    [6,12,20,15,12]
    )

non_lin_dict = {
    'ReLU': nn.ReLU(),
    'LeakyReLU': nn.LeakyReLU(),
    'Sigmoid': nn.Sigmoid(),
    'Tanh': nn.Tanh()
    }
print(non_lin_dict.keys())
#%%    Different Neural Networks
# ============================================
# ============================================
# ============================================
# ============================================
all_nets_params_grid = {
    'net1': NetParameters(
        init_params= {**init_default,
                      'model': NeuralNetwork_first_try,
                      'loss': 'my', 
                      **exp_activ_dict,
                      'w_smoo': w_smoo},
        fit_params={'n_epochs': 3,
                    'momentum': 0.1,
                    'validation_coeff': 0.1,
                    **fit_default,
                    'lr': 1e-1, 'batch_size': 16,}
        ),
    'net_deep': NetParameters(
        init_params= {**init_default,
                      'model': NeuralNetwork_deep,
                      'loss': 'my', 
                      **exp_activ_dict,
                      'w_smoo': w_smoo},
        fit_params={'n_epochs': 3,
                    'momentum': 0.1,
                    'validation_coeff': 0.1,
                    **fit_default,
                    'lr': 1e-1, 'batch_size': 16,}
        ),
    'net2': NetParameters(
        init_params= {**init_default,
                      'model': NeuralNetwork_controlled,
                      'loss': 'my', 
                      **exp_activ_dict,
                      'w_1': 1e-4,
                      'w_2': 1e-4,
                       # 'multipliers': [6,12,20,12],
                       # 'multipliers': [5,7,11,13,15,19,2],
                      'non_linearity': nn.ReLU()
                      },
        fit_params={'n_epochs': 3,
                    'momentum': 0.1,
                    'validation_coeff': 0.1,
                    **fit_default,
                    'lr': 1e-1, 'batch_size': 16,}
        ),
        'net3': NetParameters(
        init_params= {**init_default,
                      'model': NeuralNetwork_controlled,
                      'loss': 'my', 
                      **exp_activ_dict,
                      'w_1': 1e-4,
                      'w_2': 1e-4,
                      'multipliers': [2,8],
                      'non_linearity': nn.ReLU()
                      },
        fit_params={'n_epochs': 3,
                    'momentum': 0.1,
                    'validation_coeff': 0.1,
                    **fit_default,
                    'lr': 1e-1, 'batch_size': 16,}
        ),
        'net4_log': NetParameters(
        init_params= {**init_default,
                      'model': NeuralNetwork_controlled,
                      'loss': PriorSmoothLogLoss, 
                      **exp_activ_dict,
                      'multipliers': [6,12,20,12],
                      'loss_kwargs': {'k_1': 0.1,
                                      'k_2': 0.1,},
                      'non_linearity': nn.ReLU()
                      },
        fit_params={'n_epochs': 3,
                    'momentum': 0.1,
                    'validation_coeff': 0.1,
                    **fit_default,
                    'lr': 1e-1, 'batch_size': 16,}
        ),
        'net4_nonlog': NetParameters(
        init_params= {**init_default,
                      'model': NeuralNetwork_controlled,
                      'loss': PriorSmoothLoss, 
                      **exp_activ_dict,
                      'multipliers': [6,12,20,12],
                      'loss_kwargs': {'w_1': 1e-4,
                                      'w_2': 1e-4,},
                      'non_linearity': nn.ReLU()
                      },
        fit_params={'n_epochs': 3,
                    'momentum': 0.1,
                    'validation_coeff': 0.1,
                    **fit_default,
                    'lr': 1e-1, 'batch_size': 16,}
        ),
        'net5_log': NetParameters(
        init_params= {**init_default,
                      'model': NeuralNetwork_controlled,
                      'loss': PriorSmoothLogLoss, 
                      **exp_activ_dict,
                      # 'multipliers': [6,12,20,12],
                      'multipliers': [1, 1],
                      'loss_kwargs': {'k_1': 0.01,
                                      'k_2': 0.1,
                                      'n_max': n_max},
                      'non_linearity': nn.ReLU()
                      },
        fit_params={'n_epochs': 3,
                    'momentum': 0.1,
                    'validation_coeff': 0.1,
                    **fit_default,
                    'lr': 1e-1, 'batch_size': 16,}
        ),
        'net6_anls_loss': NetParameters(
        init_params= {**init_default,
                      'model': NeuralNetwork_controlled,
                      'loss': AnlsLoss, 
                      **exp_activ_dict,
                      'multipliers': [6,12,20,12],
                      'loss_kwargs': anls_loss_kwargs,
                      'non_linearity': nn.ReLU()
                      },
        fit_params={'n_epochs': 2,
                    'momentum': 0.1,
                    'validation_coeff': 0.1,
                    **fit_default,
                    'lr': 1e-1, 'batch_size': 16,}
        ),
        'net7_anls_loss': NetParameters(
        init_params= {**init_default,
                      'model': NeuralNetwork_controlled,
                      'loss': AnlsLoss, 
                      **exp_activ_dict,
                      'multipliers': [6,12,20,12],
                      'loss_kwargs': anls_loss_kwargs,
                      'non_linearity': nn.ReLU()
                      },
        fit_params={'n_epochs': 2,
                    'momentum': 0.1,
                    'validation_coeff': 0.1,
                    **fit_default,
                    'lr': 1e-1, 'batch_size': 16,}
        ),
        
        'net_gridded': NetParameters(
        init_params= {**init_default,
                      'model': NeuralNetwork_controlled,
                      'loss': AnlsLoss, 
                      **activ_dict,
                      'multipliers': multipliers_grid[nn_i_multiplier],
                      'loss_kwargs': anls_loss_kwargs,
                      'non_linearity': non_lin_dict[non_linearity]
                      },
        fit_params={'n_epochs': n_epochs,
                    'momentum': momentum,
                    'validation_coeff': 0.1,
                    **fit_default,
                    'lr': lr, 'batch_size': batch_size,}
        ),
        
        'net_l1': NetParameters(
        init_params= {**init_default,
                      'model': NeuralNetwork_controlled,
                      'loss': 'l1', 
                      **activ_dict,
                      'multipliers': multipliers_grid[nn_i_multiplier],
                      'loss_kwargs': anls_loss_kwargs,
                      'non_linearity': non_lin_dict[non_linearity]
                      },
        fit_params={'n_epochs': n_epochs,
                    'momentum': momentum,
                    'validation_coeff': 0.1,
                    **fit_default,
                    'lr': lr, 'batch_size': batch_size,}
        ),
        
        'net_l2': NetParameters(
        init_params= {**init_default,
                      'model': NeuralNetwork_controlled,
                      'loss': 'l2', 
                      **activ_dict,
                      'multipliers': multipliers_grid[nn_i_multiplier],
                      'loss_kwargs': anls_loss_kwargs,
                      'non_linearity': non_lin_dict[non_linearity]
                      },
        fit_params={'n_epochs': n_epochs,
                    'momentum': momentum,
                    'validation_coeff': 0.1,
                    **fit_default,
                    'lr': lr, 'batch_size': batch_size,
                    'is_log': True}
        ),
        
        'net_gridded_30_ep': NetParameters(
        init_params = {**init_default,
                      'model': NeuralNetwork_controlled,
                      'loss': AnlsLoss, 
                      **activ_dict,
                      'multipliers': multipliers_grid[nn_i_multiplier],
                      'loss_kwargs': anls_loss_kwargs,
                      'non_linearity': non_lin_dict[non_linearity]
                      },
        fit_params = {'n_epochs': 30,
                    'momentum': momentum,
                    'validation_coeff': 0.1,
                    **fit_default,
                    'lr': lr, 'batch_size': 2500,}
        ),
        
         'net_l2_var': NetParameters(
        init_params= {**init_default,
                      'model': NeuralNetwork_controlled,
                      'loss': L2VarLoss, 
                      **activ_dict,
                      'multipliers': multipliers_grid[nn_i_multiplier],
                      'loss_kwargs': {'l_max': n_max},
                      'non_linearity': non_lin_dict[non_linearity]
                      },
        fit_params={'n_epochs': n_epochs,
                    'momentum': momentum,
                    'validation_coeff': 0.1,
                    **fit_default,
                    'lr': lr, 'batch_size': batch_size,}
        ),
         
         'net_l2_var_30ep': NetParameters(
        init_params= {**init_default,
                      'model': NeuralNetwork_controlled,
                      'loss': L2VarLoss, 
                      **activ_dict,
                      'multipliers': multipliers_grid[nn_i_multiplier],
                      'loss_kwargs': {'l_max': n_max},
                      'non_linearity': non_lin_dict[non_linearity]
                      },
        fit_params={'n_epochs': 30,
                    'momentum': momentum,
                    'validation_coeff': 0.1,
                    **fit_default,
                    'lr': lr, 'batch_size': 2500,}
        ),
         
         'net_l2_var_30ep': NetParameters(
        init_params= {**init_default,
                      'model': NeuralNetwork_controlled,
                      'loss': L2VarLoss, 
                      **activ_dict,
                      'multipliers': multipliers_grid[nn_i_multiplier],
                      'loss_kwargs': {'l_max': n_max},
                      'non_linearity': non_lin_dict[non_linearity]
                      },
        fit_params={'n_epochs': 30,
                    'momentum': momentum,
                    'validation_coeff': 0.1,
                    **fit_default,
                    'lr': lr, 'batch_size': 2500,}
        ),
         
         'net_l2_sqrt': NetParameters(
        init_params= {**init_default,
                      'model': NeuralNetwork_controlled,
                      'loss': L2VarLoss,
                      'predict_sqrt': True,
                      **id_activ_dict,
                      'multipliers': multipliers_grid[nn_i_multiplier],
                      'loss_kwargs': {'l_max': n_max},
                      'non_linearity': non_lin_dict[non_linearity]
                      },
        fit_params={'n_epochs': 30,
                    'momentum': momentum,
                    'validation_coeff': 0.1,
                    **fit_default,
                    'lr': lr, 'batch_size': 2500,}
        ),
         
         'net_l2_sqrtsqrt': NetParameters(
        init_params= {**init_default,
                      'model': NeuralNetwork_controlled,
                      'loss': L2VarLoss,
                      'predict_sqrt': True,
                      **id_activ_dict,
                      'multipliers': multipliers_grid[nn_i_multiplier],
                      'loss_kwargs': {'l_max': n_max},
                      'non_linearity': non_lin_dict[non_linearity]
                      },
        fit_params={'n_epochs': 30,
                    'momentum': momentum,
                    'validation_coeff': 0.1,
                    **fit_default,
                    'lr': lr, 'batch_size': 2500,
                    'is_log': False,
                    'is_sqrt': True}
        ),
         'net_l2_sqrtsqrt_120': NetParameters(
        init_params= {**init_default,
                      'model': NeuralNetwork_controlled,
                      'loss': L2VarLoss,
                      'predict_sqrt': True,
                      **id_activ_dict,
                      'multipliers': [120 / n_bands, 120 / n_bands],
                      'loss_kwargs': {'l_max': n_max},
                      'non_linearity': non_lin_dict[non_linearity]
                      },
        fit_params={'n_epochs': 30,
                    'momentum': momentum,
                    'validation_coeff': 0.1,
                    **fit_default,
                    'lr': lr, 'batch_size': 2500,
                    'is_log': False,
                    'is_sqrt': True}
        ),
        
    }
# ============================================
# ============================================
# ============================================
# ============================================
#%%

# !!!!!
# nets_to_train = ['net_gridded', 'net5_log', 'net_l1', 'net_l2',]
# nets_to_train = ['net_gridded_30_ep', 'net_l2_sqrt']
# nets_to_train = ['net_l2_sqrtsqrt', 'net_l2_sqrt']
nets_to_train = ['net_l2_sqrtsqrt_120']
if not is_nn_trained:
    nets_to_train = []
    print('no training')

for net_name, params in all_nets_params_grid.items():
    print(net_name)
    if net_name not in nets_to_train:
        continue
    print('??????')
    print(f'Training net {net_name}')
    print(f'init_params: {params.init_params}')
    print(f'fit_params: {params.fit_params}')
    net = NeuralNetSpherical(**params.init_params)
    net.fit(band_estim_variances_train_dataset,
            modal_spectrum_train_dataset,
            **params.fit_params)
    if is_Linux:
        torch.save(net, f'NeuralNetworks/models/{net_name}{n_max}.pt')
    else:
        torch.save(net, f'NeuralNetworks/models/{net_name}{n_max}.pt', 
                   _use_new_zipfile_serialization=False)
    net.plot_loss(path_to_save, info=net_name)
    print('net trained')
    # net_l2_var_30ep = net

# assert False

#%%

DLSM_params = [
    kappa_SDxi, kappa_lamb, kappa_gamma,
                        SDxi_med, SDxi_add,
                        lambda_med, lambda_add,
                        gamma_med, gamma_add,
                        mu_NSL, n_max, angular_distance_grid_size
                        ]
c_loc_array = list(map(int, np.logspace(np.log10(500), np.log10(3000), 10)))
if n_max == 29:
    c_loc_array *= 2
# lcz_mx_from_c_loc = dict()
# for c_loc in c_loc_array:
#     lcz_mx_from_c_loc[c_loc] = construct_lcz_matrix(grid, c_loc)
variances = []
spectra = []
ensembles = []
# best_c_loc = []
# n_training_spheres_B = 10
s_array = np.zeros((len(c_loc_array), n_training_spheres_B))

for i_sphere in trange(n_training_spheres_B):
# for i_sphere in trange(1):
    total_start = process_time()
    plt.close('all')
    start = process_time()
    modal_spectrum_train, lambx_train, gammax_train, \
        Vx_train, cx_train = \
        get_modal_spectrum_from_DLSM_params(draw=False,
            seed=seed+i_sphere, *DLSM_params)
    end = process_time()
    print(f'get_modal_spectrum_from_DLSM_params time: {end - start}' )
    start = process_time()
    variance_spectrum_train = get_variance_spectrum_from_modal(
        modal_spectrum_train, n_max
        )
  
    end = process_time()
    print(f'get_variance_spectrum_from_modal time: {end - start}' )
    start = process_time()   
    # modal_spectrum_train_dataset.append(modal_spectrum_train)
 
    W_train, B_train = compute_B_from_spectrum(
        variance_spectrum_train, n_max, grid=grid,
        k_coarse=k_coarse, info='train'
        )
    end = process_time()
    print(f'compute_B_from_spectrum time: {end - start}' )
    start = process_time()  
    alpha_train = np.random.normal(size=(e_size+1, grid.npoints))
    end = process_time()
    print(f'alpha_train time: {end - start}' )
    start = process_time()  
    ensemble_train = np.matmul(W_train, alpha_train.T).T
    end = process_time()
    print(f'ensemble_train time: {end - start}' )
    start = process_time()  
    truth_train = ensemble_train[0,:]
    ensemble_train = ensemble_train[1:,:]
    ensemble_train_fields = [make_2_dim_field(field, grid.nlat,
                              grid.nlon+1) for field in ensemble_train]
    true_field_train = make_2_dim_field(truth_train, grid.nlat,
                                        grid.nlon+1)
    ensemble_train_fields_interp = [interpolate_grid(field,
                                              2) for field in ensemble_train_fields]
    true_field_train_interp = interpolate_grid(true_field_train, 2) 
    end = process_time()
    print(f'true_field_train_interp time: {end - start}' )
    start = process_time()  
    ensembles.append(ensemble_train_fields_interp)
    specoeffs = SHGrid.from_array(
        true_field_train_interp
        ).expand(normalization='ortho')
    spectra.append(
        specoeffs.spectrum()
        # convert_modal_spectrum_to_variance(specoeffs.spectrum(), n_max)
        )
    end = process_time()
    print(f'spectra time: {end - start}' )
    start = process_time()  
        
    var = integrate_gridded_function_on_S2(
            SHGrid.from_array(true_field_train_interp), np.square
            ) / 4 / np.pi
    print(f'variance: {var}')

    variances.append(var)
    print(f'variances: {variances}')

    B_sample = find_sample_covariance_matrix(ensemble_train.T)
    end = process_time()
    print(f'B_sample time: {end - start}' )
    print(f'!!!!!before c_loc time: {end - total_start}' )
    start = process_time()  
    for i_c_loc, c_loc in enumerate(c_loc_array):
        print(f'c_loc: {c_loc}')
        lcz_mx = construct_lcz_matrix(grid, c_loc)
        end = process_time()
        print(f'lcz_mx time: {end - start}' )
        start = process_time()  

        B_sample_loc = np.multiply(B_sample, lcz_mx)
        end = process_time()
        print(f'B_sample_loc time: {end - start}' )
        start = process_time()  

        s = make_analysis(B_sample_loc, true_field_train, grid, n_obs=n_obs,
            obs_std_err=obs_std_err, n_seeds=n_seeds, seed=seed)
        end = process_time()
        print(f'make_analysis time: {end - start}' )
        start = process_time()  
        print(f's: {s}')
        s_array[i_c_loc, i_sphere] = s
end = process_time()
print(f'total time: {end - total_start}' )
s_array_mean = s_array.mean(axis=1)
best_c = c_loc_array[np.argmin(s_array_mean)]
# best_s = s_array[np.argmin(s_array)]

plt.figure()
plt.plot(c_loc_array, s_array_mean)
plt.grid()
plt.title(f'best c_loc={best_c}')
plt.savefig(path_to_save + 'best_c_loc.png')
plt.show()


# plt.figure()
# plt.hist(best_c_loc)
# plt.grid()
# plt.title('c_loc')
# plt.savefig(path_to_save + 'best_c_loc.png')
# plt.show()

# best_c_loc = np.mean(best_c_loc)
precomputed_c_locs = {'c_loc': best_c}
np.save('best_c_loc.npy', precomputed_c_locs)
print('saved c_loc')
ensembles = np.array(ensembles)
# assert False
# draw_1D(ensembles.mean(axis=()))
#%%
spectra = []
for i in trange(len(ensembles)):
    for field in ensembles[i]:
        specoeffs = SHGrid.from_array(
            field
            ).expand(normalization='ortho')
        spectra.append(
            specoeffs.spectrum()
            # convert_modal_spectrum_to_variance(specoeffs.spectrum(), n_max)
            )
spectra = np.array(spectra)

draw_1D(spectra.mean(axis=0))

#%%

spectrum_mean = spectra.mean(axis=0)   

draw_1D(convert_modal_spectrum_to_variance(spectrum_mean, n_max))            
# draw_1D(spectrum_mean)
# spectrum_mean_3d = np.transpose(np.tile(
#     spectrum_mean, (*modal_spectrum_train_dataset.shape[1:], 1)
#     ), (2,0,1))
# draw_1D(spectrum_mean.mean(axis=(1,2)))
# print(spectrum_mean.shape, modal_spectrum_train_dataset.shape)
# variance_spectrum_3d = convert_modal_spectrum_to_variance(
#     spectrum_mean_3d, n_max
#     )
# variance_spectrum_3d = spectrum_mean_3d
# _, B_mean = compute_B_from_spectrum(variance_spectrum_3d, n_max,
#                                               grid=grid,
#                                               k_coarse=k_coarse, info='true')
B_mean = np.zeros(grid.rho_matrix.shape)

for x in range(grid.npoints):
    
    B_x_rho = Fourier_Legendre_transform_backward(
        n_max, convert_variance_spectrum_to_modal(spectrum_mean, n_max),
        rho_vector=grid.rho_matrix[x,:]
        )
    B_mean[x,:] = B_x_rho

# draw_2D(B_mean)

# draw_2D(B_mean_ - B_mean)
# draw_1D(np.diag(B_mean))

np.savetxt(f'B_mean {n_training_spheres_B}.txt', B_mean)
print('B_mean saved')

#%%
# specoeffs = SHGrid.from_array(
#     true_field_train_interp
#     ).expand(normalization='ortho')


# # convert_modal_spectrum_to_variance
# # convert_variance_spectrum_to_modal(variance_spectrum, n_max)
# draw_1D(convert_modal_spectrum_to_variance(specoeffs.spectrum(), n_max) )
# specoeffs.plot_spectrum(show=False)
# # specoeffs.plot_spectrum2d(cmap_rlimits=(1.e-7, 0.1),
# #                               show=False)



