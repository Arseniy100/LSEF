# -*- coding: utf-8 -*-

"""
Created on Fri Feb 28 15:47:28 2020
Last modified: April 2022

@author: Arseniy Sotskiy
"""

# cd Arseniy/LSM_S2
# git config --global http.proxy http://:@192.168.101.250:3128
# cat nohup.out | tr '\r' '\n'
# tail -f nohup.out 

import sys
if '/RHM-Lustre3.2/software/general/plang/Anaconda3-2019.10' in sys.path:
    is_Linux = True
else:
    is_Linux = False
    # sys.path.append(r'/RHM-Lustre3.2/users/wg-da/asotskiy/packages')
sys.path

if is_Linux:
    print('working on super-computer')
else:
    print('working on personal computer')


# cd myenv
# source bin/activate

 # WARNING: The scripts f2py, f2py3 and f2py3.7 are installed in '/RHM-Lustre3.2/users/wg-da/asotskiy/.local/bin' which is not on PATH.
 #  Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.

 # Linux xfront7 4.12.14-195-default #1 SMP Tue May 7 10:55:11 UTC 2019
 # (8fba516) x86_64 x86_64 x86_64 GNU/Linux

# !!! set HTTPS_PROXY=http://@192.168.101.250:3128

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 10000
import numpy as np
import pandas as pd
import os

from time import process_time, strftime

from pyshtools import SHGrid

COLORMAP = plt.cm.seismic

from configs import std_xi_mult, std_xi_add, lamb_mult, lamb_add, \
    gamma_mult, gamma_add, mu_NSL, n_training_spheres, w_smoo, \
    n_max,k_coarse, kappa_std_xi, kappa_lamb, kappa_gamma, threshold_coeff, \
    is_best_c_loc_needed, c_loc, e_size, n_obs, obs_std_err, n_seeds, \
    angular_distance_grid_size


from functions.tools import draw_1D, draw_2D, find_sample_covariance_matrix, \
     integrate_gridded_function_on_S2, _time, \
    convert_variance_spectrum_to_modal, make_matrix_sparse
from functions.FLT import Fourier_Legendre_transform_backward
from functions.process_functions import predict_spectrum_from_bands
from functions.DLSM_functions import compute_u_from_sigma, \
    compute_W_from_u, make_analysis, get_band_variances, \
    draw_MAE_and_bias_of_spectrum, get_band_varinaces_dataset_for_nn
from functions.lcz import construct_lcz_matrix, find_best_c_loc
from functions.R_progs import fit_shape_S2, CreateExpqBands #, V2b_shape_S2

  
from Band import Band_for_sphere

from RandomProcesses import get_modal_spectrum_from_DLSM_params

from Grids import coarsen_grid, flatten_field, make_2_dim_field, \
    interpolate_grid, LatLonSphericalGrid

from neural_network import NeuralNetSpherical, SquareActivation,\
    AbsActivation, ExpActivation





start = process_time()

# cd /mnt/c/Users/Арсений\ HP-15/Python/Росгидромет/LSM_S2
# cd C:/Users/user410/Arseniy/lsm_s2


# next - поиграть с бандами

#%% Making directory



grid_parameter = 'mu_NSL' # 'kappa', 'e_size'
# parameter which is optimised

path_to_save = 'images\ ' + _time() + \
    f" {n_training_spheres} training spheres, w_smoo {w_smoo}" + r'\ '
if is_Linux:
    path_to_save = path_to_save.replace(r'\ ', r'/')
path_to_save = path_to_save.replace(':', '_')
print(path_to_save)
os.mkdir(path_to_save)
# os.rmdir(r'images\2021_12_15_15_19_07 20 training spheres, w_smoo 0.0001, ')
print(f'made directory {path_to_save}')

global n_png
n_png = 0



#
# import subprocess
# #using strings
# path = "images/ Wed Jun  2 20_30_38 2021 e_size 5 "
# subprocess.run(["rm", "-rf", path])


#%% Creating some constant variables and setting random seed

n_iterations = 1
np.random.seed()
my_seed_mult = 123456 #np.random.randint(100000) # 44241
print(f'seed: {my_seed_mult}')
e_sizes = [e_size]

# path_to_save = r'images\biases in spatial covariances ' +f'{kappa_gamma}' + r'\ '
# path_to_save

gamma_med = gamma_add + gamma_mult



assert (len(e_sizes) in [1, n_iterations])




#%% Setting param grids and arrays for this grid optimization


params = {'n_max': n_max, 'nband': 6, 'halfwidth_min': 1.5  * n_max/30,
          'nc2': 1.5 * n_max / 22, 'halfwidth_max': 1.2 * n_max / 4,
          'q_tranfu': 3, 'rectang': False}
param_grid = [params]


# a_max_times = max deviation of the scale multiplier  a  from 1 in times
# w_a_fg = weight of the ||a-1||^2 weak constraint in fitScaleMagn
params_shape = {'moments': "012", 'a_max_times': 5, 'w_a_fg': 0.05}
param_shape_grid = [params_shape]


R_s_on_grid = []
R_m_on_grid = []
R_s_nn_on_grid = []
R_m_nn_on_grid = []
tlsmn_on_grid = []

# gamma_mult_grid = np.array([3, 2.5, 2])
# Gamma_grid = (gamma_mult_grid + np.array([0,1,2])) * 6 / 5

gamma_med_grid = np.array([3, 2.5, 2])
Gamma_grid = (gamma_med_grid + np.array([0, 1,2]))

#%%            PARAM GRID CYCLE


# for threshold_coeff in [0]: # , 0.0001, 0.001, 0.01, 0.1]:
# for mu_NSL in [3]:
# for kappa in [1,2,3,4]:
    # kappa_std_xi, kappa_lamb, kappa_gamma = kappa, kappa, kappa
# for params in param_grid:
    # print(params)
# for e_sizes in [[5], [10], [20], [40], [80]]:
# for params_shape in param_shape_grid:
#     print(params_shape)
# for i_gamma in range(len(gamma_med_grid)):
    # params_shape = params_shape_grid[0]
    
for _here_can_be_cycle_for_any_grid in range(1):

    n_bands = params['nband']
    print('e_size', e_sizes[0])
    
    try:
        transfer_functions, _, __ = CreateExpqBands(**params)
    except:
        print('Error')
        R_s_on_grid.append(-1)
        R_m_on_grid.append(-1)
        with open('grid_results.txt', 'a') as file:
            file.write('params = ' + str(params) + '\n')
            file.write('R_s = -1 # error - no bands \n')
        continue

    bands = [Band_for_sphere(transfer_functions[:,i]) for i in range(n_bands)]
    print(transfer_functions.shape)
    # n_bands = int(n_max / 5) # !!!!!!!!!!

    band_centers = [band.center for band in bands]

    bands_to_plot_indices =  np.arange(n_bands) # [n_bands-1, int(n_bands/2), 0]
    plt.figure()
    for i in bands_to_plot_indices:
        band = bands[i]
        plt.plot(np.arange(band.shape),
                  band.transfer_function,
                  label=f'band {i}')
    plt.grid()
    # plt.legend(loc='best')
    plt.title('transfer functions')
    plt.xlabel('wavenumber')
    n_png += 1
    plt.savefig(path_to_save + f'{n_png} transfer functions.png')
    plt.show()

    plt.figure()
    for i in bands_to_plot_indices:
        band = bands[i]
        plt.plot(
            # np.arange(band.shape),
                  np.linspace(0, np.pi, band.shape),
                  band.response_function / np.max(band.response_function),
                  label=f'band {i}')
    plt.grid()
    # plt.legend(loc='best')
    plt.title('normalized response functions')
    plt.xlabel('distance, rad.')
    n_png += 1
    plt.savefig(path_to_save + f'{n_png} normalized response functions.png')
    plt.show()

    Omega = np.array(
        [[(bands[j].transfer_function[l]**2) for l in range(n_max+1)]
          for j in range(n_bands)]
        )

    U, d, V_T = np.linalg.svd(Omega, full_matrices=False)
    V = V_T.T

    bands_to_plot_indices =  np.arange(n_bands) # [n_bands-1, int(n_bands/2), 0]

    plt.figure()
    for i in bands_to_plot_indices:
        band = bands[i]
        plt.plot(np.arange(band.shape),
                  band.transfer_function,
                  label=f'band {i}')
    plt.grid()
    plt.legend(loc='best')
    plt.title('transfer functions')
    plt.show()

    plt.figure()
    for i in bands_to_plot_indices:
        band = bands[i]
        plt.plot(np.arange(band.shape),
                  band.response_function / np.max(band.response_function),
                  label=f'band {i}')
    plt.grid()
    plt.legend(loc='best')
    plt.title('normalized response functions')
    plt.show()



    #%%
    MAE_from_e_size = np.zeros((n_bands, n_iterations))
    MAE_rel_from_e_size = np.zeros((n_bands, n_iterations))

    t_array = []
    l_array = []
    s_array = []
    m_array = []
    n_array = []

    r_dict = dict()
    
    #%%        OUTER CYCLE WITH DIFFERENT ITERATIONS

    # for iteration in range(n_iterations):  
    for flag, iteration in enumerate([16,]): 
    # for iteration in range(len(gamma_mult_grid)):  

        print('\n' * 5)
        print('=' * 20)
        print(f'iteration {iteration}')
        seed = my_seed_mult * (iteration + 1)
        np.random.seed(seed)
        if iteration == 16:
        # if flag != 0:
            draw = True
        else:
            draw = False

        #%% Get modal spectrum b_n(x)

        # After the processes V (x), lambda(x), and gamma(x) are computed
        # at each analysis grid point, Eq.(38) is used to find c(x).
        # With c(x), lambda(x), and gamma(x) in hand, we finally compute
        # the "true" modal spectrum bn(x) using Eq.(39)
        variance_spectrum, lambx, gammax, Vx, cx = \
            get_modal_spectrum_from_DLSM_params(
                kappa_std_xi, kappa_lamb, kappa_gamma,
                std_xi_mult, std_xi_add,
                lamb_mult, lamb_add,
                gamma_mult, gamma_add,
                mu_NSL, n_max, angular_distance_grid_size,
                draw=draw,
                seed=seed)
                
            
        
        #%%
        # V_q = np.quantile(np.sqrt(Vx), q=[0.1, 0.9])
        # print('std:', V_q[1] / V_q[0])
        # l_q = np.quantile(lambx, q=[0.1, 0.9])
        # print('lambda:', l_q[1] / l_q[0])
        # print(Vx.max()/ Vx.min())
        # print(lambx.max()/ lambx.min())

        # variance: 0.8959205005736177 0.911263188299299 1.050137427975056
        #%%
        modal_spectrum = convert_variance_spectrum_to_modal(variance_spectrum,
                                                            n_max)


        #% Get sigma_n(x) from modal spectrum

        # Then, following Eqs.(16) and (17), we
        # compute \tild{u}_n(x) = sigma_n(x) = sqrt(pbn(x))     (45)

        sigma = np.sqrt(modal_spectrum)
        
        #% iFLT: sigma to u(x; rho)

        # Next, we make use of Eq.(8) to apply the inverse Fourier-Legendre transform
        # and get the function u(x; rho) (at each grid point x independently)
        if draw:
            print('Making grids')
        sigma_coarsened = np.array(
            [coarsen_grid(sigma[i,:,:], k_coarse) for i in range(len(sigma))]
            )

        sigma_coarsened_flattened = np.array(
            [flatten_field(sigma_coarsened[i,:,:]) for i in range(len(sigma))]
            )

        _, nlat, nlon = sigma_coarsened.shape

        grid = LatLonSphericalGrid(nlat)

        points = [(np.random.randint(grid.nlon),
                           np.random.randint(grid.nlat)) for _ in range(5)]
        if draw:
            plt.figure()
            for point in points:
                plt.plot(variance_spectrum[:, point[0], point[1]])
            plt.grid()
            plt.title('variance_spectrum')
            # n_png += 1
            # plt.savefig(path_to_save + f'{n_png} variance spectrum.png')

            plt.show()
            plt.figure()
            for point in points:
                plt.plot(modal_spectrum[:, point[0], point[1]])
            plt.grid()
            plt.title('modal_spectrum')
            # n_png += 1
            # plt.savefig(path_to_save + f'{n_png} modal spectrum.png')

            plt.show()

        u = compute_u_from_sigma(sigma_coarsened_flattened, grid, n_max)

        #% After that, we build the W matrix using Eq.(12)

        W = compute_W_from_u(u, grid)
        # W = make_matrix_sparse(W)

        #% The W matrix is then used to compute the covariance matrix B
        # (only if absolutely necessary) using Eq.(24)

        B = np.matmul(W, W.T)

        #% Generating ensemble:

        plt.close('all')

        if len(e_sizes) == 1:
            e_size = e_sizes[0]
        else:
            e_size = e_sizes[iteration]

        alpha = np.random.normal(size=(e_size+1, grid.npoints))
        ensemble = np.matmul(W, alpha.T).T
        truth = ensemble[0,:]
        ensemble = ensemble[1:,:]

        ensemble_fields = [make_2_dim_field(field, grid.nlat,
                                            grid.nlon+1) for field in ensemble]
        true_field = make_2_dim_field(truth, grid.nlat,
                                            grid.nlon+1)
        
        ensemble_fields_interp = [interpolate_grid(field,
                                                  2) for field in ensemble_fields]
        true_field_interp = interpolate_grid(true_field, 2)
        
        if draw:
            n_png += 1
            draw_2D(true_field_interp,
                    title='X_true',
                    save_path=path_to_save + f'{n_png}_true_field_interp.png'
                    )
            n_png += 1
            draw_2D(true_field,
                    title='X_true',
                    save_path=path_to_save + f'{n_png}_true_field.png'
                    )

            draw_1D(true_field_interp[50,:],
                    title=f'true_field_interp[50,:], gamma_med={gamma_med}',
                    # save_path=path_to_save + f'true_field_interp[50], gamma_med={gamma_med}.png'
                    )

        i=2250
        draw_1D(B[i,i:i+100], title=f'B[{i},{i}:{i+100}]')

        # assert False


        shgrid = SHGrid.from_array(true_field_interp)
        # clm = shgrid.expand()
        # band_shape = clm.lmax + 1        
                
        points = []
        band_variances_dict = \
            get_band_variances(bands, n_bands, variance_spectrum, 
                       ensemble_fields_interp, points)
            # Variances are divided by band.c (!)
            
        band_mean_estim_variances = band_variances_dict['band_mean_estim_variances']
        band_estim_variances = band_variances_dict['band_estim_variances']
        band_mean_true_variances = band_variances_dict['band_mean_true_variances']
        band_true_variances = band_variances_dict['band_true_variances']
        band_estim_variances_at_points = band_variances_dict['band_estim_variances_at_points']
        band_mean_true_variances = np.array(band_mean_true_variances)

        
        #%%  Training dataset for the neural network
        
        band_true_variances_train_dataset,\
            band_estim_variances_train_dataset, \
            variance_spectrum_train_dataset = \
                get_band_varinaces_dataset_for_nn(
                    n_training_spheres, grid, n_max, k_coarse, 
                    e_size, bands, seed, 
                    DLSM_params = [
                        kappa_std_xi, kappa_lamb, kappa_gamma,
                        std_xi_mult, std_xi_add,
                        lamb_mult, lamb_add,
                        gamma_mult, gamma_add,
                        mu_NSL, n_max, angular_distance_grid_size
                        ]
                    )
        #%% !!!!!!!!!!!!!!!!!!!!!!!!
        # !!!!!!!!
        

        
        activation_funcs = {
            'ExpActivation': ExpActivation(),
            # 'SquareActivation': SquareActivation(),
            # 'Softplus': nn.Softplus(beta=1000),
            # # 'Hardswish': nn.Hardswish(),
            # 'GELU': nn.GELU(),
            # 'Sigmoid': nn.Sigmoid(), # does not work
            # # 'Mish': nn.Mish(),
            }
        
        # lr_grid = (0.1, 0.01, 0.001, )
        # batch_size_grid = (3,5,10,20)
        
        lr_grid = (1e-5, 1e-4, 1e-3, 1e-2, 1e-1, )
        batch_size_grid = (5, 10, 20)
        
        
        param_grid_2d = [
            [{'lr': lr, 'batch_size': batch_size} for  lr in lr_grid]
            for batch_size in batch_size_grid]
        param_grid = []
        for gr in param_grid_2d:
            param_grid += gr
            
        param_grid = [ {'lr': 1e-1, 'batch_size': 16, }]
                        
        
        
        for activation_name, activation_function in activation_funcs.items():
            # for rand_seed in range(3)
            for loss in [ 'my']: # , 'l1']:
                # loss = 'l1'
                net = NeuralNetSpherical(
                    n_bands, n_max, loss=loss, 
                    activation_name=activation_name, 
                    activation_function=activation_function,
                    w_smoo=w_smoo)
                for params in param_grid:
                    # net.fit(  # l1 loss
                    #     np.log(band_estim_variances_train_dataset),
                    #     # band_estim_variances_train_dataset,
                    #     # band_true_variances_train_dataset, 
                    #     variance_spectrum_train_dataset, 
                    #     n_epochs=5, momentum=0.1,
                    #     validation_coeff=0.1,
                    #     path_to_save=path_to_save,
                    #     draw=True,
                    #     **params
                    #     ) # l1 loss
                    
                    
                    net.fit(
                        band_estim_variances_train_dataset,
                        # np.log(band_estim_variances_train_dataset),
                        # np.log(band_true_variances_train_dataset),
                        variance_spectrum_train_dataset, 
                        n_epochs=1,
                        momentum=0.1,
                        validation_coeff=0.1,
                        path_to_save=path_to_save,
                        draw=True,
                        band_centers=band_centers,
                        **params
                        ) # my loss
                    #%%
                    
                    # variance_spectrum_nn = net.predict(np.log(band_estim_variances))
                    variance_spectrum_nn = net.predict(band_estim_variances)
                    
                    modal_spectrum_nn = convert_variance_spectrum_to_modal(
                        variance_spectrum_nn, n_max
                        )
            
                    # draw_1D(np.array([variance_spectrum_nn[:,0,0]] + \
                    #                  [variance_spectrum[:,i,i] for i in (5, 20, 30, 40)]).T,
                    #         title=f'nn and true var spectrum')
                    sigma_nn = np.sqrt(modal_spectrum_nn)
                    sigma_coarsened_nn = np.array(
                        [coarsen_grid(sigma_nn[i,:,:], k_coarse) for i in range(len(sigma))]
                        )
                    sigma_coarsened_flattened_nn = np.array(
                        [flatten_field(sigma_coarsened_nn[i,:,:]) for i in range(len(sigma))]
                        )
                    u_nn = compute_u_from_sigma(sigma_coarsened_flattened_nn, grid, n_max)
                    W_nn = compute_W_from_u(u_nn, grid)
                    B_nn = np.matmul(W_nn, W_nn.T)
                    
                    
            
                    #%%
            
                    i = np.random.randint(grid.nlon)
                    j = np.random.randint(grid.nlat)
                    V_e = band_estim_variances[:,i,j] * np.array([band.c for band in bands])
                    e = variance_spectrum[:,i,j]
            
                    # draw_1D(e)
            
                    # plt.figure()
                    # plt.plot(np.matmul(Omega, e))
                    # plt.plot(np.matmul(Omega, e) - V_e)
                    # plt.show()
                    # print(
                    #     np.max(np.abs(
                    #     np.matmul(Omega, e) - V_e
                    #     ))
                    #     )
            
                    draw_1D(np.array([V @ V.T @ e, e]).T, title='V @ V.T @ e, e')
            
                    draw_1D(V @ V.T @ e - e)
            
                    # continue
            
                    # assert False
            
            
            
                    #% % predict spectrum (modal) from b_mean (band variances):
            
                    # variance_spectrum_estim = predict_spectrum_from_bands(
                    #     band_estim_variances, bands=bands,
                    #     left_is_horizontal=left_is_horizontal,
                    #     loglog=loglog
                    #     )
            
                    all_bands_c = np.array([band.c for band in bands])
            
                    is_type_svd = True
            
                    variance_spectrum_estim = predict_spectrum_from_bands(
                        # band_true_variances, # all_bands_c[:, np.newaxis, np.newaxis],
                        # band_true_variances, # all_bands_c[:, np.newaxis, np.newaxis],
                        band_estim_variances, # * all_bands_c[:, np.newaxis, np.newaxis],
                        svd=is_type_svd,
                        n_max=n_max,
                        bands=bands,
                        precomputed=True,
                        Omega=Omega, U=U, d=d, V=V, nSV_discard=0
            
                        )
                    
                    # variance_spectrum_estim = spectrum_nn
            
                    # if is_type_svd:
                    #     info = 'type=svd' #', {iteration} iter'
                    # else:
                    #     info = 'type=lines' #', {iteration} iter'
                    info = ''
            
                    mean_variance_spectrum_estim = [integrate_gridded_function_on_S2(
                                SHGrid.from_array(variance_spectrum_estim[i,:,:])
                                ) / 4 / np.pi for i in range(variance_spectrum_estim.shape[0])]
                    mean_band_estim_variances = band_estim_variances.mean(axis=(1,2))
            
            
                    #%%
                    
                    points = [[
                        (np.random.randint(60), np.random.randint(60))
                        for _ in range(3)
                        ] for _ in range(3)]
                    if draw:
                        fig, axes = plt.subplots(3, 3, 
                                                 # sharex=True, sharey=True,
                                                 figsize=(8, 10))
                        fig.suptitle('Sharing x per column, y per row')
                        # np.random.seed(1111)
                        for i in range(3):
                            for j in range(3):
                                
                                point=points[i][j]
                                axes[i,j].plot(np.arange(variance_spectrum.shape[0]),
                                         variance_spectrum[:, point[0], point[1]],
                                         label='true', color='black')
                                axes[i,j].plot(np.arange(variance_spectrum_estim.shape[0]),
                                          variance_spectrum_nn[:, point[0], point[1]], label='nn',
                                          color='purple', alpha=0.9)
                                axes[i,j].scatter(
                                [band.center for band in bands],
                                # band_estim_variances[:, point[0], point[1]] #/ all_bands_c
                                band_estim_variances[:, point[0], point[1]] #/ all_bands_c
                                )
                                axes[i,j].grid()
                                axes[i,j].legend(loc='best')
                        title = 'SPECTRUM\n' + net.info
                        fig.suptitle(title)
                        title = title.replace('\n', '')
                        # n_png += 1
                        plt.savefig(path_to_save + f'{n_png} {title}.png')
                        plt.show()
                        
                        # plt.figure()
                        # plt.plot(np.log(
                        #          transform_spectrum_to_derivative_1d(
                        #              variance_spectrum_estim[:, point[0], point[1]]
                        #              ) + 0.0001), label='estim',
                        #          color='orange')
                        # plt.plot(np.log(
                        #          transform_spectrum_to_derivative_1d(
                        #              variance_spectrum[:, point[0], point[1]]
                        #              ) + 0.0001),
                        #          label='true', color='black')
                        # # plt.scatter(
                        # # [band.center for band in bands],
                        # # # band_estim_variances[:, point[0], point[1]] #/ all_bands_c
                        # # band_true_variances[:, point[0], point[1]] #/ all_bands_c
                        # # )
                        # plt.grid()
                        # plt.legend(loc='best')
                        # title = f'variance_spectrum before shape, point {point}, {info}'
                        # plt.title(title)
                        # # n_png += 1
                        # # plt.savefig(path_to_save + f'{n_png} {title}.png')
                        # plt.show()
            
            #%%
            
            point=(np.random.randint(60), np.random.randint(60))

            plt.figure()

            plt.plot(np.arange(variance_spectrum.shape[0]),
                     modal_spectrum[:, point[0], point[1]],
                     label='true', color='black')
            plt.plot(np.arange(variance_spectrum_estim.shape[0]),
                     convert_variance_spectrum_to_modal(
                         variance_spectrum_estim, n_max
                         )[:, point[0], point[1]], label='estim',
                     color='orange')
            # plt.scatter(
            # [band.center for band in bands],
            # band_true_variances[:, point[0], point[1]] #/ all_bands_c
            # )
            plt.grid()
            plt.legend(loc='best')
            title = f'modal_spectrum before shape, point {point}, {info}'
            plt.title(title)
            # n_png += 1
            # plt.savefig(path_to_save + f'{n_png} {title}.png')
            plt.show()


        plt.close('all')





        #%%

        # abs_error = np.abs(band_estim_variances - \
        #              np.array(band_mean_true_variances).reshape((-1,1,1)))
        # bias = (band_estim_variances - \
        #              np.array(band_mean_true_variances).reshape((-1,1,1)))


        abs_error = np.abs(band_estim_variances - band_true_variances)
        bias = (band_estim_variances - band_true_variances)

        MAE_b_mean = np.array([integrate_gridded_function_on_S2(
                    SHGrid.from_array(abs_error[i,:,:])
                    ) / 4 / np.pi for i in range(len(bands))])
        bias_b_mean = np.array([integrate_gridded_function_on_S2(
                    SHGrid.from_array(bias[i,:,:])
                    ) / 4 / np.pi for i in range(len(bands))])

        if draw:
            plt.figure()
            # for var in all_mean_band_estim_variances:
            #     plt.scatter([band.center for band in bands],
            #                 var,
            #                 color='orange', s=5)

            c_j_all_bands = np.array([band.c for band in bands])

            plt.plot([band.center for band in bands],
                        band_mean_true_variances * c_j_all_bands,
                        label='true', color='black')
            plt.plot([band.center for band in bands],
                      bias_b_mean * c_j_all_bands, label='bias', color='green')
            plt.plot([band.center for band in bands],
                      MAE_b_mean * c_j_all_bands, label='MAE', color='red')
            # plt.scatter([band.center for band in bands],
            #             mean_band_estim_variances * c_j_all_bands,
            #             label='estim', color='orange', s=5)
            # plt.scatter([band.center for band in bands],
            #             all_mean_band_estim_variances.mean(axis=0),
            #             label='estim mean', color='red', s=24)

            # plt.plot(np.arange(variance_spectrum.shape[0]),
            #          mean_variance_spectrum_estim, label='mean estim', color='green')
            plt.legend(loc='best')

            plt.xlabel('l')
            plt.ylabel('$V_{(j)}$')
            plt.grid()
            title = f'band variances (mean over sphere), e_size={e_size}, {info}'
            plt.title(title)
            n_png += 1
            plt.savefig(path_to_save + f'{n_png} {title}.png')
            plt.show()


        # all_mean_band_estim_variances.std(axis=0)
        # all_mean_band_estim_variances.mean(axis=0) - band_mean_true_variances







        #%%

        MAE_spectrum = [integrate_gridded_function_on_S2(
                    SHGrid.from_array(np.abs(
                        variance_spectrum_estim - variance_spectrum
                        )[i,:,:])
                    ) / 4 / np.pi for i in range(variance_spectrum_estim.shape[0])]
        bias_spectrum = [integrate_gridded_function_on_S2(
                    SHGrid.from_array(
                        (variance_spectrum_estim - variance_spectrum)[i,:,:]
                        )
                    ) / 4 / np.pi for i in range(variance_spectrum_estim.shape[0])]
        variance_spectrum_mean = [integrate_gridded_function_on_S2(
                    SHGrid.from_array(variance_spectrum[i,:,:])
                    ) / 4 / np.pi for i in range(variance_spectrum_estim.shape[0])]

        if draw:
            plt.figure()
            plt.plot(np.arange(variance_spectrum.shape[0]),
                     variance_spectrum_mean,
                     label='true (mean)', color='black')
            plt.plot(np.arange(variance_spectrum.shape[0]),
                     MAE_spectrum, label='MAE', color='red')
            plt.plot(np.arange(variance_spectrum.shape[0]),
                     bias_spectrum, label='bias', color='green')


            # for point in [(0,0), (45,45), (30,60)]:
            #     plt.plot(np.arange(variance_spectrum_estim.shape[0]),
            #               variance_spectrum_estim[:, point[0], point[1]], label='estim',
            #               color='orange')
            # plt.scatter([band.center for band in bands],
            #             band_mean_true_variances,
            #             label='true')
            # for point in [(0,0), (45,45), (30,60)]:
            #     plt.scatter([band.center for band in bands],
            #                 band_estim_variances[:, point[0], point[1]],
            #                 label='estim', color='orange')
            plt.plot(np.arange(variance_spectrum.shape[0]),
                      mean_variance_spectrum_estim,
                      label='mean estim (ensm)', color='orange')

            plt.legend(loc='best')


            plt.xlabel('l')
            plt.ylabel('b')
            plt.grid()
            title = f'variance spectrum, e_size={e_size}, type=nn'
            plt.title(title)
            # n_png += 1
            # plt.savefig(path_to_save + f'{n_png} {title}.png')
            plt.show()







        #%%
        modal_spectrum_estim = convert_variance_spectrum_to_modal(
            variance_spectrum_estim, n_max
            )
        sigma_estim = np.sqrt(modal_spectrum_estim.clip(0))
        sigma_estim_coarsened = np.array(
            [coarsen_grid(sigma_estim[i,:,:], k_coarse) for i in range(len(sigma))]
            )

        sigma_estim_coarsened_flattened = np.array(
            [flatten_field(sigma_estim_coarsened[i,:,:]) for i in range(len(sigma))]
            )

        sqs = np.array([(2*l+1) * (sigma_estim[l,:,:] -
                                   sigma[l,:,:])**2 / (4 * np.pi)
                        for l in range(sigma.shape[0])]).sum(axis=0)
        sqs_mean = integrate_gridded_function_on_S2(
            SHGrid.from_array(sqs)
            ) / (4 * np.pi)

        mae = np.array([(2*l+1) * np.abs(sigma_estim[l,:,:]**2 -
                                         sigma[l,:,:]**2) / (4 * np.pi)
                        for l in range(sigma.shape[0])]).sum(axis=0)
        mae_mean = integrate_gridded_function_on_S2(
            SHGrid.from_array(mae)
            ) /(4 * np.pi)

        # if draw:
        #     draw_2D(sqs,
        #             title = f'sqs, {type_line}, {type_loglog}, mean='\
        #                 + f'{sqs_mean:.4}')
        #     draw_2D(mae,
        #             title = f'mae, {type_line}, {type_loglog}, mean='\
        #                 + f'{mae_mean:.4}')
        #     print(mae_mean)




        #%

        u_estim = compute_u_from_sigma(sigma_estim_coarsened_flattened,
                                       grid, n_max)

        W_estim = compute_W_from_u(u_estim, grid)



        if draw:
            # draw_2D(u, title='u')
            # draw_2D(u_estim, title='u_estim')
            # draw_2D(u - u_estim, title='u - u_estim')
            # draw_1D(u[600,:] - u_estim[600,:], title='u - u_estim [20,:]')
            # print(np.max(np.abs(u - u_estim)))

            # draw_2D(W, title='W')
            # draw_2D(W_estim, title='W_estim')
            # draw_2D(W - W_estim, title='W - W_estim')
            # draw_1D(W[600,:] - W_estim[600,:], title='W - W_estim [20,:]')

            # print(np.max(np.abs(W - W_estim)))

            # plt.figure()
            # plt.plot(np.arange(grid.npoints),
            #           np.diag(W_estim), label='estim',
            #           color='orange')
            # plt.plot(np.arange(grid.npoints),
            #           np.diag(W), label='true')
            # plt.legend(loc='best')
            # plt.title(f'W, diagonal')
            # plt.grid()
            # plt.show()


            # plt.figure()
            # plt.plot(np.arange(grid.npoints),
            #           W_estim[0,:], label='estim',
            #           color='orange') # , s=6)
            # plt.plot(np.arange(grid.npoints),
            #           W[0,:], label='true') # , s=6)
            # plt.legend(loc='best')
            # plt.title(f'W[0,:]')
            # plt.grid()
            # plt.show()

            # plt.figure()
            # plt.plot(np.arange(grid.npoints),
            #           W_estim[:,0], label='estim',
            #           color='orange')
            # plt.plot(np.arange(grid.npoints),
            #           W[:,0], label='true')
            # plt.legend(loc='best')
            # plt.title(f'W[:,0]')
            # plt.grid()
            # plt.show()
            pass




        #%%
        B_estim = np.matmul(W_estim, W_estim.T)


        B_theor = Fourier_Legendre_transform_backward(
            n_max, modal_spectrum[:,0,0], rho_vector_size = grid.npoints
            )
        # for modal spectrum; must be changed to non-stationary case
        # and sigma(x), sigma(x').

        if draw:
            # draw_2D(B, title='B')
            # draw_2D(B_estim, title='B_estim')
            # draw_2D((B - B_estim)[:400,:400], title='B - B_estim')
            # draw_1D(np.diag(B - B_estim)[:4000], title='B - B_estim')
            # print(np.max(np.abs(B - B_estim)))

            # plt.figure()
            # plt.plot(np.arange(grid.npoints),
            #          np.diag(B_estim), label='estim',
            #          color='orange')
            # plt.plot(np.arange(grid.npoints),
            #          np.diag(B), label='true')
            # plt.legend(loc='best')
            # plt.title(f'B, diagonal')
            # plt.grid()
            # plt.show()

            plt.figure()
            
            plt.scatter(grid.rho_matrix[:,::50].flatten(), B_nn[:,::50].flatten(),
                     label='estim (nn)',
                     color='purple' , s=0.1)
            plt.scatter(grid.rho_matrix[:,::50].flatten(), B_estim[:,::50].flatten(),
                     label='estim (svd)',
                     color='orange' , s=0.1, alpha=0.5)
            plt.plot(np.linspace(0, np.pi, grid.npoints),
                # np.arange(grid.npoints),
                     B_theor, label='theor at point [0,0]') # , s=6)
            # plt.plot(np.arange(grid.npoints),
            #          B[0,:], label='B[0,:]', color='green') # , s=6)
            plt.scatter(grid.rho_matrix[0,:].flatten(), B[0,:].flatten(),s=2,
                        color='green', label='B')
            plt.legend(loc='best')
            plt.grid()
            title = f'cvf, {info}'
            plt.title(title)
            # n_png += 1
            # plt.savefig(path_to_save + f'{n_png} {title}.png')
            plt.show()


            plt.figure()
            plt.plot(np.arange(len(B[::grid.nlon,1000])-1),
                     B_estim[40::grid.nlon,1000], label='svd',
                     color='orange')
            plt.plot(np.arange(len(B[::grid.nlon,1000])-1),
                     B_nn[40::grid.nlon,1000], label='nn',
                     color='purple')
            plt.plot(np.arange(len(B[::grid.nlon,1000])-1),
                     B[40::grid.nlon,1000], label='true')
            plt.legend(loc='best')
            plt.title('B[:,0]')
            plt.grid()
            plt.show()


        #%% lcz sample cvm

        if is_best_c_loc_needed:
            c_loc_array = list(map(int, np.linspace(1000, 3000, 9)))
            c_loc = find_best_c_loc(ensemble, grid,
                                    c_loc_array,
                                    true_field, grid, n_obs=n_obs,
                                    obs_std_err=obs_std_err, n_seeds=n_seeds, seed=0,
                                    draw=False)

        B_sample = find_sample_covariance_matrix(ensemble.T)
        lcz_mx = construct_lcz_matrix(grid, c_loc)

        B_sample_loc = np.multiply(B_sample, lcz_mx)


        #%%
        #% uzing optimizator:

        # c_opt_ens, lamb_opt_ens = find_params_with_optimization(
        #     band_estim_variances, bands, n_max, cx*2, lambx*2, gammax*2
        #     )

        # variance_spectrum_optimized_ens = get_variance_spectrum(
        #     c_opt_ens, lamb_opt_ens, gammax, n_max
        #     )
        # modal_spectrum_optimized_ens = get_modal_spectrum_from_variance(
        #         variance_spectrum_optimized_ens, n_max
        #         )


        # draw_2D(c_opt_ens, title=f'c_opt_ens, cx={cx.mean()}')
        # draw_2D(lamb_opt_ens, title=f'lamb_opt_ens, lambx={lambx.mean()}')

        # plt.figure()
        # plt.plot(get_modal_spectrum_from_variance(variance_spectrum[:,0,0], n_max),
        #              color='black', label='True')
        # for point in [(5,5), (20,20), (30,40)]:
        #     plt.plot(
        #         modal_spectrum_optimized_ens[:,point[0], point[1]],
        #         color='blue')
        # plt.grid()
        # plt.legend(loc='best')
        # plt.title('modal_spectrum_optimized_ens')
        # plt.show()


        #%
        # MAE_opt = np.abs(modal_spectrum_optimized_ens - modal_spectrum).mean(
        #     axis=0
        #     )
        # ind_max_MAE = np.unravel_index(np.argmax(MAE_opt, axis=None),
        #                                MAE_opt.shape)
        # ind_bad_lambda = np.unravel_index(np.argmax(np.abs(lambx-lamb_opt_ens),
        #                                             axis=None),
        #                                lambx.shape)
        # # draw_2D(MAE_opt)


        # plt.figure()
        # plt.plot(modal_spectrum[:,0,0],
        #          color='black', label='True')
        # for point in [ind_bad_lambda]:
        #     plt.plot(modal_spectrum_optimized_ens[:,point[0], point[1]],
        #              color='blue', label='opt')
        #     plt.plot(variance_spectrum_estim[:,point[0], point[1]],
        #              color='green', label='linear')
        # plt.grid()
        # plt.legend(loc='best')
        # plt.title(
        #     f'''
        #     modal_spectrum_optimized_ens
        #     MAE_max = {MAE_opt[point[0], point[1]]}
        #     lambda: true = {lambx[point[0], point[1]]:.4}, opt = {lamb_opt_ens[point[0], point[1]]:.4}
        #     c: true = {cx[point[0], point[1]]:.4}, opt = {c_opt_ens[point[0], point[1]]:.4}
        #     ''')
        # plt.show()

        # sigma_param = np.sqrt(modal_spectrum_optimized_ens)
        # sigma_param_coarsened = np.array(
        #     [coarsen_grid(sigma_param[i,:,:], k_coarse) for i in range(len(sigma))]
        #     )
        # sigma_param_coarsened_flattened = np.array(
        #     [flatten_field(sigma_param_coarsened[i,:,:]) for i in range(len(sigma))]
        #     )

        # u_param = compute_u_from_sigma(sigma_param_coarsened_flattened, grid, n_max)
        # W_param = compute_W_from_u(u_param, grid)
        # B_param = np.matmul(W_param, W_param.T)
        # # B_theor = Fourier_Legendre_transform_backward(
        # #     n_max, modal_spectrum[:,0,0], rho_vector_size = grid.npoints
        # #     )

        #%%

        types_V2b = ['lines'] #, 'opt_param']
        
        point = np.random.randint(grid.npoints)
        if draw:
            # draw_2D(B, title='B true')
            # draw_2D(B_sample, title='B - sample cvm')
            # draw_2D(lcz_mx, title=f'lcz_mx, c_loc={c_loc}')
            # draw_2D(B_sample_loc, title=f'B_sample_loc, c_loc={c_loc}')


            # for type_V2b in types_V2b:
            plt.figure()
            # plt.plot(np.arange(grid.npoints),
            #          B[0,:], label='B[0,:]', color='green') # , s=6)

            plt.scatter(grid.rho_matrix[point,:].flatten(),
                        B_sample_loc[point,:].flatten(),s=2,
                        color='blue', alpha=0.7, label='B_sample_loc')
            plt.scatter(grid.rho_matrix[point,:].flatten(),
                        B[point,:].flatten(),s=4,
                        color='green', label='B true')
            plt.scatter(grid.rho_matrix[point,:].flatten(),
                        B_nn[point,:].flatten(),
                     label='estim (nn)',
                     color='purple', alpha=0.6, s=4)
            # if type_V2b == 'lines':
            plt.scatter(grid.rho_matrix[point,:].flatten(),
                        B_estim[point,:].flatten(),
                     label='estim (svd)',
                     color='red', alpha=0.6, s=4)
            # if type_V2b == 'opt_param':
            #     plt.scatter(grid.rho_matrix[point,:].flatten(),
            #                 B_param[point,:].flatten(),
            #              label='estim (ensm)',
            #              color='red', alpha=0.6, s=1)



            # plt.plot(np.linspace(0, np.pi, grid.npoints),
            #     # np.arange(grid.npoints),
            #          B_theor, label='true', color='black',
            #          linewidth=3) # , s=6)
            plt.legend(loc='best')
            plt.grid()
            title = f'covariances with point {point} before shape, {info}'
            plt.title(title)
            # n_png += 1
            # plt.savefig(path_to_save + f'{n_png} {title}.png')
            plt.show()


        #%%

        # Making B_median (with median values of fields):

        variance_spectrum_med, lambx_med, gammax_med, Vx_med, cx_med = \
            get_modal_spectrum_from_DLSM_params(
                kappa_std_xi=1, kappa_lamb=1, kappa_gamma=1,
                std_xi_mult=std_xi_mult, std_xi_add=std_xi_add,
                lamb_mult=lamb_mult, lamb_add=lamb_add,
                gamma_mult=gamma_mult, gamma_add=gamma_add,
                mu_NSL=mu_NSL, n_max=n_max, 
                angular_distance_grid_size=angular_distance_grid_size,
                draw=False)
                
        modal_spectrum_med = convert_variance_spectrum_to_modal(
            variance_spectrum_med, n_max
            )

        draw_1D(np.array([variance_spectrum_med[:,0,0]] + \
                         [variance_spectrum[:,i,i] for i in (5, 20, 30, 40)]).T,
                title=f'med and true var spectrum, {info}')
        sigma_med = np.sqrt(modal_spectrum_med)
        sigma_coarsened_med = np.array(
            [coarsen_grid(sigma_med[i,:,:], k_coarse) for i in range(len(sigma))]
            )
        sigma_coarsened_flattened_med = np.array(
            [flatten_field(sigma_coarsened_med[i,:,:]) for i in range(len(sigma))]
            )
        u_med = compute_u_from_sigma(sigma_coarsened_flattened_med, grid, n_max)
        W_med = compute_W_from_u(u_med, grid)
        B_med = np.matmul(W_med, W_med.T)



        #%%
        spectrum_estim_flat = np.array([
            flatten_field(variance_spectrum_estim[i,:,:]) for i in range(n_max + 1)
            ])
        variance_spectrum_flat = np.array([
            flatten_field(variance_spectrum[i,:,:]) for i in range(n_max + 1)
            ])
        # print(spectrum_estim_flat.shape)
        # spectrum_estim_2d = np.array([
        #     make_2_dim_field(spectrum_estim_flat[i,:],
        #                  variance_spectrum_estim.shape[1], variance_spectrum_estim.shape[2])
        #     for i in range(n_max + 1)
        #     ])

        # print(np.max(np.abs(spectrum_estim_2d - variance_spectrum_estim)))


        plt.close('all')

        #%%
        e_shape = variance_spectrum_med[:,0,0]
        if draw:
            draw_1D(e_shape, title='e_shape(variance_spectrum_med)')
        # assert False
        variance_spectrum_estim_linshape_flat = np.array(
            fit_shape_S2(
            # V2b_shape_S2(
                spectrum_estim_flat.T,
                # variance_spectrum_flat.T,
                e_shape,
                # moments #,a_max_times, w_a_fg,
                **params_shape
                )[0]
            ).reshape(spectrum_estim_flat.T.shape).T
        variance_spectrum_estim_linshape = np.array([
            make_2_dim_field(variance_spectrum_estim_linshape_flat[i,:],
                          variance_spectrum_estim.shape[1],
                          variance_spectrum_estim.shape[2])
            for i in range(n_max + 1)
            ])
        modal_spectrum_estim_linshape = convert_variance_spectrum_to_modal(
            variance_spectrum_estim_linshape, n_max
            )

        sigma_linshape = np.sqrt(modal_spectrum_estim_linshape)
        sigma_coarsened_linshape = np.array(
            [coarsen_grid(sigma_linshape[i,:,:], k_coarse) for i in range(len(sigma))]
            )
        sigma_coarsened_flattened_linshape = np.array(
            [flatten_field(sigma_coarsened_linshape[i,:,:]) for i in range(len(sigma))]
            )
        u_linshape = compute_u_from_sigma(sigma_coarsened_flattened_linshape, grid, n_max)
        W_linshape = compute_W_from_u(u_linshape, grid)
        W_linshape, threshold_descr = make_matrix_sparse(
            W_linshape, threshold_coeff=threshold_coeff, return_n_zeros=True
            )
        B_linshape = np.matmul(W_linshape, W_linshape.T)
        if draw:
            draw_2D(B_linshape, title=f'B_shape, {info}')


               #%%
        
        
        
        MAE_nn = np.abs(variance_spectrum_nn - variance_spectrum).mean(
            axis=0
            )
        MAE_shape = np.abs(variance_spectrum_estim_linshape - variance_spectrum).mean(
            axis=0
            )
        bad_points_inds = dict()
        bad_points_descr = {
            'nn': 'nn better than svd+linshape',
            'svd_shape': 'svd_shape better than nn',
            'one': 'nn is bad',
            'strange': 'nn good for spectra and bad for anls'
            }
        inds = [2,20,90,150,200,300,450,600,800]
        bad_points_inds['one'] = [
            np.unravel_index(np.argsort(MAE_nn, axis=None)[-i],
                      MAE_nn.shape) for i in inds
            ]
        bad_points_inds['nn'] = [
            np.unravel_index(np.argsort(-MAE_nn+MAE_shape, axis=None)[-i],
                      MAE_nn.shape) for i in inds
            ]
        bad_points_inds['svd_shape'] = [
            np.unravel_index(np.argsort(MAE_nn-MAE_shape, axis=None)[-i],
                      MAE_nn.shape) for i in inds
            ]
        bad_points_inds['strange'] = [(6, 2), (6, 3), (6,4),
                                      (8, 2), (8, 3), (8,4),
                                      (10, 2),(10, 3),(10,4),]
        for bad_points_type, bad_points_i in bad_points_inds.items():
        # bad_points = [np.unravel_index(np.argsort(MAE_nn, axis=None)[-i],
        #               MAE_nn.shape) for i in [2,4,6,8,11,15,29,46,90]]

            print(bad_points_i)
             #%
            if draw: 
            
                fig, axes = plt.subplots(3, 3, 
                                                     # sharex=True, sharey=True,
                                                     figsize=(8, 10))
                point_i = 0
                fig.suptitle('Sharing x per column, y per row')
                # np.random.seed(1111)
                for i in range(3):
                    for j in range(3):
                        
                        point=bad_points_i[point_i]
                        point_i += 1
                        axes[i,j].plot(np.arange(variance_spectrum_estim.shape[0]),
                         variance_spectrum_estim[:, point[0], point[1]],
                         label='svd before shape',
                         color='orange')
                        axes[i,j].plot(np.arange(variance_spectrum.shape[0]),
                                 variance_spectrum_estim_linshape[:, point[0], point[1]],
                                 label='svd + shape', color='red')
                        axes[i,j].plot(np.arange(variance_spectrum.shape[0]),
                                 variance_spectrum_nn[:, point[0], point[1]],
                                 label='nn', color='purple')
                        axes[i,j].plot(np.arange(variance_spectrum.shape[0]),
                                 variance_spectrum[:, point[0], point[1]],
                                 label='true', color='black')
                        axes[i,j].scatter(
                        [band.center for band in bands],
                        band_estim_variances[:, point[0], point[1]] #/ all_bands_c
                        )
                        axes[i,j].grid()
                        axes[i,j].legend(loc='best')
                   
                
                # plt.grid()
                # plt.legend(loc='best')
                title = f'variance_spectrum, {bad_points_descr[bad_points_type]}, {info}'
                fig.suptitle(title)
                n_png += 1
                plt.savefig(path_to_save + f'{n_png} {title}.png')
                plt.show()
    
                plt.figure()

#%%
        for bad_points_type, bad_points_i in bad_points_inds.items():
        # bad_points = [np.unravel_index(np.argsort(MAE_nn, axis=None)[-i],
        #               MAE_nn.shape) for i in [2,4,6,8,11,15,29,46,90]]

            print(bad_points_i)
             #%
            if draw: 
                fig, axes = plt.subplots(3, 3, figsize=(8, 10))
                point_i = 0
                fig.suptitle('Sharing x per column, y per row')
                # np.random.seed(1111)
                for i in range(3):
                    for j in range(3):
                        
                        
                        point = np.array(bad_points_i[point_i])
                        point_i += 1
                        ref_point_num = grid.transform_coords_2d_to_1d(point // 2)
                        axes[i,j].scatter(grid.rho_matrix[ref_point_num,:].flatten(),
                                    B[ref_point_num,:].flatten(),s=1,
                                    color='black', label='B true', alpha=1)
                        axes[i,j].scatter(grid.rho_matrix[ref_point_num,:].flatten(),
                                    B_sample_loc[ref_point_num,:].flatten(),s=0.5,
                                    color='blue', alpha=0.7, label='B_sample_loc')
                        axes[i,j].scatter(grid.rho_matrix[ref_point_num,:].flatten(),
                                    B_linshape[ref_point_num,:].flatten(),
                                 label='svd+shape',
                                 color='red', alpha=0.6, s=0.25)
                        axes[i,j].scatter(grid.rho_matrix[ref_point_num,:].flatten(),
                                    B_nn[ref_point_num,:].flatten(),s=1,
                                    color='purple', alpha=0.2, label='nn')
                        axes[i,j].grid()
                        axes[i,j].legend(loc='best')
                   
                
                # plt.grid()
                # plt.legend(loc='best')
                title = f'covariances, {bad_points_descr[bad_points_type]}, {info}'
                fig.suptitle(title)
                n_png += 1
                plt.savefig(path_to_save + f'{n_png} {title}.png')
                plt.show()
    




#%%     
        point = np.random.randint(grid.npoints)
        if draw:
            plt.figure()
            # plt.plot(np.arange(grid.npoints),
            #          B[0,:], label='B[0,:]', color='green') # , s=6)

            plt.scatter(grid.rho_matrix[point,:].flatten(),
                        B[point,:].flatten(),s=3,
                        color='black', label='B true', alpha=1)
            plt.scatter(grid.rho_matrix[point,:].flatten(),
                        B_sample_loc[point,:].flatten(),s=2,
                        color='blue', alpha=0.7, label='B_sample_loc')
            plt.scatter(grid.rho_matrix[point,:].flatten(),
                        B_linshape[point,:].flatten(),
                     label='svd+shape',
                     color='red', alpha=0.6, s=1)
            plt.scatter(grid.rho_matrix[point,:].flatten(),
                        B_nn[point,:].flatten(),s=2,
                        color='purple', alpha=0.2, label='nn')

            plt.legend(loc='best')
            plt.grid()
            title = f'covariances with point {point}, {info}'
            plt.title(title)
            n_png += 1
            plt.savefig(path_to_save + f'{n_png} {title}.png')
            plt.show()

#%%
        from statsmodels.nonparametric.smoothers_lowess import lowess

        cov_med_true = pd.DataFrame(
            index=grid.rho_matrix.ravel(),
            data=(B_med).ravel()
            ).groupby(level=0).mean()

        cov_bias_shape = pd.DataFrame(
            index=grid.rho_matrix.ravel(),
            data=(B_linshape-B).ravel()
            ).groupby(level=0).mean()
        
        cov_bias_nn = pd.DataFrame(
            index=grid.rho_matrix.ravel(),
            data=(B_nn-B).ravel()
            ).groupby(level=0).mean()

        cov_bias_sample_loc = pd.DataFrame(
            index=grid.rho_matrix.ravel(),
            data=(B_sample_loc-B).ravel()
            ).groupby(level=0).mean()

        cov_bias_sample = pd.DataFrame(
            index=grid.rho_matrix.ravel(),
            data=(B_sample-B).ravel()
            ).groupby(level=0).mean()

        #%

        frac=0.003


        filtered_shape = lowess(
            cov_bias_shape.values.flatten(), np.array(cov_bias_shape.index),
            is_sorted=True, frac=frac, it=0
            )
        filtered_sample_loc = lowess(
            cov_bias_sample_loc.values.flatten(),
            np.array(cov_bias_sample_loc.index),
            is_sorted=True, frac=frac, it=0
            )
        filtered_sample = lowess(
            cov_bias_sample.values.flatten(), np.array(cov_bias_sample.index),
            is_sorted=True, frac=frac, it=0
            )
        filtered_nn = lowess(
            cov_bias_nn.values.flatten(), np.array(cov_bias_nn.index),
            is_sorted=True, frac=frac, it=0
            )

#%%

        plt.figure()
        plt.plot(cov_bias_sample_loc,
                  # label='sample_loc', 
                  color='blue', alpha=0.3)
        plt.plot(cov_bias_sample,
                  # label='sample', 
                  color='green', alpha=0.3)
        plt.plot(cov_bias_shape,
                  # label='shape', 
                  color='red', alpha=0.3)
        plt.plot(cov_bias_nn,
                  # label='nn', 
                  color='purple', alpha=0.3)
        plt.plot(filtered_shape[:,0], filtered_shape[:,1],
                  label='shape+svd', color='red')
        plt.plot(filtered_sample_loc[:,0], filtered_sample_loc[:,1],
                  label='sample_loc', color='blue')
        plt.plot(filtered_sample[:,0], filtered_sample[:,1],
                  label='sample', color='green')
        plt.plot(filtered_nn[:,0], filtered_nn[:,1],
                  label='nn', color='purple')
        plt.plot(cov_med_true,
                  label='median true', color='black')
        plt.legend(loc='best')
        plt.xlabel('Distance, radians')
        plt.title(f'Biases in spatial covariances, \n {info}')
        plt.grid()
        # if draw:
        n_png += 1
        plt.savefig(path_to_save \
                    + f'{n_png} Biases in spatial covariances, {info}.png')
        plt.show()


#%%
        cov_mae_shape = pd.DataFrame(
            index=grid.rho_matrix.ravel(),
            data=np.abs(B_linshape-B).ravel()
            ).groupby(level=0).mean()

        cov_mae_sample_loc = pd.DataFrame(
            index=grid.rho_matrix.ravel(),
            data=np.abs(B_sample_loc-B).ravel()
            ).groupby(level=0).mean()

        cov_mae_sample = pd.DataFrame(
            index=grid.rho_matrix.ravel(),
            data=np.abs(B_sample-B).ravel()
            ).groupby(level=0).mean()
        
        cov_mae_nn = pd.DataFrame(
            index=grid.rho_matrix.ravel(),
            data=np.abs(B_nn-B).ravel()
            ).groupby(level=0).mean()

        #%

        frac=0.003


        filtered_mae_shape = lowess(
            cov_mae_shape.values.flatten(), np.array(cov_mae_shape.index),
            is_sorted=True, frac=frac, it=0
            )
        filtered_mae_sample_loc = lowess(
            cov_mae_sample_loc.values.flatten(),
            np.array(cov_mae_sample_loc.index),
            is_sorted=True, frac=frac, it=0
            )
        filtered_mae_sample = lowess(
            cov_mae_sample.values.flatten(), np.array(cov_mae_sample.index),
            is_sorted=True, frac=frac, it=0
            )
        filtered_mae_nn = lowess(
            cov_mae_nn.values.flatten(), np.array(cov_mae_nn.index),
            is_sorted=True, frac=frac, it=0
            )

#%%
        # if draw:
        plt.figure()
        plt.plot(cov_mae_sample_loc,
                  # label='sample_loc', 
                  color='blue', alpha=0.3)
        plt.plot(cov_mae_sample,
                  # label='sample', 
                  color='green', alpha=0.3)
        plt.plot(cov_mae_shape,
                  # label='shape', 
                  color='red', alpha=0.3)
        plt.plot(cov_mae_nn,
                  # label='shape', 
                  color='purple', alpha=0.3)
        plt.plot(filtered_mae_shape[:,0], filtered_mae_shape[:,1],
                  label='svd+shape', color='red')
        plt.plot(filtered_mae_sample_loc[:,0], filtered_mae_sample_loc[:,1],
                  label='sample_loc', color='blue')
        plt.plot(filtered_mae_sample[:,0], filtered_mae_sample[:,1],
                  label='sample', color='green')
        plt.plot(filtered_mae_nn[:,0], filtered_mae_nn[:,1],
                  label='nn', color='purple')
        plt.plot(cov_med_true,
                  label='median true', color='black')
        plt.legend(loc='best')
        plt.xlabel('Distance, radians')
        plt.grid()
        title = f'MAE in spatial covariances, {info}'
        plt.title(title)
        n_png += 1
        plt.savefig(path_to_save + f'{n_png} {title}.png')
        plt.show()


        # assert False
        # continue


        #%% 
        # !!!!!!!!!!
        for spectrum_estim, name in [(variance_spectrum_estim_linshape, 'svd+shape'),
                                     (variance_spectrum_nn, 'neural net')]:
            n_png = \
            draw_MAE_and_bias_of_spectrum(spectrum_estim,
                                      variance_spectrum,
                                      title=f'variance spectrum {name}, e_size={e_size}',
                                      path_to_save=path_to_save,
                                      n_png=n_png)
            
        #%

        # assert False


        #%%
        point = np.random.randint(0, grid.npoints)
        if draw:
            plt.figure()
            # plt.plot(spectrum_estim_linshape_2d[:,0,0], label='linshape')
            # plt.plot(variance_spectrum_estim[:,0,0], label='lines')
            # plt.plot(spectrum_estim_flat[:,point], label='lines')
            plt.plot(variance_spectrum_estim_linshape_flat[:,point], label='linshape')
            plt.plot(variance_spectrum_flat[:,point], label='true', color='black')
            plt.title(f'v2b, point {point}, {info}')
            plt.legend(loc='best')
            plt.grid()
            plt.show()

        #%% Draw cov functions along meridian
        
        
        for bad_points_type, bad_points_i in bad_points_inds.items():
        # bad_points = [np.unravel_index(np.argsort(MAE_nn, axis=None)[-i],
        #               MAE_nn.shape) for i in [2,4,6,8,11,15,29,46,90]]

            print(bad_points_i)
             #%
            if draw: 
                fig, axes = plt.subplots(3, 3, figsize=(8, 10))
                point_i = 0
                fig.suptitle('Sharing x per column, y per row')
                # np.random.seed(1111)
                for i in range(3):
                    for j in range(3):
                        
                        point = np.array(bad_points_i[point_i])
                        ref_point_num = grid.transform_coords_2d_to_1d(point // 2)
                        k = ref_point_num % grid.nlon
                        point_i += 1
                        axes[i,j].plot(np.arange(len(B[k::grid.nlon,ref_point_num])),
                                 B_linshape[k::grid.nlon,ref_point_num], label='linshape',
                                 color='red')
                        # axes[i,j].plot(np.arange(len(B[k::grid.nlon,ref_point_num])),
                        #          B[k::grid.nlon,ref_point_num], label='true',
                        #          color='black')
                        # axes[i,j].plot(np.arange(len(B[k::grid.nlon,ref_point_num])),
                        #            B_sample[k::grid.nlon,ref_point_num], label='sample cvm',
                        #           color='green')
                        axes[i,j].plot(np.arange(len(B[k::grid.nlon,ref_point_num])),
                                 B_sample_loc[k::grid.nlon,ref_point_num], label='loc*sample',
                                 color='blue')
                        axes[i,j].plot(np.arange(len(B[k::grid.nlon,ref_point_num])),
                                 B_med[k::grid.nlon,ref_point_num], label='median',
                                 color='seagreen')
                        axes[i,j].plot(np.arange(len(B[k::grid.nlon,ref_point_num])),
                                 B_nn[k::grid.nlon,ref_point_num], label='nn',
                                 color='purple')
                        # axes[i,j].plot(np.arange(len(B[k::grid.nlon,ref_point_num])),
                        #           lcz_mx[k::grid.nlon,ref_point_num], label='lcz mx',
                        #           color='orange')
                        axes[i,j].scatter(np.arange(len(B[k::grid.nlon,ref_point_num])),
                                 B[k::grid.nlon,ref_point_num], label='true',
                                 color='black')
                        axes[i,j].grid()
                        axes[i,j].legend(loc='best')
                   
                
                # plt.grid()
                # plt.legend(loc='best')
                title = f'covariance functions along meridian, {bad_points_descr[bad_points_type]}, {info}'
                fig.suptitle(title)
                n_png += 1
                plt.savefig(path_to_save + f'{n_png} {title}.png')
                plt.show()
                
                #%%
        if draw:


            # ref_point_num = np.random.randint(0, 4000)
            for ref_point_num in [1, 27, 111, 1000, 1500, 3456]:

                k = ref_point_num % grid.nlon

                plt.figure()
                plt.plot(np.arange(len(B[k::grid.nlon,ref_point_num])),
                         B_linshape[k::grid.nlon,ref_point_num], label='linshape',
                         color='red')
                # plt.plot(np.arange(len(B[k::grid.nlon,ref_point_num])),
                #          B[k::grid.nlon,ref_point_num], label='true',
                #          color='black')
                # plt.plot(np.arange(len(B[k::grid.nlon,ref_point_num])),
                #            B_sample[k::grid.nlon,ref_point_num], label='sample cvm',
                #           color='green')
                plt.plot(np.arange(len(B[k::grid.nlon,ref_point_num])),
                         B_sample_loc[k::grid.nlon,ref_point_num], label='loc*sample',
                         color='blue')
                plt.plot(np.arange(len(B[k::grid.nlon,ref_point_num])),
                         B_med[k::grid.nlon,ref_point_num], label='median',
                         color='seagreen')
                plt.plot(np.arange(len(B[k::grid.nlon,ref_point_num])),
                         B_nn[k::grid.nlon,ref_point_num], label='nn',
                         color='purple')
                # plt.plot(np.arange(len(B[k::grid.nlon,ref_point_num])),
                #           lcz_mx[k::grid.nlon,ref_point_num], label='lcz mx',
                #           color='orange')
                plt.scatter(np.arange(len(B[k::grid.nlon,ref_point_num])),
                         B[k::grid.nlon,ref_point_num], label='true',
                         color='black')
                plt.legend(loc='best')
                plt.title(
                    f'covariance functions along meridian\
                    \nc_loc={c_loc}, ref_point_num: {ref_point_num}, \n{info}')
                plt.grid()
                n_png += 1
                plt.savefig(path_to_save + f'{n_png} covariance functions along meridian\
                  c_loc={c_loc}, ref_point_num - {ref_point_num}, {info}.png')
                plt.show()

                # s = make_analysis(B_sample_loc, true_field, grid, n_obs=n_obs,
                #                   obs_std_err=obs_std_err, n_seeds=n_seeds, seed=0,
                #                   draw=False)

                # plt.figure()
                # for ref_point_num in [np.random.randint(0, 4000) for _ in range(6)]:
                #     k = ref_point_num % grid.nlon
                #     plt.plot(np.arange(len(B[k::grid.nlon,ref_point_num])),
                #          B_med[k::grid.nlon,ref_point_num], label='median',
                #          color='seagreen')
                # plt.grid()
                # plt.show()


        #%%
            # plt.figure()
            # plt.plot(np.arange(len(B[::grid.nlon,ref_point_num])-1),
            #          B_param[40::grid.nlon,ref_point_num], label='param',
            #          color='orange')
            # plt.plot(np.arange(len(B[::grid.nlon,ref_point_num])-1),
            #          B[40::grid.nlon,ref_point_num], label='true')
            # plt.legend(loc='best')
            # plt.title(f'B[:,0]')
            # plt.grid()
            # plt.show()




        #%% Analysis  
        # !!!!!!!!!!!!!!!
        
        n_seeds=1
        
        anls_error_linshape = make_analysis(B_linshape, true_field, grid, n_obs=n_obs,
                          obs_std_err=obs_std_err, n_seeds=n_seeds, seed=0,
                          draw=True, return_mode='anls_error')
        anls_error_nn = make_analysis(B_nn, true_field, grid, n_obs=n_obs,
                          obs_std_err=obs_std_err, n_seeds=n_seeds, seed=0,
                          draw=True, return_mode='anls_error')
        
        draw_2D(anls_error_linshape, title='anls_error_linshape')
        draw_2D(anls_error_nn, title='anls_error_nn')
        draw_2D(anls_error_nn - anls_error_linshape, 
                title='anls_error_nn - anls_error_linshape',
                white_zero=True)
        
        draw_2D(MAE_nn - MAE_shape, 
                title='MAE_nn - MAE_shape',white_zero=True)
        
        bad_points_anls_inds = [
            np.unravel_index(np.argsort(
                anls_error_nn - anls_error_linshape, axis=None
                )[-i],
                anls_error_nn.shape) for i in [1,2,3,5,8,15,30,45,60]
            ]
        bad_points_spectra_inds = [
            np.unravel_index(np.argsort(
                MAE_nn - MAE_shape, axis=None
                )[-i],
                MAE_nn.shape) for i in [1,2,3,5,8,15,30,45,60]
            ]

        print(2*np.array(bad_points_anls_inds))
        print(np.array(bad_points_spectra_inds))
        
        #%%[ 22  12] and [ 27  13]
        
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes
        for i, point in enumerate([(22,12), (27,13)]):
            axes[i].plot(np.arange(variance_spectrum_estim.shape[0]),
             variance_spectrum_estim[:, point[0], point[1]],
             label='svd before shape',
             color='orange')
            axes[i].plot(np.arange(variance_spectrum.shape[0]),
                     variance_spectrum_estim_linshape[:, point[0], point[1]],
                     label='svd + shape', color='red')
            axes[i].plot(np.arange(variance_spectrum.shape[0]),
                     variance_spectrum_nn[:, point[0], point[1]],
                     label='nn', color='purple')
            axes[i].plot(np.arange(variance_spectrum.shape[0]),
                     variance_spectrum[:, point[0], point[1]],
                     label='true', color='black')
            axes[i].scatter(
            [band.center for band in bands],
            band_estim_variances[:, point[0], point[1]] #/ all_bands_c
            )
            axes[i].grid()
            axes[i].legend(loc='best')
        plt.show()
        
        
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        for i, point in enumerate([(22,12), (27,13)]):
            point = np.array(point)
            ref_point_num = grid.transform_coords_2d_to_1d(point // 2)
            k = ref_point_num % grid.nlon
            point_i += 1
            axes[i].plot(np.arange(len(B[k::grid.nlon,ref_point_num])),
                     B_linshape[k::grid.nlon,ref_point_num], label='linshape',
                     color='red')
            # axes[i].plot(np.arange(len(B[k::grid.nlon,ref_point_num])),
            #          B[k::grid.nlon,ref_point_num], label='true',
            #          color='black')
            # axes[i].plot(np.arange(len(B[k::grid.nlon,ref_point_num])),
            #            B_sample[k::grid.nlon,ref_point_num], label='sample cvm',
            #           color='green')
            axes[i].plot(np.arange(len(B[k::grid.nlon,ref_point_num])),
                     B_sample_loc[k::grid.nlon,ref_point_num], label='loc*sample',
                     color='blue')
            axes[i].plot(np.arange(len(B[k::grid.nlon,ref_point_num])),
                     B_med[k::grid.nlon,ref_point_num], label='median',
                     color='seagreen')
            axes[i].plot(np.arange(len(B[k::grid.nlon,ref_point_num])),
                     B_nn[k::grid.nlon,ref_point_num], label='nn',
                     color='purple')
            # axes[i].plot(np.arange(len(B[k::grid.nlon,ref_point_num])),
            #           lcz_mx[k::grid.nlon,ref_point_num], label='lcz mx',
            #           color='orange')
            axes[i].scatter(np.arange(len(B[k::grid.nlon,ref_point_num])),
                     B[k::grid.nlon,ref_point_num], label='true',
                     color='black')
            axes[i].grid()
            axes[i].legend(loc='best')
        plt.show()
        
        
        #%%

        t_array = []
        n_array = []
        l_array = []
        s_array = []
        m_array = []
        
        

        t = make_analysis(B, true_field, grid, n_obs=n_obs,
                          obs_std_err=obs_std_err, n_seeds=n_seeds, seed=0,
                          draw=draw)
        t_array.append(t)

        l = make_analysis(
            # B_param,
            # B_estim,
            B_linshape,
            true_field, grid, n_obs=n_obs,
            obs_std_err=obs_std_err, n_seeds=n_seeds, seed=0,
            draw=draw)
        l_array.append(l)



        s = make_analysis(B_sample_loc, true_field, grid, n_obs=n_obs,
                          obs_std_err=obs_std_err, n_seeds=n_seeds, seed=0,
                          draw=draw)
        m = make_analysis(B_med, true_field, grid, n_obs=n_obs,
                          obs_std_err=obs_std_err, n_seeds=n_seeds, seed=0,
                          draw=draw)
        n = make_analysis(B_nn, true_field, grid, n_obs=n_obs,
                          obs_std_err=obs_std_err, n_seeds=n_seeds, seed=0,
                          draw=draw)
        s_array.append(s)
        m_array.append(m)
        n_array.append(n)

        R_s = (l-t)/(s-t)
        R_m = (l-t)/(m-t)
        R_s_nn = (n-t)/(s-t)
        R_m_nn = (n-t)/(m-t)
        


        print('R_S with variances, svd+shape:')
        print(f'l: {l:.4}, t: {t:.4}, s: {s:.4}, m: {m:.4}')
        print(f'R_s = (l-t)/(s-t) = {R_s}')
        print(f'R_m = (l-t)/(m-t) = {R_m}')
        
        print('R_S with variances, nn:')
        print(f'n: {n:.4}, t: {t:.4}, s: {s:.4}, m: {m:.4}')
        print(f'R_s = (n-t)/(s-t) = {R_s_nn}')
        print(f'R_m = (n-t)/(m-t) = {R_m_nn}')        

        #%%


        # make_analysis(B, true_field, grid, n_obs=n_obs,
        #                   obs_std_err=obs_std_err, n_seeds=1, seed=0, draw=draw,
        #                   time_it=True)
        # make_analysis(B_estim, true_field, grid, n_obs=n_obs,
        #                   obs_std_err=obs_std_err, n_seeds=1, seed=0, draw=draw,
        #                   time_it=True)
        # make_analysis(B_sample_loc, true_field, grid, n_obs=n_obs,
        #                   obs_std_err=obs_std_err, n_seeds=1, seed=0, draw=draw,
        #                   time_it=True)


    print(f'Done. {n_iterations} iterations')
    #%

    t_array = np.array(t_array)
    s_array = np.array(s_array)
    l_array = np.array(l_array)
    m_array = np.array(m_array)
    n_array = np.array(n_array)

    R_s_array = (l_array-t_array)/(s_array-t_array)
    R_s_nn_array = (n_array-t_array)/(s_array-t_array)


    t = np.sqrt(np.mean(t_array))
    s = np.sqrt(np.mean(s_array))
    l = np.sqrt(np.mean(l_array))
    m = np.sqrt(np.mean(m_array))
    n = np.sqrt(np.mean(n_array))
    R_s = (l-t)/(s-t)
    R_m = (l-t)/(m-t)
    R_s_nn = (n-t)/(s-t)
    R_m_nn = (n-t)/(m-t)
    tlsmn = [t,l,s,m,n]

    R_s_on_grid.append(R_s)
    R_m_on_grid.append(R_m)
    R_s_nn_on_grid.append(R_s_nn)
    R_m_nn_on_grid.append(R_m_nn)
    tlsmn_on_grid.append(tlsmn)

    # if draw:
    plt.figure()
    plt.scatter(np.arange(len(t_array)), t_array, label='t', color='black')
    plt.scatter(np.arange(len(t_array)), s_array, label='s', color='blue')
    plt.scatter(np.arange(len(t_array)), l_array, label='l', color='red')
    plt.scatter(np.arange(len(t_array)), n_array, label='n', color='purple')
    plt.ylim(bottom=0)

    plt.legend(loc='best')

    # plt.xlabel('l')
    # plt.ylabel('b')
    plt.grid()
    title = f'analysis results R_s = {R_s:.5}, e_size={e_size}, {info}'
    plt.title(title)
    n_png += 1
    plt.savefig(path_to_save + f'{n_png} {title}.png')
    plt.show()



    with open(path_to_save + f'all results {e_size}.txt', 'a') as file:
        grid_parameter
        # file.write(f'{grid_parameter} = ' + str(eval(grid_parameter)) + '\n')
        file.write(threshold_descr + '\n')
        # file.write('mu_NSL = ' + str(mu_NSL) + '\n')
        # file.write('kappa = ' + str(kappa) + '\n')
        # file.write('params = ' + str(params) + '\n')
        # file.write('params_shape = ' + str(params_shape) + '\n')
        # file.write('e_size = ' + str(e_size) + '\n')
        # file.write(f'kappa_gamma: {kappa_gamma}; gamma_mult: {gamma_mult}, gamma_add: {gamma_add}, Gamma_mult: {Gamma*5/6}' + '\n')
        file.write('R_s = ' + str(R_s) + '\n')
        file.write('R_m = ' + str(R_m) + '\n')
        file.write('R_s_nn = ' + str(R_s_nn) + '\n')
        file.write('R_m_nn = ' + str(R_m_nn) + '\n')
        # file.write('tlsm : ' + str(tlsm) + '\n')





    print(f'l: {l:.4}, t: {t:.4}, s: {s:.4}, m: {m:.4}')
    print(f'R_s = (l-t)/(s-t) = {R_s}')
    print(f'R_m = (l-t)/(m-t) = {R_m}')
    # r = (l-t)/(s-t)
    # r_dict[(band_size, band_shift)] = r

    # print(r)

    #%%
    
    for estim_array, estim_type in ((l_array, 'svd+shape'), 
                                    (n_array, 'nn')):
        n_boo = 1000
        r_boo_array = []
        for i in range(n_boo):
            if i % int(n_boo/10) == 0:
                print(f'{i} / {n_boo}')
            sample = np.random.choice(n_iterations, n_iterations)
            t = np.sqrt(np.mean(t_array[sample]))
            l = np.sqrt(np.mean(estim_array[sample]))
            s = np.sqrt(np.mean(s_array[sample]))
    
            r_boo_array.append((l-t)/(s-t))
    
        RelEE_samplNoise = np.std(r_boo_array)
        with open(path_to_save + 'RelEE_samplNoise.txt', 'a') as file:
            file.write(f'{estim_type}; RelEE_samplNoise (s): {RelEE_samplNoise}' + '\n')
    
        print(f'{estim_type}; RelEE_samplNoise (s): {RelEE_samplNoise}')
    
    
        r_boo_array = []
        for i in range(n_boo):
            if i % int(n_boo/10) == 0:
                print(f'{i} / {n_boo}')
            sample = np.random.choice(n_iterations, n_iterations)
            t = np.sqrt(np.mean(t_array[sample]))
            l = np.sqrt(np.mean(estim_array[sample]))
            m = np.sqrt(np.mean(m_array[sample]))
    
            r_boo_array.append((l-t)/(m-t))
    
        RelEE_samplNoise = np.std(r_boo_array)
    
        with open(path_to_save + 'RelEE_samplNoise.txt', 'a') as file:
            file.write(f'{estim_type}; RelEE_samplNoise (m): {RelEE_samplNoise}' + '\n')
    
        print(f'{estim_type}; RelEE_samplNoise (m): {RelEE_samplNoise}')
    
        print(f'seed: {my_seed_mult}')



    # len(R_s_on_grid) = 29

    # for i in range(len(R_s_on_grid)):
    #     if i % 3 == 0:
    #         print(f"nband = {param_grid[i]['nband']}, \
    #    halfwidth_min = {param_grid[i]['halfwidth_min']}, \
    #    nc2 = {param_grid[i]['nc2']}")
    #     print(R_s_on_grid[i])

plt.close('all')

# with open(path_to_save + f'all results (e_size).txt', 'r') as file:
#     data = file.read()
#     show_best_R_s_with_params(data=data, path=path_to_save,
#                               param_shape_grid_dict=param_grid_dict)




end = process_time()



print(f'total time: {end - start}')





#%%

# path = 'shape params/'

# for e_n in [5, 10, 20, 40, 80]:
#     with open(path + f'{e_n}.txt', 'r') as file:
#         data = file.read()
#         # print(data)
#         data = data.replace('params_shape', 'params')
#         show_best_R_s_with_params(data=data, path=path,
#                                   param_grid_dict=param_shape_grid_dict)


def plot_RMSEs(param_array, tlsm_array, xlabel, add_info='', xlog=False):
    global n_png

    fig, ax = plt.subplots()
    # ax1.plot([10, 100, 1000], [1,2,3])
    # ax1.set_xscale('log')
    # ax1.set_xticks([20, 300, 500])
    # ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    # plt.show()


    lw = 1
    ax.plot(param_array, tlsm_array[:,1], label='LSEF', color='red', linewidth=lw)
    ax.plot(param_array, tlsm_array[:,2], label='EnKF', color='blue',
              linestyle='--', linewidth=lw)
    ax.plot(param_array, tlsm_array[:,3], label='Constant B', color='green',
              # linestyle=(0,(2,1)), linewidth=2)
              linestyle=':', linewidth=lw)
    ax.plot(param_array, tlsm_array[:,0], label='True B', color='black',
             linewidth=lw * 0.75)

    if xlog:
        ax.set_xscale('log')
    ax.legend(loc='best')
    ax.set_xlabel(xlabel)
    ax.set_xticks(param_array)
    ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())

    ax.set_ylabel('RMSE')
    # plt.grid()
    title = '$S^2$ : Analysis error RMSEs'
    ax.set_title(title)
    n_png += 1
    xlabel = xlabel.replace(' ', '_')
    plt.savefig('images/1_for_workshop/' + \
                f'{n_png}_{xlabel}' + add_info + '.png')
    plt.show()

#%%




def take_R_data_and_plot(R_data: str, grid_parameter: str, *args, **kwargs):
    assert grid_parameter in ['mu_NSL', 'kappa', 'e_size']

    param_array = []
    tlsm_array = []

    R_data_strings = R_data.split('\n')
    for s in R_data_strings:
        if 'tlsm' in s:
            tlsm_array.append(eval(s[7:]))
        if grid_parameter in s:
            param_array.append(np.int(s[len(grid_parameter) + 3:]))

    param_array = np.array(param_array)
    tlsm_array = np.array(tlsm_array)
    tlsm_array = (tlsm_array.T / tlsm_array[:,0]).T

    if grid_parameter == 'kappa':
        xlabel = 'Non-stationarity strength'
    if grid_parameter == 'mu_NSL':
        xlabel = 'Non-stationarity length'
    if grid_parameter == 'e_size':
        xlabel = 'Ensemble size'
    plot_RMSEs(param_array, tlsm_array, xlabel,
               *args, **kwargs)


#%%
# R_data = \
# '''
# kappa = 1
# tlsm : [0.48912043728917654, 0.49579552454167625, 0.5441935744476959, 0.48912043728917654]
# kappa = 2
# tlsm : [0.5383483112712226, 0.5550070331698819, 0.616337701443352, 0.5942328181273917]
# kappa = 3
# tlsm : [0.5931505871855882, 0.6237378634398383, 0.6984555950280041, 0.6890990764866869]
# kappa = 4
# tlsm : [0.6278450390805799, 0.668395125101373, 0.7508185318405034, 0.74771585319471]
# '''

# take_R_data_and_plot(R_data, grid_parameter='kappa',
#                      add_info='_Gamma=3_5', xlog=False)



#%%

# # e_size_array = []
# # tlsm_array = []


# R_data = \
# '''
# e_size = 5
# tlsm : [0.5383483112712226, 0.562883221811297, 0.6966672640418815, 0.5942328181273917]
# e_size = 10
# tlsm : [0.5383483112712226, 0.5550070331698819, 0.6163377014433519, 0.5942328181273917]
# e_size = 20
# tlsm : [0.5383483112712226, 0.552102237150548, 0.5867898058938567, 0.5942328181273917]
# e_size = 40
# tlsm : [0.5383483112712226, 0.5508379629426223, 0.5677243097943943, 0.5942328181273917]
# e_size = 80
# tlsm : [0.5383483112712226, 0.5498759310455686, 0.5579653167285792, 0.5942328181273917]
# '''


# # R_data_strings = R_data.split('\n')
# # for s in R_data_strings:
# #     if 'tlsm' in s:
# #         tlsm_array.append(eval(s[7:]))
# #     if 'e_size' in s:
# #         e_size_array.append(np.int(s[9:]))

# # e_size_array = np.array(e_size_array)
# # tlsm_array = np.array(tlsm_array)
# # tlsm_array = (tlsm_array.T / tlsm_array[:,0]).T


# # plot_RMSEs(e_size_array, 'Ensemble size')

# take_R_data_and_plot(R_data, grid_parameter='e_size',
#                      add_info='_Gamma=3_5', xlog=True)


# #%%

# # mu_NSL_array = []
# # tlsm_array = []


# R_data = \
# '''
# mu_NSL = 1
# tlsm : [0.6160156595668331, 0.7243561146412096, 0.7374748217765501, 0.827869237064122]
# mu_NSL = 2
# tlsm : [0.5600483900867191, 0.5912338127992364, 0.6495765670639485, 0.6444967656432863]
# mu_NSL = 4
# tlsm : [0.5270274631234462, 0.5392794039971238, 0.5997009248722155, 0.5712468525084329]
# mu_NSL = 6
# tlsm : [0.5132831160182825, 0.5228801575382072, 0.5809643406876419, 0.5464835369565092]
# mu_NSL = 8
# tlsm : [0.5046149320983219, 0.5131856308777567, 0.5693832749691783, 0.5320050508823648]
# '''


# # R_data_strings = R_data.split('\n')
# # for s in R_data_strings:
# #     if 'tlsm' in s:
# #         tlsm_array.append(eval(s[7:]))
# #     if 'mu_NSL' in s:
# #         mu_NSL_array.append(np.int(s[9:]))

# # mu_NSL_array = np.array(mu_NSL_array)
# # tlsm_array = np.array(tlsm_array)
# # tlsm_array = (tlsm_array.T / tlsm_array[:,0]).T


# # plot_RMSEs(mu_NSL_array, 'Non-stationarity length')

# take_R_data_and_plot(R_data, grid_parameter='mu_NSL',
#                      add_info='_Gamma=3_5', xlog=True)


# #%%

# R_data_2_5_kappa = \
# '''
# kappa = 1
# tlsm : [0.4806343133480897, 0.4880446229376437, 0.5379948910061852, 0.4806343133480897]
# kappa = 2
# tlsm : [0.46165213999720073, 0.5033731403798879, 0.544911005413145, 0.5583894216338022]
# kappa = 3
# tlsm : [0.490345190593583, 0.570256467512146, 0.6036506033171184, 0.6581596084236555]
# kappa = 4
# tlsm : [0.5120559660879196, 0.6186842710513843, 0.6428165592540448, 0.7226587827199342]'''
# take_R_data_and_plot(R_data_2_5_kappa, grid_parameter='kappa',
#                      add_info='_Gamma=2_5')

# R_data_3_kappa = \
# '''
# kappa = 1
# tlsm : [0.4806343133480897, 0.48804462293764367, 0.5379948910061852, 0.4806343133480897]
# kappa = 2
# tlsm : [0.45473778672249177, 0.479293075396506, 0.5361770430631786, 0.5292515465849384]
# kappa = 3
# tlsm : [0.47770840088584837, 0.5239196892619145, 0.5860647163045838, 0.6049749858161491]
# kappa = 4
# tlsm : [0.4962003350543273, 0.5585537167016542, 0.6197515534036209, 0.6544767379093019]
# '''
# take_R_data_and_plot(R_data_3_kappa, grid_parameter='kappa',
#                      add_info='_Gamma=3')




# R_data_2_5_mu_NSL = \
# '''
# mu_NSL = 1
# tlsm : [0.5661265679257145, 0.7878345561646845, 0.6977749875684627, 0.9062514891422824]
# mu_NSL = 2
# tlsm : [0.49364318663700407, 0.5739480057491767, 0.5887789224625393, 0.6492032826466999]
# mu_NSL = 4
# tlsm : [0.443593102395521, 0.47075335794138, 0.5198396591651333, 0.5142890520926964]
# mu_NSL = 6
# tlsm : [0.4236958890700693, 0.4404342088594762, 0.49098935283299155, 0.47150538013794896]
# mu_NSL = 8
# tlsm : [0.41339430315228326, 0.4265840085272853, 0.47505696641904577, 0.4507953799566383]
# '''
# take_R_data_and_plot(R_data_2_5_mu_NSL, grid_parameter='mu_NSL',
#                      add_info='_Gamma=2_5', xlog=True)

# R_data_3_mu_NSL = \
# '''
# mu_NSL = 1
# tlsm : [0.5536380186409635, 0.7135464657213562, 0.6736698071487893, 0.8316166646976526]
# mu_NSL = 2
# tlsm : [0.4843640802661984, 0.5324155249445177, 0.5756646202483189, 0.6027059702244788]
# mu_NSL = 4
# tlsm : [0.4380971046387423, 0.455284577494327, 0.5132137698727833, 0.4945931573989398]
# mu_NSL = 6
# tlsm : [0.4195468682620666, 0.4320273092642216, 0.4856329451200968, 0.4598013429046703]
# mu_NSL = 8
# tlsm : [0.40986385760893657, 0.42065429231969453, 0.46999488774348047, 0.44204064510389646]
# '''
# take_R_data_and_plot(R_data_3_mu_NSL, grid_parameter='mu_NSL',
#                      add_info='_Gamma=3', xlog=True)

# #%

# R_data_2_5_e_size = \
# '''
# e_size = 5
# tlsm : [0.46165213999720073, 0.5089540759068879, 0.6154285027421608, 0.5583894216338022]
# e_size = 10
# tlsm : [0.46165213999720073, 0.503373140379888, 0.544911005413145, 0.5583894216338022]
# e_size = 20
# tlsm : [0.46165213999720073, 0.49970784951222474, 0.5108769097164443, 0.5583894216338022]
# e_size = 40
# tlsm : [0.46165213999720073, 0.49822003209634036, 0.4948918067179624, 0.5583894216338022]
# e_size = 80
# tlsm : [0.46165213999720073, 0.4975158485476276, 0.48638493853994386, 0.5583894216338022]
# '''
# take_R_data_and_plot(R_data_2_5_e_size, grid_parameter='e_size',
#                      add_info='_Gamma=2_5', xlog=True)

# R_data_3_e_size = \
# '''
# e_size = 5
# tlsm : [0.4547377867224918, 0.4850834222472706, 0.6043102481976295, 0.5292515465849384]
# e_size = 10
# tlsm : [0.4547377867224918, 0.47929307539650606, 0.5361770430631787, 0.5292515465849384]
# e_size = 20
# tlsm : [0.4547377867224918, 0.47588431973945233, 0.5020258339561966, 0.5292515465849384]
# e_size = 40
# tlsm : [0.4547377867224918, 0.47443961451233513, 0.4861611255305732, 0.5292515465849384]
# e_size = 80
# tlsm : [0.4547377867224918, 0.47382325567968064, 0.4775272784334712, 0.5292515465849384]
# '''
# take_R_data_and_plot(R_data_3_e_size, grid_parameter='e_size',
#                      add_info='_Gamma=3', xlog=True)


















# #%%
# # e_size_array = []
# # R_s_array = []
# # R_m_array = []

# # R_data = '''
# # e_size = 5
# # tlsm : [0.5383483112712226, 0.562883221811297, 0.6966672640418815, 0.5942328181273917]
# # e_size = 10
# # tlsm : [0.5383483112712226, 0.5550070331698819, 0.6163377014433519, 0.5942328181273917]
# # e_size = 20
# # tlsm : [0.5383483112712226, 0.552102237150548, 0.5867898058938567, 0.5942328181273917]
# # e_size = 40
# # tlsm : [0.5383483112712226, 0.5508379629426223, 0.5677243097943943, 0.5942328181273917]
# # e_size = 80
# # tlsm : [0.5383483112712226, 0.5498759310455686, 0.5579653167285792, 0.5942328181273917]
# # '''

# # R_data_strings = R_data.split('\n')
# # for s in R_data_strings:
# #     if 'R_s' in s:
# #         R_s_array.append(np.float(s[6:]))
# #     if 'R_m' in s:
# #         R_m_array.append(np.float(s[6:]))
# #     if 'e_size' in s:
# #         e_size_array.append(np.int(s[9:]))
# # e_size_array



# # e_size_array = np.array(e_size_array)
# # R_s_array = np.array(R_s_array)
# # R_m_array = np.array(R_m_array)

# # # e_size_range = np.arange(1, 5)

# # plt.figure()
# # # plt.plot(kappa_range,
# # #          np.ones(kappa_range.shape),
# # #          label='KF', color='black')
# # plt.plot(e_size_array, R_s_array, label='R_s', color='red')
# # plt.plot(e_size_array, R_m_array, label='R_m', color='green')
# # plt.legend(loc='best')
# # plt.xlabel('Ensemble size')
# # # plt.ylabel('RMSE')
# # plt.ylabel('R')
# # plt.grid()
# # title = f'Analysis error R'
# # plt.title(title)
# # # n_png += 1
# # plt.show()



# # #%%

# # mu_NSL_array = []
# # R_s_array = []
# # R_m_array = []

# # R_data = '''
# # mu_NSL = 1
# # R_s = 0.8919907984159391
# # R_m = 0.5113930874061495
# # mu_NSL = 2
# # R_s = 0.3483308134426656
# # R_m = 0.36928386729745794
# # mu_NSL = 4
# # R_s = 0.1685889261204091
# # R_m = 0.277071688326782
# # '''

# # R_data_strings = R_data.split('\n')
# # for s in R_data_strings:
# #     if 'R_s' in s:
# #         R_s_array.append(np.float(s[6:]))
# #     if 'R_m' in s:
# #         R_m_array.append(np.float(s[6:]))
# #     if 'mu_NSL' in s:
# #         mu_NSL_array.append(np.int(s[9:]))
# # e_size_array



# # mu_NSL_array = np.array(mu_NSL_array)
# # R_s_array = np.array(R_s_array)
# # R_m_array = np.array(R_m_array)

# # # e_size_range = np.arange(1, 5)

# # plt.figure()
# # # plt.plot(kappa_range,
# # #          np.ones(kappa_range.shape),
# # #          label='KF', color='black')
# # plt.plot(mu_NSL_array, R_s_array, label='R_s', color='red')
# # plt.plot(mu_NSL_array, R_m_array, label='R_m', color='green')
# # plt.legend(loc='best')
# # plt.xlabel('Non-stationarity length')
# # # plt.ylabel('RMSE')
# # plt.ylabel('R')
# # plt.grid()
# # title = f'Analysis error R'
# # plt.title(title)
# # # n_png += 1
# # plt.show()



