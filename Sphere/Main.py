# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 20:39:19 2022
Last modified: Oct 2022

@author: Arseniy Sotskiy
"""

#%% All imports

import os
#must set these before loading numpy:
os.environ["OMP_NUM_THREADS"] = '1' # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = '1' # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = '8' # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = '4' # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = '4' # export NUMEXPR_NUM_THREADS=6


import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 10000
import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from time import process_time 

from pyshtools import SHGrid

COLORMAP = plt.cm.seismic

from configs import (
    is_Linux, 
    SDxi_med, SDxi_add, lambda_med, lambda_add, gamma_med, gamma_add,
    mu_NSL, 
    n_max, k_coarse, kappa_SDxi, kappa_lamb, kappa_gamma, 
    kappa_default, gamma_fixed,
    e_size, n_obs, obs_std_err, n_seeds,
    angular_distance_grid_size,
    expq_bands_params, n_iterations,
    folder_descr, info,
    w_ensm_hybr, hybrid_type,
    my_seed_mult, n_training_spheres_B
    )

from functions.tools import (
    mkdir
    )
from functions.FLT import (
    Fourier_Legendre_transform_backward
    )
from functions.process_functions import (
    get_variance_spectrum_from_modal,
    convert_variance_spectrum_to_modal
    )
from functions.DLSM_functions import (
    make_analysis, get_band_variances,
    compute_B_from_spectrum, generate_ensemble, 
    draw_bias_MSE_band_variances
    )

from functions.R_progs import (
    CreateExpqBands 
    )
  
from Band import (
    Band_for_sphere
    )

from RandomProcesses import (
    get_modal_spectrum_from_DLSM_params
    )

from Grids import (
    coarsen_grid, 
    LatLonSphericalGrid
    )

    
from Predictors import (
    HybridMeanSamplePredictor,
    SampleCovLocPredictor, NNPredictor,
    PredictorsErrors,
    compare_predicted_variance_spectra, 
    draw_errors_in_spatial_covariances,
    draw_cvf, draw_covariances_at_points,
    )

#%%
def write_seed(seed: int, done_seeds: str):
    with open(done_seeds, 'a+') as done:
        done.write(f' {seed}')


#%%


start = process_time()



path_to_save = mkdir(folder_descr, is_Linux)


# np.random.seed(None)

# my_seed_mult_203 = 5108 # np.random.randint(10000) # 44241 # 923456
# my_seed_mult_204 = 4421
# my_seed_mult = np.random.randint(10000)

with open(path_to_save + 'setup.txt', 'a') as setup_file:
    setup_file.write(f'my_seed_mult = {my_seed_mult}\n')
    setup_file.write(f'n_max = {n_max}\n')
    setup_file.write(f'e_size = {e_size}\n')
    setup_file.write(f'gamma_fixed = {gamma_fixed}\n')
    setup_file.write(f'kappa_default = {kappa_default}\n')
    setup_file.write(f'mu_NSL = {mu_NSL}\n')
    setup_file.write(f'n_obs = {n_obs}\n')
    setup_file.write(info)

global n_png
n_png = 0



draw = False

predictors_errors = PredictorsErrors()


#%%



# expq_bands_params = {'n_max': n_max, 'nband': 6, 'halfwidth_min': 1.5  * n_max/30,
#           'nc2': 1.5 * n_max / 22, 'halfwidth_max': 1.2 * n_max / 4,
#           'q_tranfu': 3, 'rectang': False}
transfer_functions, _, __ = CreateExpqBands(**expq_bands_params)

n_bands = expq_bands_params['nband']


print('e_size', e_size)

bands = [Band_for_sphere(transfer_functions[:,i]) for i in range(n_bands)]
print(transfer_functions.shape)
# n_bands = int(n_max / 5) # !!!!!!!!!!

band_centers = [band.center for band in bands]

plt.figure()
for band in bands:
    plt.plot(np.arange(band.shape),
              band.transfer_function,)
    plt.scatter([band.center], [0], label='center')
plt.grid()
plt.xlabel('wavenumber')
# plt.legend(loc='best')
plt.title('transfer functions')
plt.show()

plt.figure()
for band in bands:
    plt.plot(np.arange(band.shape),
              band.response_function)
plt.grid()
# plt.legend(loc='best')
plt.xlabel('distance, rad.')
plt.title(f'response functions')
plt.show()
# assert False

#%% Get modal spectrum b_n(x)

# After the processes V (x), lambda(x), and gamma(x) are computed
# at each analysis grid point, Eq.(38) is used to find c(x).
# With c(x), lambda(x), and gamma(x) in hand, we finally compute
# the "true" modal spectrum bn(x) using Eq.(39)


print('\nStart iterations\n')
# c_loc_arr = []
for iteration in range(n_iterations):
# for iteration in [12, 18]:
# for iteration in range(20):
    # mu_NSL = iteration % 4 + 1
    rnd_cnt = 1
    print('==============\n'*5)
    print(f'\nIteration {iteration}\n')
    # if iteration >= 2:
    #     my_seed_mult = my_seed_mult_204
    seed = my_seed_mult * (iteration + 1)
    # seed = 42
    # seed = my_seed_mult * int(n_iterations/2 - iteration/2)
    # seed = my_seed_mult * (n_iterations - iteration)
    print(f'seed: {seed}')
    # seed = my_seed_mult * (1 + 1)
    np.random.seed(seed)
# for here_can_be_any_cycle in [True]:
    # for gamma_mult in gamma_mult_arr:
    print('get_modal_spectrum_from_DLSM_params:')
    modal_spectrum, lambx, gammax, Vx, cx = \
        get_modal_spectrum_from_DLSM_params(
            kappa_SDxi, kappa_lamb, kappa_gamma,
            SDxi_med, SDxi_add,
            lambda_med, lambda_add,
            gamma_med, gamma_add,
            mu_NSL, n_max, angular_distance_grid_size,
            draw=True)
    
    with open(path_to_save + 'R_s R_m.txt', 'a') as file:
        # lambx, gammax, Vx, cx
        file.write(f'lambx l  {lambx.var()} \n')
        file.write(f'gammax l  {gammax.var()} \n')
        file.write(f't l {0} \n')
    # continue
    print('get_modal_spectrum_from_DLSM_params - Done')
    print('get_variance_spectrum_from_modal')
    variance_spectrum = get_variance_spectrum_from_modal(modal_spectrum, n_max)
    
    print('coarsen_grid')
    nlat, nlon = coarsen_grid(variance_spectrum[0,:,:], k_coarse).shape
    
    # assert False
    
    print('LatLonSphericalGrid')
    grid = LatLonSphericalGrid(nlat)    
    
    plt.close('all')

    
    
    #%%

    print('making modal_spectrum_med')
    modal_spectrum_med, lambx_med, gammax_med, Vx_med, cx_med = \
                get_modal_spectrum_from_DLSM_params(
                    kappa_SDxi=1, kappa_lamb=1, kappa_gamma=1,
                    SDxi_med=SDxi_med, SDxi_add=SDxi_add,
                    lambda_med=lambda_med, lambda_add=lambda_add,
                    gamma_med=gamma_med, gamma_add=gamma_add,
                    mu_NSL=mu_NSL, n_max=n_max, 
                    angular_distance_grid_size=angular_distance_grid_size,
                    draw=False)
    
    variance_spectrum_med = get_variance_spectrum_from_modal(modal_spectrum_med, 
                                                             n_max)
    # draw_1D(variance_spsectrum_med[:,0,0])
    plt.close('all')
    print('modal_spectrum_med - done')
    
    #%%
    # print('compute_B_from_spectrum')
    W_true, B_true = compute_B_from_spectrum(variance_spectrum, n_max,
                                             grid=grid,
                                             k_coarse=k_coarse, info='true')
    
    # W_med, B_med = compute_B_from_spectrum(variance_spectrum_med, n_max,
    #                                          grid=grid,
    #                                          k_coarse=k_coarse, info='med')
    
    B_mean = np.loadtxt(f'B_mean {n_training_spheres_B}.txt')
    B_static = B_mean
    print('compute_B_from_spectrum - done')
    
    print('generate_ensemble')
    
    true_field_interp, ensemble_fields_interp, ensemble, true_field =\
        generate_ensemble(
            W_true, e_size, grid, draw=False, info=f' iteration {iteration} ', 
            path_to_save=path_to_save,
            k_coarse=k_coarse
        )
    #%%
    # for i, cmap in enumerate([plt.cm.inferno, plt.cm.BuPu,]):
    #     title = '$X_{true}$'
    #     draw_2D(true_field,
    #             title=title, cmap=cmap, 
    #             save_path=path_to_save + f'{iteration} {title}{info}_{i}.png',
    #             save_gray=path_to_save + f'{iteration} {title}{info}_{i}_.png'
    #             )
    
    
    #%%
    # continue # if we need only pictures of the fields
    
    with open(path_to_save + 'R_s R_m.txt', 'a') as file:
        file.write(f'true_field l  {true_field.var()} \n')
        file.write(f't l {0} \n')
    print('generate_ensemble - done')
    plt.close('all')
    # sample_cvm_loc_predictor = SampleCovLocPredictor(
    #     bands, n_max, grid, ensemble,
    #     # true_field, n_obs, obs_std_err, 
    #     c_loc = 2137
    #     )
    
    
    shgrid = SHGrid.from_array(true_field_interp)
            # clm = shgrid.expand()
            # band_shape = clm.lmax + 1        
                    
    points = []
    print('get_band_variances')
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
    
    if draw:
        print('draw_bias_MSE_band_variances')
        draw_bias_MSE_band_variances(band_estim_variances,
                                     band_true_variances,
                                     band_mean_true_variances,
                                     bands,
                                     path_to_save,
                                     info=f' iteration {iteration}')
    plt.close('all')
    # variance_spectra_from_predictors = dict()
    
    #%%

    # svd_predictor = SVDPredictor(bands, n_max, grid)

    params_shape = {'moments': "012", 'a_max_times': 5, 'w_a_fg': 0.05}
    print('making predictors')
    # svshape_predictor = SVShapePredictor(params_shape, variance_spectrum_med, 
    #                                       'orange',
    #                                       bands, n_max, grid)
    # print('svshape_modal_predictor')
    # svshape_modal_predictor = SVShapePredictor(params_shape, 
    #                                            variance_spectrum_med, 
    #                                            'red',
    #                                            bands, n_max, grid,
    #                                            predict_modal=True)
    # svshape_modal_predictor.name += '_modal'
    # nn_predictor_deep = NNPredictor(bands, n_max, grid, 'net_deep', 
    #                                 color='indigo')
    # nn_predictor_4_log = NNPredictor(bands, n_max, grid, 'net4_log', 
    #                                 color='green')
    # nn_predictor_4_nonlog = NNPredictor(bands, n_max, grid, 'net4_nonlog', 
    #                                 color='violet')
    # nn_predictor_5_log = NNPredictor(bands, n_max, grid, 
    #                                  'net5_log' + str(n_max), 
    #                                 color='indigo')
    # print('nn_predictor_anls_loss')
    # nn_predictor_anls_loss = NNPredictor(bands, n_max, grid, 
    #                                      net_name + str(n_max), 
    #                                      color='purple')
    

    plt.close('all')
    print('sample_cvm_loc_predictor')
    sample_cvm_loc_predictor = SampleCovLocPredictor(
        bands, n_max, grid, ensemble,
        true_field, n_obs, obs_std_err, 
        # c_loc = 2137,
        precomputed_dict = 'best_c_loc.npy', 
        info='c_loc'
        )

    hybrid_mean_sample_predictor = HybridMeanSamplePredictor(
        B_static, w_ensm_hybr,
        hybrid_type,
        bands, n_max, grid, ensemble,
        true_field, n_obs, obs_std_err, 
        # c_loc = 2137,
        precomputed_dict = 'best_c_loc.npy', 
        info='c_loc'
        )
    # hybrid_med_sample_predictor = HybridMeanSamplePredictor(
    #     hybrid_type='med', B_static=B_med,
    #     bands=bands, n_max=n_max, grid=grid, ensemble=ensemble,
    #     # true_field, n_obs, obs_std_err, 
    #     c_loc = 2137
    #     )

# !!!!!!!!
    # predictors_list = [svshape_predictor, nn_predictor_4_log, 
    #                    nn_predictor_4_nonlog, nn_predictor_5_log]
                       # nn_predictor_2]
                    
    colors = {
        'net_l2_sqrtsqrt': 'purple', 
        'net_l2_sqrtsqrt_120': 'blue', 
        'net_l2_sqrt': 'red',
        }          
    # predictors_list = [
        # svshape_modal_predictor, 
        # svshape_predictor,
        # nn_predictor_anls_loss,
        # nn_predictor_5_log
   
    predictors_list = [NNPredictor(bands, n_max, grid, name + str(n_max), 
                     color=colors[name]) for name in colors.keys()]
    predictors_list_with_loc = predictors_list + \
        [sample_cvm_loc_predictor, hybrid_mean_sample_predictor,]

    print('making predictors - done')
    for predictor in predictors_list:
        print(predictor.name, predictor.color)
    plt.close('all')
    #%%
    for predictor in predictors_list:
        # print('predict')
        # variance_spectra_from_predictors[predictor.name] = \
        predictor.predict(band_estim_variances)
        # print('make_spectrum_positive')
        predictor.make_spectrum_positive()
        print('compute_B')
        predictor.compute_B(k_coarse)

        if draw:
            print('draw_MAE_bias')
            predictor.draw_MAE_bias(variance_spectrum)

    if draw:
        print('compare_predicted_variance_spectra')
        compare_predicted_variance_spectra(predictors_list, variance_spectrum,
                                           band_true_variances, path_to_save,
                                           draw_modal=True,
                                           info=f' iteration {iteration}')
        plt.close('all')
        # assert(False)
        print('draw_errors_in_spatial_covariances')
        for error_type in ['bias', 'mae']:
            draw_errors_in_spatial_covariances(predictors_list,
                                                variance_spectrum,
                                                band_estim_variances,
                                                B_static,
                                                B_true,
                                                error_type=error_type,
                                                path_to_save=path_to_save,
                                                info=f' iteration {iteration}')
        plt.close('all')
        print('draw_errors_in_spatial_covariances - done')
        
    #%%
    
    
        # plt.plot(filtered['SVShape_modal'][:,0])
        # plt.show()
        # plt.plot(filtered['SVShape_modal'][:,1])
        # plt.show()
    #%%
        print('draw_cvf')
        draw_cvf(predictors_list_with_loc,
                variance_spectrum,
                B_true,
                path_to_save,
                info=' for variance spectrum' + f' iteration {iteration}')
        plt.close('all')
        print('draw_covariances_at_points')
        for is_along_meridian in [False, True]:
            draw_covariances_at_points(predictors_list_with_loc,
                                       B_true,
                                       is_along_meridian=is_along_meridian,
                                       points='random',
                                       path_to_save=path_to_save,
                                       info=f' iteration {iteration}')
            plt.close('all')
            
        modal_spectrum_true = convert_variance_spectrum_to_modal(
            variance_spectrum, n_max
            )
        B_theor = Fourier_Legendre_transform_backward(
                n_max, modal_spectrum_true[:,0,0],
                rho_vector_size = grid.npoints
                )
    #%%
        # plt.figure()
        # step = 50
        # for predictor in predictors_list_with_loc:
        #     plt.scatter(grid.rho_matrix[:,::step].flatten(),
        #                 predictor.B[:,::step].flatten(),
        #              label=predictor.name,
        #              color=predictor.color , s=0.1,
        #              alpha=0.3)
        # plt.scatter(grid.rho_matrix[:,::step].flatten(),
        #             B_mean_[:,::step].flatten(),
        #          label='mean',
        #          color='purple' , s=0.1,
        #          alpha=0.3)
    
        # plt.plot(np.linspace(0, np.pi, grid.npoints),
        #     # np.arange(grid.npoints),
        #          B_theor, label='theor at point [0,0]') # , s=6)
        # # plt.plot(np.arange(grid.npoints),
        # #          B[0,:], label='B[0,:]', color='green') # , s=6)
        # plt.scatter(grid.rho_matrix[0,:].flatten(), B_true[0,:].flatten(),s=2,
        #             color='green', label='B_true')
        # plt.scatter(grid.rho_matrix[:,::step].flatten(),
        #             B_static[:,::step].flatten(),
        #          label='med',
        #          color='black' , s=0.1,
        #          alpha=0.3)
        # plt.legend(loc='best')
        # plt.grid()
        # title = f'cvf{info}'
        # plt.title(title)
        # # n_png += 1
        # if path_to_save is not None:
        #     plt.savefig(path_to_save + f'{title}.png')
        # plt.show()
    #%%
    
    # plt.figure()
    # for frac in [0.001, 0.003, 0.01, 0.03]:
    #     filtered = compute_average_from_distance(B_true, grid, frac=frac)
    #     plt.plot(filtered[:,0], filtered[:,1], label=f'frac={frac}')
    # plt.legend(loc='best')
    # plt.grid()
    # plt.title('LOWESS of B_true')
    # plt.xlabel(r'$\rho$')
    # plt.ylabel('B')
    # plt.show()
    
    # filtered = compute_average_from_distance(B_true, grid, frac=0.01)
    
    
    #%%
    
    with open(path_to_save + 'setup.txt', 'a') as setup_file:
        setup_file.write(
            f'iter {iteration}, {rnd_cnt}) rnd: {np.random.normal()}\n'
            )
        rnd_cnt += 1
    print('make_analysis')
    t = make_analysis(B_true, true_field, grid, n_obs=n_obs,
                      obs_std_err=obs_std_err, n_seeds=n_seeds, seed=seed,
                      draw=draw)
    s = make_analysis(sample_cvm_loc_predictor.B, true_field, grid, 
                      n_obs=n_obs,
                      obs_std_err=obs_std_err, n_seeds=n_seeds, seed=seed,
                      draw=draw)
    m = make_analysis(B_static, true_field, grid, 
                      n_obs=n_obs,
                      obs_std_err=obs_std_err, n_seeds=n_seeds, seed=seed,
                      draw=draw)
    h_mean = make_analysis(hybrid_mean_sample_predictor.B, true_field, grid, 
                      n_obs=n_obs,
                      obs_std_err=obs_std_err, n_seeds=n_seeds, seed=seed,
                      draw=draw)
    # h_med = make_analysis(hybrid_med_sample_predictor.B, true_field, grid, 
    #                   n_obs=n_obs,
    #                   obs_std_err=obs_std_err, n_seeds=n_seeds, seed=seed,
    #                   draw=draw)
    plt.close('all')
    print('compute_analysis_R_s')
    for predictor in predictors_list:
        predictor.compute_analysis_R_s(t, s, ensemble, true_field, 
                                       n_obs, n_seeds, obs_std_err, draw, 
                                       m=m, h=h_mean)
        plt.close('all')

    for predictor in predictors_list:
        print(f'{predictor.name} R_S = {predictor.R_s}')
        print(f'{predictor.name} R_M = {predictor.R_m}')
        print(f'{predictor.name} R_H = {predictor.R_h}')
        predictors_errors.update(predictor)
        plt.close('all')

    with open(path_to_save + 'R_s R_m.txt', 'a') as file:
        file.write(
            f'tr_sample l  {np.diag(sample_cvm_loc_predictor.B).sum()} \n'
            )
        file.write(f't l {0} \n')
        
    # with open(path_to_save + 'R_s R_m.txt', 'a') as file:
    #     file.write(
    #         f'tr_nn l  {np.diag(nn_predictor_anls_loss.B).sum()} \n'
    #         )
    #     file.write(f't l {0} \n')

    
    predictors_errors.plot_errors(path_to_save, info=info)
    predictors_errors.compute_mean_error(path_to_save, mean=False)
    plt.close('all')

    with open(path_to_save + 'setup.txt', 'a') as setup_file:
        setup_file.write(f'iter {iteration}, rnd: {np.random.normal()}\n')
    # continue
write_seed(seed, f'all seeds{info}.txt')


# predictors_errors.compute_mean_error(path_to_save, mean=True)



