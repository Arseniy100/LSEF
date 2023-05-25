# -*- coding: utf-8 -*-
"""
Last modified on Mar 2022
@author: Arseniy Sotskiy


"""

import matplotlib.pyplot as plt
import numpy as np
import pyshtools
import typing as tp
from pyshtools import SHGrid

from time import process_time

pyshtools.utils.figstyle(rel_width=0.7)


# import configs.config as config
from functions.tools import (
    integrate_gridded_function_on_S2,
    draw_2D, convert_variance_spectrum_to_modal
    )
from functions.process_functions import (
    band_pass_filter_new, generate_obs_points, 
    get_variance_spectrum_from_modal
    )
from Grids import (
    LatLonSphericalGrid, make_2_dim_field, flatten_field, 
    coarsen_grid, interpolate_grid
    )
from RandomProcesses import (
    get_modal_spectrum_from_DLSM_params
    )
# from RandomProcesses.Sphere import RandStatioProcOnS2


from Band import Band_for_sphere

from functions.FLT import (
    Fourier_Legendre_transform_backward, 
    )







#%%

def draw_bias_MSE_band_variances(band_estim_variances: np.array,
                                 band_true_variances: np.array,
                                 band_mean_true_variances: np.array,
                                 bands: tp.List[Band_for_sphere],
                                 path_to_save: str=None,
                                 info: str=''):
    abs_error = np.abs(band_estim_variances - band_true_variances)
    bias = (band_estim_variances - band_true_variances)

    MAE_b_mean = np.array([integrate_gridded_function_on_S2(
                SHGrid.from_array(abs_error[i,:,:])
                ) / 4 / np.pi for i in range(len(bands))])
    bias_b_mean = np.array([integrate_gridded_function_on_S2(
                SHGrid.from_array(bias[i,:,:])
                ) / 4 / np.pi for i in range(len(bands))])


    plt.figure()

    c_j_all_bands = np.array([band.c for band in bands])

    plt.plot([band.center for band in bands],
                band_mean_true_variances * c_j_all_bands,
                label='true', color='black')
    plt.plot([band.center for band in bands],
              bias_b_mean * c_j_all_bands, label='bias', color='green')
    plt.plot([band.center for band in bands],
              MAE_b_mean * c_j_all_bands, label='MAE', color='red')
    plt.legend(loc='best')

    plt.xlabel('l')
    plt.ylabel('$V_{(j)}$')
    plt.grid()
    title = f'band variances (mean over sphere){info}'
    plt.title(title)
    if path_to_save is not None:
        plt.savefig(path_to_save + f'{title}.png')
    plt.show()




#%%


def generate_ensemble(W: np.array, e_size: int, grid: LatLonSphericalGrid,
                      k_coarse: int=2, 
                      draw: bool=False, info: str='', path_to_save: str=''):
    alpha = np.random.normal(size=(e_size+1, grid.npoints))
    ensemble = np.matmul(W, alpha.T).T
    truth = ensemble[0,:]
    ensemble = ensemble[1:,:]

    ensemble_fields = [make_2_dim_field(field, grid.nlat,
                                        grid.nlon+1) for field in ensemble]
    true_field = make_2_dim_field(truth, grid.nlat,
                                        grid.nlon+1)
    
    ensemble_fields_interp = [interpolate_grid(field,
                                              k_coarse) for field in ensemble_fields]
    true_field_interp = interpolate_grid(true_field, k_coarse)
    
    if draw:
        for i, cmap in enumerate([plt.cm.inferno, plt.cm.BuPu,]):
            draw_2D(true_field,
                    title='X_true', cmap=cmap, 
                    save_path=path_to_save + f'true_field{info}_{i}.png',
                    save_gray=path_to_save + f'true_field{info}_{i}_.png'
                    )

    return true_field_interp, ensemble_fields_interp, ensemble, true_field




#%%

def compute_B_from_spectrum(variance_spectrum: np.array, n_max: int, 
                            grid: LatLonSphericalGrid,
                            k_coarse: int=2, info: str=''):
    modal_spectrum = convert_variance_spectrum_to_modal(
            variance_spectrum, n_max
            )

    sigma = np.sqrt(modal_spectrum)
    sigma_coarsened = np.array(
        [coarsen_grid(sigma[i,:,:], k_coarse) for i in range(len(sigma))]
        )
    sigma_coarsened_flattened = np.array(
        [flatten_field(sigma_coarsened[i,:,:]) for i in range(len(sigma))]
        )
    u = compute_u_from_sigma(sigma_coarsened_flattened, grid, n_max)
    W = compute_W_from_u(u, grid)
    B = np.matmul(W, W.T)
    
    return W, B









#%%



def get_band_varinaces_dataset_for_nn(
        n_training_spheres: int,
        grid: LatLonSphericalGrid, 
        n_max: int,
        k_coarse: int, 
        e_size: int, 
        bands: tp.List[Band_for_sphere], 
        seed: int, 
        DLSM_params: list,
        points: list=[],
        force_generation: bool=False
        ):
    try:
        if force_generation:
            not_existing_file = np.load(
            f'nothing/there will be error and data will be generated.npy'
            )
        band_true_variances_train_dataset = np.load(
            f'data/band_true_variances_train_dataset_{n_max}_{n_training_spheres}.npy'
            )
        band_estim_variances_train_dataset = np.load(
            f'data/band_estim_variances_train_dataset_{n_max}_{n_training_spheres}.npy'
            )
        modal_spectrum_train_dataset = np.load(
            f'data/modal_spectrum_train_dataset_{n_max}_{n_training_spheres}.npy'
            )
    
    except FileNotFoundError:
    # if not is_dataset_precomputed:           
        band_true_variances_train_dataset = []
        band_estim_variances_train_dataset = []
        modal_spectrum_train_dataset = []
        for i_sphere in range(n_training_spheres):
            plt.close('all')
            modal_spectrum_train, lambx_train, gammax_train, \
                Vx_train, cx_train = \
                get_modal_spectrum_from_DLSM_params(draw=(i_sphere == 0),
                    seed=seed+i_sphere, *DLSM_params)
            variance_spectrum_train = get_variance_spectrum_from_modal(
                modal_spectrum_train, n_max
                )    
                   
            modal_spectrum_train_dataset.append(modal_spectrum_train)
  
            W_train, B_train = compute_B_from_spectrum(
                variance_spectrum_train, n_max, grid=grid,
                k_coarse=k_coarse, info='train'
                )
            alpha_train = np.random.normal(size=(e_size+1, grid.npoints))
            ensemble_train = np.matmul(W_train, alpha_train.T).T
            # truth_train = ensemble_train[0,:]
            ensemble_train = ensemble_train[1:,:]
            ensemble_train_fields = [make_2_dim_field(field, grid.nlat,
                                     grid.nlon+1) for field in ensemble_train]
            # true_field_train = make_2_dim_field(truth_train, grid.nlat,
            #                                     grid.nlon+1)
            ensemble_train_fields_interp = [interpolate_grid(field,
                                                      2) for field in ensemble_train_fields]
            # true_field_train_interp = interpolate_grid(true_field_train, 2)     
            
            band_variances_train_dict = \
                get_band_variances(bands, len(bands), variance_spectrum_train, 
                           ensemble_train_fields_interp, points)
                # Variances are divided by band.c (!)
            
            band_true_variances_train = band_variances_train_dict['band_true_variances']
            band_estim_variances_train = band_variances_train_dict['band_estim_variances']
            # band_mean_true_variances_train = band_variances_train_dict['band_mean_true_variances']
            # band_mean_estim_variances_train = band_variances_train_dict['band_mean_estim_variances']
            
            band_true_variances_train_dataset.append(band_true_variances_train)
            band_estim_variances_train_dataset.append(band_estim_variances_train)
        
#%
  
        band_true_variances_train_dataset = np.concatenate(
            band_true_variances_train_dataset, axis=1
            )
        band_estim_variances_train_dataset = np.concatenate(
            band_estim_variances_train_dataset, axis=1
            )
        modal_spectrum_train_dataset = np.concatenate(
            modal_spectrum_train_dataset, axis=1
            )
        #% %
        np.save(
            f'data/band_true_variances_train_dataset_{n_max}_{n_training_spheres}.npy',
            band_true_variances_train_dataset
            )
        
        np.save(
            f'data/band_estim_variances_train_dataset_{n_max}_{n_training_spheres}.npy',
            band_estim_variances_train_dataset
            )
        
        np.save(
            f'data/modal_spectrum_train_dataset_{n_max}_{n_training_spheres}.npy',
            modal_spectrum_train_dataset
            )

    return (band_true_variances_train_dataset,
            band_estim_variances_train_dataset, 
            modal_spectrum_train_dataset)
      
        

#%%
def draw_MAE_and_bias_of_spectrum(variance_spectrum_estim: np.array,
                                  variance_spectrum: np.array,
                                  title: str='',
                                  path_to_save: str=None,
                                  n_png=0
                                  ):
    '''
    Plots MAE and bias of estim variance spectrum 
    compared to the true one. 

    Parameters
    ----------
    variance_spectrum_estim : np.array
        estimation of the spectrum.
    variance_spectrum : np.array
        true spectrum.
    title : str, optional
        title of the image. The default is ''.
    path_to_save : str, optional
        path to save the picture. The default is None (no saving if so)

    Returns
    -------
    n_png: int

    '''
    mean_variance_spectrum_shape = [integrate_gridded_function_on_S2(
                    SHGrid.from_array(variance_spectrum_estim[i,:,:])
                    ) / 4 / np.pi for i in range(variance_spectrum_estim.shape[0])]


    MAE_spectrum_shape = [integrate_gridded_function_on_S2(
                SHGrid.from_array(np.abs(
                    variance_spectrum_estim - variance_spectrum
                    )[i,:,:])
                ) / 4 / np.pi for i in range(variance_spectrum_estim.shape[0])]
    bias_spectrum_shape = [integrate_gridded_function_on_S2(
                SHGrid.from_array(
                    (variance_spectrum_estim - variance_spectrum)[i,:,:]
                    )
                ) / 4 / np.pi for i in range(variance_spectrum_estim.shape[0])]
    variance_spectrum_mean = [integrate_gridded_function_on_S2(
                SHGrid.from_array(variance_spectrum[i,:,:])
                ) / 4 / np.pi for i in range(variance_spectrum_estim.shape[0])]

    plt.figure()
    plt.scatter(np.arange(variance_spectrum.shape[0]),
              variance_spectrum_mean,
              label='true (mean)', color='black')
    plt.plot(np.arange(variance_spectrum.shape[0]),
              MAE_spectrum_shape, label='MAE', color='red')
    plt.plot(np.arange(variance_spectrum.shape[0]),
              bias_spectrum_shape, label='bias', color='green')


    for point in [(0,0), (45,45), (30,60)]:
        plt.plot(np.arange(variance_spectrum_estim.shape[0]),
                  variance_spectrum_estim[:, point[0], point[1]],
                  label='estim',
                  color='orange')

    plt.plot(np.arange(variance_spectrum.shape[0]),
              mean_variance_spectrum_shape,
              label='mean estim (ensm)', color='gray')

    plt.legend(loc='best')

    plt.xlabel('l')
    plt.ylabel('b')
    plt.grid()
    plt.title(title)
    n_png += 1
    if path_to_save:
        plt.savefig(path_to_save + f'{n_png} {title}.png')
    plt.show()
    plt.close('all')
    
    return n_png







#%%

def compute_u_from_sigma(sigma_coarsened_flattened: np.array, 
                         grid: LatLonSphericalGrid, n_max: int) -> np.array:
    '''
    sigma = np.sqrt(modal_spectrum)
    u = Fourier_Legendre_transform_backward(sigma)

    Parameters
    ----------
    sigma_coarsened_flattened : np.array
        coarsened (nlat, nlon *= 1/2) and flattened sigma
    grid : LatLonSphericalGrid
    n_max : int

    Returns
    -------
    u: np.array
    '''
        
    if grid.rho_matrix is None:
            grid.rho(0,0)
    u = np.zeros(grid.rho_matrix.shape)

    for x in range(grid.npoints):
        
        u_x_rho = Fourier_Legendre_transform_backward(
            n_max, sigma_coarsened_flattened[:,x],
            rho_vector=grid.rho_matrix[x,:]
            )
        u[x,:] = u_x_rho

    return u


#%%
def compute_W_from_u(u: np.array, grid: LatLonSphericalGrid) -> np.array:
    r'''
    Random process :math:`\xi = W \cdot \alpha`, 
    where :math:`\alpha` - white noise
     
    :math:`B = W W^T` - covatiance matrix.
    
    We begin with the space continuous model, 
    constraining :math:`w(x, y)` to be of the form
    
    .. math:: w(x, y) = u(x, \rho(x, y))
    
    In discrete model
    
     .. math::     w_{ij} = u(x_i, \rho(x_i, y_j)) \sqrt{\Delta y_j},
    
    where :math:`\Delta y_j` is the area of :math:`j` th grid cell

    Parameters
    ----------
    u : np.array
    
    grid : LatLonSphericalGrid

    Returns
    -------
    W : np.array
    '''
    W: np.array = np.multiply(u,
                    np.matmul(
                        grid.areas_sqrt.reshape(-1,1),
                        np.ones((1, grid.npoints))
                        ).T
                    )
    return W


#%%

class Observations:
    '''
    Class of observations.
    Observations x_o are obtained from true x_true as follows:
        x_o = H * x_true + eta,
    where eta is normally distributed white noise, H - obs. operator
    Here H is matrix of shape (n_obs, grid.npoints),
    every row of H contains one "1" and (grid.npoints-1) "0".
    '''

    def __init__(self, obs_points_2d: list, x_true: np.array, std_err: float,
                 grid: LatLonSphericalGrid):
        '''
        Generates observation from given points

        Parameters
        ----------
        obs_points_2d : list[tuple[int, int]]
            list of points given as (lon_index, lat_index)
        x_true : np.array
            true array from which the observation are taken
        std_err : float
            standart deviation of the observation error (eta).
        grid : LatLonSphericalGrid

        Returns
        -------
        None.

        '''
        self.obs_points_2d = obs_points_2d
        self.n_obs = len(obs_points_2d)
        self.obs_points_1d = grid.transform_coords_2d_to_1d(obs_points_2d)

        x_true_flattened = x_true

        H = np.zeros((self.n_obs, grid.npoints))
        H[np.arange(self.n_obs), self.obs_points_1d] = 1

        self.H = H

        R = std_err**2 * np.eye(self.n_obs)
        self.R = R

        x_obs = x_true_flattened[self.obs_points_1d] \
            + np.random.normal(0, std_err, self.n_obs)

        self.x_obs = x_obs


    def apply_H_to_x(self, x: np.array):
        '''
        Computes self.H @ x efficiently
        (we know that H has only one 1 in every row)

        Parameters
        ----------
        x : np.array
            1d array of forecast.

        Returns
        -------
        np.array
            H @ x

        '''
        return x[self.obs_points_1d]

    def compute_RMSE(self, forecast: np.array):
        return np.sqrt(
            np.mean((self.x_obs - forecast[self.obs_points_1d])**2)
            )



#%% Analysis

def make_analysis(B: np.array, field_true: np.array, 
                  grid: LatLonSphericalGrid,
                  obs_std_err: float, n_obs: int, n_seeds: int=1, 
                   seed: int=None, 
                  draw: bool=False,
                  time_it: bool=False, 
                  return_mode: str='mean_mse') -> np.array:
    if seed is not None:
        np.random.seed(seed)
    assert return_mode in ['mean_mae', 'anls_error', 'mean_mse']
    all_MAE_anls = []
    all_MSE_anls = []
    all_OmF = []
    all_OmA = []
    for _ in np.arange(n_seeds):
        obs_points_2d = generate_obs_points(n_obs, grid)

        x_true=flatten_field(field_true)

        x_forecast = np.zeros((grid.npoints))


        start = process_time()
        observations = Observations(obs_points_2d, x_true,
                                    std_err=obs_std_err, grid=grid)
        end = process_time()
        if time_it:
            print(f'observations: {end - start} sec')

        start = process_time()
        # observation increment:
        y_obs = observations.x_obs - observations.apply_H_to_x(x_forecast)
        end = process_time()
        if time_it:
            print(f'observation increment: {end - start} sec')

        start = process_time()
        #analysis:
        HBHt = np.matmul(np.matmul(observations.H, B), observations.H.T)
        end = process_time()
        if time_it:
            print(f'HBHt: {end - start} sec')

        start = process_time()
        anls_increment = np.matmul(
            np.matmul(B, observations.H.T),
            np.linalg.solve(HBHt + observations.R, y_obs)
            )
        # K = np.matmul(
        #     np.matmul(B, observations.H.T),
        #     np.linalg.inv(HBHt + observations.R)
        #     )
        end = process_time()
        if time_it:
            print(f'anls_increment: {end - start} sec')

        x_anls = x_forecast + anls_increment

        x_anls_2d = make_2_dim_field(x_anls, field_true.shape[0],
                                     field_true.shape[1])
        if draw:
            draw_2D(field_true**2, title='Truth^2')
            draw_2D(x_anls_2d**2, title='Analysis^2')

        anls_error = x_anls_2d - field_true
        print('shape', anls_error.shape)
        # var of anls error:
        print(f'var(anls_error): {np.var(anls_error)}')
        # assert False, 'unravel_index(a.argmax(), a.shape)'
        print(
            f'max error point: {np.argmax(np.abs(anls_error))}, \
            max={np.max(np.abs(anls_error))}')
        # print(np.var(field_true))


        # MAE_anls = integrate_gridded_function_on_S2(
        #             SHGrid.from_array(np.abs(anls_error))
        #             ) / 4 / np.pi
        MSE_anls = \
            integrate_gridded_function_on_S2(
                SHGrid.from_array((anls_error)**2)
            ) / 4 / np.pi


        OmF = observations.compute_RMSE(x_forecast)
        OmA = observations.compute_RMSE(x_anls)

        # all_MAE_anls.append(MAE_anls)
        all_MSE_anls.append(MSE_anls)
        all_OmF.append(OmF)
        all_OmA.append(OmA)


    # all_MAE_anls = np.array(all_MAE_anls)
    all_MSE_anls = np.array(all_MSE_anls)
    all_OmF = np.array(all_OmF)
    all_OmA = np.array(all_OmA)
    # print('MAE_anls:', all_MAE_anls.mean())
    # print('OmF: ', np.sqrt(np.mean(all_OmF**2)))
    # print('OmA: ', np.sqrt(np.mean(all_OmA**2)))
    if return_mode == 'mean_mae':
        return np.mean(all_MAE_anls)
    if return_mode == 'anls_error':
        return anls_error
    if return_mode == 'mean_mse':
        return np.mean(all_MSE_anls)



#%%

def get_band_variances(bands, n_bands, variance_spectrum, 
                       ensemble_fields_interp, points=[]):
    '''
    Gets band-mean variance spectrums along each band 
    for every point 
    (and also average along all the sphere)
    Variances are divided by band.c (!)
    
    averaging weights = transfer function squared
    weights are normalized to sum up to 1

    Parameters
    ----------
    bands : tp.List[]
        list of bands.
    n_bands : int
        number of bands.
    variance_spectrum : np.array
        true variance spectrum.
    ensemble_fields_interp : tp.List[np.array]
        list of fields from the ensemble.
    points : tp.List[int], optional
        list of points at which we compute band_estim_variances. 
        The default is [].

    Returns
    -------
    dict
    

    '''

    
    V_band = np.zeros((n_bands, variance_spectrum.shape[1],
                                variance_spectrum.shape[2]))
    V_band_axis = np.zeros((n_bands))
            #      b_mean:
    band_mean_estim_variances = [] # !!! note that they are divided by c !
    band_estim_variances = []      # !!! note that they are divided by c !
    band_mean_true_variances = []  # !!! note that they are divided by c !
    band_true_variances = []       # !!! note that they are divided by c !
    band_estim_variances_at_points = []
    for band_index, band in enumerate(bands):
        variance_true = np.apply_along_axis(
            lambda x: np.matmul(x, (band.transfer_function)**2), 0,
            variance_spectrum
            )
        band_true_variances.append(variance_true / band.c)

        fields_filtered_new = [band_pass_filter_new(
            SHGrid.from_array(field), band, draw=False
            ) for field in ensemble_fields_interp]

        fields_filtered_data = np.array(
            [field.data for field in fields_filtered_new]
            )

        variance_estim = np.mean(fields_filtered_data**2,
                    axis=0)

        V_band[band_index,:,:] = np.dot(
                variance_spectrum.T,  band.transfer_function * band.transfer_function
            ).T / np.sum(band.transfer_function**2)

        V_band_axis[band_index] = np.sum(
            np.arange(
                variance_spectrum.shape[0]
                ) * band.transfer_function * band.transfer_function
            ) / np.sum(band.transfer_function**2)

        mean_variance_true = integrate_gridded_function_on_S2(
            SHGrid.from_array(variance_true)
            ) / 4 / np.pi
        band_mean_true_variances.append(mean_variance_true / band.c)
        mean_variance_estim = integrate_gridded_function_on_S2(
            SHGrid.from_array(variance_estim)
            ) / 4 / np.pi
        band_mean_estim_variances.append(mean_variance_estim  / band.c)
        band_estim_variances.append(variance_estim  / band.c)
        band_estim_variances_at_points.append(
            np.array([variance_estim[point[0], point[1]] for point in points]) \
                / band.c
            )


        MAE = integrate_gridded_function_on_S2(
            SHGrid.from_array(variance_estim - variance_true), np.abs
            ) / 4 / np.pi
        bias = integrate_gridded_function_on_S2(
            SHGrid.from_array(variance_estim - variance_true)
            ) / 4 / np.pi
        MAE_rel = integrate_gridded_function_on_S2(
            SHGrid.from_array(
                (variance_estim - variance_true)/mean_variance_true
                ), np.abs
            ) / 4 / np.pi

    
    band_estim_variances_at_points = np.array(band_estim_variances_at_points)
    band_estim_variances = np.array(band_estim_variances)
    band_true_variances = np.array(band_true_variances)
    band_mean_estim_variances = np.array(band_mean_estim_variances)
    band_mean_true_variances = np.array(band_mean_true_variances)
            
    return {
        'band_mean_estim_variances': band_mean_estim_variances,
        'band_estim_variances': band_estim_variances,
        'band_mean_true_variances': band_mean_true_variances,
        'band_true_variances': band_true_variances,
        'band_estim_variances_at_points': band_estim_variances_at_points
        }



