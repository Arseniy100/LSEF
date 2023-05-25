# -*- coding: utf-8 -*-
"""
Created on Thu Dec 30 22:06:37 2021

Last modified: Mar 2022

@author: Arseniy Sotskiy
"""

from collections import namedtuple
import typing as tp
import numpy as np
import matplotlib.pyplot as plt
from pyshtools import SHGrid
from statsmodels.nonparametric.smoothers_lowess import lowess
import pandas as pd
import torch


from Band import (
    Band_for_sphere
    )
# from neural_network import (
#     NeuralNetSpherical
#     )
from functions.process_functions import (
    predict_spectrum_from_bands
    )
from functions.tools import (
    integrate_gridded_function_on_S2, draw_1D,
    convert_variance_spectrum_to_modal,
    convert_modal_spectrum_to_variance,
    find_sample_covariance_matrix
    )
from functions.DLSM_functions import (
    compute_B_from_spectrum, make_analysis
    )
from Grids import (
    LatLonSphericalGrid, make_2_dim_field, flatten_field
    )
from functions.R_progs import (
    fit_shape_S2, CreateExpqBands
    )
from functions.FLT import (
    Fourier_Legendre_transform_backward
    )
from functions.lcz import (
    construct_lcz_matrix, find_best_c_loc
    )



#%%


class BasePredictor:

    def __init__(self, bands: tp.List[Band_for_sphere], n_max: int,
                 grid: LatLonSphericalGrid,
                 color: str='blue'):
        self.bands = bands
        self.n_max = n_max
        self.grid = grid
        self.name = 'BasePredictor'
        self.info = self.name
        self.n_bands = len(bands)
        self.color = color

    def predict(self):
        self.variance_spectrum_estim = np.array([0])
        return
    
    def make_spectrum_positive(self):
        self.variance_spectrum_estim = \
            np.maximum(0, self.variance_spectrum_estim)

    def compute_B(self, k_coarse: int=2, info: str=''):
        self.W, self.B = compute_B_from_spectrum(
            self.variance_spectrum_estim, self.n_max,
            self.grid, k_coarse, info
            )

    def draw_MAE_bias(self, variance_spectrum_true: np.array,
                      info: str=''):

        MAE_spectrum = [integrate_gridded_function_on_S2(
            SHGrid.from_array(np.abs(
                self.variance_spectrum_estim - variance_spectrum_true
                )[i,:,:])
            ) / 4 / np.pi for i in range(

                        self.variance_spectrum_estim.shape[0]
                        )]
        bias_spectrum = [integrate_gridded_function_on_S2(
                    SHGrid.from_array(
                        (self.variance_spectrum_estim - variance_spectrum_true)[i,:,:]
                        )
                    ) / 4 / np.pi for i in range(variance_spectrum_true.shape[0])]
        variance_spectrum_mean = [integrate_gridded_function_on_S2(
                    SHGrid.from_array(variance_spectrum_true[i,:,:])
                    ) / 4 / np.pi for i in range(variance_spectrum_true.shape[0])]

        mean_variance_spectrum_estim = [integrate_gridded_function_on_S2(
                                SHGrid.from_array(self.variance_spectrum_estim[i,:,:])
                                ) / 4 / np.pi for i in range(variance_spectrum_true.shape[0])]

        plt.figure()
        plt.plot(np.arange(variance_spectrum_true.shape[0]),
                 variance_spectrum_mean,
                 label='true (mean)', color='black')
        plt.plot(np.arange(variance_spectrum_true.shape[0]),
                 MAE_spectrum, label='MAE', color='red')
        plt.plot(np.arange(variance_spectrum_true.shape[0]),
                 bias_spectrum, label='bias', color='green')

        plt.plot(np.arange(variance_spectrum_true.shape[0]),
                  mean_variance_spectrum_estim,
                  label='mean estim (ensm)', color='orange')

        plt.legend(loc='best')


        plt.xlabel('l')
        plt.ylabel('b')
        plt.grid()
        title = f'variance spectrum {self.name} error{info}'
        plt.title(title)
        plt.show()

    def compute_analysis_R_s(self,
                             t: float, s: float,
                             ensemble: np.array,
                             true_field: np.array=None,
                             n_obs: int=None,
                             n_seeds: int=1,
                             obs_std_err: float=None,
                             draw: bool=False,
                             m: float=None,
                             h: float=None,
                             seed: int=None):
        l = make_analysis(
            # B_param,
            # B_estim,
            self.B,
            true_field, self.grid, n_obs=n_obs,
            obs_std_err=obs_std_err, n_seeds=n_seeds,
            draw=draw, seed=seed
        )
        if m is None:
            self.R_m = 0 
        else:
            self.R_m = (l-t)/(m-t)
        if h is None:
            self.R_h = 0 
        else:
            self.R_h = (l-t)/(h-t)
        self.R_s = (l-t)/(s-t)
        self.t = t
        self.s = s
        self.m = m
        self.l = l
        self.h = h
        



#%%

class SampleCovLocPredictor(BasePredictor):

    def __init__(self, bands: tp.List[Band_for_sphere], n_max: int,
                 grid: LatLonSphericalGrid,
                 ensemble: np.array,
                 true_field: np.array=None,
                 n_obs: int=None,
                 obs_std_err: float=None,
                 c_loc: float=None,
                 n_seeds: int=1,
                 seed: int=0,
                 draw_best_c_loc: bool=True,
                 c_loc_array: tp.List[int]=None,
                 color: str=None,
                 precomputed_dict: str=None,
                 info: str=None):
        if color is None:
            color = 'blue'
        if c_loc is None:
            assert (
                (true_field is not None)
                and (n_obs is not None)
                and (obs_std_err is not None)
                )
        
        if precomputed_dict is not None:
            assert info is not None, 'must give info about precomputed c_loc'
            try:
                precomputed_c_locs = np.load(precomputed_dict, allow_pickle='TRUE').item()
            except OSError:
                precomputed_c_locs = dict()
            if info in precomputed_c_locs.keys():
                c_loc = precomputed_c_locs[info]
                print(f'best c_loc: {c_loc}')

        super().__init__(bands, n_max, grid, color)
        self.name = 'Sample_loc'
        self.info = f'Sample  loc cvm predictor with {self.n_bands} bands'

        # c_loc is linear dependent on lambda_med
        if c_loc is None:
            if c_loc_array is None:
                c_loc_array = list(map(int, np.logspace(1, 4, 30)))
            c_loc = find_best_c_loc(ensemble, grid,
                                    c_loc_array, draw_best_c_loc,
                                    true_field, self.grid, n_obs=n_obs,
                                    obs_std_err=obs_std_err,
                                    n_seeds=n_seeds, seed=seed
                                    )
            if precomputed_dict is not None:
                precomputed_c_locs[info] = c_loc
                np.save(precomputed_dict, precomputed_c_locs)
        self.c_loc = c_loc


        B_sample = find_sample_covariance_matrix(ensemble.T)
        lcz_mx = construct_lcz_matrix(grid, c_loc)

        B_sample_loc = np.multiply(B_sample, lcz_mx)

        self.B = B_sample_loc

    def compute_B(self, k_coarse: int=2, info: str=''):
        self.W = None


#%%
class HybridMeanSamplePredictor(SampleCovLocPredictor):
    
    def __init__(self, B_static: tp.Any=None, 
                 w_ensm_hybr: float=0.5,
                 hybrid_type: str='mean', *args, **kwargs):
        assert hybrid_type in ('mean', 'med')
        assert (0 <= w_ensm_hybr) and (w_ensm_hybr <= 1), \
        'w_ensm_hybr must be between 0 and 1'
        self.hybrid_type = hybrid_type
        super().__init__(*args, **kwargs)
        self.B_sample_loc = self.B
        if (hybrid_type == 'mean') and (type(B_static) == str):
            self.B_static = np.loadtxt(B_static)
        else: # B is given
            self.B_static = B_static
        self.B = \
            w_ensm_hybr * self.B_sample_loc + (1-w_ensm_hybr) * self.B_static



#%%
class NNPredictor(BasePredictor):

    def __init__(self, bands: tp.List[Band_for_sphere], n_max: int,
                 grid: LatLonSphericalGrid, 
                 net_name: str,
                 color: str=None):
        if color is None:
            color = 'purple'
        super().__init__(bands, n_max, grid, color)
        self.name = f'NN {net_name}'
        self.info = f'NN {net_name} predictor with {self.n_bands} bands'


        net = torch.load(f'NeuralNetworks/models/{net_name}.pt')
        net.model.eval()
        self.net = net
    
    def predict(self, band_estim_variances: np.array, *args, **kwargs):
        modal_spectrum_estim = self.net.predict(band_estim_variances)
        self.variance_spectrum_estim = convert_modal_spectrum_to_variance(
            modal_spectrum_estim, self.n_max
            )
        return self.variance_spectrum_estim


#%%

class SVDPredictor(BasePredictor):

    def __init__(self, bands: tp.List[Band_for_sphere], n_max: int,
                 grid: LatLonSphericalGrid, predict_modal: bool=False,
                 color: str=None):
        if color is None:
            color = 'orange'
        super().__init__(bands, n_max, grid, color)
        self.name = 'SVD'
        self.info = f'SVD predictor with {self.n_bands} bands'

        

        self.predict_modal = predict_modal
        
        def two_l_plus_1_div_4_pi(l):
            if predict_modal:
                return (2 * l + 1) / (4 * np.pi)
            else:
                return l

        self.Omega = np.array(
            [[(bands[j].transfer_function[l]**2) * two_l_plus_1_div_4_pi(l) \
              for l in range(n_max+1)]
              for j in range(self.n_bands)]
            )

        self.U, self.d, V_T = np.linalg.svd(self.Omega, full_matrices=False)
        self.V = V_T.T

    def predict(self, band_estim_variances: np.array, *args, **kwargs):
        # self.band_estim_variances = band_estim_variances
        spectrum_estim = predict_spectrum_from_bands(
                        # band_true_variances, # all_bands_c[:, np.newaxis, np.newaxis],
                        # band_true_variances, # all_bands_c[:, np.newaxis, np.newaxis],
                        band_estim_variances, # * all_bands_c[:, np.newaxis, np.newaxis],
                        svd=True,
                        n_max=self.n_max,
                        bands=self.bands,
                        precomputed=True,
                        Omega=self.Omega, U=self.U, d=self.d, V=self.V,
                        nSV_discard=0,
                        *args, **kwargs

        )
        
        if self.predict_modal:
           self.variance_spectrum_estim = convert_modal_spectrum_to_variance(
               spectrum_estim, self.n_max
               )
        else:
            self.variance_spectrum_estim = spectrum_estim

        return self.variance_spectrum_estim
    
    




#%%


class SVShapePredictor(SVDPredictor):

    def __init__(self, params_shape: dict, variance_spectrum_med: np.array,
                 color: str=None,
                 *args, **kwargs):
        if color is None:
            color = 'red'
        super().__init__(color=color, *args, **kwargs)
        self.params_shape = params_shape
        self.variance_spectrum_med = variance_spectrum_med
        self.name = 'SVShape'

    def predict(self, band_estim_variances: np.array,

                draw: bool=False, *args, **kwargs):
        super().predict(band_estim_variances, *args, **kwargs)
        e_shape = self.variance_spectrum_med[:,0,0]
        if draw:
            draw_1D(e_shape, title='e_shape(variance_spectrum_med)')
        # assert False
        spectrum_estim_flat = np.array([
            flatten_field(self.variance_spectrum_estim[i,:,:]) \
                for i in range(self.n_max + 1)
            ])
        variance_spectrum_estim_linshape_flat = np.array(
            fit_shape_S2(
            # V2b_shape_S2(
                spectrum_estim_flat.T,
                # variance_spectrum_flat.T,
                e_shape,
                # moments #,a_max_times, w_a_fg,
                **self.params_shape
                )[0]
            ).reshape(spectrum_estim_flat.T.shape).T
        variance_spectrum_estim_linshape = np.array([
            make_2_dim_field(variance_spectrum_estim_linshape_flat[i,:],
                          self.variance_spectrum_estim.shape[1],
                          self.variance_spectrum_estim.shape[2])
            for i in range(self.n_max + 1)
            ])
        self.variance_spectrum_estim = variance_spectrum_estim_linshape



#%%

def add_to_dict(d: dict, key, val):
    if key in d.keys():
        d[key].append(val)
    else:
        d[key] = [val]

class PredictorsErrors:
    
    def __init__(self):
        self.names = set()
        self.R_s_grid = dict()
        self.R_m_grid = dict()
        self.R_h_grid = dict()
        
        self.R_s_mean = dict()
        self.R_m_mean = dict()
        self.R_h_mean = dict()

        self.t_grid = dict()
        self.s_grid = dict()
        self.m_grid = dict()
        self.h_grid = dict()
        self.l_grid = dict()

        self.colors = dict()
    
    def update(self, predictor: BasePredictor):
        add_to_dict(self.R_s_grid, predictor.name, predictor.R_s)
        add_to_dict(self.R_h_grid, predictor.name, predictor.R_h)
        add_to_dict(self.R_m_grid, predictor.name, predictor.R_m)
        add_to_dict(self.t_grid, predictor.name, predictor.t)
        add_to_dict(self.s_grid, predictor.name, predictor.s)
        add_to_dict(self.m_grid, predictor.name, predictor.m)
        add_to_dict(self.h_grid, predictor.name, predictor.h)
        add_to_dict(self.l_grid, predictor.name, predictor.l)

        self.colors[predictor.name] = predictor.color
        self.names.add(predictor.name)
    
    # def get_R_s_grid(self):
    #     return np.array(self.R_s_grid)
    
    # def get_R_m_grid(self):
    #     return np.array(self.R_m_grid)
    
    def compute_mean_error(self, path_to_save: str=None, mean: bool=True):
        for name in self.names:
            if mean:
                t = np.sqrt(np.mean(self.t_grid[name]))
                s = np.sqrt(np.mean(self.s_grid[name]))
                l = np.sqrt(np.mean(self.l_grid[name]))
                m = np.sqrt(np.mean(self.m_grid[name]))
                h = np.sqrt(np.mean(self.h_grid[name]))
                self.R_s_mean[name] = (l-t)/(s-t)
                self.R_m_mean[name] = (l-t)/(m-t)
                self.R_h_mean[name] = (l-t)/(h-t)
                to_print = f'{name}: R_s = {self.R_s_mean[name]}\n'
                to_print += f'{name}: R_m = {self.R_m_mean[name]}\n'
                to_print += f'{name}: R_h = {self.R_h_mean[name]}\n'
            else:
                t = self.t_grid[name][-1]
                s = self.s_grid[name][-1]
                l = self.l_grid[name][-1]
                m = self.m_grid[name][-1]
                h = self.h_grid[name][-1]
                if s:
                    to_print = f'{name}: R_s = {(l-t)/(s-t)}\n'
                if m:
                    to_print += f'{name}: R_m = {(l-t)/(m-t)}\n'
                if h:
                    to_print += f'{name}: R_h = {(l-t)/(h-t)}\n'
            
            to_print += f'{name}: t = {t}\n'
            to_print += f'{name}: s = {s}\n'
            to_print += f'{name}: l = {l}\n'
            to_print += f'{name}: m = {m}\n'
            to_print += f'{name}: h = {h}\n'
            if path_to_save is not None:
                with open(path_to_save + 'R_s R_m.txt', 'a') as file:
                    file.write(to_print)
    
    def plot_errors(self, path_to_save: str=None, info: str='', s:float=8):
        if len(self.names) == 0:
            return
        plt.figure()
        name = list(self.names)[0]
        plt.scatter(np.arange(len(self.t_grid[name])), self.t_grid[name], 
                    label='t', color='black', s=2*s)
        plt.scatter(np.arange(len(self.s_grid[name])), self.s_grid[name], 
                    label='s', color='blue', s=s)
        plt.scatter(np.arange(len(self.m_grid[name])), self.m_grid[name], 
                    label='m', color='gray', s=s)
        plt.scatter(np.arange(len(self.h_grid[name])), self.h_grid[name], 
                    label='h', color='yellow', s=s)
        for name in self.names:
            plt.scatter(np.arange(len(self.l_grid[name])), self.l_grid[name], 
                        label=name, color=self.colors[name], s=s)

        # plt.ylim(bottom=0)
    
        plt.legend(loc='best')
    
        # plt.xlabel('l')
        # plt.ylabel('b')
        plt.grid()
        title = f'analysis results tsml {info}'
        plt.title(title)
        if path_to_save is not None:
            plt.savefig(path_to_save + f'{title}.png')
        plt.show()
        ############################################
        plt.figure()
        for name in self.names:
            plt.scatter(np.arange(len(self.R_s_grid[name])), 
                        self.R_s_grid[name], 
                        label=name, color=self.colors[name])
        plt.ylim(bottom=0)
        plt.legend(loc='best')
        plt.grid()
        title = f'analysis results R_s {info}'
        plt.title(title)
        if path_to_save is not None:
            plt.savefig(path_to_save + f'{title}.png')
        plt.show()
        ############################################
        plt.figure()
        for name in self.names:
            plt.scatter(np.arange(len(self.R_m_grid[name])), 
                        self.R_m_grid[name], 
                        label=name, color=self.colors[name])
        plt.ylim(bottom=0)
        plt.legend(loc='best')
        plt.grid()
        title = f'analysis results R_m {info}'
        plt.title(title)
        if path_to_save is not None:
            plt.savefig(path_to_save + f'{title}.png')
        plt.show()
        ############################################
        plt.figure()
        for name in self.names:
            plt.scatter(np.arange(len(self.R_h_grid[name])), 
                        self.R_h_grid[name], 
                        label=name, color=self.colors[name])
        plt.ylim(bottom=0)
        plt.legend(loc='best')
        plt.grid()
        title = f'analysis results R_h {info}'
        plt.title(title)
        if path_to_save is not None:
            plt.savefig(path_to_save + f'{title}.png')
        plt.show()




#%%



def draw_errors_in_spatial_covariances(predictors: tp.List[BasePredictor],
                                       variance_spectrum_true: np.array,
                                       band_estim_variances: np.array,
                                       B_med: np.array,
                                       B_true: np.array,
                                       error_type: str='bias',
                                       path_to_save: str=None,
                                       info: str='',
                                       frac: float=0.003):

    assert error_type in ['bias', 'mae']

    grid = predictors[0].grid

    cov_med_true = pd.DataFrame(
        index=grid.rho_matrix.ravel(),
        data=(B_med).ravel()
        ).groupby(level=0).mean()
    cov_errors = dict()
    filtered = dict()

    for predictor in predictors:


        if error_type == 'bias':
            data=(predictor.B-B_true).ravel()
        else:
            data=np.abs(predictor.B-B_true).ravel()
        cov_errors[predictor.name] = pd.DataFrame(
            index=grid.rho_matrix.ravel(),
            data=data,
            ).groupby(level=0).mean()

        filtered[predictor.name] = lowess(
            cov_errors[predictor.name].values.flatten(),
            np.array(cov_errors[predictor.name].index),
            is_sorted=True, frac=frac, it=0
            )


#%

    plt.figure()
    for predictor in predictors:
        plt.plot(cov_errors[predictor.name],
                  # label='sample_loc',
                  color=predictor.color, alpha=0.3)

        plt.plot(filtered[predictor.name][:,0], filtered[predictor.name][:,1],
                  label=predictor.name, color=predictor.color)
    plt.plot(cov_med_true.index, cov_med_true, 
              label='median true', color='black')
    plt.legend(loc='best')
    plt.xlabel('Distance, radians')
    if error_type == 'bias':
        title = f'Biases in spatial covariances{info}'
    else:
        title = f'MAE in spatial covariances{info}'
    plt.title(title)
    plt.grid()
    plt.savefig(path_to_save \
                + f'{title}.png')
    plt.show()
    return filtered


#%%
def compute_average_from_distance(B: np.array, grid: LatLonSphericalGrid,
                                  frac: float=0.003):
    data=B.ravel()
    df = pd.DataFrame(
            index=grid.rho_matrix.ravel(),
            data=data,
            ).groupby(level=0).mean()
    filtered = lowess(
            df.values.flatten(),
            np.array(df.index),
            is_sorted=True, frac=frac, it=0
            )
    return filtered


#%%
def compare_predicted_variance_spectra(predictors: tp.List[BasePredictor],
                                       variance_spectrum_true: np.array,
                                       band_estim_variances: np.array,
                                       path_to_save: str=None,
                                       draw_modal: bool=False,
                                       info: str='',
                                       seed: int=None):
    if seed is not None:
        np.random.seed(seed)
    if len(predictors) == 0:
        return
    grid = predictors[0].grid
    bands = predictors[0].bands
    names= [predictor.name for predictor in predictors]
    points = [[
        (np.random.randint(grid.nlon),
         np.random.randint(grid.nlat))
        for _ in range(3)] for _ in range(3)]

    fig, axes = plt.subplots(3, 3,
                             # sharex=True, sharey=True,
                             figsize=(8, 10))
    fig.suptitle('Sharing x per column, y per row')
    if draw_modal:
        spectrum_true = convert_variance_spectrum_to_modal(
            variance_spectrum_true, predictors[0].n_max
            )
    else:
        spectrum_true = variance_spectrum_true
    # np.random.seed(1111)
    for i in range(3):
        for j in range(3):

            point=points[i][j]
            axes[i,j].plot(np.arange(spectrum_true.shape[0]),
                     spectrum_true[:, point[0], point[1]],
                     label='true', color='black')
            for predictor in predictors:
                if draw_modal:
                    spectrum_pred = convert_variance_spectrum_to_modal(
                        predictor.variance_spectrum_estim, predictor.n_max
                        )
                else:
                    spectrum_pred = predictor.variance_spectrum_estim
                axes[i,j].plot(
                    np.arange(spectrum_pred.shape[0]),
                    spectrum_pred[:, point[0], point[1]],
                    label=predictor.name, color=predictor.color, alpha=0.9)
            if draw_modal:
                variances = band_estim_variances[
                    :, point[0], point[1]
                    ] / np.array([band.c for band in bands])
            else:
                variances = band_estim_variances[:, point[0], point[1]] 
            # axes[i,j].scatter(
            # [band.center for band in bands],
            # # band_estim_variances[:, point[0], point[1]] #/ all_bands_c
            # variances #/ all_bands_c
            # )
            axes[i,j].grid()
            axes[i,j].legend(loc='best')
    title = f'variance spectra\n{str(names)}' + info + '\n' + str(points)
    fig.suptitle(title)
    title = title.replace('\n', '')
    # n_png += 1
    if path_to_save is not None:
        plt.savefig(path_to_save + f'{title}.png')
    plt.show()


#%%
def draw_cvf(predictors: tp.List[BasePredictor],
            variance_spectrum_true: np.array,
            B_true: np.array,
            path_to_save: str=None,
            step: int=50,
            info: str=''):

    grid = predictors[0].grid
    n_max = predictors[0].n_max

    modal_spectrum_true = convert_variance_spectrum_to_modal(
        variance_spectrum_true, n_max
        )
    B_theor = Fourier_Legendre_transform_backward(
            n_max, modal_spectrum_true[:,0,0],
            rho_vector_size = grid.npoints
            )

    plt.figure()
    for predictor in predictors:
        plt.scatter(grid.rho_matrix[:,::step].flatten(),
                    predictor.B[:,::step].flatten(),
                 label=predictor.name,
                 color=predictor.color , s=0.1,
                 alpha=0.3)

    plt.plot(np.linspace(0, np.pi, grid.npoints),
        # np.arange(grid.npoints),
             B_theor, label='theor at point [0,0]') # , s=6)
    # plt.plot(np.arange(grid.npoints),
    #          B[0,:], label='B[0,:]', color='green') # , s=6)
    plt.scatter(grid.rho_matrix[0,:].flatten(), B_true[0,:].flatten(),s=2,
                color='green', label='B_true')
    plt.legend(loc='best')
    plt.grid()
    title = f'cvf{info}'
    plt.title(title)
    # n_png += 1
    if path_to_save is not None:
        plt.savefig(path_to_save + f'{title}.png')
    plt.show()

#%%

def draw_covariances_at_points(predictors: tp.List[BasePredictor],
                               B_true: np.array,
                               points='random',
                               path_to_save: str=None,
                               is_along_meridian: bool=False,
                               info: str=''):

    grid = predictors[0].grid

    alpha_grid = [0.7, 0.6, 0.2, 0.2, 0.2]

    if points == 'random':
        points = [
            [
                (np.random.randint(grid.nlon),
                 np.random.randint(grid.nlat))
                for _ in range(3)
            ] for _ in range(3)
        ]

    fig, axes = plt.subplots(3, 3,
                             # sharex=True, sharey=True,
                             figsize=(8, 10))
    fig.suptitle('Sharing x per column, y per row')
                # np.random.seed(1111)
    for i in range(3):
        for j in range(3):

            point=np.array(points[i][j])
            ref_point_num = grid.transform_coords_2d_to_1d(point // 2)

            if not is_along_meridian:
                axes[i,j].scatter(grid.rho_matrix[ref_point_num,:].flatten(),
                                B_true[ref_point_num,:].flatten(),s=1,
                                color='black', label='B true', alpha=1)
            for p, predictor in enumerate(predictors):


                if is_along_meridian:
                    k = ref_point_num % grid.nlon
                    axes[i,j].plot(
                        predictor.B[k::grid.nlon,ref_point_num],
                        color=predictor.color,
                        label=predictor.name
                        )
                else:
                    axes[i,j].scatter(
                        grid.rho_matrix[ref_point_num,:].flatten(),
                        predictor.B[ref_point_num,:].flatten(),s=0.5,
                        color=predictor.color, alpha=alpha_grid[p],
                        label=predictor.name
                        )
            if is_along_meridian:
                axes[i,j].scatter(
                    np.arange(len(B_true[k::grid.nlon,ref_point_num])),
                    B_true[k::grid.nlon,ref_point_num], label='true',
                    color='black'
                    )
            axes[i,j].grid()
            axes[i,j].legend(loc='best')


            # plt.grid()
            # plt.legend(loc='best')
    title = \
        f'covariances{" along meridian" if is_along_meridian else ""}{info}'

    fig.suptitle(title)
    if path_to_save is not None:
        plt.savefig(path_to_save + f'{title}.png')
    plt.show()



#%%


# def predict_variance_spectrum_nn():

#     net = NeuralNetSpherical(
#         n_bands, n_max, loss=loss,
#         activation_name=activation_name,
#         activation_function=activation_function,
#         w_smoo=w_smoo)
#         # net.fit(  # l1 loss
#         #     np.log(band_estim_variances_train_dataset),
#         #     # band_estim_variances_train_dataset,
#         #     # band_true_variances_train_dataset,
#         #     variance_spectrum_train_dataset,
#         #     n_epochs=5, momentum=0.1,
#         #     validation_coeff=0.1,
#         #     path_to_save=path_to_save,
#         #     draw=True,
#         #     **params
#         #     ) # l1 loss


#     net.fit(
#         band_estim_variances_train_dataset,
#         # np.log(band_estim_variances_train_dataset),
#         # np.log(band_true_variances_train_dataset),
#         variance_spectrum_train_dataset,
#         n_epochs=3,
#         momentum=0.1,
#         validation_coeff=0.1,
#         path_to_save=path_to_save,
#         draw=True,
#         band_centers=band_centers,
#         **params
#         ) # my loss
#     #%

#     # variance_spectrum_nn = net.predict(np.log(band_estim_variances))
#     variance_spectrum_nn = net.predict(band_estim_variances)
#     return variance_spectrum_nn