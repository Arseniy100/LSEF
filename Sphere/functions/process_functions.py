# -*- coding: utf-8 -*-
"""
Last modified on Apr 2022
@author: Arseniy Sotskiy


"""

# import importlib

import matplotlib.pyplot as plt
import numpy as np
import pyshtools
import typing as tp

pyshtools.utils.figstyle(rel_width=0.7)


# import configs.config as config
from functions.tools import convert_diag_matrix_to_aligned
from functions.tools import draw_1D, draw_2D, draw_3D, svd_pseudo_inversion, \
    convert_modal_spectrum_to_variance, convert_variance_spectrum_to_modal
from Grids import LatLonSphericalGrid
# from RandomProcesses.Sphere import RandStatioProcOnS2

from scipy.special import legendre
from numpy.polynomial.legendre import legval

import time

# importlib.reload(config)

from Band import Band, Band_for_sphere



#%% Generate observations

def generate_obs_points_old(n_obs: int, grid: LatLonSphericalGrid, 
                        seed: int=None) -> tp.List[tp.Tuple[int, int]]:
    '''Generates observation points on sphere.    

    Parameters
    ----------
    n_obs : int
        number of observations.
    grid : LatLonSphericalGrid
        spherical grid from which we take points.
    seed : int, optional
        random seed. The default is None.

    Returns
    -------
    points : tp.List[tp.Tuple[int, int]]
        tuples of points (theta, phi). theta < grid.nlat, phi < grid.nlon

    '''
    if seed is not None:
        np.random.seed(seed)
    points = []
    colats = list(set(grid.colats))
    colats.sort()
    north_included = False
    south_included = False
    while len(points) < n_obs:
        if len(points) == grid.npoints:
            break
        v = np.random.uniform(-1, 1)
        theta = np.arccos(v)
        theta = np.where(colats <= theta)[0].max() # make closest, not <=
        phi = np.random.randint(0, grid.nlon-1)
        if (theta, phi) not in points:
            if theta == 0:
                if not north_included:
                    points.append((0, 0))
                    north_included = True
            elif theta == grid.nlat - 1:
                if not south_included:
                    points.append((grid.nlat - 1, 0))
                    south_included = True
            else:
                points.append((theta, phi))
    return points


def generate_obs_points(
        n_obs: int,
        grid: LatLonSphericalGrid
        ) -> tp.List[tp.Tuple[int, int]]:
    '''Generates observation points on sphere.    

    Parameters
    ----------
    n_obs : int
        number of observations.
    grid : LatLonSphericalGrid
        spherical grid from which we take points.
    seed : int, optional
        random seed. The default is None.

    Returns
    -------
    points : tp.List[tp.Tuple[int, int]]
        tuples of points (theta, phi). theta < grid.nlat, phi < grid.nlon

    '''
    
    points_1d = np.random.choice(grid.npoints, n_obs, 
                                 p=(grid.areas / grid.areas.sum()))
    points_2d = grid.transform_coords_1d_to_2d(points_1d)
    return points_2d
    


if __name__ == '__main__':
    nlat = 49
    grid = LatLonSphericalGrid(nlat)
    nlon = (nlat - 1) * 2
    field = np.zeros((grid.nlat, grid.nlon))
    for n_obs in np.logspace(2, 3, 6):
    # n_obs = 100
        n_obs = int(n_obs)
        points = generate_obs_points(n_obs, grid)
        for point in points:
            field[point[0], point[1]] = 1
        # draw_2D(field)
        draw_3D(field, title = f'n_obs = {n_obs}')
    n_obs = 4800
    points = generate_obs_points(n_obs, grid)
    for point in points:
        field[point[0], point[1]] = 1
    # draw_2D(field)
    draw_3D(field, title = f'n_obs = {n_obs}')

#%%

def predict_spectrum_1d_svd(band_mean_estim_variances: np.array,
                            Omega: np.ndarray,
                            bands: tp.List[Band_for_sphere],
                            *args, **kwargs
                            ) -> np.array:
    '''Predicts variance_spectrum from band_mean_estim_variances.
    Prediction type: svd (applies svd_pseudo_inversion)
    

    Parameters
    ----------
    band_mean_estim_variances : np.array
        band-mean variance spectrums along each band 
        Variances are divided by band.c (!)
    Omega : np.ndarray
        Omega = np.array(
        [[(bands[j].transfer_function[l]**2) for l in range(n_max+1)]
          for j in range(n_bands)]
        ).
    bands : tp.List[Band_for_sphere]
        list of bands.
    *args
        args to pass into function svd_pseudo_inversion.
    **kwargs
        kwargs to pass into function svd_pseudo_inversion.

    Returns
    -------
    np.array
        estimated variance_spectrum.

    '''
    return svd_pseudo_inversion(Omega, band_mean_estim_variances *\
                                np.array([band.c for band in bands]),
                                *args, **kwargs)





def predict_spectrum_1d(band_mean_estim_variances: np.array,
                        bands: tp.List[Band_for_sphere],
                        n_max: int,
                        left_is_horizontal: bool=False,
                        left_is_parabolic: bool=False,
                        loglog: bool=False,
                        *args, **kwargs) -> np.array:
    '''Predicts variance_spectrum from band_mean_estim_variances.
    Prediction type: linear interpolation (piecewise-linear approximation)   

    Parameters
    ----------
    band_mean_estim_variances : np.array
        band-mean variance spectrums along each band 
        Variances are divided by band.c (!).
    bands : tp.List[Band_for_sphere]
        list of bands..
    n_max : int
        n_max from config.
    left_is_horizontal : bool, optional
        Type of the left line. The default is False.
    left_is_parabolic : bool, optional
        Type of the left line. The default is False.
    loglog : bool, optional
        If True, then done in log-log coordinates. The default is False.
    *args 
        Are not passed anywhere.
    **kwargs
        Are not passed anywhere.

    Returns
    -------
    np.array
        estimated variance_spectrum.

    '''
    space = np.arange(bands[0].shape)
    centers = np.array([band.center for band in bands])
    band_transfer_functions = np.array([band.transfer_function for band in bands])
    band_c_array = np.array([band.c for band in bands])

    # band_mean_estim_variances = band_mean_estim_variances #/ band_c_array
    if loglog:
        space = np.log(space)
        # space[0] =
        centers = np.log(centers)
        band_mean_estim_variances = np.log(band_mean_estim_variances)
    k = (band_mean_estim_variances[0] - band_mean_estim_variances[1]) \
        / (centers[0] - centers[1])
    b = band_mean_estim_variances[0] - centers[0] * k
    if left_is_horizontal:
        b = band_mean_estim_variances[0]
    k_ = (band_mean_estim_variances[-1] - band_mean_estim_variances[-2]) \
        / (centers[-1] - centers[-2])
    b_ = band_mean_estim_variances[-1] - centers[-1] * k_
    variance_spectrum_estim_1d = np.interp(space, [0] + list(centers) + [bands[0].shape - 1],
                [b] + list(band_mean_estim_variances) + \
                    [b_ + k_ * (bands[0].shape - 1)])

    if loglog:
        variance_spectrum_estim_1d = np.exp(variance_spectrum_estim_1d)


    return np.maximum(0, variance_spectrum_estim_1d)

#%%
# if __name__ == '__main__':
#     spectrum_estim = predict_spectrum_1d(band_mean_estim_variances, bands,
#                                             # left_is_horizontal=True,
#                                             loglog=True,
#                                             n_max=n_max
#                                           )
#     plt.figure()
#     plt.plot(np.arange(variance_spectrum.shape[0]),
#               variance_spectrum[:, 0, 0], label='true variance_spectrum')
#     plt.plot(np.arange(spectrum_estim.shape[0]),
#               spectrum_estim, label='lines', color='orange')
#     plt.scatter([band.center for band in bands],
#                 band_mean_true_variances/np.array([band.c for band in bands]),
#                 label='band_mean_true_variances')
#     plt.scatter([band.center for band in bands],
#                 band_mean_estim_variances/np.array([band.c for band in bands]),
#                 label='band_mean_estim_variances')
#     plt.legend(loc='best')
#     plt.title(f'')
#     plt.xlabel('l')
#     plt.ylabel('b_mean')
#     plt.grid()
#     plt.show()

    # plt.figure()
    # plt.plot(np.arange(modal_spectrum.shape[0]),
    #           modal_spectrum[:, 0, 0], label='true')
    # plt.plot(np.arange(spectrum_estim.shape[0]),
    #           spectrum_estim, label='estim', color='orange')
    # plt.scatter([band.center for band in bands],
    #             band_mean_true_variances,
    #             label='true')
    # plt.scatter([band.center for band in bands],
    #             band_mean_estim_variances,
    #             label='estim')
    # plt.legend(loc='best')
    # # plt.title(f'average over sphere, size {e_size}')
    # plt.xlabel('l')
    # plt.ylabel('b_mean')
    # plt.grid()
    # plt.xscale('log')
    # plt.yscale('log')
    # plt.show()

#%%
def predict_spectrum_from_bands(band_estim_variances: np.array,
                                svd: bool=False,
                                *args, **kwargs) -> np.array:
    '''
    

    Parameters
    ----------
    band_estim_variances : np.array
        band-mean variance spectrums along each band 
        Variances are divided by band.c (!).
    svd : bool, optional
        if True, then type = svd. If False, then linear. 
        The default is False.
    *args
    **kwargs

    Returns
    -------
    np.array
        estimated variance_spectrum.
    '''
    
    if svd:
        return np.apply_along_axis(predict_spectrum_1d_svd, 0, 
                                   band_estim_variances,
                                   *args, **kwargs)
    return np.apply_along_axis(predict_spectrum_1d, 0, band_estim_variances,
                                   *args, **kwargs)

# if __name__ == '__main__':
#     spectrum_estim = predict_spectrum_from_bands(band_estim_variances, bands,
#                                             # left_is_horizontal=True,
#                                             # loglog=True
#                                          )
#     plt.figure()
#     plt.plot(np.arange(modal_spectrum.shape[0]),
#              modal_spectrum[:, 0, 0], label='true')
#     for point in [(0,0), (45,45), (30,60)]:
#         plt.plot(np.arange(spectrum_estim.shape[0]),
#                  spectrum_estim[:, point[0], point[1]], label='estim',
#                  color='orange')
#     plt.scatter([band.center for band in bands],
#                 band_mean_true_variances,
#                 label='true')
#     for point in [(0,0), (45,45), (30,60)]:
#         plt.scatter([band.center for band in bands],
#                     band_estim_variances[:, point[0], point[1]],
#                     label='estim', color='orange')
#     plt.legend(loc='best')
#     plt.title(f'')
#     plt.xlabel('l')
#     plt.ylabel('b_mean')
#     plt.grid()
#     plt.show()



#%%

def get_modal_spectrum_from_variance(variance_spectrum: np.array, 
                                     n_max: int) -> np.array:
    '''Transforms variance spectrum to modal spectrum.   

    Parameters
    ----------
    variance_spectrum : np.array
    shape = (n_max + 1, nlat, nlon)
    n_max : int
        n_max from config.

    Returns
    -------
    modal_spectrum : np.array
    '''

    if len(variance_spectrum.shape) == 3:
        modal_spectrum = 4 * np.pi * \
                variance_spectrum / \
                (2 * np.arange(n_max + 1) + 1)[:, np.newaxis, np.newaxis]
    else:
        variance_spectrum = 1 / (4 * np.pi) * \
                modal_spectrum / \
                (2 * np.arange(n_max + 1) + 1)
    return modal_spectrum


def get_variance_spectrum_from_modal(modal_spectrum: np.array, 
                                     n_max: int) -> np.array:
    '''Transforms modal spectrum to variance spectrum.   

    Parameters
    ----------
    modal_spectrum : np.array
    shape = (n_max + 1, nlat, nlon)
    n_max : int
        n_max from config.

    Returns
    -------
    variance_spectrum : np.array
    '''

    if len(modal_spectrum.shape) == 3:
        variance_spectrum = 1 / (4 * np.pi) * \
                modal_spectrum * \
                (2*np.arange(n_max + 1) + 1)[:, np.newaxis, np.newaxis]
    else:
        variance_spectrum = 1 / (4 * np.pi) * \
                modal_spectrum * \
                (2*np.arange(n_max + 1) + 1)
            
    return variance_spectrum



#%%


def generate_1D_BLim_WN(n: int, n_x: int, draw: bool=False) -> np.array:
    '''
    Generates band-limited white noise

    Parameters
    ----------
    n : int
        size of the band.
    n_x : int
        parameter (from config).
    draw : bool, optional
        If True, then will be plotted. The default is False.

    Returns
    -------
    realiz : np.array
        realization
    '''

    hi_spec_real = np.random.normal(0, 1, 2*n+1)/np.sqrt(2*n+1)
    hi_spec_imag = np.random.normal(0, 1, 2*n+1)/np.sqrt(2*n+1)
    hi_spec = hi_spec_real + 1j*hi_spec_imag
    hi_spec[0] = np.random.normal(0, 1)/np.sqrt(2*n+1)
    hi_spec[n] = np.random.normal(0, 1)/np.sqrt(2*n+1)
    for i in range(0, n+1):
        hi_spec[-i] = np.conj(hi_spec[i])
    exps = np.array([[np.exp(1j*m*x) for m in (list(range(0, n+1))+list(range(-n, 0)))] for x in range(n_x)])

    realiz = np.matmul(exps, hi_spec)
    if draw:
        plt.title('n = {}'.format(n))
        plt.plot(np.arange(len(realiz)),
                 np.real(realiz),
                 label='real')
        plt.plot(np.arange(len(realiz)),
                 np.imag(realiz),
                 label = 'imag')
        plt.legend(loc='best')
        plt.grid(True)
        plt.figure()
        plt.show()
    return realiz


#%%

def compute_macroscale_L_from_cvm(B: np.array,
                                  L_max: int, n_x: int, R_km: int,
                                  k: int=2,
                                  draw: bool=False) -> np.array:
    '''
    Macroscale (or integral scale) L is a parameter of a stationary
    random process such that B(L) << B(0). It is computed as follows:

    L = 2/B(0)*\integral_0^{\inf}B(t)dt

    In case of non-stationary process we have to change the definition.
    we localize the covariation matrix and only then compute the integral
    (for every x). The result is the array L[x].

    Parameters
    ----------
    B : 2d np.array
        covariance matrix.
    L_max : int
        parameter (I don't know what does it mean).
    k : int, optional
        such that c = L_max*k/dx, dx=2*pi*R/n_x. The default is 2.
        parameter c - for banding the matrix
    draw : bool, optional
        If True, then will be plotted. The default is False.

    Returns
    -------
    L : 1d np.array
        macroscale
    '''
    dx=2*np.pi*R_km/n_x
    c = L_max*k/dx  # parameter c - for banding the matrix

    Band_matrix = np.zeros((n_x, n_x)) # 'localization matrix',
    for i in range(n_x):        # 1 near the diag and 0 otherwise
        for j in range(n_x):
            d = np.abs(i-j)
            if d > n_x/2:
                d = n_x - d
            if d <= c:
                Band_matrix[i,j] = 1

    L = [np.sum(np.multiply(B,Band_matrix)[i,:]/B[i,i]) for i in range(n_x)]
    L = np.array(L)*np.pi*R_km/n_x
    if draw:
        plt.title("Macroscale L (computed from cvm)")
        plt.plot(np.arange(n_x), L, label = 'L(x)')
        plt.xlabel("x")
        plt.ylabel("L, km")
        plt.legend(loc='best')
        plt.grid(True)
        plt.figure(figsize=(12, 5))
        plt.show()
    return L


#%%

def compute_microscale_from_crm(
        corr_matrix: np.array, m_der: int, dx_km: int,
        order: int=1, draw: bool=False):
    '''
    Computes microscale at every x on the grid in the following way:
    microscale = median([d_1, ..., d_{m_der}]),
    where d_i is an approach to L (delta x = i * dx_km);
    L_1 = dx_km / corr_func'(0) ,
                           ' stands for derivative;
    L_2**2 = -1 / corr_func''(0)

    Parameters
    ----------
    corr_matrix : np.array
        correlation matrix.
    m_der : int
    order : int, optional
        order of the derivative. The default is 1. Can be 1 or 2
    draw : bool, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    microscale : np.array
    '''

    microscales = np.zeros((m_der+1, len(corr_matrix)))
    corr_function = convert_diag_matrix_to_aligned(corr_matrix)
    if draw:
        draw_2D(corr_function)
        draw_1D(corr_function[41,:], title = 'corr_function[41,:]')

    for i_der in range(1, m_der+1):
        if order == 1:
            microscales[i_der,:] = (i_der * dx_km) / (1 - corr_function[:,i_der])
        elif order == 2:
            second_der = -2 * (1 - corr_function[:,i_der]) / (i_der * dx_km)**2
            microscales[i_der,:] = np.sqrt(-1 / second_der)
        else:
            return

    if draw:
        #
        draw_1D(corr_function[:,1], title = 'corr_function[:,1]')
        #
        draw_2D(microscales, xlabel='x', ylabel='k', title = '$d_k = (1-crf(k))/k$')
    microscale = np.mean(microscales[1:, :], axis=0)
    if draw:
        draw_1D(microscale, title = 'microscale')

    return microscale


#%%
# from configs import n_max

def make_bands(first_band_length: int, num_of_bands: int, 
               n_x) -> tp.List[Band]:
    ''' Generates Bands 
    total_size = int(n_x/2)
    we have to solve the equation
    first_band_length*(1 + k + ... + k**(num_of_bands - 1)) = total_size
    equiv to
    k**num_of_bands - total_size/first_band_length * k + total_size/first_band_length - 1 = 0

    Parameters
    ----------
    first_band_length : int
    num_of_bands : int
    n_x : int, optional
        The default is n_max + 1.

    Returns
    -------
    list
        list of generated bands.
    '''
    
    total_size = n_x # + 1

    if num_of_bands == 1:
        band = Band(0, int(total_size), n_x)
        return [band]
    if first_band_length * num_of_bands > total_size:
        print("can't make bands")
        return None
    equation_coeffs = [1.0] + [0.0 for _ in range(num_of_bands - 2)]
    equation_coeffs += [-total_size/first_band_length, total_size/first_band_length - 1]
    k = np.real(np.max(np.roots(equation_coeffs)))
    if first_band_length * num_of_bands == total_size:
        k = 1
    bands = [Band(0, first_band_length - 1, n_x)]
    for band_index in range(1, num_of_bands - 1):
        start = bands[band_index-1].right + 1
        end = start + int(first_band_length * k**(band_index)) - 1
        bands.append( Band(start, end, n_x) )
    band_index = num_of_bands - 1
    start = bands[band_index-1].right + 1
    end = int(total_size)
    assert start < end, "start >= end, something went wrong"
    bands.append( Band(start, end, n_x) )

    return bands



#%% BAND-PASS FILTER

def band_pass_filter(grid: pyshtools.SHGrid,
                     band: Band,
                     draw: bool = False) -> pyshtools.SHGrid:
    '''
    Filters spectrum of random field

    Parameters
    ----------
    grid : pyshtools.SHGrid
        random field on a sphere given by grid.
    band : Band
        this part of spectrum should be all spectrum of the new field
    draw: bool
        if true, then function plots spectrums and grids

    Returns
    -------
    filtered_grid : pyshtools.SHGrid
        filtered random field
    '''
    specoeffs = grid.expand(normalization='ortho')

    specoeffs_filtered = specoeffs.copy()

    filtered_coeffs = specoeffs_filtered.coeffs
    for i in range(2):
        filtered_coeffs[i,:,:] = np.multiply(
            specoeffs_filtered.coeffs[i,:,:], band.indicator
            )
    specoeffs_filtered = pyshtools.SHCoeffs.from_array(filtered_coeffs,
                                                      normalization='ortho')

    grid_filtered = specoeffs_filtered.expand()


    if draw:
        specoeffs.plot_spectrum2d(vrange=(1.e-7,0.1), show=False)
        specoeffs_filtered.plot_spectrum2d(vrange=(1.e-7,0.1), show=False)
        grid.plot(show=False)
        grid_filtered.plot(show=False)
    return grid_filtered





#%% NEW BAND-PASS FILTER

def band_pass_filter_new(grid: pyshtools.SHGrid,
                         band: Band_for_sphere,
                         draw: bool = False) -> pyshtools.SHGrid:
    '''
    Filters spectrum of random field

    Parameters
    ----------
    grid : pyshtools.SHGrid
        random field on a sphere given by grid.
    band : Band_for_sphere
        this part of spectrum should be all spectrum of the new field
    draw: bool
        if true, then function plots spectrums and grids

    Returns
    -------
    filtered_grid : pyshtools.SHGrid
        filtered random field
    '''
    specoeffs = grid.expand(normalization='ortho')

    specoeffs_filtered = specoeffs.copy()

    filtered_coeffs = specoeffs_filtered.coeffs
    for i in range(filtered_coeffs.shape[0]):
        for j in range(filtered_coeffs.shape[2]):
            filtered_coeffs[i,:,j] = \
                specoeffs_filtered.coeffs[i,:,j] * band.transfer_function

    specoeffs_filtered = pyshtools.SHCoeffs.from_array(filtered_coeffs,
                                                      normalization='ortho')
    grid_filtered = specoeffs_filtered.expand()

    if draw:
        specoeffs.plot_spectrum2d(vrange=(1.e-7,0.1), show=False)
        specoeffs_filtered.plot_spectrum2d(vrange=(1.e-7,0.1), show=False)
        grid.plot(show=False)
        grid_filtered.plot(show=False)
        plt.show()
    return grid_filtered

