# -*- coding: utf-8 -*-
"""
Defines function get_modal_spectrum_from_DLSM_params.


Created on Wed Jul 31 15:30:20 2019
Last modified: Feb 2022
@author: Arseniy Sotskiy
"""

# from configs import R_km
from functions import draw_1D, convert_variance_spectrum_to_modal, draw_2D, \
    sigmoid

import numpy as np
# import matplotlib.pyplot as plt
import typing as tp

from RandomProcesses import  RandStatioProcOnS2



#%%
#================================================
#================================================
#================================================

def get_modal_spectrum_from_DLSM_params(
        kappa_SDxi: float, kappa_lamb: float, kappa_gamma: float,
        SDxi_med: float, SDxi_add: float,
        lambda_med: float, lambda_add: float,
        gamma_med: float, gamma_add: float,
        mu_NSL: float, n_max: int, angular_distance_grid_size: int,
        is_cvm_to_be_converted_to_crm: bool=False, 
        draw: bool=False, seed: int=None
        ) -> tp.Tuple:
    r'''Computes the modal spectrum with the given parameters.
    
    The variance spectrum is defined as follows:
        
        .. math::
            f_\ell(x) =  \frac{c(x)} {1 + (\lambda(x) l)^{\gamma(x)}}

    
    Here :math:`\gamma >1` defines the `shape` of the spectrum 
    (and, correspondingly, the shape
    of the covariance function),
    :math:`\lambda` controls the length scale of the process, and :math:`c`
    is the normalizing constant such that, given the parameters 
    :math:`\gamma` and :math:`\lambda`,
    the field's standard deviation :math:`std(\xi)`
    equals the pre-specified value :math:`S`.
 
    The parameter processes :math:`S(x)`, :math:`\lambda(x)`, and
    :math:`\gamma(x)` are defined as
    transformed Gaussian processes generically written as :math:`g(\chi(x))`,
    where :math:`g` is the transformation function and :math:`\chi(x)`
    stands for a stationary Gaussian process.
    We define the ''pre-transform'' Gaussian processes :math:`\chi(x)`
    to have the same shape of the modal spectrum as specified for the
    stationary model but with a larger length scale :math:`\lambda`
    than in the model for :math:`\xi`.
    
    Specifying the same shape of the spectra simplifies the setup and allows 
    an unambiguous comparison of the length scales of the parameter 
    processes :math:`S=S(x)`, :math:`\lambda=\lambda(x)`, and 
    :math:`\gamma=\gamma(x)` on the one hand and the process 
    :math:`\xi` in question on the other hand.
    This latter argument is important because we need to control those 
    length scales as our approach relies on the assumption that 
    the structure of the field in question, :math:`\xi`, changes in space 
    on a significantly larger scale than the length scale of :math:`\xi` 
    itself.
    In DLSM, we ensure this by specifying :math:`\lambda` for the processes 
    :math:`S=S(x)`, :math:`\lambda=\lambda(x)`, and :math:`\gamma=\gamma(x)`
    several times as large as the median :math:`\lambda` for :math:`\xi` 
    as detailed just below.
    
    Specifically, we postulate that
    
    .. math::
      S(x) := {S}_{\rm add} + {S}_{\rm mult}\cdot g(\log\varkappa_S\cdot \chi_S(x, \mu_{\rm NSL})),
      
      \lambda(x) := \lambda_{\rm add} + {\lambda}_{\rm mult} \cdot g(\log\varkappa_\lambda\cdot \chi_\lambda(x, \mu_{\rm NSL})),

      \gamma(x) := \gamma_{\rm add} + \gamma_{\rm mult} \cdot g(\log\varkappa_\gamma\cdot \chi_\gamma(x, \mu_{\rm NSL})),

    where :math:`g` is the transformation function (sigmoid) such that 
    :math:`g(0)=1` ,
    :math:`\chi_S,\chi_\lambda,\chi_\gamma` are the three independent 
    pre-transform stationary Gaussian processes 
    (also defined below), the coefficients 
    :math:`\varkappa_S, \varkappa_\lambda, \varkappa_\gamma`, 
    along with the parameters 
    with subscripts :math:`_{\rm add}` and :math:`_{\rm mult}`, determine
    the strength of the spatial non-stationarity, and
    :math:`\mu_{\rm NSL}` is the ratio (common for all three 
    parameter processes) of their length scale :math:`\Lambda`
    to the median length scale :math:`\lambda` of :math:`\xi`.
    
    In more detail, each of the above three  pre-transform processes,
    :math:`\chi_S,  \chi_\lambda,  \chi_\gamma`
    is a realization of the zero mean, unit variance stationary process 
    :math:`\chi(x)`
    whose variance spectrum is 
    
     .. math::
      f_\ell^\chi \propto \frac{1} {1 + (\Lambda l)^\Gamma},
    
    where :math:`\Gamma=\gamma_{\rm add} + \gamma_{\rm mult}` and
    :math:`\Lambda=(\lambda_{\rm add} + {\lambda}_{\rm mult}) \cdot  \mu_{\rm NSL}`.
    
    With :math:`\varkappa_\bullet=1`, the respective spectrum does not 
    depend on :math:`x`: :math:`f_\ell(x)=f_\ell`.
    The higher :math:`\varkappa_\bullet`, the more variable in space
    becomes the respective parameter:
    :math:`S(x)` (the standard deviation of the process at the given :math:`x`),
    :math:`\lambda(x)` (the spatially variable length scale of the process), and
    :math:`\gamma(x)` (the spatially variable shape of the local correlations).
    We specify :math:`\varkappa_\bullet` to lie between 1 (stationarity) 
    and 4 (wild non-stationarity),
    with 2 being the default value.
    
    The greater the parameter :math:`\mu_{\rm NSL}`, 
    the smoother in space the parameter processes and, thus,
    the weaker the spatial non-stationarity of :math:`\xi(x)`.
    We specify  :math:`\mu_{\rm NSL}` in range from 1 to 10, 
    with 3 being the default value.
    
    After generating :math:`S(x), \lambda (x), \gamma(x)` 
    we calculate :math:`c(x)`
    and finally compute the variance spectrum.

    

    Parameters
    ----------
    kappa_SDxi : float
        non-stationarity coefficient of field standard deviation
        (1 if statio)
    kappa_lamb : float
        non-stationarity coefficient of the length scale of the process
        (1 if statio)
    kappa_gamma : float
        non-stationarity coefficient of the shape of the covariance function
        (1 if statio)
    SDxi_med : float
        mult part of variance of the process
    SDxi_add : float
        min allowable variance of the process
    lambda_med : float
        mult part of  length scale :math:`\lambda(x)`
    lambda_add : float
        min allowable :math:`\lambda(x)`.
    gamma_med : float
        mult part of the shape of the cvf - :math:`\gamma(x)`
    gamma_add : float
        min allowable :math:`\gamma(x)`.
    mu_NSL : float
        parameter of smoothness of fields :math:`S(x), \gamma(x), \lambda(x)`.
    n_max : int
        n_max from config.
    angular_distance_grid_size : int
        angular distance grid size
    draw : bool, optional
        If True, then pictures are plotted. The default is False.
    seed : int, optional
        Random seed for numpy. The default is None.

    Returns 
    -------
    spectrum: np.array
        variance spectrum, shape = (n_max + 1, nlat, nlon)
    lambx: np.array
        field :math:`\lambda(x)` on sphere, shape = (nlat, nlon)
    gammax: np.array
        field :math:`\gamma(x)` on sphere, shape = (nlat, nlon)
    Vx: np.array
        field :math:`V(x)` on sphere, shape = (nlat, nlon)
    cx: np.array
        normalizing constant :math:`c(x)` on sphere, shape = (nlat, nlon)

    '''
    
    # .. math::
    #     c = \frac{1} {\sum_{\ell=0}^{L}   \frac{1}{1 + (\lambda l)^\gamma} }
    
    
    def frac(lamb, n, gamma_mult):
        return 1/(1+(lamb*n)**gamma_mult)
    #####
    Gamma = gamma_med
    Lamb = lambda_med * mu_NSL
    
    SDxi_mult = SDxi_med - SDxi_add
    lambda_mult = lambda_med - lambda_add
    gamma_mult = gamma_med - gamma_add
    #####
    print('Lamb:', Lamb)
    spectrum_hi = np.array([
        1 / (1 + (Lamb * l)**Gamma) for l in range(n_max+1)
        ])
    if draw:
        draw_1D(spectrum_hi, title='spectrum_hi')
    modal_spectrum_hi = convert_variance_spectrum_to_modal(spectrum_hi, n_max)
    
    if seed is not None:
        np.random.seed(seed)
    Red_Noise = RandStatioProcOnS2(modal_spectrum_hi, n_max,
                                     angular_distance_grid_size)

    Red_noise_realiz_std_xi = Red_Noise.generate_one_realization(draw=draw)
    Red_noise_realiz_lamb = Red_Noise.generate_one_realization()
    Red_noise_realiz_gamma = Red_Noise.generate_one_realization()

    if draw:
        draw_2D(Red_noise_realiz_lamb.data,
                title=f'Red_noise_realiz_lamb, Gamma={Gamma}',)
                
    Vx = (SDxi_add + SDxi_mult * sigmoid(np.log(kappa_SDxi) * Red_noise_realiz_std_xi.data))**2

    lambx: np.array = lambda_add + lambda_mult * sigmoid(np.log(kappa_lamb) * Red_noise_realiz_lamb.data)
    gammax: np.array = gamma_add + gamma_mult * sigmoid(np.log(kappa_gamma) \
                                              * Red_noise_realiz_gamma.data)

    if draw:
        draw_2D(lambx, title="$ lamb(x)$")
        draw_2D(gammax, title="$ \gamma(x)$")
        draw_2D(Vx, title="$ V(x)$")

        # print(lambx.shape)

    spectrum = 1 / (1 + np.power(
        lambx[np.newaxis,:,:] * np.arange(n_max + 1)[:,np.newaxis,np.newaxis],
        gammax
        )
        )
    if draw:
        for n in [3]:
            draw_2D(spectrum[n], title='b_n(x), n={}'.format(n))
    cx = 4 * np.pi * Vx / (spectrum * (2 * np.arange(n_max + 1)[:,np.newaxis,np.newaxis] + 1)).sum(axis=0)

    spectrum = spectrum * cx

    return spectrum, lambx, gammax, Vx, cx


    # ###    !! spectrum.shape is (n_max, lat, lon) !!




