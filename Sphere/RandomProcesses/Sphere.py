# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 13:40:59 2020
Last modified: March 2020

@author: Arseniy Sotskiy
"""


# from configs import n_max

from functions.FLT import Fourier_Legendre_transform_backward

import numpy as np
import matplotlib.pyplot as plt
import pyshtools

pyshtools.utils.figstyle(rel_width=0.7)

# def get_internal_params_from_external(V: float, L: float, r: int):
#     gamma: float = 2 * r + 2  # not sure, can be changed
#     lamb: float = 0  # need to find lamb such that B(cos(L)) = 0.5

#     assert False, 'Function not ready'


def get_spectrum_from_data(V: float, lamb: float, gamma: float,
                           n_max: int) -> np.array:
    '''
    Gets modal spectrum for STATIONARY (isotropic) process on sphere S_2.

    Parameters
    ----------
    V : float
        Variance of the random field
    lamb : float
        Controls the length scale of the process.
    gamma : float
        Defines the shape of the spectrum
        (and, correspondingly, the shape of the covariance function).
        Should be > 1


    # c, lamb and gamma:
    #     To be computed from the three external parameters we are interested in:
    #     the variance V , the macro-scale L, and the smoothness of the process
    #     (quantified as the number r of mean-square derivatives
    #     of the underlying process xi(x)).

    #     For the process covariances to be reasonably resolvable
    #     on the grid, we make sure that L is at least several times as large
    #     as the mesh size in latitude d_theta so that L / d_theta >= 5.

    n_max : int
        The array to be returned should start from 0 and end with n_max
        (len is n_max + 1)

    Returns
    -------
    spectrum : np.array
        [b_n] = [sigma_n ** 2] - modal spectrum
    '''

    spectrum = np.array([
        1 / (1 + (lamb * n) ** gamma) for n in range(n_max + 1)
        ])

    c = 4 * np.pi * V / (np.sum([
        spectrum[n] * (2 * n + 1) for n in range(n_max + 1)
        ]))

    spectrum = spectrum * c
    # The normalizing constant needed to ensure
    # the desired variance of the random field in question.


    return spectrum


class                                                                                                                                                                                                                                                                                                                                                           RandStatioProcOnS2:
    '''
    Class of real-valued random processes on the sphere $S^2$.
    '''

    def __init__(self, modal_spectrum: np.array,
                 n_max: int, angular_distance_grid_size: int):
        '''
        The function which creates an object.
        '''

        self.modal_spectrum = modal_spectrum
        self.power_spectrum = 1 / (4 * np.pi) * np.array(
        [(2*n+1) * self.modal_spectrum[n] for n in range(n_max + 1)]
        )
        self.cvf = Fourier_Legendre_transform_backward(
            n_max,
            rho_vector_size=angular_distance_grid_size,
            modal_spectrum=modal_spectrum
            )
        print('process variance = ', self.cvf[0])

    def generate_one_realization(self, draw: bool=False, **kwargs) -> np.array:
        # coeff[l, m] :
        clm = pyshtools.SHCoeffs.from_random(self.power_spectrum,
                                      normalization='ortho',
                                      **kwargs
                                     )
        # if draw:
        #     fig, ax = clm.plot_spectrum2d(vrange=(1.e-7,0.1), show=False)

        field = clm.expand(grid='DH2')
        # if draw:
        #     fig, ax = field.plot(show=False)
        return field  # maybe should return clm?..

    def generate_multiple_realizations(self,
                                       ensemble_size: int,
                                       draw: bool=False,
                                        **kwargs):
        fields = []
        for i in range(ensemble_size):
            fields.append(
                self.generate_one_realization(draw=draw, **kwargs)
                )
        return fields



#%%

def generate_band_lim_WN_on_sphere(NSW: int, n_max: int,
                                   angular_distance_grid_size: int,
                                   draw: bool = False, seed: int = None
                                   ) -> pyshtools.SHGrid:

    '''
    Generates band-limited white noise

    Parameters
    ----------
    NSW : int
        size of the band.
    n_max : int
        size of spectrum - 1.
    angular_distance_grid_size: int
    draw : bool, optional
        If True, then will be plotted. The default is False.

    Returns
    -------
    realiz_grid : pyshtools.SHGrid
        realization
    '''

    spectrum_white_noise = np.zeros((n_max + 1))
    spectrum_white_noise[0:NSW] = np.ones((NSW))

    white_noise = RandStatioProcOnS2(spectrum_white_noise, n_max,
                                     angular_distance_grid_size)
    realiz_grid = white_noise.generate_one_realization(draw=draw, seed = seed)
    if draw:
        plt.title('random realization of white noise, NSW = {}'.format(NSW))
        plt.show()
    return realiz_grid



# Надо сделать аналогично
def get_spectrum_from_LSEF_data():
    pass


class Proc_LSEF_S2:
    r'''
    Class of the random process ESM on a sphere $S^2$.


    Can make one realisation.
    '''

    def __init__(self, sigma, draw=False):
        '''
        sigma is sigma[n, x] - 2d np.ndarray
        shape = (n_x, n_x)
        '''
        pass
        # self.L =

        # self.sigma = sigma
        # w = iFFT(sigma.T) / np.sqrt(2 * np.pi * R_km)
        # self.w = np.real(w)
        # self.W = convert_aligned_matrix_to_diag(self.w) * np.sqrt(dx_km) #(dx_km) * np.sqrt(n_x)
        # exp = np.array([[np.exp(1j*n*x*2*np.pi/n_x) for n in range(n_x)] for x in range(n_x)])
        # sigma_exp = exp * sigma
        # B_true_from_matrix = np.dot(sigma_exp.T, np.conj(sigma_exp))
        # B_true_from_w = np.matmul(self.W, self.W.T)
        # self.B_true = B_true_from_matrix
        # self.spectrum = self.sigma**2
        # if draw:
        #     draw_2D(sigma, title = 'sigma')
        #     draw_2D(self.spectrum, title = 'spectrum')
        #     draw_2D(self.W, title = 'W')
        #     draw_2D(B_true_from_matrix, title = 'B_true_from_matrix')
        #     draw_2D(B_true_from_w, title = 'B_true_from_w')
        #     draw_2D(B_true_from_matrix - B_true_from_w, title = 'difference')




