# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 14:16:41 2020
Last modified: Mar 2022

@author: Arseniy Sotskiy
"""

import numpy as np
from numpy.polynomial.legendre import legval
from scipy.special import legendre


def Fourier_Legendre_transform_backward_3D(
        # x_array: np.array(float),
        n_max: int,
        modal_spectrum: np.array(float),
        rho_vector_size: int=None,
        rho_vector: np.array=None
        ) -> np.array(float):
    '''
    Transforms modal_spectrum to covariance function.
    B(\rho) = \frac{1}{4 \pi} \sum_0^{n_max} (2 n + 1) b_n P_n(\cos \rho)

    Parameters
    ----------
    n_max : int
        size of modal_spectrum
    modal_spectrum : np.array(float)
        modal_spectrum  [b_n], n = 0, 1, ..., n_max
    rho_vector_size : int
        size of grid of points at which we want to compute B
        grid: [rho_1, rho_2, ..., rho_rho_vector_size] - points on segment
            [0, pi]
    rho_vector : np.array, optional
        vector of points at which we want to compute B
        default: regular from 0 to pi

    Returns
    -------
    B : np.array(float)
        values of cvf on the grid: [B(\rho_1), ..., B(\rho_{gridsize})]
    '''
    if rho_vector is None:
        rho_vector = np.linspace(0, np.pi, rho_vector_size)
    else:
        rho_vector_size = len(rho_vector)

    two_n_plus_1 = (np.arange(n_max + 1) * 2 + 1)[:,np.newaxis,np.newaxis]
    B = legval(np.cos(rho_vector),
                c=modal_spectrum * two_n_plus_1,
                tensor=True) / (4 * np.pi)
    return B

def Fourier_Legendre_transform_backward(
        # x_array: np.array(float),
        n_max: int,
        modal_spectrum: np.array(float),
        rho_vector_size: int=None,
        rho_vector: np.array=None
        ) -> np.array(float):
    '''
    Transforms modal_spectrum to covariance function.
    B(\rho) = \frac{1}{4 \pi} \sum_0^{n_max} (2 n + 1) b_n P_n(\cos \rho)

    Parameters
    ----------
    n_max : int
        size of modal_spectrum
    modal_spectrum : np.array(float)
        modal_spectrum  [b_n], n = 0, 1, ..., n_max
    rho_vector_size : int
        size of grid of points at which we want to compute B
        grid: [rho_1, rho_2, ..., rho_rho_vector_size] - points on segment
            [0, pi]
    rho_vector : np.array, optional
        vector of points at which we want to compute B
        default: regular from 0 to pi

    Returns
    -------
    B : np.array(float)
        values of cvf on the grid: [B(\rho_1), ..., B(\rho_{gridsize})]
    '''
    if rho_vector is None:
        rho_vector = np.linspace(0, np.pi, rho_vector_size)
    else:
        rho_vector_size = len(rho_vector)

    two_n_plus_1 = np.arange(n_max + 1) * 2 + 1
    B = legval(np.cos(rho_vector),
                c=modal_spectrum * two_n_plus_1) / (4 * np.pi)
    return B


def Fourier_Legendre_transform_forward(
        n_max: int, B: np.array(float),
        ) -> np.array(float):
    '''
    Transforms covariance function to modal_spectrum.
    b_n = 2 \pi \integral_0^{\pi} (B(\rho) P_n(\cos(\rho)) sin(\rho) d \rho)

    Parameters
    ----------
    n_max : int
        size of modal_spectrum
    B : np.array(float)
        covariance function on the regular grid from 0 to pi

    Returns
    -------
    modal_spectrum : np.array(float)
        array of modal_spectrum [b_n], n = 0, 1, ..., n_max
    '''
    rho_vector_size = len(B)
    rho_vector = np.linspace(0, np.pi, rho_vector_size)
    delta_rho = np.pi / (rho_vector_size - 1)
    Legendre = []
    for n in range(n_max + 1):
        Legendre.append(legendre(n))
    P_n_k = np.array(
        [[Legendre[n](np.cos(rho_vector[k])) * np.sin(rho_vector[k])\
          for k in range(rho_vector_size)]
         for n in range(n_max + 1)]
        )
    modal_spectrum = 2 * np.pi * np.matmul(P_n_k, B) * delta_rho
    return modal_spectrum
