# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 15:44:22 2020
Last modified: March 2020

@author: Arseniy Sotskiy
"""

import matplotlib.pyplot as plt
import numpy as np
import pyshtools
from scipy.special import legendre


pyshtools.utils.figstyle(rel_width=0.7)

from functions import Fourier_Legendre_transform_forward
# from functions import Fourier_Legendre_transform_backward_scipy
from functions import Fourier_Legendre_transform_backward
from RandomProcesses import get_spectrum_from_data
from functions import draw_1D


all_RMSEs = dict()
possible_n_max = [30, 60, 90, 120, 240, 360]

for n_max in possible_n_max:
    grid_size = n_max * 10 + 1
    x_array = np.linspace(-1, 1, grid_size)
    
    
    rho_array = np.linspace(0, np.pi, grid_size)
    n = n_max
    legendre_rho = np.array([legendre(n)(np.cos(rho)) for rho in rho_array])
    plt.plot(rho_array, legendre_rho)
    plt.grid(True)
    plt.title('legendre')
    plt.xlabel(r'$\rho$, rad')
    # plt.ylabel(ylabel)
    plt.figure()
    plt.show()
    
    gamma = 3
    
    n_lat = n_max + 2
    dx = 2 * np.pi / (2 * n_lat) # in radians
    lamb = dx * 3
    V = 1
    
    spectrum_old = get_spectrum_from_data(V, lamb, gamma, n_max) 
    # modal spectrum
    
    draw_1D(spectrum_old, title=f'variance spectrum_old, n_max={n_max}')
    
    
    
    B = Fourier_Legendre_transform_backward(
        n_max, rho_vector_size=grid_size, modal_spectrum=spectrum_old
        )
    
    plt.plot(x_array, B)
    plt.grid(True)
    plt.title(f'B, n_max={n_max}')
    # plt.xlabel(xlabel)
    # plt.ylabel(ylabel)
    plt.figure()
    plt.show()
    
    # B_sp = Fourier_Legendre_transform_backward_scipy(n_max, grid_size, spectrum_old)
    
    # plt.plot(x_array, B_sp, label = 'scipy')
    # plt.plot(x_array, B, label = 'my')
    # plt.grid(True)
    # plt.title('B')
    # plt.legend(loc='best')
    # # plt.xlabel(xlabel)
    # # plt.ylabel(ylabel)
    # plt.figure()
    # plt.show()
    
    spectrum_new = Fourier_Legendre_transform_forward(n_max, B)
    draw_1D(spectrum_new, title=f'spectrum_new, n_max={n_max}')
    
    
    
    plt.plot(np.arange(n_max + 1), spectrum_old, label = 'spectrum_old')
    plt.plot(np.arange(n_max + 1), spectrum_new, label = 'spectrum_new')
    plt.grid(True)
    plt.title(f'Both, n_max={n_max}')
    # plt.xlabel(xlabel)
    # plt.ylabel(ylabel)
    plt.legend(loc = 'best')
    plt.figure()
    plt.show()
    
    
    RMSE = np.linalg.norm(spectrum_new - spectrum_old) / np.linalg.norm(spectrum_old)
    print('RMSE = ', RMSE)
    all_RMSEs[n_max] = RMSE
    
for n_max, RMSE in all_RMSEs.items():    
    print('n_max = {}, RMSE = {}'.format(n_max, RMSE))






