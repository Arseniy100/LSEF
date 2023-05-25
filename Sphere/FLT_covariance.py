# -*- coding: utf-8 -*-
"""
Created on Mon Mar  16 15:44:22 2020
Last modified: March 2020

@author: Arseniy Sotskiy
"""

import matplotlib.pyplot as plt
import numpy as np
import pyshtools
from scipy.special import legendre


pyshtools.utils.figstyle(rel_width=0.7)

from functions import Fourier_Legendre_transform_forward
from functions import Fourier_Legendre_transform_backward
from RandomProcesses import get_spectrum_from_data
from functions import draw_1D


all_RMSEs = dict()
n_max = 60
grid_size = n_max * 10 + 1
x_array = np.linspace(-1, 1, grid_size)
rho_array = np.linspace(0, np.pi, grid_size)



lamb = 0.04
V = 1
possible_gamma = np.linspace(2, 8, 13)
possible_lamb = np.linspace(0.1, 1, 10)
gamma = 4
# print(possible_gamma)

for gamma in possible_gamma:
# for lamb in possible_lamb:
    spectrum = get_spectrum_from_data(V, lamb, gamma, n_max)
    spectrum = spectrum / np.array([2*n + 1 for n in range(n_max + 1)])
    # draw_1D(spectrum, title='spectrum')    
    
    B = Fourier_Legendre_transform_backward(n_max, grid_size, spectrum)
    
    plt.plot(rho_array, B)
    plt.plot([0], [0])
    plt.grid(True)
    plt.title('B, gamma = {}, lamb = {}'.format(gamma, lamb))
    plt.xlabel(r'$\rho$')
    plt.ylabel('covariance')
    plt.figure()
    # plt.savefig('images/B, gamma = {}, lamb = {}.png'.format(gamma, lamb))
    plt.show()
    
    
    
    






