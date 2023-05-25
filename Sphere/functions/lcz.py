# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 19:22:18 2019
Last modified: Dec 2021

@author: Arseniy Sotskiy
"""

import numpy as np
import matplotlib.pyplot as plt

from configs import R_km
from Grids import LatLonSphericalGrid


from functions.tools import find_sample_covariance_matrix
from functions.DLSM_functions import make_analysis





def gasp_cohn_loc_func(zdc):
    # return 1
    if zdc>=0 and zdc<=1:
        return -1/4*zdc**5 + 1/2*zdc**4 + 5/8*zdc**3 - 5/3*zdc**2 + 1
    if zdc>=1 and zdc<=2:
        return 1/12*zdc**5 - 1/2*zdc**4 + 5/8*zdc**3 + 5/3*zdc**2 - 5*zdc + 4 -2/3/zdc
    return 0


vect_gasp_cohn_loc_func = np.vectorize(gasp_cohn_loc_func)


def construct_lcz_matrix(grid: LatLonSphericalGrid,
                         c, R=R_km):

    #---------------------------------------
    # Create a lcz mx using Gaspari-Cohn lcz function, see
    # Gaspari and Cohn (1999, Eq.(4.10))
    #
    # Args
    #
    # n - number of grid points on the circular domain
    # c - lcz length scale (NB: the lcz functions vanishes at distances > 2*c), m
    # R - Earth radius, m
    #
    # Return: C (the lcz mx)
    #---------------------------------------
    # todo: optimization (every distance - once)


    chordal_distance_matrix = 2*np.sin(grid.rho_matrix/2) * R


    C = vect_gasp_cohn_loc_func(chordal_distance_matrix/c)
    return C


def construct_lcz_matrix_fast(grid: LatLonSphericalGrid,
                              c, R=R_km):

    #---------------------------------------
    # Create a lcz mx using Gaspari-Cohn lcz function, see
    # Gaspari and Cohn (1999, Eq.(4.10))
    #
    # Args
    #
    # c - lcz length scale (NB: the lcz functions vanishes 
    #                       at distances > 2*c), m
    # R - Earth radius, m
    #
    # Return: C (the lcz mx)
    #---------------------------------------
    

    chordal_distance_matrix = 2*np.sin(grid.rho_matrix/2) * R
    zdc = chordal_distance_matrix / c
    lcz = np.where(np.logical_or(zdc < 0, zdc > 2), 0, zdc)
    lcz = np.where(np.logical_and(zdc>=0, zdc<=1), 
                 -1/4*zdc**5 + 1/2*zdc**4 + 5/8*zdc**3 - 5/3*zdc**2 +1,
                 lcz)
    lcz = np.where(np.logical_and(zdc>=1, zdc<=2), 
                 1/12*zdc**5 - 1/2*zdc**4 + 5/8*zdc**3 + 5/3*zdc**2 - 5*zdc + 4 -2/3/zdc,
                 lcz)

    return lcz



c_loc_array = list(map(int, np.linspace(1000, 3000, 9)))

def find_best_c_loc(ensemble, grid,
                    c_loc_array=c_loc_array,
                    draw: bool=False,
                    *analysis_args, **analysis_kwargs):

    s_array = []

    for c_loc in c_loc_array:
        print(f'c_loc: {c_loc}')
        B_sample = find_sample_covariance_matrix(ensemble.T)
        lcz_mx = construct_lcz_matrix(grid, c_loc)

        B_sample_loc = np.multiply(B_sample, lcz_mx)

        s = make_analysis(B_sample_loc, *analysis_args, **analysis_kwargs)
        print(f's: {s}')
        s_array.append(s)

    
    best_c = c_loc_array[np.argmin(s_array)]
    # best_s = s_array[np.argmin(s_array)]
    if draw:
        plt.figure()
        plt.plot(c_loc_array, s_array)
        plt.grid()
        plt.title(f'best c_loc={best_c}')
        plt.show()
    return best_c


# lcz_matrix = construct_lcz_matrix(n_x, L_0*scale_coeff, R_km)

# C_for_fun = construct_lcz_matrix(60 ,7000, 6400)


#%%
# if __name__ == '__main__':
#     best_c = find_best_c_loc(ensemble, grid,
#                             c_loc_array,
#                             true_field, grid, n_obs=n_obs,
#                             obs_std_err=obs_std_err, n_seeds=n_seeds, seed=0,
#                             draw=False)


# #%%
# if __name__ == '__main__':
#     start = process_time()
#     C = construct_lcz_matrix(grid, 1400, R_km)
#     end = process_time()
#     print(f'constructing {end-start} sec')
#     from functions import draw_2D
#     draw_2D(C)
    
#     start = process_time()
#     C_ = construct_lcz_matrix_fast(grid, 1400, R_km)
#     end = process_time()
#     print(f'constructing {end-start} sec')
#     from functions import draw_2D
#     draw_2D(C_)
    
#     draw_2D(C_ - C)