# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 19:22:18 2019
Last modified: June 2020

@author: Arseniy Sotskiy
"""

import numpy as np

from configs import n_x, L_0, scale_coeff, R_km

def construct_lcz_matrix(n, c, R=6400):

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

    C = np.zeros((n,n))

    hrad = 2*np.pi/n  # mesh size, rad.

    for i in range(n):
        for j in range(n):

            if np.abs(i-j)<n/2:
                z = np.abs(i-j)             # greate-circle distance, mesh sizes
            else:
                z = n - np.abs(i-j)         # greate-circle distance, mesh sizes


            zrad = z * hrad            # greate-circle distance, rad

            z = 2*np.sin(zrad/2)*R # chordal distance, m

            zdc=z/c

            if z>=0 and z<=c:
                C[i,j] = -1/4*zdc**5 + 1/2*zdc**4 + 5/8*zdc**3 - 5/3*zdc**2 +1
            if z>=c and z<=2*c:
                C[i,j] = 1/12*zdc**5 - 1/2*zdc**4 + 5/8*zdc**3 + 5/3*zdc**2 - 5*zdc + 4 -2/3*c/z

      # C[i,j] = exp(-0.5*zdc**2)

    return C

lcz_matrix = construct_lcz_matrix(n_x, L_0*scale_coeff, R_km)

C_for_fun = construct_lcz_matrix(60 ,7000, 6400)

if __name__ == '__main__':
    C = construct_lcz_matrix(600 ,7000, 6400)
    from functions import draw_2D
    draw_2D(C)