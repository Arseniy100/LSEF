# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 13:51:17 2019
Last modified on Apr 2022
@author: Arseniy Sotskiy
"""

# from configs import n_max
import numpy as np
import matplotlib.pyplot as plt

from functions.FLT import Fourier_Legendre_transform_forward, \
    Fourier_Legendre_transform_backward

# n_x = n_max + 1

class Band_for_circle:
    def __init__(self, left, right, n_x):
        self.left = left
        self.right = right
        self.middle = (left + right)/2
        self.borders = (left, right)
        self.size = right - left + 1

        self.full_size = 2 * self.size
        indicator = np.zeros((n_x, n_x))
        if self.left == 0:
            self.full_size -= 1
            indicator[:,self.left+1:self.right+1] = 1
            indicator[:, n_x-self.right:n_x-self.left] = 1
            indicator[:,self.left] = 1
        elif self.right == n_x/2:
            self.full_size -= 1
            indicator[:,self.left:self.right] = 1
            indicator[:, n_x-self.right+1:n_x-self.left+1] = 1
            indicator[:,self.right] = 1
        else:
            indicator[:,self.left:self.right+1] = 1
            indicator[:, n_x-self.right:n_x-self.left+1] = 1
        self.indicator = indicator.T
#        indicator is indicator[n, x]!

    def __repr__(self):
        return "(" + str(self.left) + ", " + str(self.right) + ")"

class Band:
    # TODO: transfer-function and square instead of indicator
    def __init__(self, left, right, n_x):
        self.left = left
        self.right = right
        self.middle = (left + right)/2
        self.borders = (left, right)
        self.size = right - left + 1

        # self.full_size = 2 * self.size
        indicator = np.zeros((n_x, n_x))
        if self.left == 0:
            # self.full_size -= 1
            indicator[:,self.left+1:self.right+1] = 1
            # indicator[:, n_x-self.right:n_x-self.left] = 1
            indicator[:,self.left] = 1
        elif self.right == n_x/2:
            # self.full_size -= 1
            indicator[:,self.left:self.right] = 1
            # indicator[:, n_x-self.right+1:n_x-self.left+1] = 1
            indicator[:,self.right] = 1
        else:
            indicator[:,self.left:self.right+1] = 1
            # indicator[:, n_x-self.right:n_x-self.left+1] = 1
        self.indicator = indicator.T
#        indicator is indicator[n, x]!

    def __repr__(self):
        return "(" + str(self.left) + ", " + str(self.right) + ")"


class Band_for_sphere:
    def __init__(self, transfer_function: np.array):
        self.transfer_function = transfer_function
        self.shape = len(transfer_function)
        # self.c_old = np.sum(self.transfer_function**2 / 4 / np.pi * \
        #     np.array([2 * l + 1 for l in range(self.shape)]))
        self.c = np.sum(self.transfer_function**2)
        self.weight_function = self.transfer_function**2 / self.c
            # p_(j) - probality measure (l)
        self.center = np.sum(self.weight_function * np.arange(self.shape))
        self.response_function = Fourier_Legendre_transform_backward(
            self.shape-1, self.transfer_function,
            rho_vector_size=self.shape
            )


    # def __repr__(self):
    #     return "(" + str(self.left) + ", " + str(self.right) + ")"


def make_exp_band(n_max, l_0, d):
    trans_f = np.exp(np.array(
        [-(l - l_0)**2 / (2 * d**2) for l in range(0, n_max + 2)]
        ))
    return Band_for_sphere(trans_f)

#%%

def make_rectangular_spherical_bands(n_max, n_bands, draw=False):
    
    band_len = int((n_max+1) / (n_bands))
    bands = []
    for i in range(n_bands):
        transfer_function = np.zeros((n_max+1))
        transfer_function[i*band_len:(i+1)*band_len] = 1
        if i == n_bands - 1:
            transfer_function[i*band_len:-1] = 1
        band = Band_for_sphere(transfer_function)
        
        bands.append(band)
    
    if draw:
        plt.figure()
        for i, band in enumerate(bands):
            plt.plot(np.arange(band.shape),
                      band.transfer_function,
                      label=f'{i}')
        plt.scatter([band.center], [0], label='center')
        plt.grid()
        plt.legend(loc='best')
        plt.title(f'transfer functions')
        plt.show()
    return bands


if __name__ == '__main__':
    make_rectangular_spherical_bands(47, 10, draw=True)
    make_rectangular_spherical_bands(47, 8, draw=True)
    
#%%


# make rectang from exp such that integral is equal
# (rectang: l_0 +- d*sqrt(pi/2))


if __name__ == '__main__':
    from configs import n_max
    d = 10
    d_list = [2, 5, 10]
    l_0_list_full = [0,2,4,9,15]
    l_0 = 5

    for l_0 in l_0_list_full:
    # for d in d_list:
        band = make_exp_band(n_max, l_0, d)
        print('band.weight_function.sum()', band.weight_function.sum())

        new_transf_func = Fourier_Legendre_transform_forward(
            band.shape-1, band.transfer_function
            )



        plt.figure()
        plt.plot(np.arange(band.shape),
                  band.transfer_function,
                  label=f'l_0={l_0}')
        plt.scatter([band.center], [0], label='center')
        plt.grid()
        plt.legend(loc='best')
        plt.title(f'transfer functions, d={d}')
        plt.show()

        plt.figure()
        plt.plot(np.arange(band.shape),
                  band.response_function,
                  label=f'l_0={l_0}')
        plt.grid()
        plt.legend(loc='best')
        plt.title(f'response functions, d={d}')
        plt.show()





    # n_x = 36
    # transfer_function = np.zeros((n_x))
    # transfer_function[10:25] = 1
    # # print(transfer_function)
    # band = Band_for_sphere(transfer_function)
    # draw_1D(band.transfer_function, title='transfer_function')
    # draw_1D(band.response_function, title='response_function')



    # print('c:', band.c)
    # draw_1D(band.weight_function, title='weight_function')
    # print('shape', band.shape)
    # print('center', band.center)
    # b = Band(10, 18)
    # print(b.indicator)
    # # draw_2D(b.indicator)
    # # plt.imshow(b.indicator);
    # # plt.colorbar()
    # # plt.figure()
    # # plt.show()

    # b = Band(0, 1)
    # print(b.indicator)
    # # draw_2D(b.indicator)
    # # plt.imshow(b.indicator);
    # # plt.colorbar()
    # # plt.figure()
    # # plt.show()
    # print(b.size)
