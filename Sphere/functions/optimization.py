# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 2020
Last modified: Nov 2020

@author: Arseniy Sotskiy
"""

import numpy as np
import scipy
from scipy.optimize import minimize
from functions.tools import draw_1D

from time import process_time

def compute_band_vars_from_spectrum(bands, variance_spectrum):
    # band_variances = np.zeros((len(bands),
    #                            # variance_spectrum.shape[1],
    #                            # variance_spectrum.shape[2]
    #                            ))
    band_transfer_functions_sq = np.array(
        [band.transfer_function for band in bands]
        )**2
    band_cs = np.array([band.c for band in bands])

    band_variances = np.matmul(band_transfer_functions_sq,
                               variance_spectrum) / band_cs

    # for band_index, band in enumerate(bands):
    #     # variance_true = np.apply_along_axis(
    #     #     lambda x: np.matmul(x, (band.transfer_function)**2), 0,
    #     #     variance_spectrum
    #     #     )
    #     # variance_true = np.sum(variance_spectrum)
    #     variance_true =  np.matmul(variance_spectrum,
    #                                 (band.transfer_function)**2)
    #     # band_variances[band_index,:,:] = variance_true
    #     band_variances[band_index] = variance_true / band.c
    # assert (np.max(np.abs(band_variances_ - band_variances)))<1/1e15, \
    #     (band_variances_, band_variances)
    return band_variances




def get_modal_spectrum(c, lamb, gamma, n_max):
    # 1 / (1 + (lambx[j,i] * n) ** gammax[j,i])
    return np.array([c / (1 + (lamb * l)**gamma) for l in range(n_max + 1)])

def get_variance_spectrum(c, lamb, gamma, n_max):
    # return np.array([c+lamb+gamma for n in range(n_max + 1)])
    # modal_spectrum = np.array(
    #     [c / (1 + (lamb * l)**gamma) for l in range(n_max + 1)]
    #     )
    # variance_spectrum = 1 / (4 * np.pi) * np.array(
    #         [(2*n+1) * modal_spectrum[n] for n in range(n_max + 1)]
    #         )
    variance_spectrum = 1 / (4 * np.pi) * np.array(
            [c * (2*l+1) / (1 + (lamb * l)**gamma) for l in range(n_max + 1)]
            )
    # print (variance_spectrum_ - variance_spectrum)
    # assert \
    #     np.max(np.abs(variance_spectrum_ - variance_spectrum)) < 1/1e15,\
    #     (variance_spectrum_, variance_spectrum,
    #      np.max(np.abs(variance_spectrum_ - variance_spectrum)))

    return variance_spectrum


def compute_loss_function(x, arguments):
    # c, lamb, gamma = x[0], x[1], x[2]
    c, lamb = x[0], x[1]
    gamma = gamma_add
    # print(arguments)
    n_max, band_variances_true, bands = arguments[0], arguments[1], arguments[2]
    # return np.linalg.norm(x)
    # variance_spectrum = get_variance_spectrum(c, lamb, gamma, n_max)
    # variance_spectrum_ =\
    # 1 / (4 * np.pi) * np.array(
    #         [c * (2*l+1) / (1 + (lamb * l)**gamma) for l in range(n_max + 1)]
    #         )
    l_arr = np.arange(n_max + 1)
    variance_spectrum =\
    1 / (4 * np.pi) * \
            c * (2*l_arr+1) / (1 + (lamb * l_arr)**gamma)
    # assert (np.max(np.abs(variance_spectrum_ - variance_spectrum)))<1/1e15

    # return np.linalg.norm(x)
    band_variances_estim = compute_band_vars_from_spectrum(bands,
                                                            variance_spectrum)
    # band_variances_estim = c+lamb
    return np.linalg.norm(band_variances_estim - band_variances_true, ord=1)


def simple_loss_function(x, args):
    return np.linalg.norm(x)


def find_params_with_optimization_at_one_point(x0, band_variances, bands,
                                               n_max, tol_coeff=0.1):
    # x0 = [c_0, lamb_0, gamma_0]
    arguments = ([
                  n_max,
                  band_variances,
                  bands
                  ])
    fatol = compute_loss_function(x0, arguments)*tol_coeff
    res = minimize(compute_loss_function,
        # simple_loss_function,
                  x0=x0,
                  args = arguments,
                  method='Nelder-Mead',
                  options={'maxiter': 150, 'disp': False, 'fatol': fatol})
    # c, lamb, gamma = res.x[0], res.x[1], res.x[2]
    c, lamb = res.x[0], res.x[1]

    return res.x

# find_params_with_optimization_vect = np.vectorize(
#     find_params_with_optimization_at_one_point,
#     signature='(m,n,k)->(m,n,k)'
#     )


def find_params_with_optimization(band_variances, bands,
                                  n_max, c_0, lamb_0, gamma_0):
    shape = c_0.shape
    print(shape)
    c_opt = np.zeros(shape)
    lamb_opt = np.zeros(shape)
    gamma_opt = np.zeros(shape)
    for i in range(shape[0]):
        print(i)
        for j in range(shape[1]):
            # if j%10 == 0:
            #     print(j)
            # x0 = np.array([c_0[i,j], lamb_0[i,j], gamma_0[i,j]])
            x0 = np.array([c_0[i,j], lamb_0[i,j]])
            x = find_params_with_optimization_at_one_point(
                x0, band_variances[:,i,j], bands, n_max
                )
            # x=[0,0]
            # c_opt[i,j], lamb_opt[i,j], gamma_opt[i,j] =\
            #     x[0], x[1], x[2]
            c_opt[i,j], lamb_opt[i,j] =\
                x[0], x[1]
    return c_opt, lamb_opt #, gamma_opt


    # x = np.apply_over_axes(find_params_with_optimization_at_one_point, x0,
    #                        [1,2],
    #                        band_variances_true,
    #                        bands, n_max)





#%%
if __name__ == '__main__':

    c_true = cx.copy()
    lamb_true = lambx.copy()
    gamma_true = gammax.copy()

    # x0 = np.array([c_true, lamb_true, gamma_true])*2
    x0 = np.array([c_true, lamb_true])*2

    variance_spectrum_true = variance_spectrum.copy()
    modal_spectrum_true = modal_spectrum.copy()
    start = process_time()
    # find_params_with_optimization_vect(x0, band_variances_true, bands,
    #                               n_max)
    c_opt, lamb_opt, gamma_opt = find_params_with_optimization(
        band_variances_true, bands, n_max, cx*2, lambx*2, gammax*2
        )
    end = process_time()
    print(end-start, 'seconds')

#%%

if __name__ == '__main__':

    c_true = cx[1,1]
    lamb_true = lambx[1,1]
    gamma_true = gammax[1,1]

    x0 = np.array([c_true, lamb_true, gamma_true])*2

    variance_spectrum_true = variance_spectrum[:,1,1]
    variance_spectrum_truee = get_variance_spectrum(c_true, lamb_true,
                                                    gamma_true, n_max)
    modal_spectrum_true = modal_spectrum[:,1,1]
    modal_spectrum_truee = get_modal_spectrum(c_true, lamb_true,
                                                    gamma_true, n_max)
    plt.figure()
    plt.plot(np.arange(len(variance_spectrum_true)),
             variance_spectrum_true,
             label='true')
    plt.plot(np.arange(len(variance_spectrum_true)), variance_spectrum_truee,
             label='now')

    plt.legend(loc='best')
    plt.title(
        f'variance spectrum'  # ', c: {c:.5}, $\lambda$: {lamb:.5}, $\gamma$: {gamma:.5}'
        )
    plt.grid()
    plt.show()
    band_variances_true = compute_band_vars_from_spectrum(bands,
                                                         variance_spectrum_true)


    x = find_params_with_optimization_at_one_point(x0, band_mean_estim_variances, bands,
                                               n_max, tol_coeff=0.1)
    # print(res.message)


draw_1D(np.array([band_mean_estim_variances,
                  band_mean_true_variances, band_variances_true]).T)

#%%
if __name__ == '__main__':
    # print(res.x)
    c, lamb, gamma = x[0], x[1], x[2]
    print(f'c: {c}, lambda: {lamb}, gamma: {gamma}')
    variance_spectrum_opt = get_variance_spectrum(c, lamb, gamma, n_max)

    plt.figure()
    plt.plot(np.arange(len(variance_spectrum[:,0,0])),
             variance_spectrum[:,0,0],
             label='true')
    plt.plot(np.arange(len(variance_spectrum_opt)), variance_spectrum_opt,
             label='optim')
    plt.legend(loc='best')
    plt.title(
        f'variance spectrum, c: {c:.5}, $\lambda$: {lamb:.5}, $\gamma$: {gamma:.5}'
        )
    plt.grid()
    plt.show()

    band_variances_opt = compute_band_vars_from_spectrum(bands,
                                                         variance_spectrum_opt)

    band_variances_true = compute_band_vars_from_spectrum(bands,
                                                         variance_spectrum_true)

    plt.figure()
    # plt.plot(np.arange(modal_spectrum.shape[0]),
    #           modal_spectrum[:, 0, 0], label='true')
    plt.scatter([band.center for band in bands],
                band_mean_true_variances,
                label='true')
    plt.scatter([band.center for band in bands],
                band_variances_opt,
                label='optim')
    plt.legend(loc='best')
    plt.title(
        f'band variances, c: {c:.5}, $\lambda$: {lamb:.5}, $\gamma$: {gamma:.5}'
        )
    plt.xlabel('l')
    plt.ylabel('b_mean')
    plt.grid()
    plt.show()










