# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 15:30:20 2019

@author: Arseniy Sotskiy
"""

from configs import *

import numpy as np
import matplotlib.pyplot as plt

from RandomProcesses import RandStatioProcOnS1


def make_sigma_corr_from_sigma(sigma):
#    sigma is a np.array[n,x]. we norm it:
#    sigma_corr_n(x) = sigma_n(x) / \sqrt{sum_{k=0}^{n_x-1}sigma_k(x)^2}
#    test: b = np.array([[1. , 0. , 0. ],
#                        [2. , 3, 6. ],
#                        [2. , 4. , 8. ]])
    factor = np.sqrt((sigma*sigma).sum(axis=0))
    return sigma/factor

#================================================
#================================================
#================================================

def get_spectrum_from_ESM_data(NS_width, kappa_V, kappa_L, kappa_gamma,
                         V=std_fcst**2, L=L_mean, gamma_mean=gamma_mean,
                         is_cvm_to_be_converted_to_crm=False, draw=False):

    '''
    External parameters: gamma, L, V.
    gamma: defines shape of the cvf (and the spectrum)
        and the smoothness of the process
        varies from 1 to 4
    L: macro-scale (L*n_x/(2*pi*R_km) = 5-10)
    V: variance of the process

    Internal parameters: gamma, lambda, c
    gamma: the same as above
    lambda: scale parameter such that
    $$ L(\lambda) =
        = \frac{\pi R}{1 + 2 \sum_{n>0} \frac{1}{1+(\lambda n)^{\gamma}}} $$
    c: normalizing coefficient such that the variance of the process is V

    Returns  sigma[n, x] - 2d np.array
    '''

    def frac(lamb, n, gamma_mean):
        return 1/(1+(lamb*n)**gamma_mean)

    kappa_V = kappa_V
    kappa_L = kappa_L
    kappa_gamma = kappa_gamma
    V=V
    L=L
    gamma_mean=gamma_mean
    # Band limited Gaussian white noise:
    spectrum_ = np.array([1/(2*NS_width+1) for _ in range(NS_width+1)] + [0 for _ in range(NS_width+1, int(n_x/2)+1)])
    White_Noise = RandStatioProcOnS1(spectrum_)
    WN_realiz_V = White_Noise.generate_one_realization()
    WN_realiz_L = White_Noise.generate_one_realization()
    WN_realiz_gamma = White_Noise.generate_one_realization()
#        if draw:
#            plt.title("BL WN spectrum")
#            plt.plot(np.arange(len(spectrum_)), spectrum_, label = 'spectrum_')
#            plt.xlabel("x")
#            plt.legend(loc='best')
#            plt.grid(True)
#            plt.figure(figsize=(12, 5))
#            plt.show()
#
#            plt.title("White_Noise")
#            plt.plot(np.arange(len(WN_realiz_V)), WN_realiz_V, label = 'WN_realiz_V')
#            plt.xlabel("x")
#            plt.legend(loc='best')
#            plt.grid(True)
#            plt.figure(figsize=(12, 5))
#            plt.show()

#        print(np.min(WN_realiz_V)*kappa_V)
#        print(sigmoid(np.min(WN_realiz_V)*kappa_V))
#        print(V*sigmoid(np.min(WN_realiz_V)*kappa_V))

    Vx = V * sigmoid(np.log(kappa_V) * WN_realiz_V)
    Lx = L_min + L * sigmoid(np.log(kappa_L) * WN_realiz_L)
    gammax = gamma_min + gamma_mean * sigmoid(np.log(kappa_gamma) * WN_realiz_gamma)

    if draw:
        plt.title("$ L(x)$")
        plt.plot(np.arange(len(Lx)), Lx, label = 'L(x)')
        plt.xlabel("x")
        plt.legend(loc='best')
        plt.grid(True)
        plt.figure(figsize=(12, 5))
        plt.show()

        plt.title("$V(x)$")
        plt.plot(np.arange(len(Vx)), Vx, label = 'V(x)')
        plt.xlabel("x")
        plt.legend(loc='best')
        plt.grid(True)
        plt.figure(figsize=(12, 5))
        plt.show()

        plt.title("$\gamma(x)$")
        plt.plot(np.arange(len(gammax)), gammax, label = '$\gamma_mean(x)$')
        plt.xlabel("x")
        plt.legend(loc='best')
        plt.grid(True)
        plt.figure(figsize=(12, 5))
        plt.show()

    lambx = lamb_from_L(gammax, Lx)
    cx = np.array(
            [Vx[x]/(1 + 2*np.sum(
                    np.array(
                            [frac(lambx[x], n, gammax[x]) for n in range(1, int(n_x/2))]
#                           [1/(1+(lambx[x]*n)**gammax[x]) for n in range(1, int(n_x/2))]
                            )
                    ) + frac(lambx[x], int(n_x/2), gammax[x])) for x in range(n_x)]
#                   ) + 1/(1+(lambx[x]*int(n_x/2))**gammax[x])) for x in range(n_x)]
                    )
#        cx = [Vx[x]*L/np.pi/R_km for x in range(n_x)]
    spectrum = np.array([[cx[x]/(1+(n*lambx[x])**gammax[x]) for n in range(int(n_x/2)+1)]for x in range(n_x)]).T
    ###    !! spectrum is spectrum[n, x] !!
    spectrum_fft = convert_spectrum2fft(spectrum)

    sigma = np.sqrt(spectrum_fft)
    if is_cvm_to_be_converted_to_crm:
        draw_2D(sigma, title = 'sigma')
        sigma_corr = make_sigma_corr_from_sigma(sigma)
        draw_2D(sigma_corr, title = 'sigma_corr')
        draw_2D(sigma_corr - sigma, title = 'difference')
        draw_2D(sigma_corr / sigma, title = 'factor')
        sigma = sigma_corr
    # modifying sigma so that B would be corr matrix
    sigma_norm = sigma/np.tile(sigma[0,:], (n_x, 1))
#
#        B_true = np.sum(np.array(
#                [[[sigma[n,x]*sigma[n,y]*np.exp(1j*n*(x-y)*2*np.pi/n_x) for x in range(n_x)
#                ] for y in range(n_x)] for n in range(n_x)]), axis = 0)
    exp = np.array([[np.exp(1j*n*x*2*np.pi/n_x) for n in range(n_x)] for x in range(n_x)])
    sigma_exp = exp * sigma
    B_true_from_matrix = np.dot(sigma_exp.T, np.conj(sigma_exp))
    B_true = B_true_from_matrix
#        draw_2D(B_true_from_matrix, title = 'B_true_from_matrix')
#        draw_2D(B_true, title = 'B_true')
#        draw_2D(B_true - B_true_from_matrix, title = 'B_true - B_poss')



#        ---Possible only if the process is stationary:---
#        B_true_fft = np.zeros((n_x, n_x))
#        for i in range(n_x):
#            B_true_fft[:,i] = np.roll(iFFT(spectrum_fft[:,i]), i)
#        B_true = B_true_fft
#        -------------------------------------------------

    c = np.mean(cx)
    lamb = np.mean(lambx)
    gamma = np.mean(gammax)

    if draw:

        plt.imshow(np.abs(B_true));
        plt.colorbar()
        plt.title("cvm")
#            plt.xlabel("x")
#            plt.ylabel("y")
        plt.figure()
        plt.show()

        plt.title("$B(x)$: min and max $\gamma$")
        min_, max_ = np.argmin(gammax), np.argmax(gammax)
        plt.plot(np.arange(len(B_true[min_,:])), B_true[min_,:], color = 'red', label = 'min_gamma = ' + str(gammax[min_]))
        plt.plot(np.arange(len(B_true[max_,:])), B_true[max_,:], color = 'red', label = 'max_gamma = ' + str(gammax[max_]))
        min_, max_ = np.argmin(Lx), np.argmax(Lx)
        plt.plot(np.arange(len(B_true[min_,:])), B_true[min_,:], color = 'green', label = 'min_L = ' + str(Lx[min_]))
        plt.plot(np.arange(len(B_true[max_,:])), B_true[max_,:], color = 'green', label = 'max_L = ' + str(Lx[max_]))

        plt.xlabel("x")
        plt.legend(loc='best')
        plt.grid(True)
        plt.figure(figsize=(12, 5))
        plt.show()


        plt.title("$c(x)$")
        plt.plot(np.arange(len(cx)), cx, label = 'c(x)')
        plt.xlabel("x")
        plt.legend(loc='best')
        plt.grid(True)
        plt.figure(figsize=(12, 5))
        plt.show()


        plt.title("$ lambda(x)$")
        plt.plot(np.arange(len(lambx)), lambx, label = 'lambda(x)')
        plt.xlabel("x")
        plt.legend(loc='best')
        plt.grid(True)
        plt.figure(figsize=(12, 5))
        plt.show()



#        draw_2D(sigma_norm)
#        draw_2D(spectrum[0:20,:])

#        plt.title('std')
#        plt.plot(np.arange(len(sigma[:,0])), sigma[:,0], label = 'spectrum')
#        plt.xlabel("x")
#        plt.legend(loc='best')
#        plt.grid(True)
#        plt.show()
#        descr = 'RP_S1\nA random processes on a circle $S^1$ \
#\nLength:\n' + str(n_x)
    return sigma #    !! sigma is sigma[n, x] !!


#================================================
#================================================
#================================================

def make_w_from_sigma(sigma):
    w = iFFT(sigma.T)/(2*np.pi*R_km)
    return w



##########################################
###############       ####################
############### CLASS ####################
###############       ####################
##########################################



class proc_ESM: # RandLSMProcOnS1
    r'''
    Class of the random process ESM on a circle $S^1$.


    Can make multiple realisations.
    '''

    def __init__(self, sigma, draw=False):
        '''
        sigma is sigma[n, x] - 2d np.ndarray
        shape = (n_x, n_x)
        '''

        self.sigma = sigma
        w = iFFT(sigma.T) / np.sqrt(2 * np.pi * R_km)
        self.w = np.real(w)
        self.W = convert_aligned_matrix_to_diag(self.w) * np.sqrt(dx_km) #(dx_km) * np.sqrt(n_x)
        exp = np.array([[np.exp(1j*n*x*2*np.pi/n_x) for n in range(n_x)] for x in range(n_x)])
        sigma_exp = exp * sigma
        B_true_from_matrix = np.dot(sigma_exp.T, np.conj(sigma_exp))
        B_true_from_w = np.matmul(self.W, self.W.T)
        self.B_true = B_true_from_matrix
        self.spectrum = self.sigma**2
        if draw:
            draw_2D(sigma, title = 'sigma')
            draw_2D(self.spectrum, title = 'spectrum')
            draw_2D(self.W, title = 'W')
            draw_2D(B_true_from_matrix, title = 'B_true_from_matrix')
            draw_2D(B_true_from_w, title = 'B_true_from_w')
            draw_2D(B_true_from_matrix - B_true_from_w, title = 'difference')




    def generate_one_realization(self, draw=False):
        '''
        Makes one real random process with random spectral coefficients.
        Array xi_spec consists of the spectral coefficients.
        If draw == True, then the process will be plotted.
        '''

        nu_real = np.random.normal(0, 1/np.sqrt(2), n_x)
        nu_imag = np.random.normal(0, 1/np.sqrt(2), n_x)
        nu = nu_real + 1j*nu_imag

#        xi_spec = np.array([np.random.normal(0, np.sqrt(b/2)) +
#                            np.random.normal(0, np.sqrt(b/2)) * 1j for b in self.spectrum_fft])
        nu[0] = np.random.normal(0, 1)
        nu[int(n_x/2)] = np.random.normal(0, 1)
        for i in range(0, int(n_x/2)):
            nu[-i] = np.conj(nu[i])

        RP = np.array([
                [self.sigma[n, int(x/2/np.pi*n_x)]*nu[n]*np.exp(1j*n*x) for n in range(-int(n_x/2-1), int(n_x/2+1))]
                for x in np.linspace(0, 2*np.pi, n_x, endpoint=False)]).sum(axis = 1)
#        RP = np.matmul(RP, np.array([[np.exp(1j*m*x) for m in (list(range(0, int(n_x/2)))+list(range(-int(n_x/2), 0)))] for x in range(n_x)]))
#        RP = np.matmul(RP, np.array([[np.exp(1j*m*x) for m in range(-int(n_x/2), int(n_x/2))] for x in range(n_x)]))
        if draw:
            fig, ax = plt.subplots(figsize=(12, 5))
            X = np.arange(0, n_x, 1)
            ax.set_title('Simulation of  RP on $S^1$')
            ax.set_ylabel(r'$RP(x)$')
            ax.set_xlabel('x')
            ax.plot(X, np.real(RP))
            ax.plot(X, np.imag(RP))
            plt.grid(True)
            plt.show()
        return np.real(RP)

    def generate_multiple_realizations_from_sigma(
            self, n_realiz, draw=False):
        '''
        Makes n_realiz process realisations with random spectral coefficients.
        If draw == True, then the processes will be plotted.

        shape: (n_realiz, n_x)
        '''
        realizations = np.zeros((n_realiz, n_x))
        nu_real = np.random.normal(0, 1/np.sqrt(2), (n_realiz, n_x))
        nu_imag = np.random.normal(0, 1/np.sqrt(2), (n_realiz, n_x))
        nu = nu_real + 1j*nu_imag
        nu[:,0] = np.random.normal(0, 1, n_realiz)
        nu[:,int(n_x/2)] = np.random.normal(0, 1, n_realiz)
        for i in range(0, int(n_x/2)):
            nu[:,-i] = np.conj(nu[:,i])

        realizations = np.array([
                [
                [self.sigma[n, int(x/2/np.pi*n_x)]*nu[i,n]*np.exp(1j*n*x) for n in range(-int(n_x/2-1), int(n_x/2+1))]
                for x in np.linspace(0, 2*np.pi, n_x, endpoint=False)]
                for i in range(n_realiz)]).sum(axis = 2)

#        for i in range(n_realiz):
#            realizations[i,:] = iFFT(xi_spec[i,:])
        if draw:
            plt.imshow(np.real(np.array(realizations)));
            plt.colorbar()
            plt.figure(figsize=(6, 6))
            plt.show()
        return np.array(realizations)

    def generate_multiple_realizations(self, n_realiz, draw=False):
        '''
        Makes n_realiz process realisations with random spectral coefficients.
        If draw == True, then the processes will be plotted.

        shape: (n_realiz, n_x)
        '''
        realizations = np.random.multivariate_normal(np.zeros((n_x)), self.B_true, n_realiz)
        if draw:
            plt.imshow(np.real(np.array(realizations)));
            plt.colorbar()
            plt.figure(figsize=(6, 6))
            plt.show()
        return realizations

#    def generate_multiple_realizations_from_w(self, n_realiz, draw=False):
#        '''
#        Makes n_realiz process realisations with random spectral coefficients.
#        If draw == True, then the processes will be plotted.
#
#        shape: (n_realiz, n_x)
#        '''
#        pass

    def test(self, n_realiz=1000, draw=True): # same, to be deleted
        '''
        Estimates the covariance matrix
        using n_realiz realisations of the process
        and compares with the truth: self.B_true
        '''
        realizations = self.generate_multiple_realizations(n_realiz)
        B_estim = np.cov(realizations, rowvar=False)

        if draw:
            draw_2D(B_estim, title = 'B from realizations')
            draw_2D(self.B_true, title = 'B_true')
            draw_2D(B_estim - self.B_true, title = 'difference')
            draw_2D(B_estim / self.B_true, title = 'factor')


if __name__ == '__main__':
    print ('Testing...')
    sigma = get_spectrum_from_ESM_data(NS_width, kappa_V, kappa_L, kappa_gamma,
                         V=std_fcst**2, L=L_mean, gamma_mean=gamma_mean,
                         is_cvm_to_be_converted_to_crm=False, draw=True)
    process = proc_ESM(sigma)
    process.test(100000)
