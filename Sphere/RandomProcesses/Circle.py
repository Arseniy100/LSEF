# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 12:09:38 2019

@author: Арсений HP-15
"""

from configs import *

import numpy as np
import matplotlib.pyplot as plt






class RandStatioProcOnS1:
    '''
    Class of real-valued random processes on the circle $S^1$.

    Can make K realisations and find their covariance function
    (comparing with the ifft(spectrum_fft)).
    '''

#    def __init__(self, spectrum = spectrum):
    def __init__(self, spectrum, n_x=None):
        '''
        The function which creates an object.

        spectrum must be a real-valued non-negative
        one-dimensional numpy.ndarray.
        (spectrum is the array of variances of spectral coefficients.)
        spectrum is only from 0 to n_x/2;
        self.spectrum_fft will be full
        (from 0 to n_x/2 and then from -n_x/2+1 to -1).
        self.spectrum_fft[-i] = self.spectrum_fft[i]
        (The spectrum is an even function of the wavenumber)
        '''
        self.spectrum = spectrum
        if n_x is None:
            self.n_x = 2*len(self.spectrum)-2
        else:
            self.n_x = n_x
        self.spectrum_fft = convert_spectrum2fft(self.spectrum)
        self.std = [np.sqrt(b_n) for b_n in self.spectrum_fft]
        self.cvf = iFFT(self.spectrum_fft)
        B_true = np.zeros((self.n_x, self.n_x))
        for i in range(self.n_x):
            B_true[:,i] = np.roll(self.cvf, i)
#        self.B_true = np.matrix(B_true)
        self.B_true = B_true
        self.A_true = estimate_A(self.B_true)
        self.K_true = estimate_K(self.A_true)


        self.descr = 'RP_S1\nA random processes on a circle $S^1$ \
\nLength:\n' + str(self.n_x)


    def generate_one_realization(self, draw=False):
        '''
        Makes one real random process with random spectral coefficients.
        Array xi_spec consists of the spectral coefficients.
        If draw == True, then the process will be plotted.
        '''

        xi_spec_real = np.random.normal(0, 1, self.n_x)*self.std/np.sqrt(2)
        xi_spec_imag = np.random.normal(0, 1, self.n_x)*self.std/np.sqrt(2)
        xi_spec = xi_spec_real + 1j*xi_spec_imag

#        xi_spec = np.array([np.random.normal(0, np.sqrt(b/2)) +
#                            np.random.normal(0, np.sqrt(b/2)) * 1j for b in self.spectrum_fft])
        xi_spec[0] = np.random.normal(0, np.sqrt(self.spectrum[0]))
        xi_spec[int(self.n_x/2)] = np.random.normal(0, np.sqrt(self.spectrum_fft[int(self.n_x/2)]))
        for i in range(0, int(self.n_x/2)):
            xi_spec[-i] = np.conj(xi_spec[i])

        RP = iFFT(xi_spec)
        if draw:
            fig, ax = plt.subplots(figsize=(12, 5))
            X = np.arange(0, self.n_x, 1)
            ax.set_title('Simulation of  RP on $S^1$')
            ax.set_ylabel(r'$RP(x)$')
            ax.set_xlabel('x')
            ax.plot(X, RP)
            plt.grid(True)
            plt.show()
        return np.real(RP)

    def generate_multiple_realizations(self, K, draw=False):
        '''
        Makes K process realisations with random spectral coefficients.
        If draw == True, then the processes will be plotted.
        '''
        Processes = np.zeros((K, self.n_x))
        xi_spec_real = np.random.normal(0, 1, (K, self.n_x))*self.std/np.sqrt(2)
        xi_spec_imag = np.random.normal(0, 1, (K, self.n_x))*self.std/np.sqrt(2)
        xi_spec = xi_spec_real + 1j*xi_spec_imag
        xi_spec[:,0] = np.random.normal(0, np.sqrt(self.spectrum[0]), K)
        xi_spec[:,int(self.n_x/2)] = np.random.normal(0, np.sqrt(self.spectrum_fft[int(self.n_x/2)]), K)
        for i in range(0, int(self.n_x/2)):
            xi_spec[:,-i] = np.conj(xi_spec[:,i])

        for i in range(K):
            Processes[i,:] = iFFT(xi_spec[i,:])
        if draw:
            plt.imshow(np.real(np.array(Processes)));
            plt.colorbar()
            plt.figure(figsize=(6, 6))
            plt.show()
        return np.array(Processes)

    def test(self, K=1000, draw=True):
        '''
        Computes the covariance function using K realisations of the process.
        '''
        Processes = self.generate_multiple_realizations(K)
        cvf_2 = [np.mean(np.array(
                [Processes[:,i]*Processes[:,i+k] for i in range(n_x-k)]
                ))for k in range(int(n_x/2))]

        if draw:
            plt.title("$S^1$ covariances")
            plt.plot(np.arange(len(cvf_2)), cvf_2, label = 'experimental')
            plt.plot(np.arange(len(cvf_2)),
                     iFFT(self.spectrum_fft)[:len(cvf_2)],
                     label = 'predicted')
            plt.xlabel("k")
            plt.ylabel("A")
            plt.legend(loc='best')
            plt.grid(True)
            plt.figure(figsize=(12, 5))
            plt.show()
        return cvf_2




#%%

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
        shape = (n_max, n_max)
        '''

        self.sigma = sigma
        w = iFFT(sigma.T) / np.sqrt(2 * np.pi * R_km)
        self.w = np.real(w)
        self.W = convert_aligned_matrix_to_diag(self.w) * np.sqrt(dx_km) #(dx_km) * np.sqrt(n_max)
        exp = np.array([[np.exp(1j*n*x*2*np.pi/n_max) for n in range(n_max)] for x in range(n_max)])
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

        nu_real = np.random.normal(0, 1/np.sqrt(2), n_max)
        nu_imag = np.random.normal(0, 1/np.sqrt(2), n_max)
        nu = nu_real + 1j*nu_imag

#        xi_spec = np.array([np.random.normal(0, np.sqrt(b/2)) +
#                            np.random.normal(0, np.sqrt(b/2)) * 1j for b in self.spectrum_fft])
        nu[0] = np.random.normal(0, 1)
        nu[int(n_max/2)] = np.random.normal(0, 1)
        for i in range(0, int(n_max/2)):
            nu[-i] = np.conj(nu[i])

        RP = np.array([
                [self.sigma[n, int(x/2/np.pi*n_max)]*nu[n]*np.exp(1j*n*x) for n in range(-int(n_max/2-1), int(n_max/2+1))]
                for x in np.linspace(0, 2*np.pi, n_max, endpoint=False)]).sum(axis = 1)
#        RP = np.matmul(RP, np.array([[np.exp(1j*m*x) for m in (list(range(0, int(n_max/2)))+list(range(-int(n_max/2), 0)))] for x in range(n_max)]))
#        RP = np.matmul(RP, np.array([[np.exp(1j*m*x) for m in range(-int(n_max/2), int(n_max/2))] for x in range(n_max)]))
        if draw:
            fig, ax = plt.subplots(figsize=(12, 5))
            X = np.arange(0, n_max, 1)
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

        shape: (n_realiz, n_max)
        '''
        realizations = np.zeros((n_realiz, n_max))
        nu_real = np.random.normal(0, 1/np.sqrt(2), (n_realiz, n_max))
        nu_imag = np.random.normal(0, 1/np.sqrt(2), (n_realiz, n_max))
        nu = nu_real + 1j*nu_imag
        nu[:,0] = np.random.normal(0, 1, n_realiz)
        nu[:,int(n_max/2)] = np.random.normal(0, 1, n_realiz)
        for i in range(0, int(n_max/2)):
            nu[:,-i] = np.conj(nu[:,i])

        realizations = np.array([
                [
                [self.sigma[n, int(x/2/np.pi*n_max)]*nu[i,n]*np.exp(1j*n*x) for n in range(-int(n_max/2-1), int(n_max/2+1))]
                for x in np.linspace(0, 2*np.pi, n_max, endpoint=False)]
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

        shape: (n_realiz, n_max)
        '''
        realizations = np.random.multivariate_normal(np.zeros((n_max)), self.B_true, n_realiz)
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
#        shape: (n_realiz, n_max)
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
    sigma = get_modal_spectrum_from_DLSM_params(NS_width, kappa_V, kappa_lamb, kappa_gamma,
                         V_mult=std_fcst**2, lamb=lamb_0, gamma_mult=gamma_mult,
                         is_cvm_to_be_converted_to_crm=False, draw=True)
    process = proc_ESM(sigma)
    process.test(100000)



#%% something old, perhaps should be deleted


#     sigma = np.sqrt(spectrum)


#     if is_cvm_to_be_converted_to_crm:
#         draw_2D(sigma, title = 'sigma')
#         sigma_corr = make_sigma_corr_from_sigma(sigma)
#         draw_2D(sigma_corr, title = 'sigma_corr')
#         draw_2D(sigma_corr - sigma, title = 'difference')
#         draw_2D(sigma_corr / sigma, title = 'factor')
#         sigma = sigma_corr
#     # modifying sigma so that B would be corr matrix
#     # sigma_norm = sigma/np.tile(sigma[0,:], (n_max, 1))
# #
# #        B_true = np.sum(np.array(
# #                [[[sigma[n,x]*sigma[n,y]*np.exp(1j*n*(x-y)*2*np.pi/n_max) for x in range(n_max)
# #                ] for y in range(n_max)] for n in range(n_max)]), axis = 0)
#     exp = np.array([[np.exp(1j*n*x*2*np.pi/n_max) for n in range(n_max)] for x in range(n_max)])
#     sigma_exp = exp * sigma
#     B_true_from_matrix = np.dot(sigma_exp.T, np.conj(sigma_exp))
#     B_true = B_true_from_matrix
# #        draw_2D(B_true_from_matrix, title = 'B_true_from_matrix')
# #        draw_2D(B_true, title = 'B_true')
# #        draw_2D(B_true - B_true_from_matrix, title = 'B_true - B_poss')



# #        ---Possible only if the process is stationary:---
# #        B_true_fft = np.zeros((n_max, n_max))
# #        for i in range(n_max):
# #            B_true_fft[:,i] = np.roll(iFFT(spectrum_fft[:,i]), i)
# #        B_true = B_true_fft
# #        -------------------------------------------------

#     c = np.mean(cx)
#     lamb = np.mean(lambx)
#     gamma = np.mean(gammax)

#     if draw:

#         plt.imshow(np.abs(B_true));
#         plt.colorbar()
#         plt.title("cvm")
# #            plt.xlabel("x")
# #            plt.ylabel("y")
#         plt.figure()
#         plt.show()

#         plt.title("$B(x)$: min and max $\gamma$")
#         min_, max_ = np.argmin(gammax), np.argmax(gammax)
#         plt.plot(np.arange(len(B_true[min_,:])), B_true[min_,:], color = 'red', label = 'min_gamma = ' + str(gammax[min_]))
#         plt.plot(np.arange(len(B_true[max_,:])), B_true[max_,:], color = 'red', label = 'max_gamma = ' + str(gammax[max_]))
#         min_, max_ = np.argmin(Lx), np.argmax(Lx)
#         plt.plot(np.arange(len(B_true[min_,:])), B_true[min_,:], color = 'green', label = 'min_L = ' + str(Lx[min_]))
#         plt.plot(np.arange(len(B_true[max_,:])), B_true[max_,:], color = 'green', label = 'max_L = ' + str(Lx[max_]))

#         plt.xlabel("x")
#         plt.legend(loc='best')
#         plt.grid(True)
#         plt.figure(figsize=(12, 5))
#         plt.show()


#         plt.title("$c(x)$")
#         plt.plot(np.arange(len(cx)), cx, label = 'c(x)')
#         plt.xlabel("x")
#         plt.legend(loc='best')
#         plt.grid(True)
#         plt.figure(figsize=(12, 5))
#         plt.show()


#         plt.title("$ lambda(x)$")
#         plt.plot(np.arange(len(lambx)), lambx, label = 'lambda(x)')
#         plt.xlabel("x")
#         plt.legend(loc='best')
#         plt.grid(True)
#         plt.figure(figsize=(12, 5))
#         plt.show()



# #        draw_2D(sigma_norm)
# #        draw_2D(spectrum[0:20,:])

# #        plt.title('std')
# #        plt.plot(np.arange(len(sigma[:,0])), sigma[:,0], label = 'spectrum')
# #        plt.xlabel("x")
# #        plt.legend(loc='best')
# #        plt.grid(True)
# #        plt.show()
# #        descr = 'RP_S1\nA random processes on a circle $S^1$ \
# #\nLength:\n' + str(n_max)
#     return sigma #    !! sigma is sigma[n, x] !!
