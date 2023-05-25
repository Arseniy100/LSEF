# -*- coding: utf-8 -*-

"""
Last modified on Mar 2022
@author: Arseniy Sotskiy
"""

import torch
import numpy as np

from torch import nn
import matplotlib.pyplot as plt

"""https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#creating-a-custom-dataset-for-your-files"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


def transform_spectrum_to_derivative_1d(spectrum):
    return (spectrum[:-1] - spectrum[1:]) # [::-1]

def transform_spectrum_to_derivative(spectrum, axis=0, torch=False):
    if torch:
        if axis == 0:
            return spectrum[:-1,:,:] - spectrum[1:,:,:]
        elif axis == 1:
            return spectrum[:,:-1,:] - spectrum[:,1:,:]
        elif axis == 2:
            return spectrum[:,:,:-1] - spectrum[:,:,1:]
        else:
            assert False, 'axis is too big'
    return np.apply_along_axis(
        transform_spectrum_to_derivative_1d, axis, spectrum
        )
    

def transform_derivative_to_spectrum_1d(derivative):
    return np.append(np.cumsum(derivative[::-1])[::-1], 0)
    # return np.append(np.cumsum(derivative)[::-1], 0)

def transform_derivative_to_spectrum(derivative, axis=0, torch=False):
    if torch:
        return 
    return np.apply_along_axis(
        transform_derivative_to_spectrum_1d, axis, derivative
        )


if __name__ == '__main__':  
    sp = np.array([10, 9, 7, 3, 1, 0.5, 0.3, 0.1, 0.01, 0])
    der = transform_spectrum_to_derivative_1d(sp)
    sp_ = transform_derivative_to_spectrum_1d(der)
    print(sp)
    print(der)
    print(sp_)
    assert (sp_ == sp).all()


#%%

def compute_grid_cell_size(a: list):
    a_list = list(a)
    sum = - np.array(
        [a_list[0], a_list[0]] + a_list
        ) + np.array(
            a_list + [a_list[-1], a_list[-1]]
            )
    return torch.from_numpy(sum[1:-1] / 2)

            
if __name__ == '__main__':            
    x = np.array([0,1,2,5,8])
    print(compute_grid_cell_size(x))
    
    


def compute_discrete_derivative(function: np.array, x_grid: np.array=None, 
                                order: int=1) -> np.array:
    assert type(order) == int
    assert order > 0
    if x_grid is None:
        x_grid = np.arange(len(function))
        # print(function[:,1:].shape, x_grid[1:].shape)
    der = (function[:,1:] - function[:,:-1]) / (x_grid[1:] - x_grid[:-1]) 
    if order == 1:
        return der
    else:
        return compute_discrete_derivative(der, 
                                x_grid[:-1], order=order-1)


#%% Loss
# !!!!!!!!!!!!!!!!!!

class PriorSmoothLoss(nn.Module):
    def __init__( self, device, n_max, w_1=1e-4, w_2=1e-4, l_0=1,  
                 **kwargs):
        super().__init__()
        self.log_wavenumbers = torch.from_numpy(
            np.log(np.arange(n_max+1) + l_0)          # s_l
            ) 

        plt.figure()
        plt.plot(self.log_wavenumbers)
        plt.title('log_wavenumbers')
        plt.grid()
        plt.show()
  
        self.grid_cell_size = compute_grid_cell_size(self.log_wavenumbers)
        
        plt.figure()
        plt.plot(self.grid_cell_size)
        plt.title('grid_cell_size')
        plt.grid()
        plt.show()

        self.w_1 = w_1
        self.w_2 = w_2
        self.w_clim = 1

    def forward(self, inputs, targets, draw=False): 

        s_l = self.log_wavenumbers
        delta_l = self.grid_cell_size
        error = inputs - targets
        loss_clim = self.w_clim / 2 * \
            error**2 * delta_l
            
        
        if draw:
            
            plt.figure()
            plt.plot((error**2).detach().numpy().T, 
                     label='(inputs - targets)**2')
            plt.legend(loc='best')
            plt.grid()
            plt.show()

        der_1 = compute_discrete_derivative(error, 
                                          s_l, order=1)
        der_2 = compute_discrete_derivative(error, 
                                          s_l, order=2)

        loss_smoo_1 = self.w_1 / 2 * der_1**2 * (s_l[1:] - s_l[:-1])#[:-1]

        loss_smoo_2 = self.w_2 / 2 * der_2**2 * (s_l[1:] - s_l[:-1])[:-1]

        if draw:
            plt.figure()
            plt.plot(loss_smoo_1.detach().numpy().T, 
                     label='loss_smoo_1', color='blue')
            plt.plot(loss_smoo_2.detach().numpy().T, 
                     label='loss_smoo_2', color='cyan')
            plt.plot(loss_clim.detach().numpy().T, 
                      label='loss_clim', color='green')
            plt.legend(loc='best')
            plt.grid()
            plt.show()

            
            
        loss = loss_clim.mean() + loss_smoo_1.mean() + loss_smoo_2.mean()

        return loss


class PriorSmoothLogLoss(PriorSmoothLoss):
    
    def __init__(self, device, k_1: float=0.5, k_2: float=0.5, 
                 S_1: float=None, S_2: float=None, *args, **kwargs):
        super().__init__(device, *args, **kwargs)
        self.log_max_wavenumber = self.log_wavenumbers.max() # = A = S_max
        self.radius = self.log_max_wavenumber / np.pi
        if S_1 is None:
            S_1 = self.log_max_wavenumber * k_1
        if S_2 is None:
            S_2 = self.log_max_wavenumber * k_2
        self.w_1 = (self.radius * S_1)**2 
        self.w_2 = (self.radius * S_2)**4
        
    
    def forward(self, inputs, targets, draw=False): 
        log_inputs = torch.log(inputs)  # lambda
        log_targets = torch.log(targets)
        return super().forward(log_inputs, log_targets, draw=draw) * np.random.rand()


class L2VarLoss(nn.Module):
    def __init__(self, l_max: int, sqrt: bool=True,
                 *args, **kwargs):
        '''
        l2 variance loss

        Parameters
        ----------
        l_max : int
            usually n_max.
        sqrt : bool, optional
            If nn predicts spectrum, then we should put sqrt in loss (True). 
            If nn predicts sqrt(spectrum), then we do not need it (False).
            The default is True.
        *args 
        **kwargs
        '''
        super().__init__()
        self.two_l_plus_1 = (torch.arange(l_max + 1) * 2 + 1).to(
            torch.float64
            ) / 4 / np.pi
        self.sqrt = sqrt

    def forward(self, inputs, targets, draw=False): 
        # inputs.shape = (batch_size, n_max+1)
        if self.sqrt:
            loss = ((torch.sqrt(inputs) - torch.sqrt(targets))**2 @ \
                self.two_l_plus_1).mean()
        else:
            loss = ((inputs - targets)**2 @ \
                self.two_l_plus_1).mean()

        return loss

class AnlsLoss(nn.Module):
    def __init__(self, obs_std_err, l_obs,
                 *args, **kwargs):
        super().__init__()
        self.l_obs = l_obs
        self.r = \
            obs_std_err**2 * 4 * np.pi / np.sum(2 * np.arange(l_obs + 1) + 1)
         
    
    def deviance(self, b, t):
        return (self.r**2 * (t - b)**2) / ((self.r + b)**2 * (t + self.r))


    def forward(self, inputs, targets, draw=False): 
        # inputs.shape = (batch_size, n_max+1)
        inputs = inputs[:,:self.l_obs + 1]
        targets = targets[:,:self.l_obs + 1]

        loss = (2 * torch.arange(self.l_obs + 1) + 1).to(
            torch.float64
            ) @ self.deviance(inputs, targets).T
        return loss.mean()


class RandomLoss(nn.Module):
    def __init__(self, 
                 *args, **kwargs):
        super().__init__()

    def forward(self, inputs, targets, draw=False): 
        # inputs.shape = (batch_size, n_max+1)
        
        return torch.randn()
    
