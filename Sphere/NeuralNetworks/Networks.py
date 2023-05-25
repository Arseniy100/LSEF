# -*- coding: utf-8 -*-
"""
Last modified on Mar 2022
@author: Arseniy Sotskiy
"""

import torch
import numpy as np
import torch.optim as optim
from time import ctime

from collections import OrderedDict
from torch import nn

import matplotlib.pyplot as plt

"""https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#creating-a-custom-dataset-for-your-files"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import typing as tp

from functions.tools import _time

from NeuralNetworks.DataPreparation import (
    prepare_data
    )


#%%

class NeuralNetwork_controlled(nn.Module):
    def __init__(self, n_in, n_out, activation, non_linearity=None,
                 multipliers = (6, 20, 16)):
        super(NeuralNetwork_controlled, self).__init__()
        if non_linearity is None:
            non_linearity = nn.ReLU()
        # non_linearity = nn.ELU
        self.flatten = nn.Flatten()
        layers = []
        multipliers = [1] + list(multipliers)
        for i in range(len(multipliers) - 1):
            layers += [
                (f'linear{i+1}', 
                 nn.Linear(int(multipliers[i]*n_in), 
                           int(multipliers[i+1]*n_in))),
                (f'non_linearity{i+1}', 
                 non_linearity)
            ]
        layers += [
                (f'linear{i+2}',
                 nn.Linear(int(multipliers[-1]*n_in), n_out)),
                ('activation',
                 activation)
            ]
        print(layers)

        self.linear_relu_stack = nn.Sequential(          
            OrderedDict(layers)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


class NeuralNetSpherical:
    # !!!!!!!
    def __init__(self, model, n_input: int, n_output: int, 
                 loss: tp.Any='l1',  
                 activation_name: str='ReLU', 
                activation_function: tp.Callable=nn.ReLU(),
                device: str=None,
                loss_kwargs: dict=dict(),
                predict_sqrt: bool=False,
                *init_args, **init_kwargs):
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            assert device in ['cuda', 'cpu']
            self.device = device
        self.predict_sqrt = predict_sqrt
        if predict_sqrt:
            loss_kwargs['sqrt'] = False
        self.activation_name = activation_name
        self.activation_function = activation_function
        print('Using {} device'.format(self.device))
        self.model = model(
            n_input, n_output+1, activation_function,
            *init_args, **init_kwargs
            ).to(self.device)

        self.trained = False
        self.model = self.model.double()
        self.loss_type = loss
        if loss == 'l1':
            self.loss = nn.L1Loss()
        elif loss == 'l2':
            self.loss = nn.MSELoss()
        else:
            self.loss = loss(device=self.device, **loss_kwargs)
        
        
        
    def fit(self, data, spectrum, 
            batch_size=4, n_epochs=5, lr=0.001, momentum=0.1,
            validation_coeff=0.1, path_to_save=None, draw=False,
            is_log: bool=False,
            is_sqrt: bool=False,
            band_centers=None):
        self.is_log = is_log
        print(f'fit: {is_log}')
        self.is_sqrt = is_sqrt
        func = lambda x: x
        func_inv = lambda x: x
        if self.is_log:
            func_inv = np.exp
            func = np.log
        if self.is_sqrt:
            func_inv = np.square
            func = lambda x: np.sqrt(np.abs(x))
        full_size = data.shape[1]
        valid_size = int(full_size * validation_coeff)

        train_dataloader = prepare_data(data[:,valid_size:,:], 
                                        spectrum[:,valid_size:,:], 
                                        batch_size=batch_size,
                                        is_log=is_log,
                                        is_sqrt=is_sqrt,
                                        predict_sqrt = self.predict_sqrt)
        validation_dataloader = prepare_data(data[:,:valid_size,:], 
                                        spectrum[:,:valid_size,:], 
                                        # batch_size=valid_size)
                                        batch_size=batch_size,
                                        is_log=is_log,
                                        is_sqrt=is_sqrt,
                                        predict_sqrt = self.predict_sqrt)
        # optimizer = optim.SGD(self.model.parameters(), 
        #                       lr=lr, momentum=momentum)
        optimizer = optim.Adam(self.model.parameters())
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=15,
            factor=0.1,
            threshold=1e-6,
            verbose=True
        )
        step = int(len(train_dataloader) / 10)

        loss_history = []
        valid_loss_history = []
        
        for epoch in range(n_epochs):  # loop over the dataset multiple times

            running_loss = 0.0
            running_valid_loss = 0.0
            for i, data_load in enumerate(train_dataloader, 0):
                plt.close('all')
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data_load
        
                # zero the parameter gradients
                optimizer.zero_grad()
        
                # forward + backward + optimize
                outputs = self.model(inputs) ##

                try:
                    loss = self.loss(outputs, labels,
                                        # draw=True
                                        draw=((i % step == step - 1) and draw)
                                     )
                except TypeError:
                    loss = self.loss(outputs, labels,)
                loss.backward()
                optimizer.step()

                
                running_loss += loss.item() / batch_size
                
                if i % step == step - 1:    # print every 2000 mini-batches
                    if draw:
                        plt.figure()
                        plt.grid()
                        if band_centers:
                            plt.scatter(np.tile(np.array(band_centers), 
                                                (batch_size,1)), 
                                        inputs[:,:].detach().numpy(),                                        color='green')
                        plt.plot(func(outputs[:,:].detach().numpy().T), color='red')
                        plt.plot(func(labels[:,:].detach().numpy().T), color='black')
                        plt.show()
                        
                        plt.figure()
                        plt.grid()
                        if band_centers:
                            plt.scatter(np.tile(np.array(band_centers), (batch_size,1)), 
                                        func_inv(inputs[:,:].detach().numpy()), # \
                                        #     if is_log else \
                                        # np.log(inputs[:,:].detach().numpy()),
                                        color='green')
                        
                        print(f'train: {self.is_log}')
                        plt.plot(outputs[:,:].detach().numpy().T, color='red')
                        plt.plot(labels[:,:].detach().numpy().T, color='black')
                        plt.show()
                        print('[%d, %5d] loss: %.7f' %
                            (epoch + 1, i + 1, running_loss / step))
                    loss_history.append(running_loss / step)
                    running_loss = 0.0
                    if validation_coeff > 0:
                        with torch.no_grad():
                            for k, valid_data in enumerate(validation_dataloader, 0):
                                # print(_)
                                inputs_valid, labels_valid = valid_data
                                outputs_valid_ = self.model(inputs_valid) ##
                                # outputs_valid = self.predict(data[:,:valid_size,:], 
                                #                               raw=True)
                                # print(data.shape, inputs_valid.size(),
                                #       outputs_valid.size(), outputs_valid_.size(), 
                                #       labels_valid.size())
                                valid_loss = self.loss(outputs_valid_, labels_valid)
                                running_valid_loss += valid_loss.item() / batch_size
                            valid_loss_history.append(running_valid_loss / k)
                            try:
                                scheduler.step()
                            except:
                                scheduler.step(loss)
                            print('[%d, %5d] valid loss: %.7f' %
                                (epoch + 1, i + 1, running_valid_loss / k))
                            running_valid_loss = 0.0
                    
        print('Finished Training')
        self.loss_history = loss_history
        self.trained = True
        self.n_epochs = n_epochs
        plt.close('all')
                    
        plt.figure()
        title = f'''Loss history  
activation function {self.activation_name},
batch_size={batch_size}, n_epochs={n_epochs}, 
lr={lr}, momentum={momentum}'''
        self.info = title
        plt.grid()
        plt.plot(np.linspace(0, n_epochs, len(loss_history)),
                 loss_history, label='train loss')
        plt.xlabel('Epochs')
        plt.ylabel('loss')
        plt.yscale('log')
        if validation_coeff > 0:
            plt.plot(np.linspace(0, n_epochs, len(valid_loss_history)),
                      valid_loss_history, label='valid loss')
        plt.title(title)
        plt.legend(loc='best')
        if path_to_save is not None:
            path = path_to_save + f'{ctime()}{title}.png'
            path = path.replace(':', '_')
            path = path.replace('\n', '')
            plt.savefig(path)
        plt.show()

        
        plt.close('all')
        
        
    def plot_loss(self, path_to_save: str=None, info: str=''):
        plt.figure()
        plt.grid()
        plt.plot(np.linspace(0, self.n_epochs, len(self.loss_history)),
                 self.loss_history, label='train loss')
        plt.xlabel('Epochs')
        plt.ylabel('loss')
        # plt.plot(valid_loss_history, label='valid loss')
        title='loss history' + info
        plt.title(title)
        plt.legend(loc='best')
        if path_to_save is not None:
            path = path_to_save + f'{_time()}{title}.png'

            plt.savefig(path)
        plt.show()
        plt.close('all')

        
    def predict(self, data, raw=False):
        # assert self.trained, 'Not trained'
        # for p in self.model.parameters():
        #     if p.grad is not None:
        #         print(p.grad.data)
        test_dataloader = prepare_data(data, is_test=True,
                                       is_log=self.is_log,
                                       is_sqrt=self.is_sqrt)
        with torch.no_grad():
            for dataload in test_dataloader: # actually only one iteration
                # calculate outputs by running images through the network
                outputs = self.model(dataload)
        if self.predict_sqrt:
            outputs = outputs**2
        if raw:
            return outputs

        outputs = outputs.numpy().T.reshape((-1, data.shape[1], data.shape[2]))
        plt.close('all')
        return outputs



#%%
class NeuralNetwork_first_try(nn.Module):
    def __init__(self, n_in, n_out, activation):
        super(NeuralNetwork_first_try, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(n_in, 2*n_in),
            # nn.LeakyReLU(negative_slope=0.1),
            nn.ELU(),
            nn.Linear(2*n_in, 8*n_in),
            # nn.LeakyReLU(negative_slope=0.1),
            nn.ELU(),
            # nn.Linear(4*n_in, 8*n_in),
            # nn.ELU(),
            # nn.Linear(8*n_in, 16*n_in),
            # nn.ELU(),
            nn.Linear(8*n_in, n_out),
            # nn.Softplus(beta=1000)
            # nn.Hardswish()
            # nn.GELU()
            # nn.Sigmoid() # does not work
            # nn.Mish()
            # SquareActivation()
            # AbsActivation()
            activation
        )
        
    def forward(self, x):
        x = self.flatten(x)
        # print(x[:,0].shape)
        # x = (x.T / x[:,0]).T
        logits = self.linear_relu_stack(x)
        return logits
        # return (logits.T * x[:,0]).T



class NeuralNetwork_small(nn.Module):
    def __init__(self, n_in, n_out):
        super(NeuralNetwork_small, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(n_in, n_out),
            nn.Softplus(beta=1000)
        )
        
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


class NeuralNetwork_deep(nn.Module):
    def __init__(self, n_in, n_out, activation):
        super(NeuralNetwork_deep, self).__init__()
        non_linearity = nn.ReLU
        # non_linearity = nn.ELU
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(          
            nn.Linear(n_in, 6*n_in),
            non_linearity(),
            nn.Linear(6*n_in, 20*n_in),
            non_linearity(),
            nn.Linear(20*n_in, 16*n_in),
            non_linearity(),
            nn.Linear(16*n_in, n_out),
            activation
            
            # nn.ReLU()
            # nn.Hardswish()
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    
