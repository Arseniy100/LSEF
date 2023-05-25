# -*- coding: utf-8 -*-

"""
Last modified on Mar 2022
@author: Arseniy Sotskiy
"""



from torch.utils.data import DataLoader, Dataset
import numpy as np


"""https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#creating-a-custom-dataset-for-your-files"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

class CustomDataset(Dataset):
    '''Dataset class.
    if is for training (self.is_test == False),
        then contains data (band variances) and true spectrum
    if self.is_test == True, then only data.
    '''
    def __init__(self, data, spectrum=None, is_test=False):
        self.is_test = is_test
        if not self.is_test:
            self.spectrum = spectrum
        self.data = data

    def __len__(self):
        return self.data.shape[1]

    def __getitem__(self, idx):
        if self.is_test:
            return self.data[:, idx]
        return self.data[:, idx], self.spectrum[:, idx]


def prepare_data(data: np.array, spectrum: np.array=None, *, 
                 is_test: bool=False,
                 is_log: bool=False,
                 is_sqrt: bool = False,
                 batch_size: int=4, 
                 shuffle: bool=True,
                 predict_sqrt: bool=False) -> DataLoader:
    '''
    

    Parameters
    ----------
    data : np.array
        data for the nn (band variances).
    spectrum : np.array, optional
        True spectrum. The default is None.
    is_test : bool, optional
        if True, then data is prepared only for prediction (without labels).
        The default is False.
    is_log : bool, optional
        if True, then log(data) is used. The default is False.
    is_sqrt : bool, optional
        if True, then sqrt(data) is used. The default is False.
    batch_size : int, optional
        size of the batch. The default is 4.
    shuffle : bool, optional
        if True, then data is shuffled. The default is True.
    predict_sqrt : bool, optional
        if True, then we predict sqrt(spectrum). The default is False.

    Returns
    -------
    dataloader : DataLoader
    '''
    
    assert is_sqrt + is_log <= 1, 'One transform should be applied'
    data = data.reshape(data.shape[0], -1)
    if is_log:
        data = np.log(data)
    if is_sqrt:
        data = np.sqrt(data)
    print(f'prepare_data: {is_log}')
    if not is_test:
        spectrum = spectrum.reshape(spectrum.shape[0], -1)

        if predict_sqrt:
            
            spectrum_transformed = np.sqrt(spectrum)
        else:
            spectrum_transformed = spectrum
        
        dataset = CustomDataset(data, spectrum_transformed)
        dataloader = DataLoader(dataset, batch_size=batch_size, 
                                shuffle=shuffle,
                                # num_workers=1
                                )
    else:
        dataset = CustomDataset(data, is_test=is_test)
        dataloader = DataLoader(dataset, batch_size=data.shape[1], 
                                shuffle=False,
                                # num_workers=1
                                )  
    return dataloader


