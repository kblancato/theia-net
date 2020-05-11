import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import torch
from torch.utils.data import TensorDataset, DataLoader

import modules.mutils as mutils


def data_split(len_data, log):
    """
    Split the data three ways into a training, validation, and test set.
    
    Parameters
    ----------
    len_data : int
        Number of stars in sample.
        
    log :_io.TextIOWrapper
        Log file.
        
    Returns
    ----------
    train_idx : array_like, int
        Training set indices.
        
    val_idx : array_like, int
        Validation set indices.
        
    test_idx : array_like, int
        Test set indices.
    """
    idx = np.arange(0, len_data)
    Xtrainval, Xtest, ytrainval, ytest = train_test_split(idx, idx,
                                    test_size=0.15, random_state=23)
                                    
    idx = np.arange(0, len(Xtrainval))
    Xtrain, Xval, ytrain, yval = train_test_split(idx, idx,
                                    test_size=0.15, random_state=23)
                                    
    train_idx = Xtrainval[Xtrain]
    val_idx = Xtrainval[Xval]
    test_idx = Xtest
    assert len(train_idx)+len(val_idx)+len(test_idx) == len_data
    
    print('Total number of stars: %s' % len_data, file=log)
    print('%s train, %s validation, %s test' % (len(train_idx),
                                                len(val_idx),
                                                len(test_idx)), file=log)
    print(r'%s train, %s validation, %s test' % (len(train_idx)/len_data,
                                        len(val_idx)/len_data,
                                        len(test_idx)/len_data), file=log)
                                
    return train_idx, val_idx, test_idx


def scale_data(data):
    """
    Scale the input time series data, as well as the target stellar properties.
        
    Parameters
    ----------
    data : array_like, floats
        Light curve flux values.
        
    Returns
    ----------
    X_scaled : array_like, floats
        Scaled time series data.
    """
    X_scaled = StandardScaler().fit_transform(data.T)
    
    return X_scaled


def scale_stds(stds):
    """
    Scale the light curve standard deviations.
        
    Parameters
    ----------
    stds : array_like, float
        Array of light curve standard deviations.
        
    Returns
    ----------
    stds_scaled : array_like, floats
        Scaled standard deviations
    """
    stds_scaled = StandardScaler().fit_transform(np.log10(stds).reshape(-1, 1))

    return stds_scaled


def data_for_torch(data, stds, label, batch_size, device, log):
    """
    Prepare and batch data for torch.
        
    Parameters
    ----------
    data : array_like, floats
        Scaled light curve flux values.
        
    stds : array_like, float
        Scaled light curve standard deviations.
        
    label : array_like, floats
        Scaled stellar property array
        
    batch_size : int
        Training batch size.
        
    device : string
        The device on which the torch tensor will be allocated.
        
    log : _io.TextIOWrapper
        Log file.
        
    Returns
    ----------
    loader : torch.utils.data.dataloader.DataLoader
        Iterable to load batched data.
    """
    X = torch.tensor(data.T, dtype=torch.float, device=device).unsqueeze(1)
    STDS = torch.tensor(stds, dtype=torch.float, device=device)
    Y = torch.tensor(label, dtype=torch.long, device=device)
    
    zipped = [[X[i], STDS[i], Y[i]] for i in range(len(X))]
    loader = DataLoader(zipped, batch_size=batch_size)

    n_batches = len(loader)
    batch_shape = [_[0].shape for _ in loader][0]
    print('%s batches with shape %s' % (n_batches,batch_shape), file=log)

    return loader
