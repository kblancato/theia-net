import numpy as np
import pickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score

import torch
from torch.autograd import Variable


def predictions(device, model, data_loader):
    """
    Make stellar property predictions based on saved model.
        
    Parameters
    ----------
    device : string
        The device on which the torch tensor will be allocated.
    
    model : torch.nn
        Saved torch model.
        
    data_loader : torch.utils.data.dataloader.DataLoader
        Iterable to load batched data.
        
    Returns
    ----------
    predictions : array_like, float
        Stellar property predictions.
    """
    for ndata, (xdata, sdata, ydata) in enumerate(data_loader):
        xdata = Variable(xdata).to(device)
        sdata = Variable(sdata).to(device)
    assert ndata == 0

    model.eval()
    return model(xdata, sdata).cpu().detach().numpy().flatten()


def inverse_scale(dir, set, data):
    """
    Transform the stellar properties back to the original data range.
        
    Parameters
    ----------
    dir : string
        Directory where the data scaler is saved.
        
    set : string
        The dataset to transform, either train, validation, or test.
        
    data : array_like, floats
        Scaled stellar property predictions.
        
    Returns
    ----------
    predictions : array_like, float
        Stellar property predictions in the original data range.
    """
    scaler = pickle.load(open(dir+'y_%s_scaler.sav' % set, 'rb'))

    return scaler.inverse_transform(data.reshape(-1,1)).flatten()


def r2(y_pred, y_true):
    """
    Compute the r2 score.
        
    Parameters
    ----------
    y_pred : array_like, float
        The model predicted stellar properties.
        
    y_true : array_like, float
        The true stellar properties.

    Returns
    ----------
    r2_score : float
        The r2 score.
    """
    return r2_score(y_true, y_pred)


def bias(y_pred, y_true):
    """
    Compute the bias.
        
    Parameters
    ----------
    y_pred : array_like, float
        The model predicted stellar properties.
        
    y_true : array_like, float
        The true stellar properties.
        
    Returns
    ----------
    bias : float
        The bias.
    """
    return np.sum(np.subtract(y_pred,y_true))/len(y_pred)


def rms(y_pred, y_true):
    """
    Compute the rms.
        
    Parameters
    ----------
    y_pred : array_like, float
        The model predicted stellar properties.
        
    y_true : array_like, float
        The true stellar properties.
        
    Returns
    ----------
    rms : float
        The rms.
    """
    return np.sqrt(np.sum((y_true-y_pred)**2.)/len(y_true))


def plot_pred_true(save_path, label, pid, y_pred, y_true, metrics):
    """
    Plot the model predicted versus true stellar properties.
        
    Parameters
    ----------
    save_path : string
        Path where to save plot.
        
    label : string
        Name of predicted stellar property.
        
    pid : int
        Hyperparameter ID number.
        
    y_pred : array_like, float
        The model predicted stellar properties.
        
    y_true : sarray_like, float
        The true stellar properties.
        
    metrics : dictionary
        Dictionary containing the evaluation metrics.
    """
    fig = plt.figure()
    ax = plt.axes()
    plt.title(r'label: %s, pid: %s' %(label, pid))
    
    ax.scatter(y_true, y_pred, s=1, c='k')
    ax.plot(y_true, y_true)
    ax.set_ylabel('predicted')
    ax.set_xlabel('true')
    ax.text(0.05,0.93, r'r2:%s, $\Delta$:%s, rms:%s' % (np.round(metrics['r2'],2),
                                                           np.round(metrics['bias'],2),
                                                           np.round(metrics['rms'],2)),
            fontsize=10, ha='left', transform=ax.transAxes)
    ax.text(0.5,0.02, r'true: $\mu$=%s, $\sigma$=%s, \
            pred: $\mu$=%s, $\sigma$=%s' % (np.round(np.mean(y_true),3),
                                            np.round(np.std(y_true),3),
                                            np.round(np.mean(y_pred),3),
                                            np.round(np.std(y_pred),3)),
            fontsize=10, ha='left', transform=ax.transAxes)
    plt.savefig(save_path+'true_v_pred%s.png' %pid)