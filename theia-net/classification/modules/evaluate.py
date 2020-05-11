import numpy as np
from scipy import interp

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import label_binarize

import torch
from torch.autograd import Variable


def predictions(device, model, data_loader):
    """
    Make stellar evolutionary state predictions based on saved model.
        
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
    predictions : array_like, int
        Categorical evolutionary state prediction.
    """
    for ndata, (xdata, sdata, ydata) in enumerate(data_loader):
        xdata = Variable(xdata).to(device)
        sdata = Variable(sdata).to(device)
    assert ndata == 0
    model.eval()
    
    y_pred = model(xdata, sdata)
    softmax = torch.exp(y_pred).cpu()
    probs = list(softmax.detach().numpy())
    predictions = np.argmax(probs, axis=1)
    
    return predictions


def probs(device, model, data_loader):
    """
    Make stellar property probability predictions based on saved model.
        
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
    probs : array_like, float
        Evolutionary state probabilities.
    """
    for ndata, (xdata, sdata, ydata) in enumerate(data_loader):
        xdata = Variable(xdata).to(device)
        sdata = Variable(sdata).to(device)
    assert ndata == 0
    model.eval()
    
    y_pred = model(xdata, sdata)
    softmax = torch.exp(y_pred).cpu()
    probs = np.array(list(softmax.detach().numpy()))
    probs = probs/probs.sum(axis=1, keepdims=1)
    
    return probs


def AUROC(y_true, probs):
    """
    Compute the multi-class area under the receiver operator curve.
        
    Parameters
    ----------
    y_true : array_like, int
        The true evolutionary states.
    
    probs : array_like, float
        Evolutionary state probabilities.
        
    Returns
    ----------
    auroc : float
        Multi-class AUROC.
    """
    y = label_binarize(y_true, classes=[0, 1, 2])
    n_classes = y.shape[1]
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y[:, i], probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr["micro"], tpr["micro"], _ = roc_curve(y.ravel(), probs.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    return (roc_auc[0]+roc_auc[1]+roc_auc[2])/n_classes


def average_precision(y_true, probs):
    """
    Compute the average precision across classes.
        
    Parameters
    ----------
    y_true : array_like, int
        The true evolutionary states.
        
    probs : array_like, float
        Evolutionary state probabilities.
        
    Returns
    ----------
    average_precision : float
        Multi-class average precision.
    """
    y = label_binarize(y_true, classes=[0, 1, 2])
    n_classes = y.shape[1]
    
    precision = dict()
    recall = dict()
    ap = dict()
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y[:, i], probs[:, i])
        ap[i] = average_precision_score(y[:, i], probs[:, i])

    return (ap[0]+ap[1]+ap[2])/n_classes


def accuracy(y_pred, y_true):
    """
    Compute the multi-class accuracy.
        
    Parameters
    ----------
    y_true : array_like, int
        The true evolutionary states.
        
    y_pred : array_like, int
        The predicted evolutionary states.
        
    Returns
    ----------
    accuracy : float
        Multi-class accuracy.
    """
    return accuracy_score(y_true, y_pred)