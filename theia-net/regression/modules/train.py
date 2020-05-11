import time
start_time = time.time()
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
from torch.autograd import Variable


def training(model, device, n_stop, tol, optimizer, loss_func, n_epochs, \
             train_loader, n_train, val_loader, n_val, save_path, log, pid):
    """
    The model training loop.
        
    Parameters
    ----------
    model : torch.nn
        Network architecure to train.
        
    device : string
        The device on which the torch tensor will be allocated.
        
    n_stop : int
        Number of epochs to stop training after if no improvement.
        
    tol : float
        Early stopping tolerance.
        
    optimizer : torch.optim
        Torch optimizer.
        
    loss_func : torch.nn.modules.loss
        Torch loss function.
        
    n_epochs : array_like, float
        The model predicted stellar properties.
        
    train_loader : torch.utils.data.dataloader.DataLoader
        Iterable to load batched training data.
        
    n_train : int
        Number of samples in the training set.
        
    val_loader : torch.utils.data.dataloader.DataLoader
        Iterable to load batched training data.
        
    n_val : int
        Number of samples in the validation set.
        
    save_path : string
        Path to save training progress and model.
        
    log : _io.TextIOWrapper
        Log file.
        
    pid : int
        Hyperparameter ID number.
    """
    start_time = time.time()
    losses = np.zeros((2, n_epochs))

    current_val_loss = np.inf
    early_stopped = False
    no_improve = 0
    
    for epoch in range(n_epochs):
        running_loss = np.zeros((2, n_epochs))

        # evaluate on validation set
        model.eval()
        with torch.no_grad():
            for nval, (xval, sval, yval) in enumerate(val_loader):
                xval = Variable(xval).to(device)
                sval = Variable(sval).to(device)
                yval = Variable(yval).to(device)
                
                valpred = model(xval, sval)
                running_loss[0,epoch] += loss_func(valpred, yval)
        losses[0,epoch] = 1e5*(running_loss[0,epoch]/n_val)
        
        if losses[0,epoch] > current_val_loss:            
            no_improve += 1
        if losses[0,epoch] <= current_val_loss and \
                current_val_loss-losses[0,epoch] > tol*losses[0,epoch]:
            no_improve = 0

        if epoch > 0:
            if losses[0,epoch] <= np.min(losses[0,:epoch]):
                torch.save(model.state_dict(), save_path+'model.pt')
            
        current_val_loss = losses[0,epoch]
        if no_improve >= n_stop:
            early_stopped = True
            print('=========EARLY STOPPING=========', file=log)
            break

        # train with training set
        model.train()
        for n, (x, s, y) in enumerate(train_loader):
            optimizer.zero_grad()
            x = Variable(x).to(device)
            s = Variable(s).to(device)
            y = Variable(y).to(device)
        
            prediction = model(x, s)
            loss = loss_func(prediction, y)
            loss.backward()
            optimizer.step()

            running_loss[1,epoch] += loss
        losses[1,epoch] = 1e5*(running_loss[1,epoch]/n_train)

        if epoch % 10 == 0:
            timenow = time.time()-start_time
            print('[EPOCH %s]: run time: %.3f s' % (epoch, timenow), file=log)

    np.save(save_path+'losses.npy', losses)
    plot_losses(losses, save_path, pid)

    if not early_stopped:
        torch.save(model.state_dict(), save_path+'model.pt')


def plot_losses(losses, save_path, pid):
    """
    Plot the loss of the training and validation set at each epoch.
        
    Parameters
    ----------
    losses : array_like, float
        The model predicted stellar properties.
        
    save_path : array_like, float
        The true stellar properties.
        
    pid : int
        Hyperparameter ID number.
    """
    fig = plt.figure(figsize=(16,4))
    ax = plt.axes()
    ax.semilogy(losses[0,:], label='validation set')
    ax.semilogy(losses[1,:], label='training set')
    
    ax.set_xticks(np.arange(0, losses.shape[1], 10))
    ax.set_ylabel('loss')
    ax.set_xlabel('epoch')
    plt.xticks(fontsize=10, rotation=90)
    plt.legend(loc=1, fancybox=False, edgecolor='k')
    
    plt.tight_layout()
    plt.savefig(save_path+'loss%s.png' %pid)