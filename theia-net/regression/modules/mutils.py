import os
import torch


def device_status(device):
    """Print device status"""
    print('Using device:', device)

    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:',
              round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
        print('Cached:   ',
              round(torch.cuda.memory_cached(0)/1024**3,1), 'GB')


def create_log(dir, name):
    """
    Create log. If it already exists, delete it and overwrite.
        
    Parameters
    ----------
    dir : string
        Path to where to create log.
        
    name : string
        Name of the log.
        
    Returns
    ----------
    log : _io.TextIOWrapper
        Log file.
    """
    try:
        os.remove(dir+'%s.log' % name)
    except:
        pass
    log = open(dir+'%s.log' % name, 'a')
    return log

