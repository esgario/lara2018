# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 15:25:41 2019

@author: Guilherme
"""

import torch
import numpy as np

# Between-Class ------------------------------------------------------------

def between_class(x, y, y2=None, C=5):
    # Cuda
    if torch.cuda.is_available():
        device = torch.device("cuda:" + str(torch.cuda.current_device()))
    else:
        device = torch.device("cpu")
        
    # Amount of images
    batch_size = x.size()[0]
    
    # Generate random values based on uniform distribution
    r = torch.tensor(np.random.rand(batch_size), dtype=torch.float32).to(device)
    
    index = torch.randperm(batch_size).to(device)
    
    # Compute mean and std
    u = x.view(batch_size, -1).mean(dim=1)
    g = x.view(batch_size, -1).std(dim=1)
    
    # Image - Mean
    x = x - u.view( (batch_size,) + (len(x.shape)-1) * (1,) )
    
    # Compute p and put on image shape
    p = 1.0 / (1 + g / g[index] * (1 - r) / r)
    p = p.view( (batch_size,) + (len(x.shape)-1) * (1,) )
    
    # Combining images
    mixed_x = (p * x + (1 - p) * x[index, :]) / torch.sqrt(p ** 2 + (1-p) ** 2)
    
    # Convert to one-hot format
    y = torch.eye(C).index_select(dim=0, index=y.cpu()).to(device)
    y2 = torch.eye(C).index_select(dim=0, index=y2.cpu()).to(device) if y2 is not None else None
     
    r = r.view(-1, 1)
    y = r * y + ( 1 - r ) * y[index]
    y2 = r * y2 + ( 1 - r ) * y2[index] if y2 is not None else None
    
    return mixed_x, y, y2

# MixUp --------------------------------------------------------------------
def mixup_data(x, y, y2=None, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        if np.random.rand() > 0.5:
            lam = 1
        else:
            lam = 0.5

    batch_size = x.size()[0]
    
    if torch.cuda.is_available():
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    
    if y2 is None:
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam
    else:        
        y_a, y_b, y2_a, y2_b = y, y[index], y2, y2[index]
        return mixed_x, y_a, y_b, y2_a, y2_b, lam    

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def mixup_criterion_multilabel(criterion, pred, y_a, y_b, lam):
    y = torch.clamp( 2*(lam * y_a + (1 - lam) * y_b), 0, 1)
    return criterion(pred, y)

# Sample Pairing -----------------------------------------------------------
def sample_pairing(x):
    '''Returns mixed inputs'''
    batch_size = x.size()[0]
    
    if torch.cuda.is_available():
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = 0.5 * x + 0.5 * x[index, :]
    
    return mixed_x
    