# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 11:09:18 2019

@author: Guilherme
"""

import numpy as np
from sklearn import preprocessing
from sklearn.metrics import accuracy_score as exact_match_ratio, precision_score, recall_score, hamming_loss, f1_score

def eval_metrics(metric='acc', y_pred=None, y_true=None, y2_true=None, lam=None, _binarize_labels=False, bits=5):
    
    if type(y_pred) != np.ndarray:
        y_pred, y_true = np.array(y_pred), np.array(y_true)

    if _binarize_labels== True:
        y_true, y_pred = binarize_labels(y_true, y_pred, bits)
        
    if y2_true is None:
    
        if metric == 'acc':
            return accuracy(y_true, y_pred)
            
        if metric == 'emr':
            return exact_match_ratio(y_true, y_pred)
            
        if metric == 'hl':
            return hamming_loss(y_true, y_pred)
        
        if metric == 'prc':
            return precision_score(y_true, y_pred, average='samples')
            
        if metric == 'rec':
            return recall_score(y_true, y_pred, average='samples')
        
        if metric == 'f1':
            return f1_score(y_true, y_pred, average='macro')
        
        if metric == 'f1_ml':
            return f1_score(y_true, y_pred, average='samples')
        
    else:
        
        if metric == 'acc':
            return (lam * accuracy(y_true, y_pred) + (1 - lam) * accuracy(y2_true, y_pred))
            
        if metric == 'emr':
            return (lam * exact_match_ratio(y_true, y_pred) + (1 - lam) * exact_match_ratio(y_true, y_pred))
            
        if metric == 'hl':
            return (lam * hamming_loss(y_true, y_pred) + (1 - lam) * hamming_loss(y_true, y_pred))
        
    return (accuracy(y_true, y_pred),
            exact_match_ratio(y_true, y_pred),
            hamming_loss(y_true, y_pred))
    
def binarize_labels(y_true, y_pred, bits):
    lb = preprocessing.LabelBinarizer()
    lb.fit(np.zeros((1, bits)))
    
    z_pred = np.empty((y_pred.shape[0], 0))
    z_true = np.empty((y_true.shape[0], 0))
    
    for i in range(y_true.shape[1]):
        z_pred = np.concatenate( (z_pred, lb.transform(y_pred[:,i])), axis=1 )
        z_true = np.concatenate( (z_true, lb.transform(y_true[:,i])), axis=1 )
    
    return z_true, z_pred
    
def accuracy(y_true, y_pred):
    '''
    Compute the Accuracy/Hamming score for the multi-label case
    '''
    acc_list = []
    for i in range(y_true.shape[0]):
        set_true = set( np.where(y_true[i])[0] )
        set_pred = set( np.where(y_pred[i])[0] )
        #print('\nset_true: {0}'.format(set_true))
        #print('set_pred: {0}'.format(set_pred))
        tmp_a = None
        if len(set_true) == 0 and len(set_pred) == 0:
            tmp_a = 1
        else:
            tmp_a = len(set_true.intersection(set_pred))/\
                    float( len(set_true.union(set_pred)) )
        #print('tmp_a: {0}'.format(tmp_a))
        acc_list.append(tmp_a)
    return np.mean(acc_list)