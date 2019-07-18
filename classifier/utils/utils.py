#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  2 09:49:37 2019

@author: esgario
"""

from bokeh.plotting import figure
from bokeh.io import show
from bokeh.models import LinearAxis, Range1d
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

def line_graph(train, val):
    
    x = list(range(1, len(train)+1))
    
    fig = plt.figure(figsize=(4,4))
    plt.ylim(40, 100)
    plt.plot(x, train, color='b', )
    plt.plot(x, val, color='r')
    plt.xlabel('Época')
    plt.ylabel('Acurácia')
    plt.legend(['Treinamento','Teste'],loc=0)
    plt.grid()
    plt.show()
    fig.savefig('result.png', dpi=200)

# Generate an static plot from results
def static_graph(train, val):
    
    train = np.array(train)
    
    val = np.array(val)
    
    min_value = np.clip( min(min(train), min(val)) - 0.05, 0, 1 )
    max_value = np.clip( max(max(train), max(val)) + 0.05, 0, 1 )
    
    # Create figure
    p = figure(x_axis_label='Epoch', y_axis_label='Accuracy', width=750, height=500,
               y_range=(min_value, max_value), title='PyTorch ConvNet results')
    
    # Add train line
    p.line(np.arange(len(train)), train, color='blue')
    
    # val line
    p.line(np.arange(len(val)), val, color='red')
    
    show(p)

def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          output_name='confusion_matrix',
                          cmap=None,
                          figsize=(5,4)):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import itertools
    
    cm_orig = cm.copy()
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    if cmap is None:
        cmap = plt.get_cmap('Blues')
        
        
#    plt.figure(figsize=figsize)
    fig = plt.figure(figsize=figsize)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    thresh = cm.max() / 1.5 # if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, "{:d}\n{:.3f}".format(cm_orig[i, j], cm[i,j]),
                 horizontalalignment="center",
                 verticalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    
    accuracy = np.trace(cm_orig) / float(np.sum(cm_orig))
    balanced_accuracy = (cm * np.identity(len(cm))).sum() / len(cm)
    misclass = 1 - accuracy

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\nacc={:0.4f}; error={:0.4f}; bac={:0.4f}'.format(accuracy, misclass, balanced_accuracy))
    #plt.show()
    
    fig.savefig('results/' + output_name + '.png', bbox_inches='tight', dpi=200)
    plt.close(fig)
    
def multilabel_confusion_matrix(y_true, y_pred):
    
    labels = [0000, # 0  - Saudável
              1000, # B  - Bicho mineiro
              100,  # F  - Ferrugem
              10,   # P  - Phoma
              1,    # C  - Cercóspora
              1100, # BF - Bicho mineiro e Ferrugem
              1010, # BP - Bicho mineiro e Phoma
              1001, # BC - Bicho mineiro e Cercóspora
              110,  # FP - Ferrugem e Phoma
              101,  # FC - Ferrugem e Cercóspora
              11]   # PC - Phoma e Cercóspora
    
    target_names = ['0', 'B', 'F', 'P', 'C', 'BF', 'BP', 'BC', 'FP', 'FC', 'PC' ]
    
    true_label = []
    pred_label = []
    
    for i in range(0, len(y_true)):
        true_label.append(sum([ y_true[i][j]*10**(3-j) for j in range(0,4) ]))
        pred_label.append(sum([ y_pred[i][j]*10**(3-j) for j in range(0,4) ]))
        
    cm = confusion_matrix(true_label, pred_label, labels)
    print(cm)
    plot_confusion_matrix(cm=cm, target_names=target_names, title=' ', normalize=True, figsize=(14,11))
    
    