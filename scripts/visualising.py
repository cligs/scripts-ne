# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 08:54:05 2017

@author: jose
"""
import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(labels_eval, labels_pred, class_values, class_, wdir,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    cnf_matrix = confusion_matrix(labels_eval, labels_pred)
    np.set_printoptions(precision=2)
    if len(class_values) < 5:
        plt.figure(figsize=(5,5))
    else:
        plt.figure(figsize=(len(class_values)-2,len(class_values)-2))
        
    if normalize:
        cnf_matrix = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cnf_matrix)

    plt.imshow(cnf_matrix, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.grid(None)
    #plt.colorbar()
    tick_marks = np.arange(len(class_values))
    plt.xticks(tick_marks, class_values, rotation=45)
    plt.yticks(tick_marks, class_values, rotation=0)

    fmt = '.2f' if normalize else 'd'
    thresh = cnf_matrix.max() / 2.
    for i, j in itertools.product(range(cnf_matrix.shape[0]), range(cnf_matrix.shape[1])):
        plt.text(j, i, format(cnf_matrix[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cnf_matrix[i, j] > thresh else "black")

    if normalize == True:
        name_vis = class_+"_conf_mat_norm"
    else:
        name_vis = class_+"_conf_mat_raw"
        
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(wdir+"visualisations/"+name_vis+".png",bbox_inches='tight', dpi=300)
    plt.show()
    
    