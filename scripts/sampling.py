# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 07:37:36 2017

@author: jose
"""

 


import sys
import os
sys.path.append(os.path.abspath("/home/jose/Dropbox/Doktorarbeit/"))
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter
from reading_robot import cull_data

def under_sample(labels, filtered_raw_features, under_sample_method, maximum_cases = 5000):
    print("undersampling", under_sample_method)

    if under_sample_method == "None":
        sampled_labels = labels
        sampled_filtered_raw_features = filtered_raw_features

    elif under_sample_method.lower() in ["lpc","lpc-max"]: #"Least Populated Class"
        labels_counter = Counter(labels).most_common()
        
        least_frequent_class_value = labels_counter[-1][1]
        
        sampled_labels = pd.concat([labels.loc[labels == class_[0]].sample(n = least_frequent_class_value) for class_ in labels_counter])
        
        if (under_sample_method.lower() == "lpc-max") & (least_frequent_class_value > maximum_cases):
            sampled_labels = sampled_labels.sample(n = maximum_cases)
            
        else:
            sampled_labels = sampled_labels.sample(frac = 1)
        
        sampled_filtered_raw_features = filtered_raw_features.loc[sampled_labels.index.tolist()]

        
    sampled_filtered_raw_features = cull_data.drop_empty_features(sampled_filtered_raw_features) 

    try:
        sampled_filtered_raw_features.index.tolist() == sampled_labels.index.tolist()
        print("undersampled", sampled_labels.shape, sampled_filtered_raw_features.shape, under_sample_method)
    except:
        print("after undersampling, the index of the data and the labels are different :(", sampled_labels.shape, sampled_filtered_raw_features.shape, under_sample_method)

    return sampled_labels, sampled_filtered_raw_features
    
def standard_sampling(data, labels, test_size, verbose = True):
    print("* Sampling data in standard way")
    data_train, data_eval, labels_train, labels_eval = train_test_split(data, labels, test_size = test_size, stratify=labels) #, random_state = 0 )
    print(data_train.shape, data_eval.shape, labels_train.shape, labels_eval.shape) if verbose == True else 0
    print(Counter(labels_train), Counter(labels_eval)) if verbose == True else 0
    
    return data_train, data_eval, labels_train, labels_eval

