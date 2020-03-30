# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 14:38:36 2017

@author: jose
"""
import sys
import os
sys.path.append(os.path.abspath("/home/jose/Dropbox/Doktorarbeit/"))
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


from mytoolbox.reading_robot import  load_data, visualising, cull_data

def drop_only_hypothetical_class_values(non_default_features_class_labels, df_labels):
    """
        It cheks if the labels that we have passed are actually in the metadata. If not, it deletes it:
    """
    print("* Checking if the class values passed as rules are actually in the labels")
    for feature_key, feature_value in non_default_features_class_labels.items():
        if feature_value not in list(set([item for item in df_labels.values.tolist()])):
            print("-> Deleted the label \"" + feature_value + "\" that was passed as argument, because none of the sample belongs to this label")
            print(list(set([item for item in df_labels.values.tolist()])))
            non_default_features_class_labels_c = dict(non_default_features_class_labels)
            del non_default_features_class_labels_c[feature_key]

            non_default_features_class_labels = non_default_features_class_labels_c
    return non_default_features_class_labels

def make_lists_features(non_default_features_class_labels, default_feature_class_labels):
    """
    It makes lists of features
    """
    non_default_features = list(non_default_features_class_labels.keys())
    default_feature = list(default_feature_class_labels.keys())
    features = non_default_features + default_feature
    non_default_labels_values = list(non_default_features_class_labels.values())
    labels_values = sorted(list(default_feature_class_labels.values()) + non_default_labels_values)
    return non_default_features, default_feature, features, non_default_labels_values, labels_values

def append_extra_feature(data, features, corpus):
    """
        It appends a feature extra to the matrix if the feature was not to  be  found
    """
    for feature in features:
        if feature not in data.columns.tolist():
            if feature not in corpus.columns.tolist():
                # TODO: esto no es realmente el corpus, son los datos duplicados
                print(corpus.shape)
                print(data.shape)
                print("* Following feature is neither in data nor in corpus: ", feature)
            else:
                print("* Appending following feature from corpus to data: ", feature)
                data[feature] = corpus[feature]
    return data 

def append_empty_pred_class(class_, data_features, default_feature_class_labels):
    """
        It addes a new column with the deafult value
    """

    # It creates a column for the predicted labels of the class
    class_predicted = class_+"-pred"

    # The default label of the class is asigned
    data_features[class_predicted] = list(default_feature_class_labels.values())[0]
    return data_features, class_predicted

def apply_rule(data_features, features, default_features_threshold, default_feature, class_predicted):
    for index, row in data_features[features].iterrows():
        if list(row[default_feature])[0] < default_features_threshold:
            row_wo = row.drop(default_feature)
            data_features = data_features.set_value(row.name, class_predicted, row_wo.idxmax())
    return data_features


def rule_tokens(document_data_raw, df_labels, default_feature_class_labels, non_default_features_class_labels, class_, wdir, corpus, default_features_threshold = "None"):
    # TODO: changed its parameters! Check in other scripts
    """
    """
    print("* Applying rule")
    
    non_default_features_class_labels = drop_only_hypothetical_class_values(non_default_features_class_labels, df_labels)

    # If a threshold was not passed, it gets the proportion of the number of the classes
    if default_features_threshold == "None":
        default_features_threshold = 1-(1/(len(non_default_features_class_labels) +len(default_feature_class_labels)))


    feature_class_labels = non_default_features_class_labels.copy()
    feature_class_labels.update(default_feature_class_labels)

    # It makes lists of features
    non_default_features, default_feature, features, non_default_labels_values, labels_values = make_lists_features(non_default_features_class_labels, default_feature_class_labels)

    document_data_raw_append = append_extra_feature(document_data_raw, features, corpus)

    # It creates a dataframe only with the selected features (tokens) and another one with the labels of the class
    data_features = document_data_raw_append[features].copy()
    
    # It calculates the proportion of these features inside each file
    data_features[data_features.columns] = data_features[data_features.columns].div(data_features[data_features.columns].sum(axis=1), axis=0)

    data_features, class_predicted = append_empty_pred_class(class_, data_features, default_feature_class_labels)

    data_features = apply_rule(data_features, features, default_features_threshold, default_feature, class_predicted)
    
    data_features = data_features.replace({class_predicted: feature_class_labels})

    # It creates a single table for features and labels

    data_features = pd.merge(data_features, df_labels.to_frame(), right_index=True, left_index=True)
    print(data_features.head())

    visualising.plot_confusion_matrix( labels_eval = data_features[class_], labels_pred = data_features[class_predicted], class_values = labels_values, class_ = class_, normalize=True, wdir = wdir)
    visualising.plot_confusion_matrix( labels_eval = data_features[class_], labels_pred = data_features[class_predicted], class_values = labels_values, class_ = class_, normalize=False, wdir = wdir)
    score = f1_score(data_features[class_], data_features[class_predicted], average = "micro") # Average? Micro? It seems ok. Macro makes weird things, I would say.  http://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html

    print(class_)
    print("Minimum value for default feature (if smaller, gets another label-value):", default_features_threshold)
    print("Score: ", score)
    return score



