# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 07:59:38 2017

@author: jose
"""

 # TODO: Lasso, Ridge, LinearRegression, 

import sys
import os
import pandas as pd
sys.path.append(os.path.abspath("/home/jose/Dropbox/Doktorarbeit/"))
from reading_robot import load_data, text2features

 
from sklearn.linear_model import (LinearRegression, Ridge, Lasso, HuberRegressor,
ARDRegression, BayesianRidge, ElasticNet, Lars, LassoLars, PassiveAggressiveRegressor,
RANSACRegressor, TheilSenRegressor, lars_path, ridge_regression)

from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.svm import SVR


def choose_regression_algorithm(method = "LR"):

    if method == "LR":
        regression_algorithm = LinearRegression()

    elif method == "Lasso":
        regression_algorithm = Lasso()

    elif method == "Ridge":
        regression_algorithm = Ridge()

    elif method == "HR":
        regression_algorithm = HuberRegressor()        
        
    elif method == "SVR":
        regression_algorithm = SVR()
        
    elif method == "LL":
        regression_algorithm = LassoLars()

    elif method == "ARDR":
        regression_algorithm = ARDRegression()

    elif method == "BR":
        regression_algorithm = BayesianRidge()

    elif method == "ElasticNet":
        regression_algorithm = ElasticNet()

    elif method == "Lars":
        regression_algorithm = Lars()

    elif method == "PA":
        regression_algorithm = PassiveAggressiveRegressor()

    elif method == "RANSAC":
        regression_algorithm = RANSACRegressor()

    elif method == "TS":
        regression_algorithm = TheilSenRegressor()

    elif method == "LP":
        regression_algorithm = lars_path()

    elif method == "RR":
        regression_algorithm = ridge_regression()
        
    else:
        print("You haven't chosen a valide classifier!!!")
    print("method used:\t", method)   
    return regression_algorithm

def regressing(wdir, features, outputs, classes, methods_lt, max_MFFs, text_representations, make_relative=False, cv=10):
    results_lt = []
    if features.shape[0] != outputs.shape[0]:
        print("Features and output do not have the same shape!")
        return
    print("features ", features.head())
    if make_relative == True:
        features = text2features.calculate_relative_frequencies(features)
    for class_ in classes:
        print("\n\nanalysed class:\t", class_)
        
        for text_representation in text_representations:
            transformed_features = text2features.choose_features(features, text_representation)
            for MFW in max_MFFs:
                print("MFW", MFW)
                transformed_features_cut = load_data.cut_corpus(transformed_features, min_MFF = 0, max_MFF = MFW)
                for method_st in methods_lt:
                    try:
                        regression_algorithm = choose_regression_algorithm(method = method_st)

                        results_dc = cross_validate(regression_algorithm, transformed_features_cut,
                                       outputs[class_], cv=10)
                        mean_results_fl = results_dc["test_score"].mean().round(3)
                        print(mean_results_fl)
                        results_lt.append([class_, text_representation, MFW, method_st, mean_results_fl, "R2"])
                    except:
                        print("problems with ", method_st)
    results_df = pd.DataFrame(results_lt, columns = ["class", "text_representation", "MFW", "method", "mean_results", "scoring"])
    results_df.sort_values(by="mean_results",ascending=False, inplace=True)
    
    return results_df
                        
                        

                            
        