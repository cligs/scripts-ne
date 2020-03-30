# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 07:37:36 2017

@author: jose
"""


import sys
import os
sys.path.append(os.path.abspath("/home/jose/Dropbox/Doktorarbeit/"))
from reading_robot import  metadata2numbers
import scipy.stats as stats
import pandas as pd


def test_normality(series):
    s, p = stats.normaltest(series)
    
    if p < 0.05:
        result = "distribution is not normal"
    else:
        result = "distribution is normal"
    
    results = pd.DataFrame([s,p,result], columns =  [series.name], index=["statistic","p-value","result"]).T
    return results
        


def calculate_chi2_from_dataframe(cross_table, class_1, class_2):
    print("Chi2 test:")
    chi2, p, dof, expected  = stats.chi2_contingency(cross_table)
    if(p < 0.05):
        print("* the classes are related")
    else:
        print("the  classes are independent")
    return chi2, p, dof, expected 


def calculate_regression(df, serie1_name, serie2_name):
    print(df.shape)
    if df[serie1_name].dtype not in ["int64","float64"]:
        df = df.loc[df[serie1_name].isin(["8","7","6","5","4","3","2","1","0","-1","-2"])].copy()
        df[serie1_name] = df[serie1_name].astype(int)

    if df[serie2_name].dtype not in ["int64","float64"]:
        df = df.loc[df[serie2_name].isin(["8","7","6","5","4","3","2","1","0","-1","-2"])].copy()
        df[serie2_name] = df[serie2_name].astype(int)

    print(df.shape)
    results = stats.linregress(df[serie1_name], df[serie2_name])

    return results




def add_significance(df, class_pvalue = "test_result_pvalue"):
    df["significance"] = ""
    df.loc[df[class_pvalue] < 0.05,"significance"] = "*"
    df.loc[df[class_pvalue] < 0.01,"significance"] = "**"
    df.loc[df[class_pvalue] < 0.001,"significance"] = "***"
    return df

def test_differences_columns(df, column_class, column_value, equal_var=False):
    results_lt = []
    seen_values = []
    for value1 in sorted(list(set(df[column_class]))):
        for value2 in sorted(list(set(df[column_class]))):
            if value2 not in seen_values and value1 != value2:

                statistic, pvalue = stats.ttest_ind(
                            df.loc[df[column_class]==value1][column_value],
                            df.loc[df[column_class]==value2][column_value],
                    equal_var=equal_var
                            )
                seen_values.append(value1)
                results_lt.append([value1,value2,pvalue])
    results_df = pd.DataFrame(results_lt, columns=["value1","value2","pvalue"])
    results_df = add_significance(results_df, class_pvalue = "pvalue")
    
    return results_df

