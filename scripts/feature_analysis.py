# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 07:37:36 2017

@author: jose
"""

import numpy as np
import sys
import os
sys.path.append(os.path.abspath("/home/jose/Dropbox/Doktorarbeit/"))
import pandas as pd
from reading_robot import load_data, sampling, text2features, visualising, cull_data, classify
from sklearn.model_selection import cross_val_score
from scipy import stats
from collections import Counter
import re
from scipy import stats


def get_coef(wdir, wsdir="corpus/", freq_table=[], metadata="metadata.csv", sep="\t",
             class_="class", verbose=True, method="SVC", max_MFF=5000,
             text_representation="zscores", problematic_class_values=["n.av."],
             minimal_value_samples=2, make_relative=True, under_sample_method="None",
             maximum_cases=5000, sampling_times=1):
    if (type(freq_table) == list) & (type(metadata) == str):
        cut_raw_features, metadata = load_data.load_corpus_metadata(wdir, wsdir, sep, verbose, 0, max_MFF, freq_table,
                                                                    metadata)
    else:
        cut_raw_features = freq_table

    if make_relative == True:
        cut_raw_features = text2features.calculate_relative_frequencies(cut_raw_features)

    filtered_raw_features, labels = cull_data.cull_data(cut_raw_features, metadata, class_, verbose,
                                                        problematic_class_values=problematic_class_values,
                                                        minimal_value_samples=minimal_value_samples)

    document_data_model_cut = load_data.cut_corpus(filtered_raw_features, min_MFF=0, max_MFF=max_MFF)

    document_data_model = text2features.choose_features(document_data_model_cut, text_representation)

    coef_df = pd.DataFrame(columns=document_data_model.columns.tolist())

    intercept_lt = []

    print("The ten first MFWs: ", document_data_model.columns.tolist()[0:10])
    print("The ten first MFWs: ", document_data_model.columns.tolist()[-10:])
    # Meter sampling loop

    for sampling_i in range(sampling_times):
        sampled_labels, sampled_document_data_model = classify.under_sample(labels, document_data_model, under_sample_method,
                                                                   maximum_cases)

        classifier = classify.choose_classifier(method=method)

        model = classifier.fit(sampled_document_data_model, sampled_labels)
        #print(model.coef_)
        print(model.coef_.shape)
        sampled_coef_df = pd.DataFrame(data=model.coef_.tolist(), columns=sampled_document_data_model.columns.tolist())
        print(model.intercept_)
        intercept_lt.append(float(model.intercept_))

        coef_df = pd.concat([coef_df, sampled_coef_df])

    coef_df = coef_df.reindex_axis(coef_df.mean().sort_values().index, axis=1)

    print(coef_df.shape)
    print(intercept_lt)

    return coef_df, intercept_lt


def calculate_scores_subgenres(wdir, df, metadata_df, subgenres, max_MFF= 1000,
                               text_representation = "log", sampling_times = 100):
    df = df.iloc[:,0:max_MFF]
    
    coef_subgenres_df = pd.DataFrame(columns=df.columns.tolist())

    print(coef_subgenres_df.shape)
    print(coef_subgenres_df)

    for subgenre in subgenres:
        print("subgenre", subgenre)
        coef_df, intercept_lt = get_coef(wdir, freq_table = df, metadata = metadata_df,
                     class_= subgenre, verbose=True, method="LR", max_MFF = max_MFF,
                     text_representation = text_representation, problematic_class_values=["n.av."],
                     minimal_value_samples = 2, make_relative=False, under_sample_method="lpc",
                     sampling_times = sampling_times)

        mean_scores_df = pd.DataFrame(coef_df.mean(),columns=[subgenre]).T
        print("mean_scores_df",mean_scores_df.head(3))


        coef_subgenres_df = pd.concat([coef_subgenres_df,mean_scores_df]).fillna(0)

    coef_subgenres_df.to_csv(wdir+str(max_MFF)+"_"+str(sampling_times)+"scores_subgenre.csv",sep="\t")
    
    print(coef_subgenres_df)
    return coef_subgenres_df

def calculate_tendencies_coefs(scores_subgenres_df, freq_table):
    tendencies_coefs_df = pd.DataFrame([scores_subgenres_df.abs().mean(), scores_subgenres_df.abs().std()], index=["mean-coef","std-coef"]).T
    tendencies_coefs_df.sort_values(by="mean-coef", inplace=True, ascending=False)
    
    tendencies_coefs_df["mean_relative_freq"] = freq_table.mean()

    print(tendencies_coefs_df.head())
    return tendencies_coefs_df  
  


def extract_type_of_feature(tendencies_coefs_df, categories = ['@case','@gen', '@lemma', '@mariax', '@mood', '@num', '@person', '@pos', '@possessornum', '@tense', '@type', '@wnlex',
                                                    "@ord_ent",
                                                   #r'(_nr|_ds)',
                                                   "dis-part@",
                                                   "proverbs@",
                                                   r"am\.",
                                                   '_verb', '_adposition', '_pronoun', '_adjective', '_determiner', '_noun', '_number', '_conjunction', '_adverb', '_interjection', '_punctuation'
                                                  ]):
    print("len types", len(categories))
    types_tendencies_coefs_df = tendencies_coefs_df.copy()
    types_tendencies_coefs_df["features"] = types_tendencies_coefs_df.index

    for category in categories:
        column = re.sub(r"[\\\.\(\)\|(.*?)]+",r"", category)+"s"
        print(column)
        types_tendencies_coefs_df[column] = 0
        types_tendencies_coefs_df.loc[types_tendencies_coefs_df["features"].str.contains(category)==True,column] = 1

    types_tendencies_coefs_df["token"] = 1
    types_tendencies_coefs_df.loc[types_tendencies_coefs_df["features"].str.contains(r"[\.@]")==True,"token"] = 0
    
    types_tendencies_coefs_df.rename(columns=lambda x: re.sub(r'^[@_](.*?)',r'\1',x), inplace=True)

    types_tendencies_coefs_df.rename(columns=lambda x: re.sub(r'(.*?)@',r'\1',x), inplace=True)

    types_tendencies_coefs_df.rename(columns={"am":"tags"}, inplace=True)

    
    #types_tendencies_coefs_df.rename(columns=lambda x: re.sub(r'nrds\?',r'nr-ds?',x), inplace=True)
    #types_tendencies_coefs_df.rename(columns={"nrds?":"nr-ds?"}, inplace=True)
    
    return types_tendencies_coefs_df
    


def calculate_welch_test_types_coef(types_tendencies_coefs_df):
    results = []
    for column in types_tendencies_coefs_df.columns.tolist():
        if "mean" not in column and "std" not in column and "features" not in column:
            #print(type_feature_mean_coef_df.loc[type_feature_mean_coef_df[column]==1])

            statistic, pvalue = stats.ttest_ind(
                                types_tendencies_coefs_df.loc[types_tendencies_coefs_df[column]==1]["mean-coef"],
                                types_tendencies_coefs_df.loc[types_tendencies_coefs_df[column]==0]["mean-coef"],
                                equal_var = False,
                                )
            difference = types_tendencies_coefs_df.loc[types_tendencies_coefs_df[column]==1]["mean-coef"].mean() - types_tendencies_coefs_df["mean-coef"].mean()
            
            results.append([column, pvalue, difference])
        else:
            print("ignoring", column)
    results_df = pd.DataFrame(results,columns=["type-feature","p-value","difference"]).fillna(0)
    results_df["percetange-gain"] = ((results_df["difference"] * 100)/types_tendencies_coefs_df["mean-coef"].max()).round(2)


    results_df.sort_values(by="difference",ascending=False, inplace=True)

    return results_df


def compare_narr_ds_features(narr_scores_subgenres_df):
    already_seen = []
    results = []
    for feature in narr_scores_subgenres_df.columns.tolist():
        if feature not in already_seen:
            try:
                neutral_feature = re.sub(r"(.*?)_..",r"\1",feature)

                if "_ds_" in feature:
                    same_feature = re.sub(r"(.*?)_ds",r"\1_nr",feature)
                else:
                    same_feature = re.sub(r"(.*?)_nr",r"\1_ds",feature)
                r, pvalue = stats.pearsonr(narr_scores_subgenres_df[feature],narr_scores_subgenres_df[same_feature])
                already_seen.append(feature)
                already_seen.append(same_feature)
                results.append([neutral_feature, r, pvalue])
            except:
                pass
    results_correlation_features_df = pd.DataFrame(results,columns=["feature","r","pvalue"])
    results_correlation_features_df["absolute-r"] = results_correlation_features_df["r"].abs()
    results_correlation_features_df.sort_values(by="absolute-r",inplace=True)
    results_correlation_features_df.head()
    return results_correlation_features_df
