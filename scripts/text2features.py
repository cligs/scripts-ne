# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 09:15:17 2017

@author: jose 

This script converts a Bag of Words model into something different:

# In Spyder

To develope this script in Spyder you need a corpus:

import pandas as pd
import sys
import os
sys.path.append(os.path.abspath("/home/jose/Dropbox/Doktorarbeit/"))
from reading_robot import load_data
corpus = load_data.corpus2table(
wdir = "/home/jose/cligs/experiments/20170725 reading robot/" ,
wsdir = "corpus/",
verbose = True, min_MFF = 0, max_MFF = 5000,  normalized = False)

"""
from sklearn.feature_extraction.text import TfidfTransformer
import pandas as pd
import numpy as np
import re


def from_chapters_df_make_novels_parameter_df(chapters_df, parameter="mean"):
    list_novels = list(set(
        [re.sub(r"^(.*?)_.*?$", r"\1", id_chapter, flags=re.MULTILINE) for id_chapter in chapters_df.index.tolist()]))
    print("amount of novels: ", len(list_novels))

    chapters_rel_df = calculate_relative_frequencies(chapters_df).copy()

    novels_rel_df = pd.DataFrame(index=list_novels, columns=chapters_df.columns.tolist())

    for novel in list_novels:
        if parameter == "mean":
            novels_rel_df.loc[novel] = chapters_rel_df.loc[novel + "_chap1":novel + "_chap999"].mean()
        elif parameter == "std":
            novels_rel_df.loc[novel] = chapters_rel_df.loc[novel + "_chap1":novel + "_chap999"].std()
        elif parameter == "median":
            novels_rel_df.loc[novel] = chapters_rel_df.loc[novel + "_chap1":novel + "_chap999"].median()
        elif parameter == "sum":
            novels_rel_df.loc[novel] = chapters_rel_df.loc[novel + "_chap1":novel + "_chap999"].sum()

    novels_rel_df.sort_index(inplace=True)

    return novels_rel_df


def from_chapters_df_make_author_zscores_df(chapters_df, metadata, parameter="mean"):
    chapters_df = calculate_relative_frequencies(chapters_df.copy())

    novels_df = from_chapters_df_make_novels_parameter_df(chapters_df, parameter = parameter)

    print(novels_df.head(2))
    novels_author_df = novels_df.copy()

    for author in list(set(metadata["author.name"].tolist())):
        chapters_by_author = [chapter for idno in metadata.loc[metadata["author.name"] == author]["idno"].tolist() for chapter in chapters_df.loc[idno + "_chap1":idno + "_chap999"].index.tolist()]

        novels_by_author = list(set(metadata.loc[metadata["author.name"] == author]["idno"].tolist()))

        print(author, len(chapters_by_author), len(novels_by_author))

        novels_author_df.loc[novels_by_author] = (novels_df.loc[novels_by_author] - chapters_df.loc[chapters_by_author].mean()) / (chapters_df.loc[chapters_by_author].std() + 0.00001)
    print(novels_author_df.head())
    novels_author_df.sort_index(inplace=True)
    return novels_author_df


def delete_textual_hapax(df):
    df.drop([col for col, val in (df > 0).astype(int).sum().iteritems() if val < 2], axis=1, inplace=True)
    return df


def replace_per_row_mode(df):
    for i in range(df.shape[0]):
        df.iloc[i].fillna(df.iloc[i].mode()[0])
    return df


def calculate_tfidf(corpus, log10_tf=False):
    idf = np.log((corpus.shape[0]) / ((corpus[corpus > 0].count()) + 1))
    if log10_tf == True:
        corpus_transformed = calculate_log(corpus)
    else:
        corpus_transformed = corpus.copy()
    corpus = corpus_transformed * idf

    return corpus


def calculate_tfidf_sklearn(corpus, norm="l2"):
    tf_transformer = TfidfTransformer(norm=norm).fit(corpus)
    X_train_tf = tf_transformer.transform(corpus)
    corpus = pd.DataFrame(X_train_tf.toarray(), columns=corpus.columns, index=corpus.index)
    return corpus


def calculate_relative_frequencies(corpus):
    corpus = corpus.loc[:].div(corpus.sum(axis='columns'), axis="index")
    return corpus


def calculate_zscore(corpus):
    means = corpus.mean(axis="index")
    stds = corpus.std(axis="index")
    corpus = (corpus - means) / stds
    # print(corpus.head())
    return corpus


def calculate_lang_zscore(corpus, relative=True, relative_corde = True,
                          corde_dir="../../ne_data/chap3_3/corde_7000_1860-1960.pqt"):
    corde = pd.read_parquet(corde_dir)

    corpus_lang_zscore = corpus.copy()

    if relative == True:
        corde = calculate_relative_frequencies(corde)
        corpus = calculate_relative_frequencies(corpus)
    else:
        if relative_corde == True:
            corde = calculate_relative_frequencies(corde)


    means = corpus_lang_zscore.mean(axis="index")
    stds = corpus_lang_zscore.std(axis="index")
    corpus_lang_zscore = (corpus_lang_zscore - means) / stds

    for column in corpus.columns.tolist():
        if column in corde.columns.tolist():
            means = corde[column].mean(axis="index")
            stds = corde[column].std(axis="index")
            corpus_lang_zscore[column] = (corpus[column] - means) / stds

    return corpus_lang_zscore


def calculate_qscores(corpus):
    medians = corpus.median(axis="index")
    iqrs = (corpus.quantile(q=0.75, axis="index") - corpus.quantile(q=0.25, axis="index"))
    corpus = (corpus - medians) / (iqrs + 0.00001)
    return corpus


def convert_inf_to_int(corpus):
    corpus = corpus.replace([[np.inf]],
                            round(float(np.array([value for value in corpus.max() if value != np.inf]).max()), 4))
    corpus = corpus.replace([[-np.inf]],
                            #round(float(np.array([value for value in corpus.min() if value != -np.inf]).min()), 4))
                            float(np.array([value for row in corpus.values for value in row if value != -np.inf]).min()))

    return corpus


def calculate_log(corpus):
    corpus = np.log(corpus)
    corpus = convert_inf_to_int(corpus)
    return corpus


def calculate_log2(corpus):
    corpus = np.log2(corpus)
    corpus = convert_inf_to_int(corpus)
    return corpus


def calculate_log10(corpus):
    corpus = np.log10(corpus)
    corpus = convert_inf_to_int(corpus)
    return corpus

def calculate_log1000(corpus):
    corpus = np.log(corpus)/ np.log(10000)
    corpus = convert_inf_to_int(corpus)
    return corpus

def calculate_log10zscores(corpus):
    corpus = np.log10(corpus)
    corpus = convert_inf_to_int(corpus)

    means = corpus.mean(axis="index")
    stds = corpus.std(axis="index")
    corpus = (corpus - means) / stds
    return corpus


def calculate_zscoreslog(corpus):
    means = corpus.mean(axis="index")
    stds = corpus.std(axis="index")
    corpus = (corpus - means) / stds
    corpus = np.log(corpus).fillna(0)
    return corpus


def calculate_logqscores(corpus):
    corpus = np.log(corpus)
    corpus = convert_inf_to_int(corpus)
    medians = corpus.median(axis="index")
    iqrs = (corpus.quantile(q=0.75, axis="index") - corpus.quantile(q=0.25, axis="index"))
    corpus = (corpus - medians) / (iqrs + 0.00001)
    return corpus


def calculate_binary(corpus):
    corpus = (corpus > 0).astype(int)
    return corpus


from sklearn.preprocessing import MinMaxScaler


def calculate_minmax(corpus):
    scaler = MinMaxScaler()
    min_max_arrays = scaler.fit_transform(corpus)
    corpus = pd.DataFrame(min_max_arrays, index=corpus.index, columns=corpus.columns)
    return corpus


def choose_features(corpus, text_representation, verbose=False):
    if text_representation in ["raw", "term frequency", "relative-mean", "relative-std","authorial-zscore","relative-median","authorial-zscore-rel-log"]:
        # If we need to convert the corpus into something different, it would be here
        data = corpus
    elif text_representation == "relative":
        data = calculate_relative_frequencies(corpus)
    elif text_representation in ["tfidf", "tf-idf", "tf-idf-norm", "tfidf-norm"]:
        data = calculate_tfidf(corpus)
    elif text_representation == "zscores":
        data = calculate_zscore(corpus)
    elif text_representation == "qscores":
        data = calculate_qscores(corpus)
    elif text_representation in ["logzscores", "log-zscores"]:
        data = calculate_log(corpus)
        data = calculate_zscore(data)
    elif text_representation in ["rellogzscores", "rel-log-zscores"]:
        data = calculate_relative_frequencies(corpus)
        data = calculate_log(data)
        data = calculate_zscore(data)
    elif text_representation in ["rel-log10-zscores"]:
        data = calculate_relative_frequencies(corpus)
        data = calculate_log10(data)
        data = calculate_zscore(data)

    elif text_representation in ["rel-zscores", "relzscores"]:
        data = calculate_relative_frequencies(corpus)
        data = calculate_zscore(data)
    elif text_representation == "logqscores":
        data = calculate_logqscores(corpus)
    elif text_representation == "zscoreslog":
        data = calculate_zscoreslog(corpus)
    elif text_representation == "log":
        data = calculate_log(corpus)
    elif text_representation == "log2":
        data = calculate_log2(corpus)
    elif text_representation == "rel-log":
        data = calculate_relative_frequencies(corpus)
        data = calculate_log(data)
    elif text_representation == "log10":
        data = calculate_log10(corpus)
    elif text_representation == "rel-log10":
        data = calculate_relative_frequencies(corpus)
        data = calculate_log10(data)
    elif text_representation == "rel-log1000":
        data = calculate_relative_frequencies(corpus)
        data = calculate_log1000(data)
    elif text_representation == "log1000":
        data = calculate_log1000(corpus)
    elif text_representation == "minmax":
        data = calculate_minmax(corpus)
    elif text_representation == "binary":
        data = calculate_binary(corpus)
    elif text_representation in ["log10zscores","log10-zscores"]:
        data = calculate_log10zscores(corpus)
    elif text_representation == "rel-lang-zscore":
        data = calculate_lang_zscore(corpus)

    elif text_representation == "lang-zscore":
        data = calculate_lang_zscore(corpus)
    elif text_representation == "log-tfidf":
        data = calculate_tfidf(corpus, log10_tf=True)
    elif text_representation == "rel-tfidf":
        data = calculate_relative_frequencies(corpus)
        data = calculate_tfidf(data)
    elif text_representation == "rel-tfidf-zscores":
        data = calculate_relative_frequencies(corpus)
        data = calculate_tfidf(data)
        data = calculate_zscore(data)
    elif text_representation == "tfidf-zscores":
        data = calculate_tfidf(corpus)
        data = calculate_zscore(data)
    elif text_representation == "rel-zscores-tfidf":
        data = calculate_relative_frequencies(corpus)
        data = calculate_zscore(data)
        data = calculate_tfidf(data)
    elif text_representation == "rel-log10-tfidf":
        data = calculate_relative_frequencies(corpus)
        data = calculate_tfidf(data, log10_tf=True)
    elif text_representation == "rel-log10-tfidf-zscores":
        data = calculate_relative_frequencies(corpus)
        data = calculate_tfidf(data, log10_tf=True)
        data = calculate_zscore(data)

    elif text_representation == "rel-tfidf-log10-zscores":
        data = calculate_relative_frequencies(corpus)
        data = calculate_tfidf(data) + 1
        data = calculate_log10(data)
        data = calculate_zscore(data)
    elif text_representation == "rel-tfidf-log10":
        data = calculate_relative_frequencies(corpus)
        data = calculate_tfidf(data) + 1
        data = calculate_log10(data)
    elif text_representation == "rel-log10-zscores-tfidf":
        data = calculate_relative_frequencies(corpus)
        data = calculate_log(data)
        data = calculate_zscore(data)
        data = calculate_tfidf(data)
    elif text_representation == "rel-log10-tfidf-zscores":
        data = calculate_relative_frequencies(corpus)
        data = calculate_tfidf(data, log10_tf=True)
        data = calculate_zscore(data)
    else:
        print("your transformation of the data " , text_representation, " is wrong. Bug coming!")

    print(data.head()) if verbose == True else 0

    print("textual representation: ", text_representation)
    print("Columns that are empty: ", data.columns[data.isnull().any()].tolist())

    return data


from scipy.stats import pearsonr
from scipy import stats


def describe_transformation(corpus, transformations, MFWs=2000, text_name1="Pazos",
                            make_relative=True, use_other_corpus_to_compare=False, comparing_corpus=""):
    corpus = corpus.iloc[:, 0:MFWs].copy()

    if use_other_corpus_to_compare == False:
        comparing_corpus = corpus.copy()
    else:
        comparing_corpus = comparing_corpus.iloc[:, 0:MFWs].copy()

    print(corpus.shape)
    results_list = []
    for transformation in transformations:
        print(transformation)
        if transformation not in ["raw", "term frequency", "relative"] and make_relative == True:
            transformed_corpus = choose_features(corpus, "relative")

        else:
            transformed_corpus = corpus.copy()

        transformed_corpus = choose_features(transformed_corpus, transformation)
        mean = transformed_corpus.loc[text_name1].mean()
        max_ = transformed_corpus.loc[text_name1].max()
        min_ = transformed_corpus.loc[text_name1].min()
        std = transformed_corpus.loc[text_name1].std()
        median = transformed_corpus.loc[text_name1].median()
        iqr = stats.iqr(transformed_corpus.loc[text_name1])
        pearsonsr_results = pearsonr(comparing_corpus.loc[text_name1], transformed_corpus.loc[text_name1])
        pearsonsr, r_pvalue = pearsonsr_results[0], pearsonsr_results[1]

        gaussian_results = stats.normaltest(transformed_corpus.loc[text_name1])
        normal_st, normal_pvalue = gaussian_results[0], gaussian_results[1]
        
        skew = stats.skew(transformed_corpus.loc[text_name1])


        results_list.append([transformation, min_, max_, mean, std, median, iqr, pearsonsr, r_pvalue, normal_st, normal_pvalue, skew])

    results_trasnformation_df = pd.DataFrame(results_list,
                                             columns=["transformation", "min", "max", "mean", "std", "median", "IQR",
                                                      "Pearson's R", "R p-value","Gaussian Test","Gaussian p-value", "skew"]).fillna(0)
    results_trasnformation_df.index = results_trasnformation_df["transformation"]
    del results_trasnformation_df["transformation"]
    print(corpus.head())
    return results_trasnformation_df


