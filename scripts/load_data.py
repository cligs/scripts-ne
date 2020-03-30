# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 08:19:24 2017

@author: jose

This script takes a path in which there are texts files, it reads it and converts it to a MFW dataframe

Example of how to use it:

import sys
import os
sys.path.append(os.path.abspath("/home/jose/Dropbox/MTB/investigacion/mytoolbox/"))
sys.path.append(os.path.abspath("/home/jose/"))
from reading_robot import load_data
corpus = load_data.load_corpus(wdir ="/home/jose/Dropbox/MTB/investigacion/mytoolbox/reading_robot/data/")
metadata = load_metadata(wdir ="/home/jose/Dropbox/MTB/investigacion/mytoolbox/reading_robot/data/",sep = ",")

"""

from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import os
import glob
import re
from scipy import stats
from collections import Counter
import sys
import os
import numpy as np
import itertools
sys.path.append(os.path.abspath("/home/jose/Dropbox/Doktorarbeit/"))
from reading_robot import text2features


def load_corpus_metadata(wdir, wsdir="corpus/", sep="\t", verbose=True, min_MFF=0, max_MFFs=[5000],
                         freq_table="freq_table_raw__T.parquet", metadata="metadata_beta-opt-obl-structure.csv",
                         does_relative=True):
    max_MFF = np.array(max_MFFs).max()
    if (type(freq_table) == str):
        # We load the corpus dataframe (texts as rows, words as columns)
        cut_raw_features = load_corpus(wdir=wdir, wsdir=wsdir, freq_table=freq_table, sep=sep, verbose=verbose,
                                       min_MFF=min_MFF, max_MFF=max_MFF)
    elif (type(freq_table) == pd.DataFrame):
        cut_raw_features = cut_corpus(freq_table, min_MFF=min_MFF, max_MFF=max_MFF)
    else:
        print("there is something wrong with the corpus")
    print(cut_raw_features.shape) if verbose == True else 0

    if type(metadata) == str:
        # We load the metadata
        metadata = load_metadata(wdir=wdir, metadata_table=metadata, sep=sep, verbose=verbose)

    print(metadata.shape) if verbose == True else 0
    if does_relative == True:
        cut_raw_features = text2features.calculate_relative_frequencies(cut_raw_features)

    return cut_raw_features, metadata


def load_corpus(wdir, wsdir="corpus/", freq_table="", sep="\t", verbose=True, min_MFF=0, max_MFF=5000, lowercase=True,
                keep_punctuation=False, text_format="txt"):
    """
    * wdir
    * wsdir = "corpus/"
    * freq_table: string or dataframe
    * sep = "\t"
    * verbose = True
    * min_MFF = 0
    * max_MFF = 5000 or False
    * lowercase = True
    * keep_punctuation = False
    * text_format = "txt"
    """
    print("* Loading corpus")
    # We load the corpus dataframe (texts as rows, words as columns)
    if freq_table == "" or freq_table == " ":
        print("* Loading texts from folder")
        corpus = corpus2df(wdir=wdir, wsdir=wsdir, verbose=verbose, min_MFF=min_MFF, max_MFF=max_MFF,
                           lowercase=lowercase, text_format=text_format, keep_punctuation=keep_punctuation)
    else:
        print("* Opening table")
        corpus = open_freq_table(wdir=wdir, min_MFF=min_MFF, max_MFF=max_MFF, freq_table=freq_table)
    print("corpus' shape: \t", corpus.shape)
    return corpus


def corpus2df(wdir, wsdir="corpus/", text_format="txt", keep_punctuation=False, save_files=True, min_MFF=0,
              max_MFF=5000, verbose=True, normalized=True, lowercase=True, freq_table_name="freq_table_raw"):
    """
    Main function. It takes a wdir with texts and make a dataframe of relative frequencies.
    """
    print("reading texts!")

    # First, we open the files and convert them into a dictionary (file-names as keys, texts as values)
    corpus = corpus2dict(wdir, wsdir, text_format, verbose=verbose)

    print("corpus with: \t", len(corpus), " samples ")

    # Second, we convert it to a dataframe with the relative frequency
    corpus = corpusdict2df(wdir, corpus, keep_punctuation, save_files, verbose=verbose, normalized=normalized,
                           lowercase=lowercase, freq_table_name=freq_table_name)
    # print(corpus.index)

    print(max_MFF)
    if max_MFF != False:
        print("cuting features of corpus!")

        # Third, we take only as much features as we want
        corpus = cut_corpus(corpus, min_MFF, max_MFF)
        print("final corpus shape: ", corpus.shape)

    if save_files == True:
        if max_MFF == False:
            max_MFF_str = ""
        else:
            max_MFF_str = str(max_MFF)

        print("saving")
        if corpus.shape[0] > 100:

            corpus.T.to_parquet(wdir + wsdir[:-1] + "_" + freq_table_name + "_" + max_MFF_str + "_T.parquet")
        else:
            corpus.to_csv(wdir + wsdir[:-1] + "_" + freq_table_name + "_" + max_MFF_str + ".csv", sep='\t',
                          encoding='utf-8', index=True)

    return corpus


def corpus2dict(wdir, wsdir, text_format, verbose=True):  # , case_sensitive, keep_puntuaction):
    """
    This function creates an empty dictionary and adds all the texts in a corpus (file-name as key, text as value)
    """
    # It creates an empty dictionary in which the names of the files are keys and the texts are the values
    corpus = {}
    # It iterates over the files of the folder
    print(wdir + wsdir + "*." + text_format)

    for doc in glob.glob(wdir + wsdir + "*." + text_format):
        print(doc)
        # It opens the files and extract file-names and the text
        file_name, text = open_file(doc, verbose=verbose)
        # It adds both thing into the corpus-dictionary
        corpus[file_name] = text
    # It gives back the corpus-dictionary
    return corpus


def open_file(doc, verbose=True):
    """
    This function opens a file and gives back the name of the file and the text
    """
    file_name = os.path.splitext(os.path.split(doc)[1])[0]
    # print(file_name) if verbose == True else 0
    with open(doc, "r", errors="replace", encoding="utf-8") as fin:
        text = fin.read()
        fin.close()
    return file_name, text


def corpusdict2df(wdir, corpus, keep_punctuation, save_files=True, normalized=True, verbose=True, lowercase=True,
                  freq_table_name="freq_table_raw"):
    """
    This function takes the corpus-dictionary and converts it into a dataframe
    """
    if len(corpus) > 5000:
        max_features = 10000
        print("The corpus has too many rows, we are *only* going to use 10,000 columns (which should be enough)")
    else:
        max_features = 99999999999
    # We define how we want the tokeniser
    if keep_punctuation == True:
        vec = CountVectorizer(
            token_pattern=r"(?u)\b[\w_@]+\b|[¶\(»\]\?\.\–\!’•\|“\>\)\-\—\:\}\*\&…¿\/=¡\"\'·+\{\#\[;­,«~]",
            lowercase=lowercase,
            max_features = max_features)
    elif keep_punctuation == "features":
        vec = CountVectorizer(
            token_pattern=r"[\w_@¶\(»\]\?\.\–\!’•\|“\>\)\-\—\:\}\*\&…¿\/=¡\"\'·+\{\#\[;\­,«~]+",
            lowercase=lowercase,
            max_features = max_features)
    elif keep_punctuation == "tags":
        vec = CountVectorizer(
            token_pattern=r"(?u)\b[<\w>]+\b|[¶\(»\]\?\.\–\!’•\|“\>\)\-\—\:\}\*\&…¿\/=¡\"\'·+\{\#\[;­,«~]",
            lowercase=lowercase,
            max_features = max_features)
    else:
        vec = CountVectorizer(token_pattern=r"(?u)\b\w+\b", lowercase=lowercase, max_features = max_features)
    # The tokenizer is used
    texts = vec.fit_transform(corpus.values())

    corpus = pd.DataFrame(texts.toarray(), columns=vec.get_feature_names(), index=corpus.keys())

    print(type(corpus))
    print(corpus.head())
    # corpus-dataframe is sorted using the index (i.e. the names of the files)
    corpus = corpus.sort_index(axis="index")

    # It saves the file
    corpus = corpus.reindex(corpus.mean().sort_values(ascending=False).index, axis=1)

    if save_files == True:

        thefile = open(wdir + 'mff.txt', 'w')
        for item in corpus.columns.tolist():
            thefile.write("%s\n" % item)

    print(corpus.head() if verbose == True else 0)
    return corpus


def load_metadata(wdir, metadata_table="metadata.csv", sep="\t", verbose=True):
    metadata_df = pd.read_csv(wdir + metadata_table, encoding="utf-8", sep=sep, index_col=0)
    print("metadata and class shape: \t", metadata_df.shape)
    return metadata_df


def cut_corpus(corpus, min_MFF=0, max_MFF=5000, sort_by="median"):
    """
    This function cut the amount of MFF to be used
    """
    if sort_by == "median":
        corpus = corpus.reindex(corpus.median().sort_values(ascending=False).index, axis=1)
    if max_MFF == False:
        max_MFF = ""
    corpus = corpus.iloc[:, min_MFF: max_MFF]
    # Deleting colunms that are only 0
    corpus = corpus.loc[:, (corpus != 0).any(axis=0)]

    return corpus


def open_freq_table(wdir, min_MFF, max_MFF, freq_table):
    freq_table_format = os.path.splitext(freq_table)[1]
    if "csv" in freq_table_format:
        corpus = pd.read_csv(wdir + freq_table, encoding="utf-8", sep="\t", index_col=0)
    elif "pqt" in freq_table_format or "parquet" in freq_table_format:
        corpus = pd.read_parquet(wdir + freq_table)
        if "_T" in freq_table:
            corpus = corpus.T

    print("corpus shape: ", corpus.shape)

    if max_MFF != False:
        print("cuting features of corpus!")
        corpus = cut_corpus(corpus, min_MFF, max_MFF)
    print("corpus shape: ", corpus.shape)

    return corpus


def merge_dfs(df1, df2, append_column_1 = "", append_column_2 = "" ):
    if append_column_1 != "":
        df1 = df1.rename(columns={element: re.sub(r'(.+)',append_column_1+r'_\1', element, flags = re.MULTILINE) for element  in df1.columns.tolist()}).copy()
    if append_column_2 != "":
        df2 = df2.rename(columns={element: re.sub(r'(.+)',append_column_2+r'_\1', element, flags = re.MULTILINE) for element  in df2.columns.tolist()}).copy()
    
    merged_df = pd.merge(df1,df2, left_index = True, right_index = True)
    ordered_columns = [element for tuple_ in list(itertools.zip_longest(df1.columns.tolist(),df2.columns.tolist())) for element in tuple_ if element != None]

    merged_df = merged_df[ordered_columns]

    return merged_df
