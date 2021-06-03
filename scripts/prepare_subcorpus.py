# -*- coding: utf-8 -*-

"""
# Select a subset of text files from a larger text collection. The main function is copy_subset.

"""
import pandas as pd
from collections import Counter

def categorical_sampling(metadata_df, categorical_sample_filter_column_st, categorical_sample_filter_value_int, identifier_st):
        print(categorical_sample_filter_column_st, categorical_sample_filter_value_int)
        
        approved_values_lt = [value_frequency[0] for value_frequency in Counter(metadata_df[categorical_sample_filter_column_st] ).most_common() if value_frequency[1] >= categorical_sample_filter_value_int]
        
        metadata_df = metadata_df[metadata_df[categorical_sample_filter_column_st].isin(approved_values_lt)]
        
        approved_id_texts_lt = [item for feature in approved_values_lt for item in list(metadata_df[metadata_df[categorical_sample_filter_column_st] == feature].sample(n = categorical_sample_filter_value_int)[identifier_st]) ]

        metadata_df = metadata_df[metadata_df[identifier_st].isin(approved_id_texts_lt)]

        return metadata_df

def prepare_subcorpus(metadata_df, categorical_filters_lt = [{"genre":["novela"]}],
                      categorical_sample_filters_lt = [["author.name",3]],
                      identifier_st = "idno"):
    ## Filter the metadata_df table by one or several criteria
    for categorical_filter_dc in categorical_filters_lt:
        for filter_key_st, filter_values_lt in categorical_filter_dc.items():
            print("categorical filter", filter_key_st)
            metadata_df = metadata_df[metadata_df[filter_key_st].isin(filter_values_lt)]
        print(metadata_df.shape)

    ## Sample the metadata using a category and a fixed value
    for categorical_sample_filter_lt in categorical_sample_filters_lt:
        categorical_sample_filter_column_st = categorical_sample_filter_lt[0]
        categorical_sample_filter_value_int = categorical_sample_filter_lt[1]
        metadata_df = categorical_sampling(metadata_df, categorical_sample_filter_column_st, categorical_sample_filter_value_int, identifier_st)
    return metadata_df

