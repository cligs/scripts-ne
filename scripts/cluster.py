# -*- coding: utf-8 -*-
"""
Created on Tue May 14 07:11:03 2019

@author: jose
"""
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
warnings.filterwarnings("once", category=Warning)


import sys
import os
import datetime
import pandas as pd
from collections import Counter
sys.path.append(os.path.abspath("/home/jose/Dropbox/Doktorarbeit/"))


from reading_robot import text2features, load_data
from sklearn.metrics import adjusted_rand_score
import numpy as np
from scipy import stats as stats


def sample_unique_text_by_class(data_df, metadata_df, class_):
    sampled_metadata_df = metadata_df.sample(frac=1).groupby([class_]).head(1)
    
    sampled_data_df = data_df.loc[sampled_metadata_df.index]
    return sampled_data_df, sampled_metadata_df
    
def evaluate_cluster(wdir, freq_table_df, metadata_df, ground_truths = ["author.name","decade","subgenre.cligs.important"],
            methods = ["KMeans"], min_MFF = 0, max_MFFs = [5000], text_representations = ["rel-zscores"],
            ns_clusters = [30], under_sample_method = "author.name", sampling_times = 10, method_evaluation = "ARI",
            
           ):
    # A list for the results is initialized empty
    results_lt = []

    # Iterate over representation or trasnformation of the data    
    for text_representation in text_representations:
        document_data_model_df = text2features.choose_features(freq_table_df, text_representation)
        # Iterate over amount of MFFs
        for MFW in max_MFFs:
            print(MFW)
            document_data_model_cut_df = load_data.cut_corpus(document_data_model_df, min_MFF = min_MFF, max_MFF = MFW)
            try:
                print("first columns ", document_data_model_cut_df.columns.tolist()[0:5])
                print("last columns ", document_data_model_cut_df.columns.tolist()[-5:])
            except:
                print("first columns ", document_data_model_cut_df.columns.tolist()[0:1])
                print("last columns ", document_data_model_cut_df.columns.tolist()[-1:])

            # Iterate over clustering algorithms            
            for method in methods:
                print(method)
                    
                # This if takes care of the amount of subclusters for those algorithms that need to be defined or not
                if method not in ["KMeans","SpectralClustering","AgglomerativeClustering"]:
                    print(method)
                    actual_ns_clusters = ["-"]
                else:
                    actual_ns_clusters = ns_clusters 

                # Iterate over number of cluster (this is only relevant for the algorithms that need to be initialize with a number of subclusters; for the rest I pass 30, but they will decide the number)
                for n_clusters in actual_ns_clusters:
                    print(n_clusters)

                    # Iterate over sampling times:
                    for i in range(sampling_times):

                        # Possibility of undersampling taking only one text per author (or any other class):
                        if under_sample_method in ["author.name","authorial"]:
                            sampled_data_df, sampled_metadata_df = sample_unique_text_by_class(document_data_model_cut_df, metadata_df, class_ = "author.name")
                        else:
                            sampled_data_df, sampled_metadata_df = document_data_model_cut_df, metadata_df
                        
                        try:
                            # Make labels and take the real number of subclusters:
                            labels = choose_cluster_algorithm(method, n_clusters = n_clusters).fit(sampled_data_df).labels_
                            
                            n_clusters = len(list(set(labels)))

                            # Evaluate with 
                            for ground_truth in ground_truths:
    
                                evaluation = evalute_clustering(sampled_metadata_df[ground_truth], labels, method = method_evaluation)
                                
                                # Add everything to the list of the results
                                results_lt.append([ground_truth, evaluation, text_representation,method, n_clusters, MFW, method_evaluation,sampled_data_df.shape[0]])

                        except:
                            print("problem with ", text_representation, method, ground_truths, n_clusters)


    # Convert the list into a dataframe, sort, clean...                            
    results_df = pd.DataFrame(results_lt, columns=["ground_truth", "evaluation", "text_representation","method", "n_clusters", "MFW", "method_evaluation","sample_size"])
    results_df = results_df.sample(frac=1).sort_values(by=["evaluation"], ascending=[False])

    # Save the results 
    results_file = "results"+"_"+ "-".join(ground_truths)+"_"+ "-".join(methods)+"_"+ "-".join(str(x) for x in max_MFFs)+"_" +"-".join(text_representations)
    if len(results_file) > 100:
        results_file = "results_"+str(datetime.datetime.now().year)+str(datetime.datetime.now().month)+str(datetime.datetime.now().day)+str(datetime.datetime.now().hour)+str(datetime.datetime.now().minute)+str(datetime.datetime.now().second)
    print(results_file)
    results_df.to_csv(wdir + results_file+".csv", sep = "\t")

    return results_df
                            
def choose_cluster_algorithm(method = "KMeans", n_clusters=20):
    
    if method == "KMeans":
        from sklearn.cluster import KMeans
        clustering_algorithm = KMeans(n_clusters = n_clusters)

    elif method == "AffinityPropagation":
        from sklearn.cluster import AffinityPropagation
        clustering_algorithm = AffinityPropagation()
        
    elif method == "MeanShift":
        from sklearn.cluster import MeanShift
        clustering_algorithm = MeanShift()
        
    elif method == "SpectralClustering":
        from sklearn.cluster import SpectralClustering
        clustering_algorithm = SpectralClustering(n_clusters = n_clusters)
    
    elif method == "AgglomerativeClustering":
        from sklearn.cluster import AgglomerativeClustering
        clustering_algorithm = AgglomerativeClustering(n_clusters = n_clusters)

    elif method == "DBSCAN":
    
        from sklearn.cluster import DBSCAN
        clustering_algorithm = DBSCAN()

    elif method == "Birch":
    
        from sklearn.cluster import Birch    
        clustering_algorithm = Birch(n_clusters = n_clusters)

    """
    elif method == "OPTICS":
        
        from sklearn.cluster import OPTICS
        clustering_algorithm = OPTICS()
    """
    

    return clustering_algorithm


def evalute_clustering(labels_true, labels_pred, method="ARI", ):

    from sklearn.metrics import adjusted_rand_score
    ari = adjusted_rand_score(labels_true, labels_pred)
    return ari


def create_clusters(wdir, freq_table_df, methods = ["KMeans"], min_MFF = 0,
                    max_MFFs = [5000], text_representations = ["rel-zscores"], ns_clusters = [2],
                    sampling_times = 10, 
                   ):
    i = 0
     
    # Two lists for the results is initialized empty
    clustering_results_dict = {}
    parameters_results_dict = {}

    # Iterate over representation or trasnformation of the data    
    for text_representation in text_representations:
        document_data_model_df = text2features.choose_features(freq_table_df, text_representation)
        # Iterate over amount of MFFs
        for MFW in max_MFFs:
            print(MFW)
            document_data_model_cut_df = load_data.cut_corpus(document_data_model_df, min_MFF = min_MFF, max_MFF = MFW)
            print(document_data_model_cut_df.head())
            # Iterate over clustering algorithms            
            for method in methods:
                print(method)
                    
                # This if takes care of the amount of subclusters for those algorithms that need to be defined or not
                if method not in ["KMeans","SpectralClustering","AgglomerativeClustering"]:
                    print(method)
                    actual_ns_clusters = ["-"]
                else:
                    actual_ns_clusters = ns_clusters 

                # Iterate over number of cluster (this is only relevant for the algorithms that need to be initialize with a number of subclusters; for the rest I pass 30, but they will decide the number)
                for n_clusters in actual_ns_clusters:
                    print(n_clusters)

                    # Iterate over sampling times:
                    for j in range(sampling_times):
                        

                        try:
                            # Make labels and take the real number of subclusters:
                            labels_lt = choose_cluster_algorithm(method, n_clusters = n_clusters).fit(document_data_model_cut_df).labels_

                            n_clusters = len(list(set(labels_lt)))
                            
                            clustering_results_dict["cluster_"+str(i)] = labels_lt
                            parameters_results_dict["cluster_"+str(i)] = [text_representation, MFW, method, n_clusters, j]

                            
                        except:
                            print("problem with ", text_representation, method, n_clusters)
                        i += 1
    clustering_results_df =  pd.DataFrame.from_dict(clustering_results_dict)
    parameters_results_df =  pd.DataFrame.from_dict(parameters_results_dict)
    
    print(clustering_results_df.shape)
    print(freq_table_df.shape)
    clustering_results_df.index = freq_table_df.index
    
    return clustering_results_df, parameters_results_df



def compare_clusters_and_evaluate_with_metadata(clustering_results_df, metadata_df, classes_lt = ["author.name","decade"], subgenres_lt = ['autobiografía', 'aventura', 'biografía', 'costumbrista', 'diálogo', 'educación', 'episodio nacional', 'erótica', 'espiritual', 'fantástico', 'filosófica', 'greguería', 'guerra', 'histórica', 'humor', 'memorias', 'modernista', 'naturalista', 'nivola', 'poética', 'psicológica', 'realista', 'social', 'ficción-literaria']):
    clustering_results_df = clustering_results_df.loc[:, (clustering_results_df != 0).any(axis=0)]
    
    
    result_lt = []

    for column_1 in clustering_results_df.columns.tolist():
        scores = []
        for column_2 in clustering_results_df.columns.tolist():
            if column_1 != column_2:
                score = adjusted_rand_score(clustering_results_df[column_1],clustering_results_df[column_2])
                scores.append(score)

        scores = np.array(scores)

        ari_with_author = np.array([adjusted_rand_score(clustering_results_df[column_1],pd.get_dummies(metadata_df["author.name"])[author_st]) for author_st in pd.get_dummies(metadata_df["author.name"]).columns.tolist()  ]).max()
        ari_with_decade = np.array([adjusted_rand_score(clustering_results_df[column_1],pd.get_dummies(metadata_df["decade"])[decade_st]) for decade_st in pd.get_dummies(metadata_df["decade"]).columns.tolist()  ]).max()
        ari_with_subgenres = np.array([adjusted_rand_score(clustering_results_df[column_1],metadata_df[subgenre_st]) for subgenre_st in subgenres_lt ]).max()
        

        result_lt.append([column_1, np.mean(scores), np.median(scores), np.std(scores), ari_with_author, ari_with_decade, ari_with_subgenres])
    results_df = pd.DataFrame(result_lt, columns = ["cluster","mean","median","std", "max ARI author", "max ARI decade", "max ARI subgenres"])

    results_df["opt-score-median"] = results_df["median"] - ((results_df["max ARI author"] + results_df["max ARI decade"] + results_df["max ARI subgenres"] ) /3 )
    results_df["opt-score-mean"] = results_df["mean"] - ((results_df["max ARI author"] + results_df["max ARI decade"] + results_df["max ARI subgenres"] ) /3 )

    
    results_df.sort_values(by=["median","cluster"],inplace=True,ascending=[False,True])
    
    print(results_df.iloc[0])
    return results_df


def compare_cluster_with_metadata(cluster_arr, metadata_df, classes_lt, get_dummies=False):
    if get_dummies == True:
        metadata_df = pd.get_dummies(metadata_df[classes_lt])
        classes_lt = metadata_df.columns.tolist()
        
    results_lt = []

    for class_st in classes_lt:
        score_fl = adjusted_rand_score(cluster_arr, metadata_df[class_st])

        results_lt.append([class_st, score_fl, "ARI"])
    results_df = pd.DataFrame(results_lt, columns = ["class","score","measure"])
    results_df.sort_values(by="score", ascending=False,inplace=True)
    return results_df



def define_cluster_with_metadata(metadata_df, cluster_str):
    results_lt = []
    for column_st in metadata_df.columns.tolist():
        try:
            median_fl = metadata_df.loc[metadata_df[cluster_str] == 1][column_st].median()
            mean_fl = metadata_df.loc[metadata_df[cluster_str] == 1][column_st].mean()
            std_fl = metadata_df.loc[metadata_df[cluster_str] == 1][column_st].std()
            iqr_fl = stats.iqr(metadata_df.loc[metadata_df[cluster_str] == 1][column_st])
            
            results_lt.append([column_st,median_fl,mean_fl,std_fl,iqr_fl,"",""])

        except:
            mode_str = metadata_df.loc[metadata_df[cluster_str] == 1][column_st].mode()[0]
            mode_prop_fl = round(metadata_df.loc[(metadata_df[cluster_str] == 1 ) & (metadata_df[column_st] == mode_str )].shape[0] / metadata_df.loc[metadata_df[cluster_str] == 1].shape[0],1)
            
            

            results_lt.append([column_st, "", "", "", "",mode_str,mode_prop_fl])
    result_df = pd.DataFrame(results_lt, columns = ["class","median","mean","std","iqr","mode", "mode_proportion"])

    return result_df
    

def test_cluster_with_metadata(metadata_df, cluster_str):
    print(metadata_df.head(3))
    results_lt = []

    for column in metadata_df.columns.tolist():
        print(column)
        if metadata_df[column].dtype.kind in 'bifc':
            try:
                print("numeric")
                t_test_results_tp = stats.mannwhitneyu(
                                metadata_df.loc[metadata_df[cluster_str] == 1][column],
                                metadata_df.loc[metadata_df[cluster_str] == 0][column],
                                alternative="two-sided")
                value_positive_group = metadata_df.loc[metadata_df[cluster_str] == 1][column].median()
                value_negative_group = metadata_df.loc[metadata_df[cluster_str] == 0][column].median()

                results_lt.append([column, t_test_results_tp[1], value_positive_group, value_negative_group,"Mann-Whitney U-test"])
            except:
                pass
        else:
            chi2_results_tp = stats.chi2_contingency(
                pd.crosstab(metadata_df[column], metadata_df[cluster_str])
            )

            value_positive_group = metadata_df.loc[metadata_df[cluster_str ] == 1][column].mode()[0]
            value_negative_group = metadata_df.loc[metadata_df[cluster_str ] == 0][column].mode()[0]

            results_lt.append([column, chi2_results_tp[1], value_positive_group, value_negative_group,"Chi-square"])
            
    results_df = pd.DataFrame(results_lt, columns = ["class","p-value","median/mode-in-"+cluster_str,"median/mode-out-"+cluster_str, "test"])
    results_df.sort_values(by="p-value", inplace=True)


    return results_df


def compare_metadata_with_random_model(metadata_df, iterations=10, cluster_str="cluster_50"):
    
    import random
    random_difference_05 = []
    random_difference_01 = []
    random_difference_001 = []

    cluster_size = metadata_df[cluster_str].sum()
    non_cluster_size = metadata_df.shape[0] - cluster_size
    
    print("cluster size: ", cluster_size)
    for i in range(iterations):

        random_subgenre = [1]* cluster_size + [0] * non_cluster_size
        random.shuffle(random_subgenre, random.random)
        metadata_df["random.subgenre."+str(cluster_size)+str(i)] = random_subgenre

        test_metadata_random_subgenre = test_cluster_with_metadata(metadata_df, "random.subgenre."+str(cluster_size)+str(i))
        random_difference_05.append(test_metadata_random_subgenre.loc[test_metadata_random_subgenre["p-value"]<0.05].shape[0]-1)
        random_difference_01.append(test_metadata_random_subgenre.loc[test_metadata_random_subgenre["p-value"]<0.01].shape[0]-1)
        random_difference_001.append(test_metadata_random_subgenre.loc[test_metadata_random_subgenre["p-value"]<0.001].shape[0]-1)
        del metadata_df["random.subgenre."+str(cluster_size)+str(i)]
    return random_difference_05, random_difference_01, random_difference_001


def compare_subgenre_with_metadata(metadata_df, semantic_subgenres_lt):
    
    metadata_wo_subgenre_df = metadata_df.drop(semantic_subgenres_lt, axis=1).copy()
    subgenre_df = metadata_df[semantic_subgenres_lt]

    results_lt = []
    
    for subgenre_str in semantic_subgenres_lt:
        metadata_wo_subgenre_df[subgenre_str] = subgenre_df[subgenre_str]

        test_metadata_subgenre_df = test_cluster_with_metadata(metadata_wo_subgenre_df, subgenre_str)

        results_05 = test_metadata_subgenre_df.loc[test_metadata_subgenre_df["p-value"]<0.05].shape[0]
        results_01 = test_metadata_subgenre_df.loc[test_metadata_subgenre_df["p-value"]<0.01].shape[0]
        results_001 = test_metadata_subgenre_df.loc[test_metadata_subgenre_df["p-value"]<0.001].shape[0]
        
        results_lt.append([subgenre_str, results_05, "0.05"])
        results_lt.append([subgenre_str, results_01, "0.01"])
        results_lt.append([subgenre_str, results_001, "0.001"])
        
        del metadata_wo_subgenre_df[subgenre_str]
    results_df = pd.DataFrame(results_lt,columns=["subgenre","amount differentiating fields","p-value"])
    results_df.sort_values(by=["subgenre","amount differentiating fields"],ascending=False,inplace=True)

    return results_df
