# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 07:37:36 2017

@author: jose
"""


"""
To continue developing this script you need:
import pandas as pd
data_train = pd.read_csv("data/data_train.csv", sep="\t", encoding="utf-8", index_col=0)
data_eval = pd.read_csv("data/data_eval.csv", sep="\t", encoding="utf-8", index_col=0)
labels_train = pd.read_csv("data/labels_train.csv", sep="\t", encoding="utf-8", index_col=0, header=None)
labels_eval = pd.read_csv("data/labels_eval.csv", sep="\t", encoding="utf-8", index_col=0, header=None)
print(data_train.shape, data_eval.shape, labels_train.shape, labels_eval.shape)

classifier, training_score, evaluation_score  = classify(data_train, data_eval, labels_train, labels_eval, method = "SVC")

print(training_score, evaluation_score)

"""

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn


import numpy as np
import sys
import os
sys.path.append(os.path.abspath("/home/jose/Dropbox/Doktorarbeit/"))
import pandas as pd
from reading_robot import load_data, sampling, text2features, visualising, cull_data
from sklearn.model_selection import cross_val_score, cross_validate
from scipy import stats
from collections import Counter
import datetime
import re

import warnings
warnings.filterwarnings("once", category=Warning)



def choose_classifier(method = "SVC"):

    if method == "SVC":
        from sklearn.svm import SVC
        classifier = SVC(kernel="linear") # Default C = 1 # , C=100000000000000

    elif method == "KNN":
        from sklearn.neighbors import KNeighborsClassifier
        classifier = KNeighborsClassifier(n_neighbors=2)

    elif method == "DT":
        from sklearn.tree import DecisionTreeClassifier
        classifier = DecisionTreeClassifier()
                
    elif method == "RF":
        from sklearn.ensemble import BaggingClassifier
        from sklearn.tree import DecisionTreeClassifier
        tree = DecisionTreeClassifier()
        classifier = BaggingClassifier(tree)

    elif method == "LR":
        from sklearn.linear_model import LogisticRegression
        classifier = LogisticRegression(solver="lbfgs")

    elif method == "ordinal-LR":
        from sklearn.linear_model import LogisticRegression
        
        classifier = LogisticRegression(multi_class='multinomial',
                                        solver='newton-cg',
                                        fit_intercept=True
                                        )

    elif method == "BN":
        from sklearn.naive_bayes import BernoulliNB
        classifier = BernoulliNB()

    elif method == "GN":
        from sklearn.naive_bayes import GaussianNB
        classifier = GaussianNB()

    elif method == "Ridge":
        from sklearn.linear_model import RidgeClassifier
        classifier = RidgeClassifier() 
        
        """
    # This method doesn't accept negative values in the features, so it breaks with zscores
    # We could move the center of the zscores so that all values are positive
    # Although if we use other features, we would need to modify it in a different way...

    elif method == "MN":
        from sklearn.naive_bayes import MultinomialNB
        classifier = MultinomialNB()
        """
    else:
        print("You haven't chosen a valide classifier!!! You ")
    print("method used:\t", method)   
    return classifier
    
def classify_standard(data_train, data_eval, labels_train, labels_eval, classifier):
    
    return training_score, evaluation_score  


def make_confusion_matrix(classifier, data_train, data_eval, labels_train, labels_eval, class_, wdir):

    labels_pred = classifier.fit(data_train, labels_train).predict(data_eval)
    print(list(zip(labels_eval.values,labels_pred)))

    class_names = class_names = sorted(list(set(labels_train)))
    
    # Plot non-normalized confusion matrix
    visualising.plot_confusion_matrix(labels_eval, labels_pred, class_values = class_names, class_ = class_, title="Confusion matrix of "+ class_ +" without normalization", wdir = wdir)
    
    # Plot normalized confusion matrix
    visualising.plot_confusion_matrix(labels_eval, labels_pred, class_values=class_names, normalize=True, class_ = class_, title='Normalized confusion matrix of ' + class_, wdir = wdir)
    

def test_ttest_cross_results_baseline(results_cross, baseline):

    test_result = stats.ttest_1samp(results_cross, baseline)

    print("result of comparing cross-validation to baseline", test_result)
    return test_result

    
  
def standard_classification(wdir, least_frequent_class_value, document_data_model_cut, sampled_labels, verbose, classifier, class_):
    test_size = cull_data.calculate_test_size(least_frequent_class_value, verbose = True)
                                
    # Data and labels are splitted into train, evaluation
    data_train, data_eval, labels_train, labels_eval = sampling.standard_sampling(document_data_model_cut, sampled_labels, test_size, verbose = verbose) #[class_])

    classifier.fit(data_train, labels_train)
    training_score = classifier.score(data_train, labels_train)
    evaluation_score = classifier.score(data_eval, labels_eval)

    results = training_score, evaluation_score
    make_confusion_matrix(classifier = classifier, data_train = data_train, data_eval = data_eval, labels_train = labels_train, labels_eval = labels_eval, class_ = class_, wdir = wdir)

    return results

from sklearn import preprocessing

def label_encoder(df):
    for column in df.columns.tolist():
        print(column)
        print(df[column].dtype)
        if df[column].dtype not in [np.float64, np.int64, np.float32, np.int32]:
            le = preprocessing.LabelEncoder()
            le.fit(list(set(df[column])))
            df[column] = le.transform(df[column]) 
    return df

def classify_cross(data, labels, classifier, cv = 10, scoring = {'f1': 'f1', 'rec': 'recall', "prec":"precision", 'f1_macro': 'f1_macro', 'f1_micro': 'f1_micro' }):

    if len(set(labels)) > 2:
        print("more than 2 classes!")
        
        scoring = {'f1': 'f1_micro', 'rec': 'recall_micro', "prec":"precision_micro", 'f1_macro': 'f1_macro','f1_micro': 'f1_micro',}
    else:
        labels = labels.astype(int)
        
    scores_dc = cross_validate(classifier, data, labels, scoring = scoring, cv = cv, return_train_score = False)
    #print(scores_dc)
    scores_df = pd.DataFrame(list(scores_dc.values()),index=list(scores_dc.keys())).T.rename(columns={"test_rec":"rec", "test_f1":"f1","test_prec":"prec",'test_f1_macro': 'f1_macro','test_f1_micro': 'f1_micro'})[["f1","rec","prec","f1_macro","f1_micro"]]
        
    return scores_df


def classify(wdir, wsdir = "corpus/", freq_table  = [], metadata = "metadata.csv",
             sep = "\t", classes = ["class"], verbose = True, methods = ["SVC"], min_MFF = 0,
             max_MFFs = [5000], text_representations = ["zscores"], typographies = [True],
             sampling_mode = "cross", problematic_class_values = ["n.av.", "other", "mixed", "?", "unknown","none", "second-person"],
             minimal_value_samples = 2, make_relative = True,
             under_sample_method = "None", maximum_cases = 5000, sampling_times = 1, outdir_results = "", sort_by="median"
             ):
    """
     *  wdir
     *  wsdir = "corpus/"
     *  freq_table  = []
     *  metadata = "metadata.csv"
     *  sep = "\t"
     *  classes = ["class"]
     *  verbose = True
     *  method = ["SVC"]
     *  min_MFF = 0
     *  max_MFFs = [5000]
     *  text_representations = ["zscores"]
     *  typographies = [True,False]
     *  sampling_mode = "cross"
     *  problematic_class_values = ["n.av.", "other", "mixed", "?", "unknown","none", "second-person"]
     *  minimal_value_samples = 2
     *  make_relative = True
     *  scoring = "f1"
     *  under_sample_method = "None"
     *  maximum_cases = 5000,
     *  sampling_times = 1

    """
    cut_raw_features = freq_table
    
    print("cut_raw_features ", cut_raw_features.head())
    print("in classify, cut_raw_features, ", cut_raw_features.shape)
    if make_relative == True:
        cut_raw_features = text2features.calculate_relative_frequencies(cut_raw_features)
        print("cut_raw_features after relative normalization", cut_raw_features.head())
    
    results = []


    for class_ in classes:
        print("\n\nanalysed class:\t", class_)
        # This step deletes too small classes
        filtered_raw_features, labels = cull_data.cull_data(cut_raw_features, metadata, class_, verbose, problematic_class_values = problematic_class_values, minimal_value_samples = minimal_value_samples)


        print("size after culling data:", filtered_raw_features.shape, labels.shape)

        for typography in typographies:
            filtered_raw_features_typo = cull_data.cull_typography(filtered_raw_features, keep_typography = typography)
            print("typography ", typography)

            for text_representation in text_representations:
                # The corpus is modeled somehow (raw, relative frequencies, tf-idf, z-scores...)
                document_data_model = text2features.choose_features(filtered_raw_features_typo, text_representation)


                print(document_data_model.shape) if verbose == True else 0
                for MFW in max_MFFs:
                    print("MFW", MFW)
                    document_data_model_cut = load_data.cut_corpus(document_data_model, min_MFF = min_MFF, max_MFF = MFW, sort_by = sort_by)
                    print("The three first MFWs: ",document_data_model_cut.columns.tolist()[0:3])
                    print("The three last MFWs: ",document_data_model_cut.columns.tolist()[-3:])
                    if len(set(labels.values.tolist())) < 2:
                        print("After culling the class", class_, " can't be divided in two groups. This category is going to be ignored" )
                    else:
                        for method in methods:
                            classifier = choose_classifier(method = method)

                            f1s_over_sampling = np.array([])
                            scores_over_sampling_df = pd.DataFrame(columns=["f1","rec","prec"])

                            for sampling_i in range(sampling_times):
                                print(labels.shape)
                                print(document_data_model_cut.shape)

                                sampled_labels, sampled_document_data_model_cut = sampling.under_sample(labels,
                                                                                             document_data_model_cut,
                                                                                             under_sample_method,
                                                                                             maximum_cases)
                                baseline = cull_data.calculate_baseline(sampled_labels)

                                least_frequent_class_value = Counter(sampled_labels).most_common()[-1][1]

                                if sampling_mode == "standard":
                                    print("standard sampling, bug coming!")

                                    results = standard_classification(wdir, least_frequent_class_value, document_data_model_cut, sampled_labels, verbose, classifier, class_)
                                    return results

                                elif sampling_mode == "cross":
                                    cv = cull_data.calculate_cv(least_frequent_class_value)
                                    print("cross validation sampling of ", class_)

                                    scores_df = classify_cross(sampled_document_data_model_cut, sampled_labels, classifier, cv = cv)
                                    
                                    f1s_over_sampling = np.append(f1s_over_sampling,scores_df["f1"])
                                    
                                    scores_over_sampling_df = pd.concat([scores_df, pd.DataFrame(scores_df.mean()).T],axis=0)
                                    
                            #print(scoring + ": %0.2f (+/- %0.2f)" % (evaluation_over_sampling.mean(), evaluation_over_sampling.std() * 2))
                            test_result_param, test_result_pvalue = test_ttest_cross_results_baseline(f1s_over_sampling, baseline)

                            # Creo que aquí hay que descender en los loops
                            print("Class: \t", class_)
                            print("Scores:\n \t", scores_over_sampling_df.mean().round(3))
                            print("p-value: ",round(test_result_pvalue,4))
                            print("Baseline: \t\t", round(baseline,2))
                            print(method)

                            f1_baseline = scores_over_sampling_df.mean()["f1"].round(3) - baseline
                            print(scores_over_sampling_df.mean()["f1"].round(3) - baseline)
                            results.append([class_, scores_over_sampling_df.mean()["f1"].round(3), scores_over_sampling_df.mean()["rec"].round(3), scores_over_sampling_df.mean()["prec"].round(3), scores_over_sampling_df.mean()["f1_macro"].round(3), scores_over_sampling_df.mean()["f1_micro"].round(3), baseline, f1_baseline,  method, text_representation, MFW, typography, f1s_over_sampling.round(2),  test_result_pvalue, sampled_labels, sampled_labels.shape[0], cv, sampling_times, classifier])
    results_df = pd.DataFrame(results, columns=["class",'mean_f1','mean_rec',"mean_prec", "f1_macro","f1_micro", 'baseline', "f1-baseline" ,  'classifier_name','text_representation', 'MFW', 'typography', "f1s", 'test_result_pvalue', 'labels', "sample_size","cv","sampling_times",'classifier'])
    print(results_df.head())
    
    results_df = results_df.sample(frac=1)
    results_df.sort_values(by=["f1-baseline","MFW"], ascending=[False,True], inplace=True)
    if outdir_results == "":
        outdir_results = wdir
    results_file = "results"+"_"+ "-".join(classes)+"_"+ "-".join(methods)+"_"+ "-".join(str(x) for x in max_MFFs)+"_" +"-".join(text_representations)
    if len(results_file) > 100:
        results_file = "results_"+str(datetime.datetime.now().year)+str(datetime.datetime.now().month)+str(datetime.datetime.now().day)+str(datetime.datetime.now().hour)+str(datetime.datetime.now().minute)+str(datetime.datetime.now().second)

    results_df.to_csv(outdir_results + results_file+".csv", sep = "\t")
    print("done!")
    return results_df




def predict(wdir, entire_raw_features, metadata, class_ = "class", predict_class_values = ["?"], verbose = True, method = "SVC", min_MFF = 0, max_MFF = 5000, text_representation="relative", make_relative = True, iterations = 1, do_scores = False, type_classes = "binary"):
    if make_relative == True:
        entire_raw_features = text2features.calculate_relative_frequencies(entire_raw_features)

    #print(entire_raw_features.columns.tolist()[0:10])
    entire_raw_features = load_data.cut_corpus(entire_raw_features, min_MFF = min_MFF, max_MFF = max_MFF)
    print(entire_raw_features.columns.tolist()[0:10])

    print("corpus and metadata are coherent") if entire_raw_features.index.tolist() == metadata.index.tolist() else "corpus and metadata are NOT coherent"

    train_class_values = [set_label for set_label in list(set(metadata[class_])) if set_label not in predict_class_values]
    print("train classes", train_class_values)

    smallest_class = Counter(metadata.loc[metadata[class_].isin(train_class_values)][class_]).most_common()[-1]
    print("smallest class", smallest_class)
    
    document_data_model = text2features.choose_features(entire_raw_features, text_representation)

    metadata_predict = metadata.loc[metadata[class_].isin(predict_class_values)].copy()#.sort_index()

    metadata_predict_iterations = pd.DataFrame(index=metadata_predict.index, columns = [i for i in range(iterations)])
    if type_classes == "binary":
        metadata_predict["sum_prediction_" + class_] = 0

    
    document_data_model_predict = document_data_model.loc[metadata_predict.index.tolist()]#.sort_index()
    #print("document data model to predict\n", document_data_model_predict.head(3))
    print("metadata and data to predict coherent?", metadata_predict.index.tolist() == document_data_model_predict.index.tolist())

    for i in range(iterations):
        metadata_sample = pd.concat([metadata.loc[(~metadata[class_].isin(predict_class_values) ) & (metadata[class_] != smallest_class[0])], metadata.loc[metadata[class_] == smallest_class[0]].sample(n = smallest_class[1])]).sample(frac = 1)
        document_data_model_sample = document_data_model.loc[metadata_sample.index.tolist()]
        #print("document_data_model_sample\n", document_data_model_sample.head(3))
        print("metadata and texts coherent") if metadata_sample.index.tolist() == document_data_model_sample.index.tolist() else print("metadata and corpus are not coherent")
        print("metadata's shape", metadata_sample.shape)
        classifier = choose_classifier(method = method)
        #print(set(metadata_sample[class_]))
        #print(document_data_model_sample.head())
        classifier.fit(document_data_model_sample, metadata_sample[class_].astype(str))
        
        if do_scores == True:        
            scores = classify_cross(document_data_model_sample, metadata_sample[class_].astype(str), classifier, cv = 10, scoring = "f1")
            
            print("scores", scores)
        
        print(document_data_model_predict.index.tolist())
        results = classifier.predict(document_data_model_predict)
        print(i, metadata_sample.index[0:3], results, )
        
        metadata_predict_iterations[i] = results

        if type_classes == "binary":
            metadata_predict["sum_prediction_" + class_] = np.array(results).astype(int) + metadata_predict["sum_prediction_" + class_]
        
            metadata_predict_iterations[i] = metadata_predict_iterations[i].astype(int)
            
    if type_classes == "binary":
        metadata_predict["sum_prediction_" + class_] = metadata_predict["sum_prediction_" + class_] / iterations

        print(metadata_predict["sum_prediction_" + class_])
    
    return metadata_predict, results, metadata_predict_iterations

