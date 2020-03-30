# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 16:32:50 2018

@author: jose
"""
import pandas as pd
import os
from scipy import stats
from sklearn.metrics import cohen_kappa_score
import networkx as nx
import sys
import matplotlib.pyplot as plt
import re
import numpy as np
from collections import Counter

sys.path.append(os.path.abspath("/home/jose/Dropbox/Doktorarbeit/"))

from reading_robot  import st_tests , load_data, describe_data


def get_set_labels(wdir, metadata_labels, save = False):
    complete_list_labels =  get_list_labels(wdir, metadata_labels, print_ = False)

    complete_list_labels = sorted(list(set(complete_list_labels)))

    if save == True:    
        with open(wdir+'set_labels.txt', "w", encoding="utf-8") as fout:
            fout.write("\n".join(complete_list_labels))

    return complete_list_labels

def get_list_labels(wdir, metadata_labels, print_=True):
    complete_list_labels = []
    for column in metadata_labels.columns:
        list_labels = [label.strip().lower() for grouped_label in metadata_labels[column].values.tolist() for label in re.split("[,;]", grouped_label) if label not in ["", "n.av.","?"]]

        complete_list_labels = complete_list_labels + list_labels
    complete_list_labels = complete_list_labels

    unvalid_labels = [""," "]
    for unvalid_label in unvalid_labels:
        if unvalid_label in complete_list_labels:
            complete_list_labels.remove(unvalid_label)
    
    if print_ == True:
        with open(wdir+'list_labels.txt', "w", encoding="utf-8") as fout:
            fout.write("\n".join(complete_list_labels))
    
    return complete_list_labels
    

def check_label_vs_reference(set_labels, reference_labels, wdir="/home/jose/Dropbox/Doktorarbeit/thesis/data/"):
    i = 0
    new_label = []
    for label in set_labels:
        if label.lower() not in [item.lower() for item in reference_labels["label"].values.tolist()]:
            print("following lable not in refence: " + label)
            new_label.append(label.lower())
    if i == 0:
        print("all labels in the metadata are already contained in the reference!")
    with open(wdir+'new_labels.txt', "w", encoding="utf-8") as fout:
        fout.write("\n".join(new_label))

def get_set_from_column(metadata_labels, column):
    list_labels = list(set( get_list_from_column(metadata_labels, column) ))
    return list_labels

def get_list_from_column(metadata_labels, column):
    list_labels = [label.strip().lower() for grouped_label in metadata_labels[column].values.tolist() for label in re.split("[,;]", grouped_label) if label not in ["", "n.av.","?"]]
    return list_labels



def modelize_metadata_labels(metadata_labels, wdir, reference_labels, reference_column = "semantic_label_reference"):

    metadata_labels_normalized = metadata_labels.copy()
    
    for index, row in reference_labels.iterrows():
        for column in metadata_labels_normalized.columns.tolist():
            for i in range(2):
                metadata_labels_normalized[column] = metadata_labels_normalized[column].str.replace(r"(^|[,;])\s*" + row['label'] + r"\s*($|[,;])", r"\1" + row[reference_column] + r"\2", flags=re.MULTILINE|re.IGNORECASE)
    
    metadata_labels_normalized.to_csv(wdir+"labelsNormalized-" + reference_column +".csv", sep="\t")

    return metadata_labels_normalized

def delete_specific_label(metadata, label_to_delete):
    metadata = metadata.replace(label_to_delete, "")
    for column_name, column in metadata.iteritems():
        metadata.loc[metadata[column_name] == label_to_delete] = ""
        i = 0
        while i < 4:
            metadata[column_name] = metadata[column_name].str.replace(r"(\s*"+label_to_delete+"\s*[;,]|^\s*"+label_to_delete+"\s*$|[;,]\s*"+label_to_delete+"$)", r"", flags=re.MULTILINE|re.IGNORECASE)
            i +=1
        
    return metadata


def create_cross_table(dataframe, class_1, class_2, wdir = "", do_print = False, verbose=False):
    cross_dataframe = pd.crosstab(dataframe[class_1], dataframe[class_2])
    print("cross tab done") if verbose == True else 0
    if do_print == True and wdir != "":
        cross_dataframe.to_csv(wdir+"cross-table_" + class_1 + "_" + class_2 + ".csv", sep = "\t")
    return cross_dataframe

def calculate_chi2_from_dataframe(dataframe, class_1, class_2, verbose = False):
    print("\nChi2 test: for " + class_1 + " and "+ class_2) if verbose == True else 0
    cross_table = create_cross_table(dataframe, class_1, class_2)
    chi2, p, dof, expected  = stats.chi2_contingency(cross_table)
    print(p) if verbose == True else 0
    if(p < 0.055):
        print("* the classes are related") if verbose == True else 0
    else:
        print("the  classes are independent") if verbose == True else 0
    return chi2, p, dof, expected 


def create_df_of_labels(set_labels, metadata_labels, wdir, reference_labels = ""):
    if type(reference_labels) != str:
        columns = ["used_by", "used_for", "is_in_thema"]
    else:
        columns = ["used_by", "used_for"]
    df_labels_sources = pd.DataFrame(index = set_labels, columns = columns).fillna(0)
    #print(df_labels_sources)
    for label in set_labels:
        df_labels_sources.at[label, "used_by"] = metadata_labels.apply(lambda row: row.astype(str).str.contains(r"(?:^|[,;])\s*" + label + r"\s*(?:$|[,;])", flags=re.IGNORECASE|re.MULTILINE).any(), axis=0).sum()
        df_labels_sources.at[label, "used_for"] = metadata_labels.apply(lambda row: row.astype(str).str.contains(r"(?:^|[,;])\s*" + label + r"\s*(?:$|[,;])", flags=re.IGNORECASE|re.MULTILINE).any(), axis=1).sum()
        if type(reference_labels) != str:
            if reference_labels.loc[(reference_labels["semantic_label_reference"] == label) & (reference_labels["thema_code"] != "-")].shape[0] > 0:
                
                df_labels_sources.at[label, "is_in_thema"] = 1
            else:
                df_labels_sources.at[label, "is_in_thema"] = 0
            
    df_labels_sources = df_labels_sources.sort_values(by = "used_by")
    df_labels_sources.to_csv(wdir+"labels_used_by_for.csv", sep = "\t")
    print(df_labels_sources.shape)
    return df_labels_sources


def create_df_of_sources(metadata_labels, wdir):
    df_sources_labels = pd.DataFrame(index = metadata_labels.columns, columns = ["different_labels", "labeled_books"]).fillna(0)
    for column in metadata_labels.columns.tolist():

        list_labels = get_set_from_column(metadata_labels, column)

        df_sources_labels.at[column, "different_labels"] = len(list_labels)
        df_sources_labels.at[column, "labeled_texts"] = len( metadata_labels.loc[~metadata_labels[column].isin(["", "n.av.","?"]) ].values.tolist())
        df_sources_labels.at[column, "mean-labels-texts"] = metadata_labels.loc[metadata_labels[column] != ""][column].str.count(",").mean()+1
        df_sources_labels.at[column, "std-labels-texts"] = metadata_labels.loc[metadata_labels[column] != ""][column].str.count(",").std()
        
        #print(Counter(get_list_from_column(metadata_labels, column)).most_common(1)[0][0])
        df_sources_labels.at[column, "most-common-label"] = Counter(get_list_from_column(metadata_labels, column)).most_common(1)[0][0]
        


    df_sources_labels["ratio"] = df_sources_labels["labeled_texts"] / df_sources_labels["different_labels"]
    df_sources_labels = df_sources_labels.sort_values(by = "ratio")

    df_sources_labels = df_sources_labels[["different_labels","labeled_texts","ratio","mean-labels-texts","std-labels-texts","most-common-label"]]
    
    df_sources_labels.to_csv(wdir+"sources_labels.csv", sep = "\t")
    print(df_sources_labels.shape)
    return df_sources_labels
    
"""
wdir = "/home/jose/Dropbox/Doktorarbeit/thesis/data/"
metadata = load_data.load_metadata(wdir, metadata_table = "metadata_beta-opt-obl-subgenre-structure.csv", sep = ",")

columns_labels_sources = ["subgenre", "genre-label","subgenre-lithist-MdLE","subgenre-lithist-HdLE", "subgenre-edit-epublibre", "subgenre-edit-wikidata", "genre-label-source-bne"]

metadata_labels = cleanup_labels(metadata, wdir, columns_labels_sources) 

columns_labels_sources = ["subgenre", "genre-label","subgenre-lithist-MdLE","subgenre-lithist-HdLE", "epub-1", "subgenre-edit-wikidata", "genre-label-source-bne"]

reference_labels = pd.read_csv("/home/jose/Dropbox/MTB/investigacion/mytoolbox/reading_robot/data/reference_labels.csv", sep="\t")

metadata_labels_sem_normalized = modelize_metadata_labels(metadata_labels, wdir, reference_labels, "semantic_label_reference", columns = columns_labels_sources)
"""

def calculate_shared_labels(labels_df, source1, source2, is_proportion):
    total_amount_source1 = len(labels_df.loc[labels_df[source1] != ""].values.tolist())
    total_amount_source2 = len(labels_df.loc[labels_df[source2] != ""].values.tolist())
    shared_amount = sorted([total_amount_source1,total_amount_source2])[1]
    if is_proportion  == True:
        value = len(labels_df.loc[ (labels_df[source1] == labels_df[source2]) & (labels_df[source1] != "")].values.tolist()) / shared_amount
    else:
        value = len(labels_df.loc[ (labels_df[source1] == labels_df[source2]) & (labels_df[source1] != "")].values.tolist()) 
    return value

def calculate_chi2(labels_df, source1, source2):
    chi2, p, dof, expected = calculate_chi2_from_dataframe(labels_df, source1, source2)
    if p < 0.05:
        value = 1
    else:
        value = 0
    return value

def calcultae_kappa(labels_df, source1, source2):
    value = cohen_kappa_score(labels_df[source1], labels_df[source2])
    return value

def make_edges_df_from_labels(labels_df, columns_labels_sources, wdir, dataset, kind = "kappa", is_proportion = True, threshold = 0.05, verbose = True, apply_threshold = True):
    describe_data.create_dir(wdir, "edges")
        
    labels_df = labels_df.replace("n.av.","")
    edges = []
    already_seen_sources = []
    for source1 in columns_labels_sources:
        already_seen_sources.append(source1)
        for source2 in columns_labels_sources:
            if source2 not in already_seen_sources:
                print(source1, source2) if verbose == True else 0
                
                if kind == "shared":
                    value = calculate_shared_labels(labels_df, source1, source2, is_proportion)

                elif kind == "kappa":
                    value = cohen_kappa_score(labels_df[source1], labels_df[source2])

                elif kind == "chi2":
                    value = calculate_chi2(labels_df, source1, source2)

                if apply_threshold == True:
                    if value > threshold:
                        edges.append([source1, source2, value])
                else:
                    edges.append([source1, source2, value])
                    
    df_edges = pd.DataFrame(edges, columns = ["Source", "Target", "Weight"])
    df_edges["Type"] = "Undirected"
    df_edges = df_edges.sort_values(by=["Weight"], ascending=False)
    df_edges.to_csv(wdir+"edges/edges_" + dataset +"_" + kind + ".csv", sep="\t")
    print(df_edges) if verbose == True else 0
    return df_edges

def make_edges_df_from_each_label(labels_df, labels, columns_labels_sources, wdir, dataset, kind = "kappa", threshold = 0.05):
    labels_df = labels_df.replace("n.av.","")
    edges = []
    already_seen_sources = []
    for source1 in columns_labels_sources:
        already_seen_sources.append(source1)
        for source2 in columns_labels_sources:
            if source2 not in already_seen_sources:
                values = []
                for label in labels:
                    metadata_labels_single_class = labels_df.copy()
                    #print(label)
                    metadata_labels_single_class = maintain_specific_label(metadata_labels_single_class, label,  columns_labels_sources)
                    #print(label, source1, metadata_labels_single_class[source1].values.tolist())
                    if kind == "kappa":
                        value = calcultae_kappa(metadata_labels_single_class, source1, source2)

                    #print(value)
                    if value > threshold:
                        values.append(value.astype(float))
                values = np.array(values)
                print(source1,source2,values,values.mean())
                if len(values) >= threshold:
                    edges.append([source1, source2, np.mean(values)])
                    
    df_edges = pd.DataFrame(edges, columns = ["Source", "Target", "Weight"]).fillna(0)
    df_edges["Type"] = "Undirected"
    df_edges = df_edges.sort_values(by=["Weight"], ascending=False)
    df_edges.to_csv(wdir+"edges-mean-each-class_" + dataset +"_" + kind + ".csv", sep="\t")
    print(df_edges)
    return df_edges


def make_network(df_edges, wdir, dataset, edge_vmin = 0, edge_vmax = 1, multiple_value_edges_by = 10, threshold=0.05):
    df_edges = df_edges.loc[df_edges["Weight"] >= threshold]
    graph = nx.from_pandas_edgelist(df = df_edges, source = "Source", target = "Target", edge_attr = ['Weight','Type'] )
    widths = [w['Weight'] * multiple_value_edges_by for (u, v, w) in graph.edges(data=True)]
    pos = nx.spring_layout(graph)
    nx.draw_networkx(graph, pos, width = widths, alpha=0.6, edge_color='#87CEFA', edge_vmin = edge_vmin, edge_vmax = edge_vmax)
    plt.title(dataset)
    plt.savefig(wdir + "visualisations/"+ dataset + "_" + str(threshold)+".png", dpi=300)

    return graph


def summary_labels(label_agreement_df, outdir):
    
    classes_accepted_kappas = label_agreement_df.apply(lambda x: [1 if y > 0 else 0 for y in x]).sum().sort_values()
    
    sumary_label_agreement_df = pd.DataFrame([classes_accepted_kappas[classes_accepted_kappas > 0], label_agreement_df.replace(0, np.NaN).mean(), label_agreement_df.replace(0, np.NaN).median()], columns=label_agreement_df.columns.tolist(), index=["number pairs kappa","mean kappa","median kappa"]).T.fillna(0)

    sumary_label_agreement_df.to_csv(outdir+"sumary_label_agreement_df.csv", sep="\t")

    return sumary_label_agreement_df


def calculate_agreement_per_label(metadata_labels, classes, columns_labels_sources, wdir, dataset, does_filter = True, threshold = 0.05):
    results_df = pd.DataFrame(columns = classes)
    
    for class_ in classes:
        print(class_)
        metadata_labels_single_class = maintain_specific_label(metadata_labels, class_, columns_labels_sources)
        #print(metadata_labels_single_class.head())

        df_edges_kappa = make_edges_df_from_labels(metadata_labels_single_class, columns_labels_sources, wdir, class_+dataset, kind = "kappa", verbose=False, threshold = "", apply_threshold = False)

        results_df[class_] = df_edges_kappa["Weight"]
    
    results_df.to_csv(wdir+"labels_agreement.csv", sep="\t")

    if does_filter == True:
        
        results_df = results_df.mask(results_df < threshold).fillna(0)
    results_df = results_df.fillna(0)
    print(results_df)
    return results_df 



def maintain_specific_label(metadata_labels, labels, sources):
    if isinstance(labels, str) == True:
        labels = [labels]
    metadata_labels_binarized = metadata_labels.copy()
    for label in labels:
        for index, row in metadata_labels.iterrows(): 
            for source in sources:
                value = 0
                value += row[source].count(label)
                if value > 0:
                    metadata_labels_binarized.at[index, source] = label
                else:
                    metadata_labels_binarized.at[index, source] = ""
                    
                
    #print(metadata_labels_binarized.head())
    return metadata_labels_binarized

def count_labels_from_sources(labels, metadata_labels, sources, wdir):
    metadata_labels_quantified = metadata_labels.copy()

    metadata_labels_quantified = pd.DataFrame(columns = labels, index = metadata_labels.index.tolist())
    
    for label in labels:
        for index, row in metadata_labels.iterrows(): 
            value = 0
            for source in sources:
                print(source, label)
                if label in row[source]:
                    value_sorce = 1
                else:
                    value_sorce = 0
                value = value + value_sorce
            metadata_labels_quantified.at[index, label] = value
    metadata_labels_quantified.to_csv(wdir+"quantified_subgenreClasses.csv", sep="\t")
    return metadata_labels_quantified

def make_edges_about_labels_sources(labels, metadata_labels, sources, wdir):
    edges = []
    already_seen_labels = []
    lists = [";".join(item) for item in metadata_labels[sources].values.tolist()]
    for label1 in labels:
        already_seen_labels.append(label1)
        for label2 in labels:
            if label2 not in already_seen_labels:
                
                use_label1 = sum([int(bool(len(re.findall(r'(.*?'+label1 + ".*?)", list_)))) for list_ in lists])
                use_label2 = sum([int(bool(len(re.findall(r'(.*?'+label2 + ".*?)", list_)))) for list_ in lists])
                
                couse = sum([int(bool(len(re.findall(r'(.*?'+label1 + ".*?" + label2+'.*?|.*?'+label2 + ".*?" + label1+'.*?)', list_)))) for list_ in lists])
                """
                weight += int(bool(couse ))
                
                
                for list_ in lists:
                    found = re.findall(r'(.*?'+label1 + ".*?" + label2+'.*?|.*?'+label2 + ".*?" + label1+'.*?)', list_, flags = re.DOTALL)
                    weight += int(bool(len(found)))
                """
                print(label1, label2, couse, use_label1, use_label2)
                relative_couse = couse / sorted( [use_label1, use_label2])[0] 
                if couse > 0:
                    edges.append([label1, label2, relative_couse])
    df_edges = pd.DataFrame(edges, columns = ["Source", "Target", "Weight"])
    df_edges["Type"] = "Undirected"
    df_edges = df_edges.sort_values(by=["Weight"], ascending=False)
    df_edges.to_csv(wdir+"edges-labels_.csv", sep="\t")
    print(df_edges)
    return df_edges
    

def make_edges_about_labels_metadata(labels, metadata_labels, wdir):
    edges = []
    already_seen_labels = []
    metadata_labels = metadata_labels[labels].copy()
    metadata_labels = (metadata_labels > 0).astype(int)
    print(metadata_labels.head())
    for label1 in labels:
        already_seen_labels.append(label1)
        for label2 in labels:
            if label2 not in already_seen_labels:
                
                weight =  metadata_labels.loc[(metadata_labels[label1]==1) & (metadata_labels[label2]==1)].shape[0]
                
                use = sorted([metadata_labels.loc[(metadata_labels[label1]==1)].shape[0], metadata_labels.loc[(metadata_labels[label2]==1)].shape[0]])[0]
                edges.append([label1, label2, weight/use])
                print(label1, label2, weight,use)
    df_edges = pd.DataFrame(edges, columns = ["Source", "Target", "Weight"])
    df_edges["Type"] = "Undirected"
    df_edges = df_edges.sort_values(by=["Weight"], ascending=False)
    df_edges.to_csv(wdir+"edges-labels_.csv", sep="\t")
    print(df_edges)
    return df_edges


def make_interval_subgenres_binary(metadata_df,
                                   labels = ['guerra', 'diálogo', 'histórica', 'humor', 'aventura', 'nivola', 'poética', 'memorias', 'naturalista',
                 'erótica', 'greguería', 'fantástico', 'episodio nacional', 'costumbrista', 'social', 'espiritual',
                 'realista', 'autobiografía', 'psicológica', 'modernista', 'biografía', 'educación', 'filosófica']):
    metadata_df[labels] = ( metadata_df[labels] > 0).astype(int, inplace=True)
    return metadata_df

