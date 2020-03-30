# -*- coding: utf-8 -*-
"""
Created on Mon May  7 06:55:45 2018

@author: jose
"""

import sys
import os
import pandas as pd
sys.path.append(os.path.abspath("/home/jose/Dropbox/Doktorarbeit/"))
#from reading_robot import load_data, tei2text, text2features, classify, process_labels, describe_data,feature_analysis, cull_data, st_tests, metadata2numbers, cluster, call_stylo, prepare_subcorpus
from reading_robot import text2features
import scipy.stats as stats
import matplotlib.pyplot as plt
import networkx as nx


def select_features(features_df):
    
    selected_columns = []
    for column in features_df.columns.tolist():
        if "@person" in column or "mariax" in column or "am." in column or "@wnlex" in column or column == "0@ord_ent" or column == "1@ord_ent"  or "madrid" in column or column == "españa@form_noun@pos" or "@type" in column:
            selected_columns.append(column)
    selected_features_annotation_df = features_df[selected_columns]
    
    #selected_features_annotation_df.columns = selected_features_annotation_df.columns.str.replace("(@mariax|@wnlex)","",regex=True)
    
    return selected_features_annotation_df




def results_features(new_metadata_df, semantic_subgenres_lt, selected_features_df):
    results_features_df = pd.DataFrame([],)

    for subgenre in semantic_subgenres_lt:

        results_features_df[subgenre] = selected_features_df.loc[new_metadata_df.loc[new_metadata_df[subgenre]>0].index.tolist()].mean()

    results_features_df = results_features_df.T
    return results_features_df


def make_subgenre_nodes(bool_new_metadata_df, new_metadata_df, semantic_subgenres_lt, wdir):
    subgenres_nodes_df = pd.DataFrame([],)

    for subgenre in semantic_subgenres_lt:
        results_df = pd.DataFrame(bool_new_metadata_df.groupby(subgenre).mean().loc[1],).T
        results_df.index = [subgenre]

        results_df["number.texts"] = new_metadata_df.loc[new_metadata_df[subgenre] > 0].shape[0]
        results_df["number.authors"] = len(set(new_metadata_df.loc[new_metadata_df[subgenre] > 0]["author.name"]))

        subgenres_nodes_df = subgenres_nodes_df.append(results_df)
    subgenres_nodes_df.sort_index(inplace=True)
    
    combined_results_metadata_df = pd.read_csv(wdir+"combined_results_metadata.csv", sep="\t", index_col=0)
    
    subgenres_nodes_df["mean.f1.ling-lit"] = combined_results_metadata_df.groupby(["class"]).head(1).groupby(["class"]).mean()["mean_f1"].sort_index()
    subgenres_nodes_df["node"] = subgenres_nodes_df.index
    subgenres_nodes_df["label"] = subgenres_nodes_df.index#.str.replace("^", "_", flags=re.MULTILINE, regex=True)
    subgenres_nodes_df["size"] = (text2features.calculate_minmax(subgenres_nodes_df[["number.texts"]])+0.01)*50
    subgenres_nodes_df["size"] = (text2features.calculate_minmax(subgenres_nodes_df[["mean.f1.ling-lit"]])+0.01)*20
    subgenres_nodes_df["type"] = "subgenre"
    return subgenres_nodes_df[["node","size","type","label","mean.f1.ling-lit", "litHist.pages","number.authors","number.texts"]]

def make_texts_nodes(new_metadata_df):
    
    texts_nodes_df = pd.DataFrame([new_metadata_df["title"]+ ", de " +new_metadata_df["author.name"].tolist(), new_metadata_df["litHist.pages"], new_metadata_df["title.main"]]).T
    texts_nodes_df = texts_nodes_df.rename(index=str, columns={"title":"label","litHist.pages": "size"})
    texts_nodes_df["node"] = texts_nodes_df.index
    texts_nodes_df["type"] = "text"
    texts_nodes_df["size"] = (text2features.calculate_minmax(texts_nodes_df[["size"]])+0.01)*50
    return texts_nodes_df

def make_annotation_nodes(selected_features_annotation_df):
    annotation_nodes_df = pd.DataFrame((selected_features_annotation_df.mean()+selected_features_annotation_df.abs().mean().max()).sort_values(ascending=False), columns=["mean-log-frequency"])
    annotation_nodes_df["feature"] = annotation_nodes_df.index.tolist()
    annotation_nodes_df = annotation_nodes_df.rename(index=str, columns={"feature":"node","mean-log-frequency": "size"})
    annotation_nodes_df["type"] = "feature"
    annotation_nodes_df["label"] = annotation_nodes_df["node"]
    
    annotation_nodes_df["label"] = annotation_nodes_df["label"].replace("@mariax"," ",regex=True)
    annotation_nodes_df["size"] = text2features.calculate_minmax(annotation_nodes_df[["size"]])
    return annotation_nodes_df



def make_subgenre_features_edges(results_features_zscores_df, amount_of_features_per_subgenre, std_deviation_of_feature_in_subgenre, semantic_subgenres_lt = ["autobiografía","aventura","biografía","costumbrista","diálogo","educación","episodio nacional","erótica","espiritual","fantástico","filosófica","greguería","guerra","histórica","humor","memorias","modernista","naturalista","nivola","poética","psicológica","realista","social","bucólica","mono-diálogo","ficción-literaria"]):
    subgenre_features_edges_lt = []
    for subgenre in semantic_subgenres_lt:
        subgenre_features_edges_lt = subgenre_features_edges_lt + [[subgenre, feature, value, "subgenre-feature"] for feature, value in results_features_zscores_df.T.loc[results_features_zscores_df.T[subgenre] > std_deviation_of_feature_in_subgenre].sort_values(by= subgenre, ascending=False)[subgenre][0:amount_of_features_per_subgenre].items()]
    subgenre_features_edges_df = pd.DataFrame(subgenre_features_edges_lt,columns=["source","target","weight","type"])
    return subgenre_features_edges_df


def make_subgenres_novels_edges(new_metadata_df, semantic_subgenres_lt, amount_of_texts_per_subgenre, texts_split):
    rel_labels_df = new_metadata_df[semantic_subgenres_lt]
    #rel_labels_df = text2features.calculate_minmax(rel_labels_df)
    rel_labels_df["author.name"] = new_metadata_df["author.name"]
    rel_labels_df["title"] = new_metadata_df["title"]
    rel_labels_df["litHist.pages"] = new_metadata_df["litHist.pages"]
    rel_labels_df["title.main"] = new_metadata_df["title.main"]
    
    edges_subgenres_novels_lt = []
    
    for subgenre in semantic_subgenres_lt:
        for index, row in rel_labels_df.loc[~rel_labels_df.index.isin(texts_split)].loc[rel_labels_df[subgenre]> 0].sort_values(by="litHist.pages",ascending=False).iloc[0:amount_of_texts_per_subgenre,:].iterrows():
            edges_subgenres_novels_lt.append([subgenre, index, row["title"]+ ", de "+row["author.name"],row[subgenre]*10, "subgenre-novel",row["title.main"]])
    
    subgenres_novels_edges_df = pd.DataFrame(edges_subgenres_novels_lt, columns=["source","target","target-label","weight","type","title.main"])
    return subgenres_novels_edges_df
    

def make_text_features_edges(zscores_selected_features_df, subgenres_novels_edges_df, texts_without_genre_lt, std_deviation_of_feature_in_text, amount_of_features_per_text, subgenre_features_edges_df, texts_split, with_texts_without_genre = True):
    if  with_texts_without_genre == True:
        zscores_selected_features_selected_works_df = zscores_selected_features_df.loc[list(set(subgenres_novels_edges_df["target"].tolist())) + texts_without_genre_lt ].sort_index()
    else:
        zscores_selected_features_selected_works_df = zscores_selected_features_df.loc[list(set(subgenres_novels_edges_df["target"].tolist()))].sort_index()

    text_features_edges_lt = []

    for feature_st in zscores_selected_features_df.columns.tolist():
        if feature_st in subgenre_features_edges_df["target"].tolist():
            text_features_edges_lt = text_features_edges_lt + [[feature_st, text, value, "feature-novel"] for text, value in zscores_selected_features_selected_works_df.loc[~zscores_selected_features_selected_works_df.index.isin(texts_split)].loc[zscores_selected_features_selected_works_df[feature_st] > std_deviation_of_feature_in_text].sort_values(by=feature_st, ascending=False)[feature_st].head(amount_of_features_per_text).items()]    
            #text_features_edges_lt = text_features_edges_lt + [[feature_st, text, value, "feature-novel"] for text, value in zscores_selected_features_selected_works_df.sort_values(by=feature_st, ascending=False)[feature_st].head(5).items()]    
    
    text_features_edges_df = pd.DataFrame(text_features_edges_lt, columns=["source","target","weight","type"])


    return text_features_edges_df


def filter_nodes(nodes_df, edges_df, texts_split, texts_without_genre_lt):
    print(nodes_df.shape)
    for node in nodes_df["node"]:
        if (node not in edges_df["source"].tolist() and node not in edges_df["target"].tolist() and node not in texts_without_genre_lt) or (node in texts_split):
            nodes_df = nodes_df.loc[nodes_df["node"] != node]        
    print(nodes_df.shape)
    return nodes_df


def make_graph(nodes_df, edges_df):
    
    graph_gp = nx.Graph()
    graph_gp.add_nodes_from(nodes_df.loc[nodes_df["type"] =="text"]["node"].tolist(), bipartite=2)
    graph_gp.add_nodes_from(nodes_df.loc[nodes_df["type"] =="feature"]["node"].tolist(), bipartite=1)
    graph_gp.add_nodes_from(nodes_df.loc[nodes_df["type"] =="subgenre"]["node"].tolist(), bipartite=0)

    nx.set_node_attributes(G = graph_gp, name = 'size', values = {k:v for (k,v) in zip(nodes_df["node"], nodes_df["size"])})
    nx.set_node_attributes(G = graph_gp, name = 'label', values = {k:v for (k,v) in zip(nodes_df["node"], nodes_df["label"])})

    graph_gp.nodes(data=True)
    
    graph_gp.add_weighted_edges_from(
    edges_df[["source","target","weight"]].values,
    weight='weight')
    return graph_gp



def make_attributes(graph_gp, color_dict = {0:'#df68fd',1:'#8db9d8', 2:"#b3e6b3"}, size_multiplicator=300):
    #print(graph_gp.nodes(data=True))
    #print(graph_gp.edges(data=True))
    colors_list = [color_dict[i[1]] for i in graph_gp.nodes.data('bipartite')]

    sizes_list = [[graph_gp.node[n]['size'] * size_multiplicator] for n in graph_gp.nodes()]
    
    weights_list = [w['weight'] for (u, v, w) in graph_gp.edges(data=True)]
    
    labels_list = {}
    for key, values in dict(graph_gp.nodes(data=True)).items():
        labels_list[key] = values["label"]

    return colors_list, sizes_list, weights_list, labels_list



def add_in_network_to_metadata(new_metadata_df, graph_gp):
    new_metadata_df["in_network"] = 0
    for novel in new_metadata_df.index.tolist():
        if novel in list(graph_gp.nodes().keys()):
            new_metadata_df.loc[new_metadata_df["idno"] == novel, "in_network"] = 1
    return new_metadata_df

def calculate_centralities(graph_gp, subgenres_nodes_df, new_metadata_df, semantic_subgenres_lt):
    eigenvector_list = nx.eigenvector_centrality_numpy(graph_gp)
    betweenness_list = nx.betweenness_centrality(graph_gp)
    degree_list = nx.degree(graph_gp)


    subgenres_nodes_df["eigenvector"] = 0
    subgenres_nodes_df["betweenness"] = 0
    subgenres_nodes_df["degree"] = 0
    #print(subgenres_nodes_df.head())
    
    #print(eigenvector_list["_naturalista"])
    for semantic_subgenre_st in semantic_subgenres_lt:
        subgenres_nodes_df.loc[semantic_subgenre_st,"eigenvector"] = eigenvector_list[semantic_subgenre_st]
        subgenres_nodes_df.loc[semantic_subgenre_st,"betweenness"] = betweenness_list[semantic_subgenre_st]
        subgenres_nodes_df.loc[semantic_subgenre_st,"degree"] = degree_list[semantic_subgenre_st]

    
    new_metadata_df = add_in_network_to_metadata(new_metadata_df, graph_gp)
    
    new_metadata_df["eigenvector"] = 0
    new_metadata_df["betweenness"] = 0
    new_metadata_df["degree"] = 0

    for novel in new_metadata_df["idno"].tolist():
        try:
            new_metadata_df.loc[new_metadata_df["idno"] == novel,"eigenvector"] = eigenvector_list[novel]
            new_metadata_df.loc[new_metadata_df["idno"] == novel,"betweenness"] = betweenness_list[novel]
            new_metadata_df.loc[new_metadata_df["idno"] == novel,"degree"] = degree_list[novel]
        except:
            pass

        
    return subgenres_nodes_df, new_metadata_df


def make_complete_graph(bool_new_metadata_df, new_metadata_df, semantic_subgenres_lt,
                       selected_features_annotation_df, metadata_features_df,
                       results_features_zscores_df,
                        amounts_of_features_per_subgenre, amounts_of_texts_per_subgenre, amounts_of_features_per_text,
                        stds_deviation_of_feature_in_subgenre, stds_deviation_of_feature_in_text,
                        zscores_selected_features_df, texts_split, wdir, texts_without_genre_lt,
                        with_texts_without_genre = True
                       ):

    results_list = []

    for amount_of_features_per_subgenre in amounts_of_features_per_subgenre:
        for amount_of_texts_per_subgenre in amounts_of_texts_per_subgenre:
            for amount_of_features_per_text in amounts_of_features_per_text:
                for std_deviation_of_feature_in_subgenre in stds_deviation_of_feature_in_subgenre:
                    for std_deviation_of_feature_in_text in stds_deviation_of_feature_in_text:

                        graph_gp = nx.Graph()

                        subgenres_nodes_df = make_subgenre_nodes(bool_new_metadata_df, new_metadata_df, semantic_subgenres_lt, wdir)
                        texts_nodes_df = make_texts_nodes(new_metadata_df)
                        annotation_nodes_df = make_annotation_nodes(selected_features_annotation_df)
                        metadata_nodes_df = make_annotation_nodes(metadata_features_df)

                        #nodes_df = pd.concat([subgenres_nodes_df[["node","size","type","label"]], texts_nodes_df[["node","size","type","label"]], metadata_nodes_df[["node","size","type","label"]], annotation_nodes_df[["node","size","type","label"]]],axis=0)
                        nodes_df = pd.concat([subgenres_nodes_df[["node","size","type","label"]], texts_nodes_df[["node","size","type","label","title.main"]], metadata_nodes_df[["node","size","type","label"]], annotation_nodes_df[["node","size","type","label"]]],axis=0).fillna("")

                        subgenre_features_edges_df = make_subgenre_features_edges(results_features_zscores_df, amount_of_features_per_subgenre, std_deviation_of_feature_in_subgenre, semantic_subgenres_lt)


                        subgenres_novels_edges_df= make_subgenres_novels_edges(new_metadata_df, semantic_subgenres_lt, amount_of_texts_per_subgenre, texts_split)

                        text_features_edges_df = make_text_features_edges(zscores_selected_features_df, subgenres_novels_edges_df, texts_without_genre_lt, std_deviation_of_feature_in_text, amount_of_features_per_text, subgenre_features_edges_df, texts_split, with_texts_without_genre = with_texts_without_genre) 

                        edges_df = pd.concat([subgenre_features_edges_df, subgenres_novels_edges_df, text_features_edges_df]).fillna("")

                        edges_df = edges_df.loc[~edges_df["target"].isin(texts_split)]

                        nodes_df = filter_nodes(nodes_df, edges_df, texts_split, texts_without_genre_lt) 
                        #print(nodes_df)

                        graph_gp = make_graph(nodes_df, edges_df)

                        colors_list, sizes_list, weights_list, labels_list = make_attributes(graph_gp)
                        #print((graph_gp.nodes()))
    
                        modified_subgenres_nodes_df, modified_new_metadata_df = calculate_centralities(graph_gp, subgenres_nodes_df.copy(), new_metadata_df.copy(), semantic_subgenres_lt)


                        r_correlation_subgenres, p_correlation_subgenres = stats.pearsonr(
                        modified_subgenres_nodes_df["eigenvector"],
                        modified_subgenres_nodes_df["mean.f1.ling-lit"],
                        )

                        r_correlation_novels, p_correlation_novels = stats.pearsonr(
                        modified_new_metadata_df.loc[modified_new_metadata_df["in_network"]==1]["eigenvector"],
                        modified_new_metadata_df.loc[modified_new_metadata_df["in_network"]==1]["litHist.pages"],
                        )
                        print(amount_of_features_per_subgenre, amount_of_texts_per_subgenre, amount_of_features_per_text, std_deviation_of_feature_in_subgenre, std_deviation_of_feature_in_text, round(r_correlation_subgenres,2), round(p_correlation_subgenres,2), round(r_correlation_novels,2), round(p_correlation_novels,2))


                        results_list.append([r_correlation_subgenres, p_correlation_subgenres, r_correlation_novels, p_correlation_novels, amount_of_features_per_subgenre, amount_of_texts_per_subgenre, amount_of_features_per_text, std_deviation_of_feature_in_subgenre, std_deviation_of_feature_in_text])

    results_df = pd.DataFrame(results_list, columns = ["r-eigenvector-mean.f1.ling-lit","pvalue-eigenvector-mean.f1.ling-lit","r-eigenvector-litHist.pages","pvalue-eigenvector-litHist.pages", "amount_of_features_per_subgenre", "amount_of_texts_per_subgenre", "amount_of_features_per_text", "std_deviation_of_feature_in_subgenre", "std_deviation_of_feature_in_text"])
    results_df["mean-r"] = results_df[["r-eigenvector-mean.f1.ling-lit","r-eigenvector-litHist.pages"]].abs().mean(axis=1).round(2)

    return modified_subgenres_nodes_df, edges_df, results_df, graph_gp, colors_list, sizes_list, weights_list, labels_list
    


def neighborhood(graph_gp, node, n):
    path_lengths = nx.single_source_dijkstra_path_length(graph_gp, node)
    return [node for node, length in path_lengths.items() if int(length) == int(n)]



def plot_graph(graph_gp, labels_list, weights_list, colors_list, sizes_list, wdir, figsize=(20,30), pos_str = "neato", k = 3, iterations = 1500, savefig = False, title=""):
    if pos_str == "neato":
        pos = nx.nx_agraph.graphviz_layout(graph_gp, prog="neato")
    elif pos_str == "spring":
        pos = nx.spring_layout(graph_gp, k = k, iterations = iterations)
    plt.figure(figsize = figsize)
    plt.axis('off')

    nx.draw_networkx_labels(graph_gp, pos, labels = labels_list, font_color = 'black', alpha = 0.8, font_size=12)
    nx.draw_networkx_edges(graph_gp, pos, width = weights_list, edge_color = "#d9d9d9")
    nx.draw_networkx_nodes(graph_gp, pos, node_color = colors_list, node_size = sizes_list)
    plt.title(title, y = 0, fontdict = { "fontsize" : 18})
    
    plt.tight_layout()
    if savefig == True:
        plt.savefig(wdir+"/visualisations/graph_"+pos_str+".png",dpi=300, format="png")
    plt.show()
    
    