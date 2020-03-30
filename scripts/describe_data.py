# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 11:35:42 2017

@author: jose

These scripts make different kind of plots (bars, scatter, histograms, heatmaps) from Pandas dataframes.
On the bottom of this script there is an example of how to use it.

"""
import sys
import os
sys.path.append(os.path.abspath("/home/jose/Dropbox/Doktorarbeit/"))
from reading_robot import load_data, cull_data
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import os
import numpy as np
import seaborn as sns
from sklearn import tree
import graphviz
import numbers

# Definition of colors ans styles
sns.set(palette = "viridis")
plt.style.use("seaborn-whitegrid")

format_ = "png"
basic_color = "#440154"
alternative_basic_color = "#a457b7"
classic_blue ="#31688e"
yellow = "#fde725"
color_boxplots = dict(boxes = basic_color, whiskers = alternative_basic_color,
                medians = basic_color, caps = basic_color, means = alternative_basic_color)
flierprops = dict(marker = 'o', markerfacecolor = alternative_basic_color, markersize = 5)
boxprops = dict(linewidth = 5)
medianprops = dict(linewidth = 5)
meanlineprops = dict(linestyle = '--', linewidth = 2)

name_colors = ['purple', 'crimson', 'ivory', 'lightgray', 'cornsilk', 'darkorange', 'blanchedalmond', 'mediumaquamarine', 'salmon', 'deepskyblue', 'forestgreen', 'darkslategrey', 'darkcyan', 'azure', 'powderblue', 'black', 'mediumblue', 'tan', 'lavenderblush', 'plum', 'darkgrey', 'lightslategrey', 'paleturquoise', 'coral', 'blue', 'pink', 'darkgray', 'grey', 'lemonchiffon', 'steelblue', 'mediumorchid', 'tomato', 'orchid', 'lightsalmon', 'fuchsia', 'firebrick', 'slategrey', 'lightgreen', 'darkslategray', 'dimgray', 'dimgrey', 'gold', 'darkred', 'darksalmon', 'papayawhip', 'midnightblue', 'beige', 'darkseagreen', 'lawngreen', 'maroon', 'orangered', 'lightcyan', 'rosybrown', 'lightslategray', 'magenta', 'yellow', 'darkmagenta', 'palevioletred', 'darkorchid', 'sienna', 'oldlace', 'yellowgreen', 'darkgreen', 'bisque', 'mediumseagreen', 'darkblue', 'seagreen', 'darkolivegreen', 'aqua', 'silver', 'whitesmoke', 'cadetblue', 'khaki', 'thistle', 'mistyrose', 'burlywood', 'darkslateblue', 'lightskyblue', 'ghostwhite', 'honeydew', 'floralwhite', 'brown', 'lightblue', 'saddlebrown', 'linen', 'palegreen', 'white', 'darkturquoise', 'olivedrab', 'goldenrod', 'greenyellow', 'slategray', 'chocolate', 'wheat', 'navajowhite', 'violet', 'peachpuff', 'antiquewhite', 'mediumslateblue', 'sandybrown', 'rebeccapurple', 'seashell', 'dodgerblue', 'slateblue', 'lightgoldenrodyellow', 'indianred', 'cyan', 'turquoise', 'aquamarine', 'chartreuse', 'limegreen', 'aliceblue', 'gray', 'mediumturquoise', 'moccasin', 'snow', 'hotpink', 'darkviolet', 'lightsteelblue', 'gainsboro', 'blueviolet', 'palegoldenrod', 'skyblue', 'mintcream', 'lavender', 'teal', 'lightcoral', 'cornflowerblue', 'darkgoldenrod', 'peru', 'lightgrey', 'mediumpurple', 'springgreen', 'olive', 'red', 'darkkhaki', 'lime', 'deeppink', 'green', 'orange', 'mediumspringgreen', 'lightseagreen', 'mediumvioletred', 'lightyellow', 'lightpink', 'royalblue', 'navy', 'indigo']           

def facetGrid(data_df, data_plot, col, row, facet, wdir, dataset, hue = "", aspect=1.5, type_=plt.scatter, alpha = 0.5, ylim = 0, rotation = 0):
    """
    Without hue:
    facetGrid(results_df, data_plot = "mean F1", col = "classifier_name", row = "text_representation", facet = "MFW", wdir = wdir, dataset = "results of evaluation of genre in CORDE")

    With hue:
    facetGrid(results_df, data_plot = "mean F1", col = "text_representation", row = "genre", hue="classifier_name", facet = "MFW", wdir = wdir, dataset = "results of evaluation of genre in CORDE")
    """
    if type_ == plt.scatter:
        if hue == "":
            g = sns.FacetGrid(data_df, col = col, row = row, aspect = aspect, margin_titles=True, ylim = ylim)
            g.map(type_, facet, data_plot, alpha = alpha)
            plt.subplots_adjust(top = 0.95, right = 0.98)
        else:
            g = sns.FacetGrid(data_df, col = col, row = row, hue = hue, aspect = aspect, margin_titles=True, legend_out = True, ylim = ylim)
            g = (g.map(type_, facet, data_plot, alpha = alpha ) .add_legend())
            plt.subplots_adjust(top = 0.95, right = 0.93)
    elif type_ == plt.boxplot or type_ == sns.boxplot:
            g = sns.FacetGrid(data_df, col = col, row = row, aspect = aspect, margin_titles=True, ylim = ylim)
            g.map(type_, facet, data_plot, color=classic_blue)
            plt.subplots_adjust(top = 0.95, right = 0.98)

    [plt.setp(ax.get_xticklabels(), rotation = rotation) for ax in g.axes.flat]

    g.fig.suptitle("Facet Grid with " + col + ", "+ row + " and " + facet + " in "+ dataset)

    create_dir(wdir, "visualisations/")

    plt.savefig(wdir + "visualisations/" + dataset + "_scatt_" + col + "_" + row + "_" + facet + "_" + hue + '.'+format_, dpi=300, format=format_)
    plt.show()

def create_dir(wdir, name):
    if not os.path.exists(wdir+name):
        os.makedirs(wdir+name)

def grouped_bars(metadata, class_, wdir, dataset, horizontal=False, rot = 45, figsize = (12,6), quantitative_axis_label="amount"):
    metadata_class = metadata.groupby([class_])[class_].count().copy()
    #print((metadata_class))
    fig, ax1 = plt.subplots()
    
    

    if horizontal == False:
        kind = "bar"
        figsize = figsize

        if metadata_class.values.max() > (metadata_class.values.sum()/3):
            print("for of data is good")
            ax1.set_ylabel('proportion')
            ax1.grid(False)
            ax2 = ax1.twinx()
            ax2.set_ylim((0,metadata_class.values.sum()))
            ax2.set_ylabel(quantitative_axis_label)
        else:
            ax1.set_ylabel(quantitative_axis_label)
        ax1.set_xlabel(class_)

    else:
        kind = "barh"
        figsize = (figsize[1],figsize[0])
        metadata_class = metadata_class.sort_values(ascending=False)
        ax1.set_xlabel(quantitative_axis_label)


    fig = metadata_class.plot(kind = kind, figsize = figsize, title="Bar plot of "+dataset+" by "+class_, color=[basic_color])
    fig.set_axisbelow(True)
    ax1.tick_params(axis="x", labelrotation = rot)
    plt.tight_layout()
    create_dir(wdir, "/visualisations")
    fig.get_figure().savefig(wdir+"/visualisations/"+dataset+"_g-bar_"+class_+'.'+format_, dpi=300, format=format_)
    
    plt.show(fig)

def simple_bars(data, feature, wdir, dataset, color_label_position = [], logx=False, xlim=""):

    fig, ax = plt.subplots()

    if len(data.shape) > 1:
        #data = data.sort_index()
        fig = data.T.plot(
                         #color = basic_color,
                         legend = False,
                         kind="barh",
                         title = "Bar plot of " + feature + " in "+ dataset,
                         logx = logx
                         )

    else:
        data = data.sort_values(ascending=False)

        fig = data.plot(
                         color = basic_color,
                         legend=False,
                         kind="barh",
                         title = "Bar plot of " + feature + " in "+ dataset
                         )

    if type(color_label_position) == int:
        fig.get_yticklabels()[color_label_position].set_color(classic_blue)
    elif type(color_label_position) == list and len(color_label_position) > 0:
        for position in color_label_position:
            fig.get_yticklabels()[position].set_color(classic_blue)

    fig.set_xlabel(feature)
    plt.tight_layout()
    create_dir(wdir, "/visualisations")
    fig.get_figure().savefig(wdir+"/visualisations/"+dataset+"_bar_"+feature+'.'+format_, dpi=300, format=format_)
    plt.show(fig)


def simple_grouped_bars(metadata, class_numerical, class_categorical, wdir, dataset, class_color, class_order = "",  use_colors = True, colors_values = [basic_color], figsize = ""):
    if class_order == "":
        class_order = class_numerical
    if figsize == "":
        figsize = ([5,metadata.shape[0]/6])
    metadata = metadata.sort_values( class_order, ascending=False)
    fig, ax = plt.subplots()
    if use_colors == True:
        colors = metadata[class_color].map(colors_values)
    else:
        colors = basic_color

    fig = metadata.plot( x = class_categorical, y = class_numerical,
                             color = colors,
                             figsize = figsize,
                             legend=False,
                             kind="barh",
                             title = "Bar plot of " + class_numerical + " by " +class_categorical
                             )
    fig.set_xlabel(class_numerical)
    plt.tight_layout()
    create_dir(wdir, "/visualisations")
    fig.get_figure().savefig(wdir+"/visualisations/"+dataset+"_s-bar_"+class_categorical+'.'+format_, dpi=300, format=format_)
    plt.show(fig)

def heatmap(wdir, df, title, absolute = True):
    correlation_df = df.corr()
    if absolute == True:
        correlation_df = correlation_df.abs()
    
    correlation_df = correlation_df.reindex(correlation_df.mean().sort_values(ascending=True).index, axis=1)
    correlation_df = correlation_df.reindex(correlation_df.mean().sort_values(ascending=True).index, axis=0)

    fig, ax = plt.subplots()
    ax = plt.pcolor(correlation_df, cmap="Reds", vmin=0.2, vmax=1)
    plt.title(title)
    plt.yticks(np.arange(0.5, len(correlation_df.index), 1), correlation_df.index)
    plt.xticks(np.arange(0.5, len(correlation_df.columns), 1), correlation_df.columns, rotation=90)
    fig.tight_layout()
    create_dir(wdir, "/visualisations")
    ax.get_figure().savefig(wdir+"/visualisations/"+title+"_heatmap_."+format_, dpi=300, format=format_)
    fig.show()

def chronological_heatmap(wdir, metadata, class_2, dataset, class_1 = "year",  amount_unities = 1):

    if amount_unities == 2:
        metadata[str(amount_unities)+"_"+class_1] = ((metadata[class_1] + (((-1)**(metadata[class_1]+1))-1)/2)+1).map(lambda x: str(x)[0:4])
        class_1 = str(amount_unities)+"_"+class_1

    metadata_classes = metadata.groupby([class_1,class_2])[class_1].count().unstack(class_2).fillna(0).copy()
    metadata_classes = metadata_classes.T
    metadata_classes = metadata_classes.sort_values(by = metadata_classes.columns.tolist(), ascending=False) #
    print(metadata_classes.shape)
    if metadata_classes.shape[0] > 20 or metadata_classes.shape[1] > 20:
        print(((metadata_classes.shape[1]/2)+1, (metadata_classes.shape[0]/2)+1))
        fig, ax = plt.subplots(figsize=(int(metadata_classes.shape[1]/3)+5, int(metadata_classes.shape[0]/3)))
    else:
        fig, ax = plt.subplots()
        
    if metadata_classes.values.max() > 10:
        vmax = metadata_classes.values.max()/2
    else:
        vmax = metadata_classes.values.max()
    fig = sns.heatmap(metadata_classes, cmap='Reds', annot=True, ax = ax, vmin = 0,vmax = vmax, linecolor="white", cbar =False, linewidths=2,) # 
    ax.set_title("Distribution of "+class_2 +"\n in "+dataset+" over "+class_1+"s")
    ax.tick_params(labeltop=False, labelright=True)
    plt.yticks(rotation=0)
    plt.xticks(rotation=45)
    plt.tight_layout()
    create_dir(wdir, "/visualisations")
    fig.get_figure().savefig(wdir+"/visualisations/"+dataset+"_heatmap_"+class_1+str(amount_unities)+"_"+class_2+'.'+format_, dpi=300, format=format_)
    plt.show(fig)

def histogram(metadata, class_, wdir, dataset, bins = 20, kind ="size"):

    fig, ax = plt.subplots()
    if kind == "size":
        metadata_class = metadata.groupby(class_).size()
    elif kind == "normal":
        metadata_class = metadata[class_].copy()
        
    ax.hist(metadata_class, facecolor=basic_color, bins = bins)

    ax.set_xlabel(class_)
    ax.set_ylabel("Number of texts")
    ax.tick_params(axis="x", labelrotation = 45)
    
    ax.set_title(r"Histogram of "+dataset+" by "+class_)
    fig.tight_layout()
    create_dir(wdir, "/visualisations")
    fig.savefig(wdir+"/visualisations/"+dataset+"_hist-count_"+class_+'.'+format_, dpi=300, format=format_)
    
    plt.show()
    
def histogram_counter(class_counter, class_, wdir, dataset, bins = 20):
    class_counter = list((class_counter).values())[1:]
    fig, ax1 = plt.subplots()
    ax1.set_xlabel(class_)
    ax1 = plt.hist(class_counter, facecolor=basic_color, bins = bins)
    plt.title(r"Histogram of "+dataset+" in "+class_)
    plt.tight_layout()
    create_dir(wdir, "/visualisations")
    plt.savefig(wdir+"/visualisations/"+dataset+"_hist-count_"+class_+'.'+format_, dpi=300, format=format_)
    
    plt.show()

def simple_boxplot(metadata, classes, wdir, dataset, do_cull = True, problematic_values = ["?","n.av.", None, "unknown","other","mixed","-"], ylabel = "", rotation=45):

    if do_cull == True:
        for class_ in classes:
            metadata = cull_data.cull_metadata(metadata, class_, problematic_values)
            metadata[class_] = pd.to_numeric(metadata[class_], errors='ignore')


    ax = metadata[classes].plot(kind="box",
                           meanline = True, showmeans = True, flierprops = flierprops,
                           boxprops = boxprops, medianprops = medianprops, meanprops = meanlineprops)
                           #, , sym='r+')
    fig = ax.get_figure()
    if len(ylabel) > 0:
        ax.set_ylabel(ylabel)
    if len(classes) > 5:
        subtitle_classes = "several classes"
        plt.xticks(rotation = rotation)
    else:
        subtitle_classes = " ".join(classes)
        
    ax.set_title(r"Box plot of "+ subtitle_classes +" in "+dataset)
    fig.tight_layout()
    create_dir(wdir, "visualisations/")
    fig.savefig(wdir+"visualisations/"+dataset+"_box_"+subtitle_classes+'.'+format_, dpi=300, format=format_)
    
    fig.show()

def boxplots(metadata, class_, class_by, wdir, dataset, rotation = 45, figsize = [], baseline = "", xlabel = "", ylabel = "", ylim = (), sort_by = "index", color_label_position = ""):
    """
    metadata, class_, class_by, wdir, dataset, rotation = 45, figsize = [], baseline = "", xlabel = "", ylabel = "", ylim =""
    """
    if len(ylim) == 0:
        ylim = (metadata[class_].values.min(),metadata[class_].values.max())
    else:
        ylim = ylim
    fig, ax1 = plt.subplots()
    ax1.tick_params(labeltop = False)
    if sort_by == "median":
        
          metadata2 = pd.DataFrame({col:vals[class_] for col, vals in metadata.groupby(class_by)})
          meds = metadata2.median().sort_values()
          ax1 = metadata2[meds.index].boxplot(rot=90,meanline = True, showmeans = True, flierprops = flierprops,
                           boxprops = boxprops, medianprops = medianprops, meanprops = meanlineprops,)


    else:
        ax1 = metadata.boxplot(column = class_, by = class_by,
                           meanline = True, showmeans = True, flierprops = flierprops,
                           boxprops = boxprops, medianprops = medianprops, meanprops = meanlineprops,
                           )
    ax1.set_ylim(ylim)
    plt.title(r"Box plot of " + class_ + " over "+ class_by +" in "+dataset)
    plt.suptitle(r"")

    
    if baseline != "":
    
        baseline = (len(set(metadata[class_]))+2) * [baseline]
        plt.plot(baseline, color = "red")

    if figsize != []:
        plt.rcParams["figure.figsize"] = figsize

    if xlabel == "":
        xlabel = class_by
    if ylabel == "":
        ylabel = class_

    ax1.set_xlabel(xlabel)
    if len(color_label_position) > 0:
        if type(color_label_position) == list:
            for color_label_position_i in color_label_position:
                ax1.get_xticklabels()[color_label_position_i].set_color(classic_blue)
        
        else:
            ax1.get_xticklabels()[color_label_position].set_color(classic_blue)
    ax1.set_ylabel(ylabel)
    plt.xticks(rotation = rotation)
    plt.tight_layout()
    create_dir(wdir, "/visualisations")
    plt.savefig(wdir+"/visualisations/"+dataset+"_box_"+"-"+class_+"-"+class_by+'.'+format_, dpi=300, format=format_)
    
    plt.show()

def describe_corpus_scatter(metadata, wdir, class_1, class_2, dataset, do_cull = True, problematic_values = ["?","n.av.", None, "unknown","other","mixed", "-"], fit_reg=True, annotation_lt = [], height = 5 , aspect = 1, title = ""):
    metadata_classes = metadata[[class_1,class_2]]
    if do_cull == True:
        metadata_classes = cull_data.cull_metadata(metadata, class_1, problematic_values)
        metadata_classes = cull_data.cull_metadata(metadata, class_2, problematic_values)

    metadata_classes[class_2] = pd.to_numeric(metadata_classes[class_2], errors='ignore')
    metadata_classes[class_1] = pd.to_numeric(metadata_classes[class_1], errors='ignore')

    #fig = metadata_classes.plot.scatter(class_1,class_2, title = , color=basic_color)
    g = sns.lmplot(x = class_1, y = class_2, data = metadata_classes, fit_reg = fit_reg, sharex = False, sharey = False, height = height, aspect = aspect )
    g.set(ylim=((metadata_classes[class_2].min()- metadata_classes[class_2].min()/10), (metadata_classes[class_2].max()+metadata_classes[class_2].max()/10)))
    
    if len(annotation_lt) > 0 :
        ax = g.axes[0,0]
    
        reduced_metadata_classes = metadata_classes.loc[annotation_lt]

        jitter_x = (reduced_metadata_classes[class_1].dropna().values.max() - reduced_metadata_classes[class_1].dropna().values.min()) / 50
        
        jitter_y = (reduced_metadata_classes[class_2].dropna().values.max() - reduced_metadata_classes[class_2].dropna().values.min() )/ 50

        for line in range(0,reduced_metadata_classes.shape[0]):
             ax.text(reduced_metadata_classes[class_1][line]  + jitter_x, reduced_metadata_classes[class_2][line] + jitter_y, reduced_metadata_classes.index[line], horizontalalignment='left', size='medium', color='black', alpha=0.8)
        
    if title == "":
        if fit_reg == True:
            subtitle = "\n(with regression line)"
        else:
            subtitle = ""
        if len(class_1) < 20:
            plt.title("Scatter plot of " +class_1 + " and "+ class_2 + subtitle)
        else:
            plt.title("Scatter plot of " +class_1 + "\n and "+ class_2 + subtitle)
    else:
        plt.title(title)
        
    plt.xticks(rotation = 45)
    plt.tight_layout()
    create_dir(wdir, "/visualisations")
    plt.savefig(wdir+"/visualisations/"+dataset+"_scatt_"+class_1+"_"+class_2+'.'+format_, dpi=300, format=format_)
    plt.show()
    return g

def scatter_color(data, wdir, class_1, class_2, metadata_color, dataset,  colors = [],  do_cull = True, problematic_values = ["?","n.av.", None, "unknown","other","mixed", "-"], cmap = "PRGn"):
    data_classes = data[[class_1,class_2]]

    if do_cull == True:
        data_classes = cull_data.cull_metadata(data_classes, class_1, problematic_values)
        data_classes = cull_data.cull_metadata(data_classes, class_2, problematic_values)

    """
    i = 0
    for class_ in list(set(metadata_color)):
        metadata_color.replace(class_, colors[i])
        i += 1
    #print(metadata_color)
    """
    fig, ax1 = plt.subplots()
    ax1.tick_params(labeltop = False)
    ax1 = data_classes.plot.scatter(class_1, class_2, c = metadata_color, cmap = cmap, alpha = 0.5, edgecolors='none')


    ax1.set_xlabel(class_1)
    ax1.set_ylabel(class_2)

    plt.tight_layout()

    create_dir(wdir, "/visualisations")
    plt.savefig(wdir+"/visualisations/"+dataset+"_scatt_color"+class_1+"_"+class_2+'.'+format_, dpi=300, format=format_)
    
    plt.show()

def bars_from_two_dataframes(metadata_1, dataset_1, metadata_2, dataset_2, class_, wdir):
    metadata_1 = metadata_1.groupby([class_])[class_].count().copy()
    metadata_2 = metadata_2.groupby([class_])[class_].count().copy()
    
    df_combined = pd.concat([metadata_1.rename(dataset_1), metadata_2.rename(dataset_2)], axis=1)

    fig = df_combined.plot(kind='bar', title="Bar plot of "+dataset_1+" and "+dataset_2+" by "+class_, grid= True,  cmap = "PiYG")
    fig.set_ylabel("Amount")
    plt.tight_layout()
    create_dir(wdir, "/visualisations")
    fig.get_figure().savefig(wdir+"/visualisations/"+dataset_1+"_"+dataset_2+"_bars_"+class_+"."+format_, dpi=300, format=format_)
    plt.legend(loc="upper left", frameon=True,edgecolor="gray")
    plt.show(fig)


def describe_corpus_stackedbars(metadata,  class_2, wdir, dataset, class_1 = "decade", in_proportions = False, legend = "", has_title = True, loc_legend = "lower left", do_cull = True, problematic_values =  ["?","n.av.", None, "unknown","other","mixed", "-"], cmap=""):
    """
        
    """
    if do_cull == True:
        metadata_classes = cull_data.cull_metadata(metadata, class_1, problematic_values)
        metadata_classes = cull_data.cull_metadata(metadata, class_2, problematic_values)
    if in_proportions == True:
        metadata_classes = (metadata_classes.groupby([class_1, class_2])[class_1].count().groupby(level = 0).transform(lambda x: x/x.sum())*100).unstack(class_2).fillna(0).copy()
        title_sub ="(in percentage)"
        ylabel = "Percentage of texts"
    else:
        metadata_classes = metadata.groupby([class_1,class_2])[class_1].count().unstack(class_2).fillna(0).copy()
        title_sub =""
        ylabel = "Number of texts"

    # Color options
    if cmap != "":
        cmap = cmap
    else:
        if len(metadata_classes.columns.tolist()) == 2:
            cmap = "viridis"
        elif "_ordi" in class_2:
            cmap = "viridis_r" # "copper_r"
        elif len(metadata_classes.columns.tolist()) >= 10:
            cmap = "tab20"
        else:
            cmap = "viridis"#'Accent'
    
    
    if has_title == True:
        title = dataset+" by "+class_1+" and "+class_2 + "\n" +title_sub
    else:
        title = ""
    
    ax = metadata_classes.plot(kind='bar', stacked=True, figsize=(12,6), title = title, grid= True,  cmap = cmap)
    ax.set_xlabel(class_1)
    ax.set_ylabel(ylabel)
    fig = ax.get_figure()
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelrotation=45)
    if legend == "":
        legend = metadata_classes
    ax.legend(legend, loc = loc_legend, frameon=True, edgecolor="gray", title = class_2)

    create_dir(wdir, "/visualisations")
    fig.savefig(wdir+"/visualisations/"+dataset+"_barstacked_"+class_1+"_"+class_2+'.'+format_, dpi=300, format=format_)
    fig.show()

def independent_boxplots_by_class(metadata, classes, class_by, class_by_values, wdir, dataset, has_title=True, do_cull = True, problematic_values=  ["?","n.av.", None, "unknown","other","mixed", "-"]):
    fig = plt.figure()
    for i in range(4):
        if i == 0:
            ax = fig.add_subplot(2, 2, i+1)
        else:
            ax = fig.add_subplot(2, 2, i+1, sharex=ax)

        if do_cull == True:
            metadata_classes = cull_data.cull_metadata(metadata, classes[i], problematic_values)
            metadata_classes[classes[i]] = metadata_classes[classes[i]].astype(int)
        
        ax.boxplot(
            [metadata_classes.loc[metadata_classes[class_by] == class_by_values[0],classes[i]].dropna(),
            metadata_classes.loc[metadata_classes[class_by] == class_by_values[1],classes[i]].dropna()],
            meanline = True, showmeans = True, flierprops = flierprops,
            boxprops = boxprops, medianprops = medianprops, meanprops = meanlineprops
        )
        # TODO: aquí podríamos en realidad sacar los valores del keywords
        ax.set_yticks(sorted(list(set(metadata_classes[classes[i]]))))
        ax.set_title(classes[i])
        ax.set_xticklabels(class_by_values)

    fig.tight_layout()
    fig.suptitle(class_by+" and other metadata",verticalalignment="top", y = 1, fontweight="bold", size=14)
    create_dir(wdir, "/visualisations")
    fig.savefig(wdir+"/visualisations/"+dataset+"_indepdendet_boxplots"+"-".join(classes)+'.'+format_, dpi=300, format=format_)

    fig.show()

def visualize_transformation(wdir, analyzed_corpus, analyzed_transformation, comparing_corpus, comparing_transformation, results_sample_df, text_name1 = "Pazos", text_name2 = "TiranoBanderas", amount_mfws_bars = 20,  amount_mfws = 2000, bins = 100):
    """
    Very specific function to visualize the transformation of features in chapter 5.2.
    * wdir: path for saving the figures
    * analyzed_corpus: transformed dataframe (rows texts, columns are features)
    * analyzed_transformation: name of the transformation
    * comparing_corpus: dataframe for comparison (normally total frequency or relative frequency)
    * comparing_transformation: name of the comparing transformation
    * results_sample_df: dataframe created with text2features.describe_transformation, in which the analyzed transformation has to be in the index
    * text_name1 = name of one text in the analyzed corpus, for example "Pazos" or "ne0033"
    * text_name2 = "TiranoBanderas",
    * amount_mfws_bars = 20,
    * amount_mfws = 2000,
    * bins = 100
    
    """
    fig, axes = plt.subplots(nrows = 2, ncols = 2, figsize = (9,9))
    fig.suptitle("Visualization of " + analyzed_transformation[0].upper() + analyzed_transformation[1:]+ " Frequency", fontsize=20)

    # plot 1
    plot1 = analyzed_corpus.loc[[text_name1,text_name2]].iloc[:,0:amount_mfws_bars].T.plot.bar(ax=axes[0,0], cmap = "PiYG") #metadata.plot.scatter("year","am.tokens")
    plot1.set_title("Sub-figure A:\n\nBar Plot with Frequency of " + str(amount_mfws_bars) + " MFW \nin Two Novels", fontsize=16)
    plot1.set_ylabel(analyzed_transformation + " frequency")
    plot1.set_xlabel(str(amount_mfws_bars)+" MFW in corpus")

    # plot2

    plot2 = analyzed_corpus.loc[[text_name1,text_name2]].iloc[:,0:amount_mfws].T.plot.hist(bins = bins, ax = axes[0,1], alpha = 0.5, cmap = "PiYG", label = text_name1)
    plot2.set_title("Sub-figure B:\nHistogram of " + str(amount_mfws) + " MFWs \n" + r'mean'+ " =  " + str(results_sample_df.loc[analyzed_transformation]["mean"].round(2)) + "\nstandard deviation = "+str(results_sample_df.loc[analyzed_transformation]["std"].round(2) ), fontsize=16)
    #plot2.legend("Frequencies of " +text_name1)
    plot2.set_xlabel(analyzed_transformation + " frequency")
    plot2.set_ylabel("frequency")

    # plot 3
    #plot3 = analyzed_corpus.loc[[text_name1, text_name2]].iloc[:,0:amount_mfws].T.plot.box(ax=axes[1,0])
    
    plot3 = analyzed_corpus.loc[[text_name1, text_name2]].iloc[:,0:amount_mfws].T.plot(kind="box",
                       meanline = True, showmeans = False, flierprops = flierprops,
                           boxprops = boxprops, medianprops = medianprops, meanprops = meanlineprops,
                       ax = axes[1,0], cmap = "PiYG")

        
    plot3.set_title('')
    plot3.set_title("Sub-figure C:\nBox Plots of " + str(amount_mfws) + " MFWs \nMedian =  " + str(results_sample_df.loc[analyzed_transformation]["median"].round(2)) + "\nIQR = "+str(results_sample_df.loc[analyzed_transformation]["IQR"].round(2)), fontsize=16)
    plot3.set_ylabel(analyzed_transformation + " frequency")

    
    # plot 4
    
    scatter_text_1_pd = pd.DataFrame([analyzed_corpus.loc[text_name1],comparing_corpus.loc[text_name1]], index = [analyzed_transformation, comparing_transformation]).T
    scatter_text_2_pd = pd.DataFrame([analyzed_corpus.loc[text_name2],comparing_corpus.loc[text_name2]], index = [analyzed_transformation, comparing_transformation]).T

    #plot4 = axes[1,1]
    axes[1,1].scatter(scatter_text_1_pd[comparing_transformation], scatter_text_1_pd[analyzed_transformation],  c = "Purple", alpha=0.3)

    axes[1,1].scatter(scatter_text_2_pd[comparing_transformation], scatter_text_2_pd[analyzed_transformation],  c = "Green", alpha=0.3)

    #.plot.scatter(x = comparing_transformation, y = analyzed_transformation, alpha=0.5, ax = axes[1,1], c = alternative_basic_color, label = text_name1)
    
    #plot4_2 = scatter_text_2_pd.plot.scatter(x = comparing_transformation, y = analyzed_transformation, alpha=0.5, ax = axes[1,1], c = alternative_basic_color, label = text_name1)
    #plot4 = plot4_1 + plot4_2
    axes[1,1].set_title("Sub-figure D:\nScatter Plot of Frequencies \nCorrelation (Pearson's r) =  " + str(results_sample_df.loc[analyzed_transformation]["Pearson's R"].round(2)) + "\np-value = "+str(results_sample_df.loc[analyzed_transformation]["R p-value"].round(2) ), fontsize=16)
    axes[1,1].set_alpha(1)
    axes[1,1].legend([text_name1,text_name2])
    axes[1,1].set_alpha(1)
    axes[1,1].set_xlabel(comparing_transformation + " frequency")
    axes[1,1].set_ylabel(analyzed_transformation + " frequency")

    fig.tight_layout(rect=[0, 0, 1, 0.95])

    create_dir(wdir, "/visualisations")
    fig.savefig(wdir+"/visualisations/"+analyzed_transformation+".png", dpi=300, format="png")
    
    fig.show()


def make_tree(class_, features_df, labels_df, wdir):

    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(features_df,labels_df[class_])

    dot_data = tree.export_graphviz(clf, out_file=None) 
    graph = graphviz.Source(dot_data) 
    dot_data = tree.export_graphviz(clf, out_file=None, 
                          feature_names=features_df.columns.tolist(),  
                          class_names=[str(item) for item in list(set(labels_df[class_]))],  
                          filled=True, rounded=True,  
                          special_characters=True)  
    graph = graphviz.Source(dot_data)
    
    graph.savefig(wdir+"/visualisations/tree_"+class_+'.'+format_, dpi=300, format=format_)
    graph.show()

    return graph


import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import ward, dendrogram


def plot_ward_dendrogram(features, wdir, figsize = (20,6), fontsize = 16, title = "Dendrogram of ", show_labels = True, xlabel = "Cluster identifiers", ylabel = "Cluster distance"):
    fig = plt.figure(figsize = figsize )
    linkage_array = ward(features)
    if show_labels == True:
        dendrogram(linkage_array, labels=features.index.tolist())
    else:
        dendrogram(linkage_array, no_labels=True)
    ax = plt.gca()
    bounds = ax.get_xbound()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    ax.set_title(title, fontsize = fontsize )

    ax.tick_params(labelsize=12)
    ax.set_axisbelow(True)
    ax.grid(axis="x")

    plt.tight_layout()
    create_dir(wdir, "/visualisations")
    fig.savefig(wdir+"/visualisations/"+title+".png", dpi=300, format="png")

    plt.show()
    

"""
wdir ="/home/jose/Dropbox/Doktorarbeit/thesis/data/"
metadata = "metadata_beta-opt-obl-subgenre-structure_ordi.csv"
dataset = "CoNSSA"
metadata = load_data.load_metadata(wdir = wdir, metadata_table = metadata, sep = "\t")

classes = ["type-end_ordi", "protagonist-age_ordi","protagonist-social-level_ordi","setting_ordi"]
class_by = "protagonist-gender"
class_by_values = ["male","female"]
independent_boxplots(metadata, classes, class_by, class_by_values,  wdir, dataset, has_title=True)
"""
"""
dataset = "CoNSSA-plus"
wdir ="/home/jose/Dropbox/Doktorarbeit/thesis/data/"

metadata = "metadata_beta-opt-obl-subgenre-structure_ordi.csv"
metadata = load_data.load_metadata(wdir = wdir, metadata_table = metadata, sep = "\t")

#describe_corpus_stackedbars(metadata = metadata, class_1 = "decade", class_2 = "setting-continent", wdir = wdir, dataset = dataset)

metadata = metadata.loc[(metadata['genre'] == "novel") & (metadata['text-histlit-pages'] >= 1)].copy()
dataset = "CoNSSA-core"
describe_corpus_stackedbars(metadata = metadata, class_1 = "decade", class_2 = "setting_ordi", wdir = wdir, dataset = dataset, in_proportions = True, legend =["boat","rural","small city","big-city"])

"""