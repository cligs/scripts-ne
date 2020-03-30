# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 09:35:43 2018

@author: jose
"""

import rpy2.robjects as ro
import pandas as pd
import numpy as np
def load_stylo():
    R = ro.r
    R.library("stylo")
    print("Stylo version: ", R.packageVersion("stylo"))
    R('''
    cosine_delta <- function(x){
        # z-scoring the input matrix of frequencies
        x = scale(x)
        # computing cosine dissimilarity
        y = as.dist( x %*% t(x) / (sqrt(rowSums(x^2) %*% t(rowSums(x^2)))) )
        # then, turning it into cosine similarity
        z = 1 - y
        # getting the results
        return(z)
    }
    ''')
    return R
    
def calculate_delta(wdir, mfwmin = 5000, mfwmax = 5000, distance_measure = "dist.eder", wsdir="corpus/", height =5):
    R = load_stylo()
    R.setwd(wdir)
    results = R.stylo(
        **{
            "gui" : False,
            "analyzed.features" : "w",
            "ngram.size" : 1,
            "preserve.case" : False,
            "mfw.min" : mfwmin,
            "mfw.max" : mfwmax,
            "mfw.list.cutoff" : mfwmax,
            "analysis.type" : "CA",
            "distance.measure" : distance_measure,
            "sampling" : "no.sampling",
            "display.on.screen" : False,
            "write.png.file" : True,
            "save.distance.tables" : True,
            "save.analyzed.features" : True,
            "save.analyzed.freqs" : True,
            "corpus.dir": wsdir,
            "plot.custom.height" : height,
        }
    )
    delta_matrix = results[0]
    delta_matrix = pd.DataFrame(np.array(delta_matrix),
                                       index=list(R.colnames(delta_matrix)),
                                       columns=list(R.colnames(delta_matrix)))
    return delta_matrix
def calculate_PCA(wdir, mfwmin = 5000, mfwmax = 5000, wsdir="corpus/"):
    R = load_stylo()
    R.setwd(wdir)
    results = R.stylo(
        **{
            "gui" : False,
            "analyzed.features" : "w",
            "ngram.size" : 1,
            "preserve.case" : False,
            "mfw.min" : mfwmin,
            "mfw.max" : mfwmax,
            "mfw.list.cutoff" : mfwmax,
            "analysis.type" : "PCR",
            "sampling" : "no.sampling",
            "display.on.screen" : False,
            "write.png.file" : True,
            "save.distance.tables" : True,
            "save.analyzed.features" : True,
            "save.analyzed.freqs" : True,
            "pca.visual.flavour" : "classic",
            "corpus.dir": wsdir,
        }
    )
    return results
#def plot_average_delta(delta_matrix):
#    delta_matrix_mean = delta_matrix.mean()
    
"""
#features = results[1]
wdir = "/home/jose/Dropbox/Doktorarbeit/thesis/corpora/75_corpus/"
delta_matrix = calculate_delta(wdir, distance_measure="cosine_delta")
"""
"""
wdir = "/home/jose/Dropbox/MTB/proyectos/externos/Moretianos/corpus/corpus/corpus-derivados/solo-moreto-seguro"
calculate_PCA(wdir, mfwmin = 2000, mfwmax = 2000)
"""