# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 07:47:09 2018

@author: jose
"""
import pandas as pd
import os
import glob
import re
from collections import Counter

import sys
sys.path.append(os.path.abspath("/home/jose/Dropbox/Doktorarbeit/"))
import numpy as np



def corde_metadata2df(wdir = "/home/jose/Dropbox/biblioteca/corpora/corde/", wsdir="corde_comp/"):
    corde = []
    for doc in glob.glob(wdir+wsdir+"*.txt"):
        
        input_name  = os.path.splitext(os.path.split(doc)[1])[0]
        with open(doc, "r", errors="replace", encoding="utf-8") as fin:
            content = fin.read()
            print(input_name)
            titulo = re.findall(r'<TITULO>\s*(.*?)\s*</TITULO>', content)
            autor = re.findall(r'<AUTOR>\s*(.*?)\s*</AUTOR>', content)[0]
            pais = re.findall(r'<PAIS>\s*(.*?)\s*</PAIS>', content)[0]
            fechacre = re.findall(r'<FECHACRE>\s*(.*?)\s*</FECHACRE>', content)[0]
            tema = re.findall(r'<TEMA>(.*?)</TEMA>', content)
            medio = re.findall(r'<MEDIO>(.*?)</MEDIO>', content)[0]
            formas = re.findall(r'Número de formas ortográficas:\s*(\d+)', content)[0]
            tokens = re.findall(r'Número de elementos \(tokens\): (\d+)', content)[0]
            types = re.findall(r'Número de elementos distintos \(types\): (\d+)', content)[0]
            if titulo == []:
                titulo = ""
            else:
                titulo = titulo[0]
            if tema == []:
                tema = ""
            else:
                tema = tema[0]
            
            corde.append([input_name, titulo, autor, pais, fechacre, tema, medio, formas, tokens, types])
            #print(titulo)
            fin.close()
    
    dfcorde = pd.DataFrame(corde, columns=["archivo","titulo", "autor", "pais", "fechacre", "tema", "medio", "formas", "tokens", "types"])


    dfcorde["prim_fecha"] = dfcorde["fechacre"].str.extract(r"^.*?(\d\d\d\d*).*?$", expand=False).astype(float).fillna(0).astype(int)
    dfcorde["siglo"] = dfcorde["prim_fecha"].astype(str).str[0:2].astype(int)
    dfcorde.loc[dfcorde["siglo"]>19,["siglo"]] = dfcorde["prim_fecha"].astype(str).str[0:1]
    #dfcorde = dfcorde[dfcorde["siglo"].apply(lambda x: x.isnumeric())]
    dfcorde["siglo"] = pd.to_numeric(dfcorde["siglo"])
    dfcorde["siglo"] = dfcorde["siglo"]+1

    dfcorde["continente"] = "América"
    dfcorde.loc[dfcorde["pais"]=="España",["continente"]] = "Europa"
    dfcorde.loc[dfcorde["pais"]=="Filipinas",["continente"]] = "Asia"

    
    
    dfcorde["matiz_fecha"] = dfcorde["fechacre"].str.extract(r"^\s*([a-z]).*?$", expand=False).fillna("")
    dfcorde["prim_tema"] = dfcorde["tema"].str.extract(r"^\s*(.*?):.*?$", expand=False).fillna("")
    dfcorde["secund_tema"] = dfcorde["tema"].str.extract(r"^\s*.*?:\s*(.*?)$", expand=False).fillna("")

    dfcorde.loc[dfcorde["secund_tema"] == "", ["secund_tema"]] = dfcorde["tema"]


    dfcorde["abs_tema"] = "Técnico"
    dfcorde.loc[dfcorde["prim_tema"].str.contains(r"Prensa"),["abs_tema"]] = "Prensa"
    dfcorde.loc[dfcorde["prim_tema"].str.contains(r"Verso"),["abs_tema"]] = "Verso"
    dfcorde.loc[dfcorde["prim_tema"].str.contains(r"Prosa (dramática|narrativa)"),["abs_tema"]] = "Prosa"
    dfcorde.loc[dfcorde["prim_tema"] == "", ["prim_tema"]] = dfcorde["tema"]


    dfcorde.to_csv(wdir+wsdir[:-1]+"_metadatos.csv", encoding="utf-8", sep="\t")
    print(dfcorde)
    return corde
#corde = corde_metadata2df()

from io import StringIO
   
def corde_clean_vocabulary(wdir = "/home/jose/Dropbox/biblioteca/corpora/corde/", file="lista_formas_corde.txt"):
    with open(wdir+file, "r", errors="replace", encoding="utf-8") as fin:
        content = fin.read()
        content = re.sub(r'^ *', r'', content, flags = re.MULTILINE)
        content = re.sub(r' *\t *', r'\t', content, flags = re.MULTILINE)
        content = re.sub(r' *FORMA   Frec. gral. Frec. norm.', r'token\tfrecGral\tfrecNorm', content, flags = re.MULTILINE)
        df = pd.read_csv(StringIO(content), sep="\t")
    df = df.sort_values(by="frecNorm", ascending = False)

    df.iloc[0:7000].to_csv(wdir+"corde_vocabulary_5000.csv", sep="\t")
    
    df.to_csv(wdir+"corde_vocabulary.csv", sep="\t")
    return df
        
#df = corde_clean_vocabulary()    

def corde_csv2vocabulary(wdir = "/home/jose/Dropbox/biblioteca/corpora/corde/", wsdir="corde_comp_csv/"):
    i = 0
    vocabulary = []

    for doc in glob.glob(wdir+wsdir+"*.csv"):
        print(i)
        df = pd.read_csv(doc, sep="\t").fillna("")

        vocabulary = list(set(vocabulary+df["token"].str.lower().tolist()))
        #vocabulary = [item for sublist in vocabulary for item in list(set(sublist))]
        
        print(len(vocabulary))
        if i in list(range(1000,35000,1000)):
            for word in vocabulary:
                if type(word) != str:
                    print(word)
            print("printing")
            vocabulary_str = "\n".join(vocabulary)
            with open(wdir+"vocabulary.csv", "w", encoding="utf-8") as fout:
                fout.write(vocabulary_str)
        i += 1
    vocabulary_str = "\n".join(vocabulary)
    with open(wdir+"vocabulary.csv", "w", encoding="utf-8") as fout:
        fout.write(vocabulary_str)

    return vocabulary

#vocabulary = corde_csv2vocabulary(wsdir="corde_comp_csv/")

def corde_csv2csv_low(wdir = "/home/jose/Dropbox/biblioteca/corpora/corde/", wsdir="corde_samp_csv/"):
    i = 0
    for doc in glob.glob(wdir+wsdir+"*.csv"):
        input_name  = os.path.split(doc)[1]

        print(i)
        df = pd.read_csv(doc, sep="\t").fillna(0)
        df["token"] = df["token"].str.lower()
        df = pd.DataFrame(df.groupby("token").sum().unstack(["token"])["freq"].sort_values(ascending=False), columns =["freq"])
        df.to_csv(wdir+wsdir[0:-1]+"_low/" + input_name, sep="\t")
        i += 1
#corde_csv2csv_low(wsdir="corde_comp_csv/")        

def corde_csv2vocabulary_freq(wdir = "/home/jose/Dropbox/biblioteca/corpora/corde/", wsdir="corde_comp_csv/"):
    if "samp" in wsdir:
        mode = "samp"
    elif "comp" in wsdir:
        mode = "comp"
        
    vocabulary = pd.read_csv(wdir + "vocabulary.csv", sep="\t").dropna()
    print(vocabulary.shape)
    print(vocabulary.head())
    vocabulary["freq"] = 0
    vocabulary = vocabulary.set_index("token")
    vocabulary.head()
    i = 21000
    doc = glob.glob(wdir+wsdir+"*.csv")[21000]
    
    df = pd.read_csv(doc, sep="\t").fillna(0)
    df["token"] = df["token"].str.lower()
    df = pd.DataFrame(df.groupby("token").sum().unstack(["token"])["freq"].sort_values(ascending=False), columns =["freq"])
    vocabulary["freq"] = vocabulary["freq"].add(df["freq"], fill_value=0)
    vocabulary = vocabulary.sort_values("freq", ascending=False)
    i += 1
    print(i)
    print(vocabulary.head())
        
    for doc in glob.glob(wdir+wsdir+"*.csv")[21001:]:
        df = pd.read_csv(doc, sep="\t").fillna(0)
        df = df.set_index("token")
        #df["token"] = df["token"].str.lower()
        #df = pd.DataFrame(df.groupby("token").sum().unstack(["token"])["freq"].sort_values(ascending=False), columns =["freq"])
        vocabulary["freq"] = vocabulary["freq"].add(df["freq"], fill_value=0)
        i += 1
        print(i)
        if i in list(range(1000,35000,1000)):
            vocabulary.to_csv(wdir+"vocabulary_freq_" + mode + "_21000-" + str(i) +".csv", sep="\t")
    print(vocabulary)
    vocabulary.to_csv(wdir+"vocabulary_freq_" + mode + "_21000.csv", sep="\t")
    return vocabulary

def corde_csv2vocabulary_punct_freq(wdir = "/home/jose/Dropbox/biblioteca/corpora/corde/", wsdir="corde_samp_csv/"):
    if "samp" in wsdir:
        mode = "samp"
    elif "comp" in wsdir:
        mode = "comp"
    i = 0
    vocabulary = pd.read_csv(wdir + "corde_puncts_empty.csv", sep="\t", index_col = 0 ).dropna()
    vocabulary = vocabulary.set_index("token")
    vocabulary["freq"] = 0
    print(vocabulary.index.tolist())
    for doc in glob.glob(wdir+wsdir+"*.csv"):
        df = pd.read_csv(doc, sep="\t").fillna(0)
        df = df.set_index("token")
        try:
            df = df.loc[vocabulary.index.tolist()]
            vocabulary["freq"] = vocabulary["freq"].add(df["freq"], fill_value=0)
            i += 1
            print(i)
        except:
            print("not found")
        if i in list(range(1000,35000,1000)):
            vocabulary.to_csv(wdir+"vocabulary_freq_punct" + mode + str(i) +".csv", sep="\t")
    #print(vocabulary)
    vocabulary.to_csv(wdir+"vocabulary_freq_punct" + mode + str(i) +".csv", sep="\t")
    return vocabulary
#corde_csv2vocabulary_punct_freq( wsdir="corde_comp_csv/")

def concat_vocabularies(wdir = "/home/jose/Dropbox/biblioteca/corpora/corde/"):

    vocabulary_punct = pd.read_csv(wdir+"vocabulary_freq_punct.csv", sep="\t")
    
    vocabulary_punct["token"] = vocabulary_punct["token"].astype(str)
    
    corde_vocabulary = pd.read_csv(wdir+"corde_vocabulary.csv", sep = "\t", index_col=0)
    
    corde_vocabulary_complete = pd.concat([vocabulary_punct, corde_vocabulary]).sort_values(by="frecGral", ascending=False)
    
    corde_vocabulary_complete["frecNormPunct"] = (corde_vocabulary_complete["frecGral"] / corde_vocabulary_complete["frecGral"].sum()) * 1000000
    
    print(corde_vocabulary_complete.head(30))
    
    corde_vocabulary_complete.to_csv(wdir+"corde_vocabulary_w_punct.csv",sep="\t")
    return corde_vocabulary_complete

#concat_vocabularies()


def corde_texts2dfs(wdir = "/home/jose/Dropbox/biblioteca/corpora/corde/", wsdir="corde_comp/"):
    outdir = wsdir[0:-1]+"_csv/"
    for doc in glob.glob(wdir+wsdir+"*.txt"):
        
        input_name  = os.path.splitext(os.path.split(doc)[1])[0]
    
    
        with open(doc, "r", errors="replace", encoding="utf-8") as fin:
            content = fin.read()
            
            #print(input_name)
            content = re.sub(r'\A.+?Número de elementos distintos \(types\):\s*\d+\n', r'freq\ttoken\n', content, flags = re.MULTILINE|re.DOTALL)
            content = re.sub(r'^\s+', r'', content, flags = re.MULTILINE)
            content = re.sub(r'(\d+)\s+', r'\1\t', content, flags = re.MULTILINE)
            content = re.sub(r'\s+\Z', r'\n', content, flags = re.MULTILINE)
            #print(content[0:100])
            df = pd.read_csv(StringIO(content[0:-1]), sep="\t", quoting=3, error_bad_lines=False).set_index('token')
            df.to_csv(wdir+outdir+input_name+".csv", sep="\t")
#corde_texts2dfs( wsdir="corde_samp/")
#corde_texts2dfs()

def cordecsv2csv_n(wdir = "/home/jose/Dropbox/biblioteca/corpora/corde/", wsdir="corde_samp_csv_low/", vocabulary_file ="corde_vocabulary_w_punct.csv", n = 7000):
    i = 0
    vocabulary = pd.read_csv(wdir + vocabulary_file, sep="\t", index_col=0).set_index('token')
    vocabulary = vocabulary.iloc[:n,:].copy()
    for doc in glob.glob(wdir+wsdir+"*.csv"):
        
        input_name  = os.path.split(doc)[1]
        print(input_name)

        df = pd.read_csv(wdir+wsdir+input_name, sep="\t")

        df = df.set_index('token')
        df["freq"] = df["freq"]

        try:
            df = df.loc[vocabulary.index.tolist()]
            #df = df.replace("", 0).fillna(0)
            df.to_csv(wdir+ wsdir[0:-1]+"_" + str(n) + "/" +input_name, sep="\t")
        except:
            pass
        print(i)
        i += 1

#cordecsv2csv_n(wsdir="corde_comp_csv_low/")

def cut_vocabulary(wdir = "/home/jose/Dropbox/biblioteca/corpora/corde/", vocabulary_file ="corde_vocabulary_w_punct.csv", i = 7000):
    corde_df = pd.read_csv(wdir + vocabulary_file, sep="\t", encoding = 'utf8', index_col = 0)
    corde_df = corde_df.sort_values(by="frecGral", ascending=False)
    corde_df = corde_df.iloc[0:i]
    print(wdir+vocabulary_file[-4:]+str(i)+".csv")
    corde_df.to_csv(wdir + vocabulary_file[0:-4] + "_" + str(i) + ".csv", sep="\t")
    print(corde_df.head())
    print(corde_df.shape)
#cut_vocabulary()

def corde_csvs2df(wdir = "/home/jose/Dropbox/biblioteca/corpora/corde/", wsdir="corde_comp_csv_low_7000/", vocabulary_file ="corde_vocabulary_w_punct_7000.csv"):

    try:
        corde_df =  pd.read_parquet(wdir+wsdir[0:-1]+".pasdasqt")
        print("corde opened")
        if corde_df.shape[0] == 3:
            print("in good shape")
        corde_df = corde_df.replace(['0', '0.0'], '', inplace=True)
        
    except:
        print("new df")
        corde_df = pd.read_csv(wdir + vocabulary_file, sep="\t", encoding = 'utf8')
        corde_df = corde_df.dropna(subset=["token"])
        corde_df.index = corde_df["token"].astype(str)
        print(corde_df.head())
        print(corde_df.shape)
    
        del corde_df["token"]

    problematic_files = []
    i = 0
    j = 0

    print(corde_df.head())
    print(corde_df.shape)
    
    #print(wdir+wsdir+"*.csv")
    for doc in glob.glob(wdir+wsdir+"*.csv"):
        print(i, j)
        
        input_name  = os.path.split(doc)[1]
        
        if input_name[0:-4] not in corde_df.columns.tolist():
            try:
    
                df = pd.read_csv(wdir+wsdir+input_name, sep="\t").set_index('token').fillna(0)
                #print(df.head())
                df[input_name[0:-4]] = df["freq"]
    
                #corde_df = pd.concat([corde_df, df[[input_name[0:-4]]]], axis=1)
                corde_df = corde_df.join(df[[input_name[0:-4]]])
                #print(corde_df.head())
                j += 1
            except:
                problematic_files.append(input_name)
                print("problematic file", input_name)
        if j in list(range(500,35000,500)):
            print("printing")
            corde_df.fillna(0).to_parquet(wdir+wsdir[0:-1]+".pqt")
        i += 1

    print(corde_df.columns.tolist())
    print(corde_df.head())
    
    print(set([type(value) for values in corde_df.values.tolist() for value in values]))
    
    del corde_df["Unnamed: 0"]
    del corde_df["frecGral"]
    del corde_df["frecNorm"]
    del corde_df["frecNormPunct"]
    
    #corde_df = corde_df.T.reindex_axis(corde_df.median().sort_values(ascending=False).index, axis=1).T
    
    corde_df.fillna(0).to_parquet(wdir+wsdir[0:-1]+".pqt")
    #corde_df.to_csv(wdir+wsdir[0:-1]+".csv", sep="\t")
    
    return corde_df, problematic_files
            
#corde_df, problematic_files = corde_csvs2df()

def corde_get_punctuation():
    wdir= "/home/jose/Dropbox/biblioteca/corpora/corde/"
    
    vocabulary_punct = pd.read_csv(wdir+"vocabulary_punct.csv",sep="\t").sort_values(by="token")
    
    vocabulary_punct.head()
    
    corde_vocabulary = pd.read_csv(wdir+"corde_vocabulary.csv",sep="\t", index_col=0)
    
    tokens = [str(item) for item in vocabulary_punct["token"].tolist()]
    
    tokens.sort(key = len)
    
    print(tokens[0:100])
    
    puncts_list = []
    for token in tokens[0:100]:
        if token not in corde_vocabulary["token"].tolist() and len(token) < 2:
            puncts_list.append([token,0])
    
    puncts_df = pd.DataFrame(puncts_list,  columns = ["token","frecGral"]).sort_values(by="token")
    
    puncts_df.to_csv(wdir+"corde_puncts_empty.csv", sep="\t")
    return puncts_df
