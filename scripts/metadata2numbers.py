# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 09:52:43 2017

@author: jose
"""
import pandas as pd
from geopy.geocoders import Nominatim
import sys
import os
from collections import Counter
sys.path.append(os.path.abspath("/home/jose/Dropbox/Doktorarbeit/"))
from reading_robot import describe_data, load_data
from lxml import etree
import numpy as np
import scipy.stats as stats

namespaces_concretos = {'tei':'http://www.tei-c.org/ns/1.0','xi':'http://www.w3.org/2001/XInclude'}

def open_geoplaces():
    df_geoplaces = pd.read_csv("/home/jose/Dropbox/Doktorarbeit/reading_robot/data/geoplaces.csv", encoding="utf-8", sep="\t", index_col=0).fillna("NaN")
    return df_geoplaces


def setting_string2geo(wdir, metadata, geo_classes, filter_class, basic_classes = ["year","author.name","title","setting.country","setting.continent","representation"]):
    """
    wdir ="/home/jose/Dropbox/Doktorarbeit/thesis/data/"
    metadata = "metadata_beta-opt-obl-subgenre-structure_ordi.csv"
    
    metadata = load_data.load_metadata(wdir = wdir, metadata_table = metadata, sep = "\t")
    print(metadata.head)
    geo_classes = ["setting-represented"]
    filter_class = {}
    metadata_existingplaces = setting_string2geo(wdir, metadata, geo_classes, filter_class)
    """
    print(filter_class)
    classes = geo_classes + basic_classes

    if len(filter_class) > 0:
        metadata_existingplaces = metadata.loc[metadata[list(filter_class.keys())[0]] == list(filter_class.values())[0]][classes].copy()
        subtitle="_filtered"
    else:
        metadata_existingplaces = metadata[classes].copy()
        subtitle=""
        
    metadata_existingplaces = metadata_existingplaces.loc[~metadata_existingplaces[geo_classes[0]].isin(["?","unknown","mixed"])]
    geolocator = Nominatim()
    df_geoplaces = open_geoplaces()

    for class_ in geo_classes:
        print(class_)
        metadata_existingplaces["Longitude"] = ""
        metadata_existingplaces["Latitude"] = ""
        for index, row in metadata_existingplaces.iterrows():
            print(index, row[class_])
            if row[class_] not in df_geoplaces["location"].values.tolist():
                print("no encontrado")
                location = geolocator.geocode([row[class_]]) 
                print(location)
                if location is not None:
                    df_new_loc = pd.DataFrame([[row[class_],location.longitude,location.latitude]], columns=["location","lon","lat"])
                    df_geoplaces = df_geoplaces.append(df_new_loc, ignore_index=True)
                    #print(df_geoplaces)
                else:
                    df_new_loc = pd.DataFrame([[row[class_],"NaN","NaN"]], columns=["location","lon","lat"])
                    df_geoplaces = df_geoplaces.append(df_new_loc, ignore_index=True)
            #print("here")
            #print(df_geoplaces.loc[df_geoplaces['location'] == row[class_], "lon"].values[0])
            metadata_existingplaces["Longitude"][index] = (df_geoplaces.loc[df_geoplaces['location'] == row[class_], "lon"].values[0])
            metadata_existingplaces["Latitude"][index] = (df_geoplaces.loc[df_geoplaces['location'] == row[class_],"lat"].values[0])

            df_geoplaces.to_csv("/home/jose/Dropbox/Doktorarbeit/reading_robot/data/geoplaces.csv", sep='\t', encoding='utf-8', index=True)

    metadata_existingplaces["Address"] = metadata_existingplaces[class_]
    metadata_existingplaces["Name"] = metadata_existingplaces[class_]
    metadata_existingplaces["Description"] = metadata_existingplaces[class_]
    metadata_existingplaces.rename(columns={'year': 'TimeStamp'}, inplace=True)
    
    #print(metadata_existingplaces)
    metadata_existingplaces.to_csv(wdir+geo_classes[0]+subtitle+".csv", sep=',', encoding='utf-8', index=True)
    df_geoplaces.to_csv("/home/jose/Dropbox/Doktorarbeit/reading_robot/data/geoplaces.csv", sep='\t', encoding='utf-8', index=True)
    return metadata_existingplaces


"""
wdir ="/home/jose/Dropbox/Doktorarbeit/thesis/data/"
metadata = "authors-place-birth.csv"
dataset = "corpus"
metadata = load_data.load_metadata(wdir = wdir, metadata_table = metadata, sep = "\t")
classes = ["place_of_death"]
filter_class = {"Revisado en manual de la literatura": 1}

metadata_existingplaces = setting_string2geo(metadata,classes,filter_class)

"""

def metadata2dummies(metadata_df):
    
    metadata_df = pd.get_dummies(metadata_df)
    
    return metadata_df

def open_keywords(dir_keywords = "/home/jose/cligs/ne/keywords/keywords.xml"):
    tree = etree.parse(dir_keywords)
    root = tree.getroot()
    
    return root



def get_values_names_keywords(root, category_name, only_numerical = True):
    category_list = []
    category = root.xpath('//tei:category[./tei:desc/@type="class"][./tei:desc/text()="' + category_name + '"]/tei:category', namespaces = namespaces_concretos)
    category_list = [ [value.attrib["n"], value.xpath('./tei:catDesc/text()', namespaces = namespaces_concretos)[0]] for value in category]
    category_df = pd.DataFrame(category_list, columns = ["value","name"]).sort_values(by="value")
    if only_numerical == True:
        category_df = category_df.loc[category_df["value"].isin([str(i) for i in range(-10,25)])]
    return category_df


def recategorize_metadata(dir_keywords = "/home/jose/cligs/ne/keywords/keywords.xml", dir_metadata = "/home/jose/cligs/ne/", metadata_table = "metadata_beta-opt-obl-structure.csv"):
    metadata_df = load_data.load_metadata(wdir = dir_metadata, metadata_table = metadata_table, sep = ",").fillna("")

    root = open_keywords(dir_keywords)

    keywords_names = root.xpath('//tei:category[./tei:desc/@type="class"]/tei:desc/text()', namespaces = namespaces_concretos)

    for keyword in keywords_names:
        if keyword in metadata_df.columns:
            class_type = root.xpath('//tei:category[./tei:desc/@type="class"][./tei:desc/text()="'+keyword+'"]/tei:desc/@subtype', namespaces=namespaces_concretos)[0]
            print(keyword)
            if class_type == "ordinal":
                class_names = root.xpath('//tei:category[./tei:desc/@type="class"][./tei:desc/text()="'+keyword+'"]//tei:catDesc/text()', namespaces=namespaces_concretos)
                class_values = root.xpath('//tei:category[./tei:desc/@type="class"][./tei:desc/text()="'+keyword+'"]//tei:category/@n', namespaces=namespaces_concretos)
                metadata_df[keyword + "_" + class_type] = metadata_df[keyword].replace(class_names, class_values)
    print(metadata_df.shape)
    metadata_df.to_csv(dir_metadata+"metadata_recategorized.csv", sep="\t")
    return metadata_df
# recategorize_metadata()

def clean_numerical_serie(serie):
    serie = [int(element) for element in serie if element in [8,7,6,5,4,3,2,1,0,-1,-2,"8","7","6","5","4","3","2","1","0","-1","-2"]]
    return serie
    
def analyze_central_tendencies(metadata_df, wdir = "/home/jose/cligs/ne/"):
    """
    wdir = "/home/jose/Dropbox/Doktorarbeit/thesis/data/"
    metadata = load_data.load_metadata(wdir = wdir, metadata_table = "metadata_recategorized.csv", sep = "\t")
    metadata = analyze_central_tendencies(metadata, wdir = wdir)
    """

    central_tendencies_results = []
    namespaces_concretos = {'tei':'http://www.tei-c.org/ns/1.0','xi':'http://www.w3.org/2001/XInclude'}

    root = open_keywords()

    keywords_names = root.xpath('//tei:category[./tei:desc/@type="class"]/tei:desc/text()', namespaces = namespaces_concretos)

    for keyword in keywords_names:
        if (keyword+"_ordinal" in metadata_df.columns) or (keyword in metadata_df.columns):
            class_type = root.xpath('//tei:category[./tei:desc/@type="class"][./tei:desc/text()="'+keyword+'"]/tei:desc/@subtype', namespaces=namespaces_concretos)[0]
            #print(keyword)
            if class_type == "ordinal":
                #print(keyword)
                serie = metadata_df[keyword+"_ordinal"].dropna()
                serie = [int(element) for element in serie if element in [8,7,6,5,4,3,2,1,0,-1,-2,"8","7","6","5","4","3","2","1","0","-1","-2"]]
                #print(serie)
                central_tendency = np.median(serie)
                variance_tendecy = stats.iqr(serie)
                
            elif class_type == "interval" or class_type == "ratio":
                #print(keyword)
                serie = metadata_df[keyword].dropna()
    
                serie = [element for element in serie if element not in ['?', "unknown","-", "mixed"]]
                if type(serie[0]) == str:
                    serie = [int(element) for element in serie]
                central_tendency = np.median(serie)
                variance_tendecy = stats.iqr(serie)
            elif class_type == "nominal":
                serie = metadata_df[keyword].dropna()
                serie = [element for element in serie if element not in ['?', "unknown","-", "mixed"]]
                central_tendency = Counter(serie).most_common(1)[0][0]
                variance_tendecy = len(set(serie))
            
            
            numb_possible_values = len(set(serie))
            metadatum_ct = Counter(serie).most_common(2)
            mode = metadatum_ct[0][0]
            freq_mode = metadatum_ct[0][1]
            prop_mode = freq_mode / metadata_df.shape[0]
            if numb_possible_values > 1:
                second_mode = metadatum_ct[1][0]
                freq_second_mode = metadatum_ct[1][1]
            else:
                second_mode = ""
                freq_second_mode = 0
                
            
            central_tendencies_results.append([keyword, class_type, central_tendency, variance_tendecy, prop_mode, numb_possible_values, mode, freq_mode, second_mode, freq_second_mode])
            
    central_tendencies_results_df = pd.DataFrame(central_tendencies_results, columns = ["keyword", "class_type", "central_tendency", "variance_tendecy", "prop_mode", "numb_possible_values", "mode", "freq_mode", "second_mode", "freq_second_mode"])
    central_tendencies_results_df = central_tendencies_results_df.sort_values(by="keyword")

    
    central_tendencies_results_df.to_csv(wdir+"central_tendencies_metadata.csv", sep="\t")
    return central_tendencies_results_df
        

def replace_non_numerical_with_mode(metadata_df, classes_lt = ["setting.type_ordinal","time.period_ordinal","protagonist.age_ordinal","protagonist.socLevel_ordinal","end_ordinal",'time.span']):
    for class_ in classes_lt:
        mode = Counter(metadata_df[class_]).most_common()[0][0]
        for value in list(set(metadata_df[class_])):
            try:
               int(value)
            except:
                metadata_df.loc[metadata_df[class_]==value, class_] = mode

        metadata_df[class_] = metadata_df[class_].astype(float)
    return metadata_df


"""
def string2ordi(metadata_df, class_, values = {'high':2, 'low':0, 'medium':1}, value2Nan = ["?","n.av.","unknown",None]):
    class_ordi_name = class_+"_ordi"
    metadata_df[class_ordi_name] = metadata_df[class_]
    metadata_df[class_ordi_name] = metadata_df[class_ordi_name].map(values)

    metadata_df.loc[metadata_df[class_].isin(value2Nan), class_ordi_name] = "NaN"
    return metadata_df

def string2int(metadata_df, class_, values = {'antiquity':50, 'middle_ages':750, 'modern_times': 1675, 'contemporary':1900, 'future': 2100 }, value2Nan = ["?","n.av.","unknown",None]):
    class_ordi_name = class_+"_int"
    metadata_df[class_ordi_name] = metadata_df[class_]
    metadata_df[class_ordi_name] = metadata_df[class_ordi_name].map(values)

    metadata_df.loc[metadata_df[class_].isin(value2Nan), class_ordi_name] = "NaN"
    return metadata_df

def allstrings2num(metadata, wdir, output_name = "metadata_beta-opt-obl-subgenre-structure_ordi.csv"):
    metadata = string2ordi(metadata_df = metadata, class_ = "type-end", values = {'partial positive':1, 'negative':-2, 'neutral':0, 'partial negative':-1, 'positive':2})
    metadata = string2ordi(metadata_df = metadata, class_ = "setting", values = {'big-city':3, 'small-city':2, 'rural':1, 'boat':0}, value2Nan = ["?","n.av.","unknown",None,'mixed'])
    metadata = string2ordi(metadata_df = metadata, class_ = "protagonist-age"   , values = {'mature':3, 'adult':2,  'child':0, 'young':1} , value2Nan = ["?","n.av.","unknown",None,'none'])
    metadata = string2ordi(metadata_df = metadata, class_ = "author-text-relation"   , values = {'medium':2, 'none':0, 'high':3, 'low':1})
    metadata = string2ordi(metadata_df = metadata, class_ = 'keywords-cert'   , values = {'medium':1, 'high':2, 'low':0} )
    metadata = string2ordi(metadata_df = metadata, class_ = "quality-listhist"   , values = {'high':2, 'medium':1,  'low':0})
    metadata = string2ordi(metadata_df = metadata, class_ = "time-period", values = {'antiquity':0, 'middle_ages':1, 'modern_times': 2, 'contemporary':3, 'future':4 })
    metadata = string2ordi(metadata_df = metadata, class_ = "protagonist-social-level"   , values = {'medium': 0, 'high': 1, 'low': -1})


    metadata = string2int(metadata_df = metadata, class_ = "time-period")

    metadata.to_csv(wdir+output_name, sep='\t', encoding='utf-8', index=True)
    return metadata
"""
"""
wdir = "/home/jose/Dropbox/Doktorarbeit/thesis/data/"
metadata = load_data.load_metadata(wdir = wdir, metadata_table = "metadata_beta-opt-obl-subgenre-structure.csv", sep = ",")
metadata = allstrings2num(metadata, wdir = wdir)
print(metadata)
"""
