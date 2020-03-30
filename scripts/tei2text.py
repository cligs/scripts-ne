# -*- coding: utf-8 -*-

"""
Submodule with functions for reading selected metadata from TEI files.
To use the function of this file you have to import extract as library
@author: Christof Schöch
"""

import re
import os
import glob
from lxml import etree
import pandas as pd
import csv


def extract_and_save(xml, xpath, namespaces, txtFolder, filename):
    ### Removes tags but conserves their text content.
    ### USER: Uncomment as needed.
    #etree.strip_tags(xml, "{http://www.tei-c.org/ns/1.0}seg")
    #etree.strip_tags(xml, "{http://www.tei-c.org/ns/1.0}said")
    #etree.strip_tags(xml, "{http://www.tei-c.org/ns/1.0}hi")

    ### Removes elements and their text content.
    ### USER: Uncomment as needed.
    #etree.strip_elements(xml, "{http://www.tei-c.org/ns/1.0}reg", with_tail=False)
    #etree.strip_elements(xml, "{http://www.tei-c.org/ns/1.0}orig", with_tail=False)
    #etree.strip_elements(xml, "{http://www.tei-c.org/ns/1.0}note", with_tail=False)
    #etree.strip_elements(xml, "{http://www.tei-c.org/ns/1.0}quote", with_tail=False)
    #etree.strip_elements(xml, "{http://www.tei-c.org/ns/1.0}l", with_tail=False)
    #etree.strip_elements(xml, "{http://www.tei-c.org/ns/1.0}p", with_tail=False)
    etree.strip_elements(xml, "{http://www.tei-c.org/ns/1.0}head", with_tail=False)
    #etree.strip_elements(xml, "{http://www.tei-c.org/ns/1.0}stage", with_tail=False)
    #etree.strip_elements(xml, "{http://www.tei-c.org/ns/1.0}speaker", with_tail=False)

    ### XPath defining which text to select
    xp_bodytext = "//tei:body//text()"
    xp_alltext = "//text()"
    xp_seg = "//tei:body//tei:seg//text()"
    xp_said = "//tei:body//tei:said//text()"
    xp_div_chapter = ".//tei:p//text()|.//tei:l//text()"
    #xp_bodyprose = "//tei:body//tei:p//text()"
    #xp_bodyverse = "//tei:body//tei:l//text()"
    #xp_castlist = "//tei:castList//text()"
    #xp_stage = "//tei:stage//text()"
    #xp_hi = "//tei:body//tei:hi//text()"
    
    ### Applying one of the above XPaths, based on parameter passed.
    ### USER: use on of the xpath values used here in the parameters.
    if xpath == "bodytext": 
        text = xml.xpath(xp_bodytext, namespaces=namespaces)
    if xpath == "alltext": 
        text = xml.xpath(xp_alltext, namespaces=namespaces)
    if xpath == "seg": 
        text = xml.xpath(xp_seg, namespaces=namespaces)
    if xpath == "said": 
        text = xml.xpath(xp_said, namespaces=namespaces)
    if xpath == "chapter":
        text = xml.xpath(xp_div_chapter, namespaces=namespaces)
    #print(filename)

    text = "\n".join(text)

    ### Some cleaning up
    text = re.sub("[ ]{2,8}", " ", text)
    text = re.sub("\n{2,8}", "\n", text)
    text = re.sub("[ \n]{2,8}", " \n", text)
    text = re.sub("\t{1,8}", "\t", text)

    # TODO: Improve whitespace handling.
    print("   ", filename)
    outtext = str(text)
    outfile = txtFolder + filename + ".txt"

    with open(outfile,"w") as output:
        output.write(outtext)
    
    return len(text)

def from_TEIP5(wdir, xpath, use_chapter = True, tei_swdir="tei/"):
    """
    Extracts selected text from TEI P5 files and writes TXT files.
    xpath (string): "alltext", "bodytext, "seg" or "said".
    
    For example:

    tei2text.from_TEIP5("/home/jose/cligs/ne/", "bodytext", use_chapter = False, tei_swdir="master/")

    """
    if use_chapter == True:
        txtFolder = wdir+"chapters_txt/"
    else:
        txtFolder = wdir+"txt/"
        
    if not os.path.exists(txtFolder):
        os.makedirs(txtFolder)
    ## Do the following for each file in the inpath.
    counter = 0
    chapter_list = []
    print(wdir+tei_swdir)
    for file in glob.glob(wdir+tei_swdir+"*.xml"):
        with open(file, "r"):
            filename = os.path.basename(file)[:-4]
            idno = filename[:6] # assumes idno is at the start of filename.
            print("* Treating " + idno)
            counter +=1
            xml = etree.parse(file)
            namespaces = {'tei':'http://www.tei-c.org/ns/1.0'}
            
            if use_chapter == False:
                extract_and_save(xml, xpath, namespaces, txtFolder, filename)
            else:
                print("  extracting chapters")
                chapters = xml.xpath('//tei:div[@type="chapter"]', namespaces=namespaces)
                len_chapters = len(chapters)
                for chapter in chapters:
                    
                    chapter_id = str(chapter.xpath('./@xml:id', namespaces=namespaces)[0])
                    #print(chapter_id)
                    len_chapter = extract_and_save(chapter, "chapter", namespaces, txtFolder, chapter_id)
                    chapter_list.append([chapter_id, idno, len_chapter, len_chapters])
    print(chapter_list)
    df_chapters = pd.DataFrame(chapter_list, columns=["id_chapter","idno","length_text", "length_chapters"])
    print(df_chapters)
    print("TEI exported. Files treated: " + str(counter))
    return df_chapters


def combine_dfs_chapters_metadata(metadata_file, df_chapters, wdir):
    metadata = pd.read_csv(metadata_file, encoding="utf-8", sep=",", index_col=0)
    metadata_chapters = pd.merge(metadata, df_chapters, on="idno")
    metadata_chapters.to_csv(wdir+"metadata_chapters.csv", sep='\t', encoding='utf-8', index=True)
    return metadata_chapters


def extract_features_from_element(element_xml, xpaths_dict, append_attribute, append_narrative, outdir, outdirs, file_name , format_, feature_separator, specific_namespaces):
    linguistic_features_lists = []
    for xpath, xpath_features in xpaths_dict.items():

        elements = element_xml.xpath(xpath, namespaces = specific_namespaces)
        # print(len(elements))

        for element in elements:
            features_list = []

            for xpath_feature in xpath_features:
                # print (xpath + "/" + xpath_feature)
                try:
                    feature = element.xpath("./" + xpath_feature, namespaces = specific_namespaces)[0]
                    if (xpath_feature == "@mariax") or ("phr" in xpath):
                        feature = re.sub(r" ", r"_", feature)

                    if append_attribute == True:
                        if xpath_feature == "@mariax":
                            if append_narrative == True:

                                if "direct-speech" in xpath:
                                    feature = " ".join([value + xpath_feature +"_ds" for value in feature.split("|")])
                                if "narrative" in xpath:
                                    feature = " ".join([value + xpath_feature +"_nr" for value in feature.split("|")])

                            else:
                                feature = " ".join([value + xpath_feature for value in feature.split("|")])

                        else:
                            feature = str(feature) + xpath_feature
                    if (append_narrative == True) and (xpath_feature != "@mariax"):
                        if "direct-speech" in xpath:
                            feature = str(feature) + "_ds"
                        if "narrative" in xpath:
                            feature = str(feature) + "_nr"

                    features_list.append(str(feature))  # , namespaces=specific_namespaces))
                except:
                    # print("bad")
                    features_list.append(" ")
                    pass

            # print(features_list)
            linguistic_features_lists.append(features_list)
    print(linguistic_features_lists[0:2])
    print(linguistic_features_lists[-2:])

    print(len(linguistic_features_lists))
    linguistic_features_str = str("\n".join([feature_separator.join(feature) for feature in linguistic_features_lists]))

    linguistic_features_str = re.sub(r"  +",r" ", linguistic_features_str)

    with open(os.path.join(outdir, outdirs, file_name + "." + format_), "w", encoding="utf-8") as fout:
        fout.write(linguistic_features_str)




def get_outdirs_from_xpaths(xpaths_dict, outdir, use_chapter, outdirs):
    if outdirs == "":
        outdirs = str(xpaths_dict)
        outdirs = re.sub(r"[\[\]\./@=,\" ':{}\(\)]+", r"_", outdirs)
        outdirs = re.sub(r"_+", r"_", outdirs)
        outdirs = re.sub(r"_tei_+", r"_", outdirs)
        if use_chapter == True:
            outdirs = "chapter" + outdirs

        if len(outdirs) > 200:
            print("cutting!")
            outdirs = outdirs[0:90] + "..." + outdirs[-90:]
    if not os.path.exists(os.path.join(outdir, outdirs)):
        os.makedirs(os.path.join(outdir, outdirs))
    return outdirs

import time



def teia_features2files(inputwdir, xpaths_dict, outdir, feature_separator = "_",
                        format_= "txt", files = "*.xml", append_attribute = False,
                        append_narrative = False, use_chapter = False, outdirs=""):
    """
inputwdir = "/home/jose/cligs/ne/annotated_test/"  # "/home/jose/Dropbox/Doktorarbeit/thesis/data/annotated/"

outdir = "/home/jose/cligs/novelasespanolas/"
unit_separator = "\n"
feature_separator = "\t"
format_ = "txt"

xpaths_dict = {"//tei:anno[@type='multiwords']/tei:w": ["@form", "@ctag", "@lemma"], "//tei:anno[@source='dpde']/tei:phr": ["@form"]}

teia_features2files(inputwdir, xpaths_dict, outdir, feature_separator=feature_separator, format_=format_, use_chapter=True)

    """
    outdirs = get_outdirs_from_xpaths(xpaths_dict, outdir, use_chapter, outdirs)
    # For every xml file in the folder
    total_length = len(glob.glob(inputwdir+"*.xml"))
    i = 1
    for doc in glob.glob(inputwdir+"*.xml"):
        start_time = time.time()

        file_name = os.path.splitext(os.path.split(doc)[1])[0]
        print(file_name, i,"th file. Done ", str((i/total_length)*100)[0:3],"%")

        if os.path.join(outdir,outdirs,file_name+".txt") in glob.glob(os.path.join(outdir,outdirs,"*.txt")):
            print("already extracted")

        else:
            # The XML file is parsed as root element
            root_document = etree.parse(doc).getroot()

            # Namespaces are defined
            specific_namespaces = {'tei':'http://www.tei-c.org/ns/1.0','xi':'http://www.w3.org/2001/XInclude', 'cligs': 'https://cligs.hypotheses.org/ns/cligs'}

            if use_chapter == False:
                with open(os.path.join(outdir, outdirs, file_name + "." + format_), "w", encoding="utf-8") as fout:
                    fout.write(" ")


                extract_features_from_element(root_document, xpaths_dict, append_attribute, append_narrative, outdir, outdirs, file_name,
                                          format_, feature_separator, specific_namespaces)
            else:
                print(root_document)
                chapters = root_document.xpath('.//tei:div[@type="chapter"]', namespaces = specific_namespaces)
                print(chapters)
                for chapter in chapters:
                    chapter_id = str(chapter.xpath('./@xml:id', namespaces=specific_namespaces)[0])
                    print(chapter_id)
                    extract_features_from_element(chapter, xpaths_dict, append_attribute, append_narrative, outdir, outdirs, chapter_id,
                                                  format_, feature_separator, specific_namespaces)


        i += 1
        print(i)
        print("--- %s seconds ---" % round((time.time() - start_time)),4)



