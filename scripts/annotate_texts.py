# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 08:30:39 2018

@author: jose
"""


import glob
import os
import re
from lxml import etree
import numpy as np
import pandas as pd
import scipy.stats as stats

###############
## FUNCTIONS ##
###############

def annotate_ds(content):
    

    content = re.sub(r'<seg(?:|.*?)>(.*?)</seg>', r'\1', content)
    content = re.sub(r'<seg(?:|.*?)>(.*?)</seg>', r'\1', content)

    while len(re.findall(r'(<p>.*?)(<stage>.*?</stage>\s*)', content)) > 0:
        
        content = re.sub(r'(<p>.*?)(<stage>.*?</stage>\s*)', r'\2\n\1', content)


    content = re.sub(r'<stage>(.*?)</stage>', r'<p>\1</p>', content)

    content = re.sub(r'<p>([.\s]*?</p>\s+</sp>)', r'<p>—\1', content)

    
    content = re.sub(r'<p>(\s*(?:<[^>]*?>)?\s*[\-—~»«\–].*?)</p>', r'<p rend="direct-speech">\1</p>', content)


    # Add said to simple ds paragraphs
    content = re.sub(r'(<p rend="direct-speech">)(\s*(?:<[^>]*?>)?\s*[\-—~»«\–][^\-—~»«\–]*?)</p>', r'\1<seg type="direct-speech">\2</seg></p>', content)

    #content = re.sub(r'(<p rend="direct-speech">)(\s*(?:<[^>]*?>)?\s*[»«].*?[»«\.])</p>', r'\1<seg type="direct-speech">\2</seg></p>', content)

    # Add said to paragraphs with island of narrative in ds paragraph
    content = re.sub(r'(<p rend="direct-speech">)(\s*(?:<[^>]*?>)?\s*[\-—~»«\–].*?[\-—~»«\–])(.+?)</p>', r'\1<seg type="direct-speech">\2</seg>\3</p>', content)

    content = re.sub(r'(</seg>.*?)([\-—~»«\–].+?)(</p>)', r'\1<seg type="direct-speech">\2</seg>\3', content)

    #content = re.sub(r'(<p(?:| .*?)>)([^<]+)(<seg type="direct-speech">)', r'\1<seg type="narrative">\2</seg>\3', content)
    content = re.sub(r'(</seg>)(.*?)(</p>|<seg type="direct-speech">)', r'\1<seg type="narrative">\2</seg>\3', content)



    content = re.sub(r'<seg type="narrative"></seg>', r'', content)

    content = re.sub(r'(<p>)(.+)(</p>)', r'\1<seg type="narrative">\2</seg>\3', content)
    content = re.sub(r'(<p>)(((?!seg).)+)(</p>)', r'\1<seg type="narrative">\3</seg>\4', content)

    content = re.sub(r'(<p rend="direct-speech">)(((?!seg).)+)(</p>)', r'\1<seg type="direct-speech">\3</seg>\4', content)

    content = re.sub(r'<p rend="direct-speech">', r'<p>', content)

    content = re.sub(r'</seg><seg type="direct-speech">([\-—\–][\.,;\?\!])', r'—\1</seg><seg type="direct-speech">', content)
    return content

    

def decompress_abbreviations(content):
    content = re.sub(r'P\. ?D\.', r'Postdata ', content)
    content = re.sub(r'S\. ?A\.', r'Sociedad Anónima ', content)

    content = re.sub(r'([>a-z])([A-Z])\. ?([A-Z])\. ?([A-Z])\.', r'\1\2_\3_\4', content)

    content = re.sub(r'D\. ', r'Don ', content)
    content = re.sub(r'Sr\. ', r'Señor ', content)
    content = re.sub(r'M\. ', r'María ', content)
    content = re.sub(r'J\. ', r'José ', content)
    content = re.sub(r'Vd\. ', r'Usted ', content)
    content = re.sub(r'V\. ', r'Usted ', content)
    content = re.sub(r'Dr\. ', r'Doctor ', content)
    
    return content

def place_sentence_milestone(content):

    content = re.sub(r'(<seg .*?>)(.*?)(</seg>)', r'\1<s>\2</s>\3', content)

    while re.findall('(<seg .*?>.*?[\.\?!:]) (\W*[A-ZÁ-ÚÑÜ¡¿])', content):
        content = re.sub(r'(<seg .*?>.*?[\.\?!:]) (\W*[A-ZÁ-ÚÑÜ¡¿])', r'\1</s><s> \2', content)
    #content = re.sub(r'(</[p|ab|stage]>)', r'<milestone unit="s"/>\1', content)
    return content

def annotate_ne(content):
    content = decompress_abbreviations(content)
    while re.findall('(<seg [^>]*?>)([^<]*?)([\-—~»«\–])([^<]*?)</seg>', content):
        content = re.sub(r'(<seg [^>]*?>)([^<]*?)([\-—~»«\–])([^<]*?)</seg>', r'\1\2</seg>\3\1\4</seg>', content)

    while re.findall('(<seg [^>]*?>)([^<]*?)([\.\?\!]+?)( [A-ZÁ-ÚÑÜ][^<]*?)</seg>', content):
        content = re.sub(r'(<seg [^>]*?>)([^<]*?)([\.\?\!]+?)( [A-ZÁ-ÚÑÜ][^<]*?)</seg>', r'\1\2</seg>\3\1\4</seg>', content)

    content = annotate_ds(content)
    
    content = place_sentence_milestone(content)
    return content
    
def create_text_anno(basic_wdir, input_wdir, output_wdir):
    """
        Example of how to use it:
        create_text_anno("/home/jose/cligs/","master/*.xml", "ds/")
    """
    if not os.path.exists(basic_wdir+output_wdir):
        os.makedirs(basic_wdir+output_wdir)

    # The we open each file
    for doc in glob.glob(basic_wdir+input_wdir):
        idno_file = os.path.basename(doc)
        print(idno_file)
        with open(doc, "r", errors="replace", encoding="utf-8") as fin:
            content = fin.read()
            
            content = annotate_ne(content)
            
            # The file is written
            with open (os.path.join(basic_wdir+output_wdir, idno_file), "w", encoding="utf-8") as fout:
                fout.write(content)
              
#create_text_anno("/home/jose/cligs/ne/","master/*.xml", "ds/")

def get_textual_metadata(annotated_content, size_kb, wsdir, master, idno_file):
    
    root_document = etree.parse( wsdir + master + idno_file ).getroot()
    specific_namespaces = {'tei':'http://www.tei-c.org/ns/1.0','xi':'http://www.w3.org/2001/XInclude', 'cligs': 'https://cligs.hypotheses.org/ns/cligs'}
    chapters = root_document.xpath("//tei:body//tei:div[@type='chapter']", namespaces=specific_namespaces)
    len_chapters = []
    for chapter in chapters:
        len_chapters.append(len(" ".join(chapter.xpath(".//text()", namespaces=specific_namespaces))))
    len_chapters = np.array(len_chapters)
    
    text_measures = ""
    
    text_measures = text_measures + '\n\t\t\t\t<measure unit="chapters.len.mean">' + str("%.2f" % round(len_chapters.mean(),2)) + r'</measure>'
    text_measures = text_measures + '\n\t\t\t\t<measure unit="chapters.len.std">' + str("%.2f" % round(len_chapters.std(),2)) + r'</measure>'
    text_measures = text_measures + '\n\t\t\t\t<measure unit="chapters.len.median">' + str("%.2f" % round(np.percentile(len_chapters, q = 50), 2)) + r'</measure>'
    text_measures = text_measures + '\n\t\t\t\t<measure unit="chapters.len.iqr">' + str("%.2f" % round(stats.iqr(len_chapters), 2)) + r'</measure>'

    
        
    
    content_abstract = re.findall(r'<abstract.*?>(.*?)</abstract>', annotated_content, flags=re.DOTALL)[0]
    content_abstract = re.sub(r'</?.*?>', r'', content_abstract, flags=re.DOTALL)
    content_abstract = re.sub(r'\s\s+', r' ', content_abstract)
    len_abstract = str(len(content_abstract))
    
    annotated_content = re.sub(r'<teiHeader>.*?</teiHeader>', r'', annotated_content, flags=re.DOTALL)
    
    # Divs and groups of lines are counted
    divs = str(annotated_content.count("<div"))
    lines = str(len(re.findall(r'\n+',annotated_content)))

        
    # Diferent TEI elements are counted
    chapters = str(len(re.findall(r'<div[^>]*?type="chapter"',annotated_content)))
    short_stories = str(len(re.findall(r'<div[^>]*?type="shortStories"',annotated_content)))
    parts = str(len(re.findall(r'<div[^>]*?type="part"',annotated_content)))
    sections = str(len(re.findall(r'<div[^>]*?type="section"',annotated_content)))
    divisions = str(len(re.findall(r'<div[^>]*?type="division"',annotated_content)))
    blocks = str(len(re.findall(r'<(l|ab|head|stage|sp|p|ab)( .+?|)>',annotated_content)))
    line_verses = str(len(re.findall(r'<(l)( .+?|)>',annotated_content)))
    heads = str(len(re.findall(r'<(head)( .+?|)>',annotated_content)))
    stages = str(len(re.findall(r'<(stage)( .+?|)>',annotated_content)))
    sps = str(len(re.findall(r'<(sp)( .+?|)>',annotated_content)))
    ps = str(len(re.findall(r'<(p)( .+?|)>',annotated_content)))
    abs_ = str(len(re.findall(r'<(ab)( .+?|)>',annotated_content)))
    lg_poems = str(len(re.findall(r'<lg type="poem">',annotated_content)))
    lg_stanzas = str(len(re.findall(r'<lg type="stanza">',annotated_content)))
    ft = str(len(re.findall(r'<(floatingText)( .+?|)>',annotated_content)))
    punctual_ss = str(len(re.findall(r'<milestone unit="s"/>',annotated_content)))
    
    # The paragraphas that have right after a punctuation mark that presents direct speech are counted
    saids = str(len(re.findall(r'<said>',annotated_content)))
    speech_ps = str(len(re.findall(r'<p rend="direct-speech">',annotated_content)))
    narrative_ps = str(len(re.findall(r'<p>',annotated_content)))

    # Then the text is converted into plaintext and the white space cleaned
    plain_body = annotated_content
    plain_body = re.sub(r'</?.*?>', r'', plain_body, flags=re.DOTALL)
    plain_body = re.sub(r'[\t ]+', r' ', plain_body)
    plain_body = re.sub(r'\n[\n]+', r'\n', plain_body)

    # Characters and words are counted
    chars = str(len(plain_body))
    tokens = str(len(re.findall(r'[\wáéíóúñü\d]+',plain_body)))

    # If we want some more info, the ammount of numbers and punctuation marks are counted
    numerals = str(len(re.findall(r'\d+',plain_body)))
    puncts = str(len(re.findall(r'[!"\#$%&\'()*+,\-./:;<=>?@\[\\\]^_`{|}~¿¡…—–~»«]',plain_body)))

    textual_metadata = r'\n\t\t\t\t<measure unit="lines">'+re.escape(lines)+r'</measure>\n\t\t\t\t<measure unit="divs">'+re.escape(divs)+r'</measure>\n\t\t\t\t<measure unit="tokens">'+re.escape(tokens)+r'</measure>\n\t\t\t\t<measure unit="chars">'+re.escape(chars)+r'</measure>\n\t\t\t\t<measure unit="size_kb">'+re.escape(size_kb)+r'</measure>\n\t\t\t\t<measure unit="chapters">'+re.escape(chapters)+r'</measure>\n\t\t\t\t<measure unit="shortStories">'+re.escape(short_stories)+r'</measure>\n\t\t\t\t<measure unit="parts">'+re.escape(parts)+r'</measure>\n\t\t\t\t<measure unit="sections">'+re.escape(sections)+r'</measure>\n\t\t\t\t<measure unit="divisions">'+re.escape(divisions)+r'</measure> \n\t\t\t\t<measure unit="blocks">'+re.escape(blocks)+r'</measure> \n\t\t\t\t<measure unit="lg.poems">'+re.escape(lg_poems)+r'</measure> \n\t\t\t\t<measure unit="lg.stanzas">'+re.escape(lg_stanzas)+r'</measure> \n\t\t\t\t<measure unit="line.verses">'+re.escape(line_verses)+r'</measure> \n\t\t\t\t<measure unit="heads">'+re.escape(heads)+r'</measure> \n\t\t\t\t<measure unit="stages">'+re.escape(stages)+r'</measure> \n\t\t\t\t<measure unit="sps">'+re.escape(sps)+r'</measure> \n\t\t\t\t<measure unit="paragraphs">'+re.escape(ps)+r'</measure> \n\t\t\t\t<measure unit="abs">'+re.escape(abs_)+r'</measure> \n\t\t\t\t<measure unit="fts">'+re.escape(ft)+r'</measure>\n\t\t\t\t<measure unit="paragraphs.ds">'+re.escape(speech_ps)+r'</measure>\n\t\t\t\t<measure unit="saids">'+re.escape(saids)+r'</measure>\n\t\t\t\t<measure unit="narrative.ps">'+re.escape(narrative_ps)+r'</measure>\n\t\t\t\t<measure unit="punctual_ss">'+re.escape(punctual_ss)+r'</measure> \n\t\t\t\t<measure unit="numerals">'+re.escape(numerals)+r'</measure> \n\t\t\t\t<measure unit="puncts">'+re.escape(puncts)+r'</measure> \n\t\t\t\t<measure unit="len.abstract">'+re.escape(len_abstract)+r'</measure>' + text_measures

    return textual_metadata


def get_linguistic_metadata(wsdir, master_anno, idno_file):
    #print(wsdir + master_anno + idno_file)
    root_document = etree.parse( wsdir + master_anno + idno_file ).getroot()
    #print(len(root_document))
    specific_namespaces = {'tei':'http://www.tei-c.org/ns/1.0','xi':'http://www.w3.org/2001/XInclude', 'cligs': 'https://cligs.hypotheses.org/ns/cligs'}

    poss = ["conjunction","determiner","noun","verb","adverb","adjective","adposition","punctuation","pronoun","date","number","interjection"]
    
    ling_measures = "\n"


    types_vaues = root_document.xpath("//tei:w//text()", namespaces=specific_namespaces)
    ling_measures = ling_measures+ '\n\t\t\t\t<measure unit="types">' + str(len(set(types_vaues))) + r'</measure>'
    ling_measures += "\n"
    
    
    
    tags = ["s","w"]            
    for tag in tags:
        #print(tag)

        tag_elements = root_document.xpath("//tei:" + tag, namespaces=specific_namespaces)

        ling_measures = ling_measures+ '\n\t\t\t\t<measure unit="' + tag + r's">' + str(len(tag_elements)) + r'</measure>'
        #print(len(tag_elements))
        
        len_tag = []
        amount_act_verbs_text = []
        
        for tag_element in tag_elements:
            len_tag.append(len(" ".join(tag_element.xpath(".//text()", namespaces=specific_namespaces))))
            if tag == "s":
                amount_active_verbs = len(tag_element.xpath("./tei:w[@cligs:ctag='VMI']", namespaces=specific_namespaces))
                amount_active_verbs += len(tag_element.xpath("./tei:w[@cligs:ctag='VSI']", namespaces=specific_namespaces))
                amount_act_verbs_text.append(amount_active_verbs)

        len_tag = np.array(len_tag)

        ling_measures = ling_measures+ '\n\t\t\t\t<measure unit="' + tag + r's.num.mean">' + str("%.2f" % round(len_tag.mean(),2)) + r'</measure>'
        ling_measures = ling_measures+ '\n\t\t\t\t<measure unit="' + tag + r's.num.std">' + str("%.2f" % round(len_tag.std(),2)) + r'</measure>'
        ling_measures = ling_measures+ '\n\t\t\t\t<measure unit="' + tag + r's.num.median">' + str("%.2f" % round(np.percentile(len_tag, q = 50), 2)) + r'</measure>'
        ling_measures = ling_measures+ '\n\t\t\t\t<measure unit="' + tag + r's.num.iqr">' + str("%.2f" % round(stats.iqr(len_tag), 2)) + r'</measure>'

        if tag == "s":
            amount_act_verbs_text = np.array(amount_act_verbs_text)
                
            ling_measures = ling_measures+ '\n\t\t\t\t<measure unit="ss.active.verbs.mean">' + str("%.2f" % round(amount_act_verbs_text.mean(),2)) + r'</measure>'
            ling_measures = ling_measures+ '\n\t\t\t\t<measure unit="ss.active.verbs.std">' + str("%.2f" % round(amount_act_verbs_text.std(),2)) + r'</measure>'
            ling_measures = ling_measures+ '\n\t\t\t\t<measure unit="ss.active.verbs.median">' + str("%.2f" % round( np.percentile(amount_act_verbs_text, q = 50), 2)) + r'</measure>'
            ling_measures = ling_measures+ '\n\t\t\t\t<measure unit="ss.active.verbs.iqr">' + str("%.2f" % round(stats.iqr(amount_act_verbs_text),2)) + r'</measure>'
            
            ling_measures += "\n"
    
    ling_measures += "\n"

    for pos in poss:
        
        pos_value = str(len( root_document.xpath("//tei:w[@pos='" + pos + "']", namespaces=specific_namespaces)))
        ling_measures = ling_measures+ '\n\t\t\t\t<measure unit="' + pos + 's">' + pos_value + r'</measure>'

    ling_measures += "\n"

    nes = ["person","organization","location","other"]
    for ne in nes:
        ne_value = str(len( root_document.xpath("//tei:w[@cligs:neclass='" + ne + "']", namespaces=specific_namespaces)))
        ling_measures = ling_measures+ '\n\t\t\t\t<measure unit="ne.' + ne + 's">' + ne_value + r'</measure>'

    ling_measures += "\n"

    wnlexs = ['noun.plant',  'verb.communication',  'noun.food',  'verb.possession',  'verb.cognition',  'noun.communication',  'noun.state',  'verb.stative',  'noun.cognition',  'noun.time',  'verb.body',  'noun.person',  'adj.all',  'noun.quantity',  'noun.phenomenon',  'verb.creation',  'adj.pert',  'adv.all',  'noun.process',  'noun.artifact',  'verb.perception',  'noun.feeling',  'verb.weather',  'noun.substance',  'noun.shape',  'verb.competition',  'verb.motion',  'noun.animal',  'noun.act',  'noun.body',  'noun.object',  'noun.motive',  'verb.social',  'noun.group',  'verb.consumption',  'noun.possession',  'noun.Tops',  'noun.relation',  'noun.attribute',  'verb.emotion',  'noun.location',  'noun.event',  'verb.contact',  'xxx',  'verb.change']

    for wnlex in wnlexs:
        wnlex_value = str(len( root_document.xpath("//tei:w[@cligs:wnlex='" + wnlex + "']", namespaces=specific_namespaces)))
        ling_measures = ling_measures+ '\n\t\t\t\t<measure unit="wnlex.' + wnlex + 's">' + wnlex_value + r'</measure>'

    
    return ling_measures

     
     
def textual_annotation_and_save_ling_metadata_in_master(wsdir, master, master_anno, master_output, textual_anno_output):
    if not os.path.exists(wsdir + master_output):
        os.makedirs(wsdir + master_output)
    if not os.path.exists(wsdir + textual_anno_output):
        os.makedirs(wsdir + textual_anno_output)

    # The we open each file
    for doc in glob.glob(wsdir + master + "*.xml"):
        idno_file = os.path.basename(doc)
        print(idno_file)
        with open( doc, "r", errors="replace", encoding="utf-8") as fin:
            content = fin.read()

            # Size in kb is calculated    
            size_kb = str(int(os.path.getsize(doc) / 1024 ) )
            
            # From the TEI File, teiHeader, front and back are deleted
            annotated_content = annotate_ne(content)
            
            textual_metadata = get_textual_metadata(annotated_content, size_kb, wsdir, master, idno_file)
            #print("textual_metadata", len(textual_metadata))
            ling_measures = get_linguistic_metadata(wsdir, master_anno, idno_file)
            #print("ling_measures", len(ling_measures))
            
            new_metadata = r'\n\t\t\t<extent>' + textual_metadata + ling_measures + r'\n\t\t\t</extent>\1'
            content = re.sub(r'\s+<extent>.*</extent>', r'', content, flags=re.DOTALL)
            content = re.sub(r'(\n[\s\t]+<publicationStmt>)',  new_metadata, content, flags=re.DOTALL)

            annotated_content = re.sub(r'\s+<extent>.*</extent>', r'', annotated_content, flags=re.DOTALL)
            annotated_content = re.sub(r'(\n[\s\t]+<publicationStmt>)',  new_metadata, annotated_content, flags=re.DOTALL)

            # The file is written
            with open (os.path.join(wsdir+master_output, idno_file), "w", encoding="utf-8") as fout:
                fout.write(content)    
            # The file is written
            with open (os.path.join(wsdir+textual_anno_output, idno_file), "w", encoding="utf-8") as fout:
                fout.write(annotated_content)    

"""
textual_annotation_and_save_ling_metadata_in_master(
    wsdir = "/home/jose/cligs/ne/",
    master = "master/",
    master_anno = "linguistic_annotated/",
    master_output = "master-2/",
    textual_anno_output = "textual_annotated/",
    )
"""

def order_metadata(basic_wdir, input_wdir, output_wdir):
    for doc in glob.glob(basic_wdir+input_wdir+"*.xml"):
        #print("aquí va doc!!!: ",doc)
        input_name  = os.path.splitext(os.path.split(doc)[1])[0]
        print(input_name)
        with open(doc, "r", errors="replace", encoding="utf-8") as fin:
            content = fin.read()
            keywords = re.findall(r'(<keywords.*?>)(.*?)</keywords>', content, flags = re.DOTALL)
            keywords_element = keywords[0][0]
            keywords_content = keywords[0][1]
            keywords_content = re.sub(r"\n\s*\n", r"\n", keywords_content)
            keywords_content = keywords_content.split('\n')
            
            while "" in keywords_content:
                keywords_content.remove("")
            while '\t\t\t\t' in keywords_content:
                keywords_content.remove('\t\t\t\t')
            joined_keywords_content = "\n".join(sorted(keywords_content))
            new_keywords = keywords_element+"\n"+joined_keywords_content+"\n\t\t\t\t</keywords>"
            new_content = re.sub(r'<keywords.*?>.*?</keywords>', new_keywords, content, flags = re.DOTALL)
            with open (basic_wdir+output_wdir+input_name+".xml", "w", encoding="utf-8") as fout:
                fout.write(new_content)

def add_genre_epublibre_idnos():
    # NOT IN USE ANY MORE!

    epublibre = pd.read_csv("/home/jose/Dropbox/biblioteca/datos/epublibre/20171130-epublibre.csv")
    
    metadata = pd.read_csv("/home/jose/cligs/ne/metadata_beta-opt-obl-subgenre-structure.csv")
    
    errores = ["Zalacain", "PazGuerra" ,"Barraca", "Marta", "Jose", "Papeles", "Tristan", "Riverita", "Ruta", "Naranjos", "Catedral", "Horda", "Sangre", "TierraTodos", "MareNostrum", "ParaisoMujeres", "Senorito", "Idilio", "MajosCadiz", "Espuma", "Fe2", "Maestrante"]
    for index, row in metadata.iterrows():
        #print(type(row["digital-source-idno"]))
        if row["digital-source-idno"] != "n.av":
            if row["digital-source-idno"] in [str(value) for value in epublibre["EPL Id"].values ]:
                if row["title"] not in errores:
                    print("\n")
                    print(row["idno"])
                    print(row["title"])
                    generos = epublibre.loc[epublibre["EPL Id"] == int(row["digital-source-idno"]), "Géneros"].values.tolist()[0]
        
                    for doc in glob.glob("/home/jose/cligs/ne/master/" + row["idno"] + ".xml"):
                        idno_file = os.path.basename(doc)
          
                        with open(doc, "r", errors="replace", encoding="utf-8") as fin:
                            content = fin.read()
                            
                            content = re.sub(r'\s*<term type="subgenre-edit" resp="epublibre"[^>]*?>.*?</term>\s*', r'', content)
    
                            content = re.sub(r'\s*</keywords>', r'\n\t\t\t\t\t<term type="subgenre-edit" resp="epublibre">'  + str(generos) + r'</term>\n</keywords>', content)
                            
        
                            with open ("/home/jose/cligs/ne/master/"+ row["idno"] + ".xml", "w", encoding="utf-8") as fout:
                                fout.write(content)

#add_genre_epublibre()

def add_genre_epublibre_title():
    # NOT IN USE ANY MORE!
    
    epublibre = pd.read_csv("/home/jose/Dropbox/biblioteca/datos/epublibre/20171130-epublibre.csv")
    
    metadata = pd.read_csv("/home/jose/cligs/ne/metadata_beta-opt-obl-subgenre-structure.csv")
    
    for index, row in metadata.iterrows():
        if row["title_main"] in epublibre["Título"].values.tolist():
                generos = epublibre.loc[epublibre["Título"] == row["title_main"], "Géneros"].values.tolist()[0]
                print(row["title"])
                print(row["author-name"])
                print(row["idno"])
                print(epublibre.loc[epublibre["Título"] == row["title_main"], "Título"])
                print(epublibre.loc[epublibre["Título"] == row["title_main"], "Autor"])
                for doc in glob.glob("/home/jose/cligs/ne/master/" + row["idno"] + ".xml"):
                    idno_file = os.path.basename(doc)
      
                    with open(doc, "r", errors="replace", encoding="utf-8") as fin:
                        content = fin.read()
                        
                        content = re.sub(r'\s*<term type="subgenre-edit" resp="epublibre"[^>]*?>.*?</term>\s*', r'', content)
    
                        content = re.sub(r'\s*</keywords>', r'\n\t\t\t\t\t<term type="subgenre-edit" resp="epublibre">'  + str(generos) + r'</term>\n\t\t\t\t5</keywords>', content)
                        
    
                        with open ("/home/jose/cligs/ne/master2/"+ row["idno"] + ".xml", "w", encoding="utf-8") as fout:
                            fout.write(content)


def substitute_teiheader(basic_wdir, text_wsdir, teiheader_wsdir, output_wsdir):
    # NOT IN USE ANY MORE!
    if not os.path.exists(basic_wdir+output_wsdir):
        os.makedirs(basic_wdir+output_wsdir)

    # The we open each file
    for doc in glob.glob(basic_wdir+text_wsdir):
        idno_file = os.path.basename(doc)
        print(idno_file)

        with open(doc, "r", errors="replace", encoding="utf-8") as fin:
            content_text = fin.read() 

            with open(basic_wdir+teiheader_wsdir+idno_file, "r", errors="replace", encoding="utf-8") as fin:
                content_header = fin.read()
                right_header = re.findall(r'(<teiHeader>.*?</teiHeader>)', content_header,  flags = re.DOTALL)[0]
            
                content_text = re.sub(r'<teiHeader>.*?</teiHeader>', right_header, content_text,  flags = re.DOTALL)
            
            
                with open (basic_wdir+output_wsdir+idno_file, "w", encoding="utf-8") as fout:
                    fout.write(content_text)
#substitute_teiheader(basic_wdir = "/home/jose/cligs/ne/", text_wsdir = "annotated/*.xml", teiheader_wsdir = "master/", output_wsdir = "annotated2/")


def add_genre_amazon():
    # NOT IN USE ANY MORE!
    metadata = pd.read_csv("/home/jose/cligs/ne/metadata_beta-opt-obl-subgenre-structure.csv")
    amazon_metadata_filtered_manually = pd.read_csv("/home/jose/cligs/ne/other data/metadata-amazon-full_automatic_filtered_manually_filtered.csv", sep="\t").fillna(" ")
    amazon_metadata_filtered_manually_correct = amazon_metadata_filtered_manually.loc[amazon_metadata_filtered_manually["correct title"] == 1].copy()
    grouped = amazon_metadata_filtered_manually_correct.groupby('title_main')['amazon_clasification'].apply(lambda x:  "{%s}" % '; '.join(x))
    new_table = pd.DataFrame({"work":grouped.index, "amazon-class":grouped.values,})
    
    for index, row in metadata.iterrows():
        if row["title.main"] in new_table["work"].values.tolist():
            print(row["idno"])
            generos = new_table.loc[new_table["work"] == row["title.main"], "amazon-class"].values.tolist()[0]
            generos = re.sub(r'(\{ ;| ; \}|[\{\}]|\s+;\s+;)', r'', generos)
            generos = re.sub(r'(^\s*;\s*|\s*;\s*$)', r'', generos, flags=re.MULTILINE)
            generos = re.sub(r'\s*;\s+;\s*', r' ; ', generos, flags=re.MULTILINE)
            generos = re.sub(r'^\s+$', r'', generos, flags=re.MULTILINE)

            
        for doc in glob.glob("/home/jose/cligs/ne/master/" + row["idno"] + ".xml"):
            idno_file = os.path.basename(doc)

            with open(doc, "r", errors="replace", encoding="utf-8") as fin:
                content = fin.read()
                content = re.sub(r'\s*<term type="genre.subgenre.edit" resp="amazon"[^>]*?>.*?</term>\s*', r'', content)
                if len(generos) > 0:
                
                    content = re.sub(r'\s*</keywords>', r'\n\t\t\t\t\t<term type="genre.subgenre.edit" resp="amazon">'  + str(generos) + r'</term>\n\t\t\t\t</keywords>', content)

                with open ("/home/jose/cligs/ne/master2/"+ row["idno"] + ".xml", "w", encoding="utf-8") as fout:
                    fout.write(content)
#add_genre_amazon()