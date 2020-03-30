# -*- coding: utf-8 -*-

"""
Submodule with functions for reading selected text from TEI files.
To use the function of this file you have to import extract as library
@author: Christof Schöch
"""

import re
import os
import glob
from lxml import etree
import pandas as pd

def from_TEIP5(wdir, inpath, metadatafile, mode="opt-obl"):
    """
    Extracts metadata from the CLiGs teiHeader and writes it to CSV.
    mode (string): "obl", "obl-opt" or "beta-opt-obl".
    Example of how to use this function:
from toolbox.extract import get_metadata        
metadata = get_metadata.from_TEIP5("/home/jose/cligs/ne/","master/*.xml","metadata","beta-opt-obl-subgenre-structure")

    """

    ## USER: Set list of metadata items to extract (see xpaths for list)
    ## We can choose only the obligatory metadata, the optional or the beta. 
    ## labels = ("idno_header","author_short","author_viaf", "author-gender", "title_short", "title_viaf", "pub_year", "supergenre", "genre", "subgenre", "genre-label", "narration", "availability")
     
    labels_obl = ["idno", "author.name", "title", "year", "supergenre", "genre", "subgenre.cligs.important", "author.country",
                  "author.viaf", "author.bne","author.gender", "title.main", "title.viaf","title.bne", "author.name.full",
                  "subtitle", "availability", "subgenre.cligs", "genre.analyzed","genre.analyzed.cert","date.file", "digital.source", "digital.source.format"]
    labels_opt = ["genre.subtitle","narrator","protagonist.gender","setting.type","form", "publication", "digital.source.idno", ]

    labels_structure = ["am.tokens","am.chars", "am.divs", "am.verses", "am.sps", "am.paragraphs","am.pds", "am.numerals", "am.puncts", "am.blocks", "am.chapters", "am.shortStories", "am.parts", "am.sections", "am.divisions","am.lg.poems","am.lg.stanzas","am.saids","am.narrative.ps","am.punctual.ss","len.abstract","am.fts", "len.chapters.mean","len.chapters.std","len.chapters.median","len.chapters.iqr",
                        "am.conjunctions", "am.determiners", "am.nouns", "am.verbs", "am.adverbs", "am.adjectives", "am.adpositions", "am.punctuations", "am.pronouns", "am.dates", "am.numbers", "am.interjections", "am.ne.persons", "am.ne.organizations", "am.ne.locations", "am.ne.others",
                        "am.ss","len.ss.mean","len.ss.std", "len.ss.median", "len.ss.iqr","am.ss.active.verbs.mean","am.ss.active.verbs.std","am.ss.active.verbs.median","am.ss.active.verbs.iqr","am.ws","am.ws.num.mean","am.ws.num.std","am.ws.num.median","am.ws.num.iqr","am.wnlex.noun.plants","am.wnlex.verb.communications",
                        "am.wnlex.noun.foods","am.wnlex.verb.possessions","am.wnlex.verb.cognitions","am.wnlex.noun.communications","am.wnlex.noun.states","am.wnlex.verb.statives","am.wnlex.noun.cognitions","am.wnlex.noun.times","am.wnlex.verb.bodys","am.wnlex.noun.persons","am.wnlex.adj.alls","am.wnlex.noun.quantitys","am.wnlex.noun.phenomenons","am.wnlex.verb.creations","am.wnlex.adj.perts","am.wnlex.adv.alls","am.wnlex.noun.processs","am.wnlex.noun.artifacts","am.wnlex.verb.perceptions","am.wnlex.noun.feelings","am.wnlex.verb.weathers","am.wnlex.noun.substances","am.wnlex.noun.shapes","am.wnlex.verb.competitions","am.wnlex.verb.motions","am.wnlex.noun.animals","am.wnlex.noun.acts","am.wnlex.noun.bodys","am.wnlex.noun.objects","am.wnlex.noun.motives","am.wnlex.verb.socials","am.wnlex.noun.groups","am.wnlex.verb.consumptions","am.wnlex.noun.possessions","am.wnlex.noun.Topss","am.wnlex.noun.relations","am.wnlex.noun.attributes","am.wnlex.verb.emotions","am.wnlex.noun.locations","am.wnlex.noun.events","am.wnlex.verb.contacts","am.wnlex.xxxs","am.wnlex.verb.changes"
                        ]

    labels_beta = [ "author.continent",  "authorText.group", "authorText.subgroup","protagonist.name",
    "protagonist.age", "protagonist.socLevel","protagonist.profession", "representation", "setting.continent", "setting.country", 
    "setting.settlement", "setting.territory",  "text.movement", 
    "time.period", "time.span", "authorText.relation", "end","time.year","keywords.cert",
    "author.movement","author.submovement", "author.date.birth", "author.date.death", "litHist.literaryQuality",  "litHist.brow", "litHist.audience",
    "author.litHist.pages", "author.nonNovelGenre", "litHist.pages",
    "author.year.change","setting.settlement.represented.exist","setting.represented","MdLE","CoNSSA",
    "CoNSSA.canon","HdLE", "subgenre.lithist.MdLE", "subgenre.lithist.HdLE",
    "subgenre.edit.epublibre","subgenre.edit.amazon", "subgenre.edit.wikidata","subgenre.edit.esdbpedia", "subgenre.subtitle.bne","subgenre.subtitle.first.edition","subgenre.cluster"]

    #labels_subgenre = [ "adventure", "erotic", "fantastic", "naturalist", "modernist", "realist", "sentimental", "social", "spiritual", "historical", "comedy", "philosophical", "memoir", "moralist", "symbolic", "political-fiction", "bildungsroman", "war-novel","autobiographical","dialogue", ]
    #labels_histnov = ["idno", "language", "author-continent", "author-country", "author-name", "title", "year", "subgenre_hist", "subgenre_x", "subgenre"]
    
    ## Dictionary of all relevant xpaths with their labels
    xpaths = {

              "title": '//tei:title[@type="short"]//text()',
              "subtitle":'//tei:title[@type="sub"]//text()',
              "title.main": '//tei:title[@type="main"]//text()',
              "author.name": '//tei:author//tei:name[@type="short"]//text()', 
              "author.name.full": '//tei:author//tei:name[@type="full"]//text()', 
              "author.viaf":'//tei:author//tei:idno[@type="viaf"]//text()',
              "author.bne":'//tei:author//tei:idno[@type="bne"]//text()',
              "author.gender":'//tei:term[@type="author.gender"]//text()',
              "language":'//tei:term[@type="text.language"]//text()',
              "title.viaf":'//tei:title//tei:idno[@type="viaf"]//text()',
              "title.bne":'//tei:title//tei:idno[@type="bne"]//text()',
              "year":'//tei:bibl[@type="edition-first"]//tei:date//text()',
              "supergenre":'//tei:term[@type="text.genre.supergenre"]//text()',
              "genre": '//tei:term[@type="text.genre"]//text()',
              "genre.analyzed": '//tei:term[@type="text.genre.analyzed"]//text()',
              "genre.analyzed.cert": '//tei:term[@type="text.genre"]/@cert',
              "idno": '//tei:idno[@type="cligs"]//text()',
              "availability": '//tei:availability//@status',
              "author.country": '//tei:term[@type="author.country"]//text()',
              "author.continent": '//tei:term[@type="author.continent"]//text()',
              "subgenre.cligs.important":'//tei:term[@type="text.genre.subgenre.litHist"][@resp="#jct"][@cligs:importance > parent::tei:keywords/tei:term[@type="text.genre.subgenre.litHist"][@resp="#jct"]/@cligs:importance or not(parent::tei:keywords/tei:term[@type="text.genre.subgenre.litHist"][@resp="#jct"][2])]//text()',
              "date.file" : '//tei:change[1]/@when',
              "digital.source" : '//tei:bibl[@type="digital-source"]/@source',
              "digital.source.format" : '//tei:bibl[@type="digital-source"]/@subtype',

            


              "genre.subtitle":'//tei:term[@type="text.genre.subtitle"]//text()',
              "narrator": '//tei:term[@type="text.narration.narrator"]//text()',
              "protagonist.gender": '//tei:term[@type="text.characters.protagonist.gender"]//text()',
              "setting.type": '//tei:term[@type="text.setting.settlement.type"]//text()',
              "form": '//tei:term[@type="text.form"]//text()',
              "publication": '//tei:term[@type="text.publication"]//text()',
              "digital.source.idno": '//tei:bibl[@type="digital-source"]/tei:idno/text()',

              "authorText.group": '//tei:term[@type="authorText.group"]//text()',
              "authorText.subgroup": '//tei:term[@type="authorText.subgroup"]//text()',
              "protagonist.name": '//tei:term[@type="text.characters.protagonist.name"]//text()',
              "protagonist.socLevel": '//tei:term[@type="text.characters.protagonist.socLevel"]//text()',
              "protagonist.age": '//tei:term[@type="text.characters.protagonist.age"]//text()',
              "protagonist.profession": '//tei:term[@type="text.characters.protagonist.profession"]//text()',
              "representation": '//tei:term[@type="text.plot.representation"]//text()',
              "setting.continent": '//tei:term[@type="text.setting.continent"]//text()',
              "setting.country": '//tei:term[@type="text.setting.country"]//text()',
              "setting.settlement": '//tei:term[@type="text.setting.settlement"]//text()',
              "setting.territory": '//tei:term[@type="text.setting.territory"]//text()',
              "setting.type": '//tei:term[@type="text.setting.settlement.type"]//text()',
              "setting.settlement.represented.exist" : '//tei:term[@type="text.setting.settlement.represented.exist"]//text()',
              "setting.represented" : '//tei:term[@type="text.setting.represented"]//text()',
              "text.movement": '//tei:term[@type="text.movement"]//text()',
              "time.period": '//tei:term[@type="text.time.period"]//text()',
              "time.span": '//tei:term[@type="text.time.span"]//text()',
              "time.year": '//tei:term[@type="text.time.year"]//text()',
              "authorText.relation": '//tei:term[@type="authorText.relation"]//text()',
              "end": '//tei:term[@type="text.plot.end"]//text()',
              "keywords.cert": '///tei:keywords/@cert',
              "author.movement": '//tei:term[@type="author.movement"]//text()',
              "litHist.literaryQuality": '//tei:term[@type="text.litHist.literaryQuality"]//text()',
              "litHist.brow": '//tei:term[@type="text.litHist.brow"]//text()',
              "litHist.audience": '//tei:term[@type="text.litHist.audience"]//text()',

              "author.date.birth": '//tei:term[@type="author.date.birth"]//text()',
              "author.date.death": '//tei:term[@type="author.date.death"]//text()',
              "author.litHist.pages": '//tei:term[@type="author.litHist.pages"]//text()',
              "author.submovement": '//tei:term[@type="author.submovement"]//text()',
              "author.nonNovelGenre": '//tei:term[@type="author.nonNovelGenre"]//text()',
              "author.year.change": '//tei:term[@type="author.year.change"]//text()',
              "litHist.pages": '//tei:term[@type="text.litHist.pages"]//text()',


              "subgenre.lithist.MdLE":'//tei:term[@type="text.genre.subgenre.litHist"][@resp="MdLE"][1]//text()',
              "subgenre.lithist.HdLE":'//tei:term[@type="text.genre.subgenre.litHist"][@resp="HdLE"][1]//text()',
              "subgenre.edit.epublibre": '//tei:term[@type="text.genre.subgenre.edit"][@resp="epublibre"]//text()', #'//*[name()="term"][@type="text.subgenre.edit"][@resp="epublibre"]//text()',
              "subgenre.edit.amazon": '//tei:term[@type="text.genre.subgenre.edit"][@resp="amazon"]//text()', #'//*[name()="term"][@type="text.subgenre.edit"][@resp="epublibre"]//text()',
              "subgenre.edit.wikidata": '//tei:term[@type="text.genre.subgenre.edit"][@resp="wikidata"]//text()',
              "subgenre.edit.esdbpedia": '//tei:term[@type="text.genre.subgenre.edit"][@resp="esdbpedia"]//text()',
              "subgenre.subtitle.bne": '//tei:term[@type="text.genre.subtitle.source"][@resp="bne"][1]//text()',
              "subgenre.subtitle.first.edition": '//tei:term[@type="text.genre.subtitle.source.first.edition"]//text()',
              "subgenre.cligs" : '//tei:term[@type="text.genre.subgenre.litHist"][@resp="#jct"]//text()',
              
              "subgenre.cluster": '//tei:term[@type="text.genre.subgenre.cluster"]//text()',

              "MdLE" : '//tei:term[@type="text.litHist.mentioned"][@resp="MdLE"]//text()',
              "HdLE" : '//tei:term[@type="text.litHist.mentioned"][@resp="HdLE"]//text()',

              "CoNSSA" : '//tei:term[@type="text.corpora.CoNSSA"]//text()',
              "CoNSSA.canon" : '//tei:term[@type="text.corpora.CoNSSA.canon"]//text()',

              "am.tokens": '//tei:extent/tei:measure[@unit="tokens"]//text()',
              "am.chars": '//tei:extent/tei:measure[@unit="chars"]//text()',
              "am.blocks": '//tei:extent/tei:measure[@unit="blocks"]//text()',
              "am.divs": '//tei:extent/tei:measure[@unit="divs"]//text()',
              "am.verses": '//tei:extent/tei:measure[@unit="line.verses"]//text()',
              "am.paragraphs": '//tei:extent/tei:measure[@unit="paragraphs"]//text()',
              "am.sps": '//tei:extent/tei:measure[@unit="sps"]//text()',
              "am.pds": '//tei:extent/tei:measure[@unit="paragraphs.ds"]//text()',
              "am.numerals": '//tei:extent/tei:measure[@unit="numerals"]//text()',
              "am.puncts": '//tei:extent/tei:measure[@unit="puncts"]//text()',
              "am.chapters": '//tei:extent/tei:measure[@unit="chapters"]//text()',
              "am.shortStories": '//tei:extent/tei:measure[@unit="shortStories"]//text()',
              "am.parts": '//tei:extent/tei:measure[@unit="parts"]//text()',
              "am.sections": '//tei:extent/tei:measure[@unit="sections"]//text()',
              "am.divisions": '//tei:extent/tei:measure[@unit="divisions"]//text()',
              "am.lg.poems": '//tei:extent/tei:measure[@unit="lg.poems"]//text()',
              "am.lg.stanzas": '//tei:extent/tei:measure[@unit="lg.stanzas"]//text()',
              "am.fts": '//tei:extent/tei:measure[@unit="fts"]//text()',
              "am.saids": '//tei:extent/tei:measure[@unit="saids"]//text()',
              "am.narrative.ps": '//tei:extent/tei:measure[@unit="narrative.ps"]//text()',
              "am.punctual.ss": '//tei:extent/tei:measure[@unit="punctual_ss"]//text()',
              "len.abstract": '//tei:extent/tei:measure[@unit="len.abstract"]//text()',

              "len.chapters.mean": '//tei:extent/tei:measure[@unit="chapters.len.mean"]//text()',
              "len.chapters.std": '//tei:extent/tei:measure[@unit="chapters.len.std"]//text()',
              "len.chapters.median": '//tei:extent/tei:measure[@unit="chapters.len.median"]//text()',
              "len.chapters.iqr": '//tei:extent/tei:measure[@unit="chapters.len.iqr"]//text()',

              "am.conjunctions": '//tei:extent/tei:measure[@unit="conjunctions"]//text()',
              "am.determiners": '//tei:extent/tei:measure[@unit="determiners"]//text()',
              "am.nouns": '//tei:extent/tei:measure[@unit="nouns"]//text()',
              "am.verbs": '//tei:extent/tei:measure[@unit="verbs"]//text()',
              "am.adverbs": '//tei:extent/tei:measure[@unit="adverbs"]//text()',
              "am.adjectives": '//tei:extent/tei:measure[@unit="adjectives"]//text()',
              "am.adpositions": '//tei:extent/tei:measure[@unit="adpositions"]//text()',
              "am.punctuations": '//tei:extent/tei:measure[@unit="punctuations"]//text()',
              "am.pronouns": '//tei:extent/tei:measure[@unit="pronouns"]//text()',
              "am.dates": '//tei:extent/tei:measure[@unit="dates"]//text()',
              "am.numbers": '//tei:extent/tei:measure[@unit="numbers"]//text()',
              "am.interjections": '//tei:extent/tei:measure[@unit="interjections"]//text()',

              "am.ne.persons": '//tei:extent/tei:measure[@unit="ne.persons"]//text()',
              "am.ne.organizations": '//tei:extent/tei:measure[@unit="ne.organizations"]//text()',
              "am.ne.locations": '//tei:extent/tei:measure[@unit="ne.locations"]//text()',
              "am.ne.others": '//tei:extent/tei:measure[@unit="ne.others"]//text()',

              "am.ss": '//tei:extent/tei:measure[@unit="ss"]//text()',
              "len.ss.mean": '//tei:extent/tei:measure[@unit="ss.num.mean"]//text()',
              "len.ss.std": '//tei:extent/tei:measure[@unit="ss.num.std"]//text()',
              "len.ss.median": '//tei:extent/tei:measure[@unit="ss.num.median"]//text()',
              "len.ss.iqr": '//tei:extent/tei:measure[@unit="ss.num.iqr"]//text()',

              "am.ss.active.verbs.mean": '//tei:extent/tei:measure[@unit="ss.active.verbs.mean"]//text()',
              "am.ss.active.verbs.std": '//tei:extent/tei:measure[@unit="ss.active.verbs.std"]//text()',
              "am.ss.active.verbs.median": '//tei:extent/tei:measure[@unit="ss.active.verbs.median"]//text()',
              "am.ss.active.verbs.iqr": '//tei:extent/tei:measure[@unit="ss.active.verbs.iqr"]//text()',

              "am.ws": '//tei:extent/tei:measure[@unit="ws"]//text()',
              "am.ws.num.mean": '//tei:extent/tei:measure[@unit="am.ws.num.mean"]//text()',
              "am.ws.num.std": '//tei:extent/tei:measure[@unit="am.ws.num.std"]//text()',
              "am.ws.num.median": '//tei:extent/tei:measure[@unit="am.ws.num.median"]//text()',
              "am.ws.num.iqr": '//tei:extent/tei:measure[@unit="am.ws.num.iqr"]//text()',
				
				
              "am.wnlex.noun.plants": '//tei:extent/tei:measure[@unit="wnlex.noun.plants"]//text()',
              "am.wnlex.verb.communications": '//tei:extent/tei:measure[@unit="wnlex.verb.communications"]//text()',
              "am.wnlex.noun.foods": '//tei:extent/tei:measure[@unit="wnlex.noun.foods"]//text()',
              "am.wnlex.verb.possessions": '//tei:extent/tei:measure[@unit="wnlex.verb.possessions"]//text()',
              "am.wnlex.verb.cognitions": '//tei:extent/tei:measure[@unit="wnlex.verb.cognitions"]//text()',
              "am.wnlex.noun.communications": '//tei:extent/tei:measure[@unit="wnlex.noun.communications"]//text()',
              "am.wnlex.noun.states": '//tei:extent/tei:measure[@unit="wnlex.noun.states"]//text()',
              "am.wnlex.verb.statives": '//tei:extent/tei:measure[@unit="wnlex.verb.statives"]//text()',
              "am.wnlex.noun.cognitions": '//tei:extent/tei:measure[@unit="wnlex.noun.cognitions"]//text()',
              "am.wnlex.noun.times": '//tei:extent/tei:measure[@unit="wnlex.noun.times"]//text()',
              "am.wnlex.verb.bodys": '//tei:extent/tei:measure[@unit="wnlex.verb.bodys"]//text()',
              "am.wnlex.noun.persons": '//tei:extent/tei:measure[@unit="wnlex.noun.persons"]//text()',
              "am.wnlex.adj.alls": '//tei:extent/tei:measure[@unit="wnlex.adj.alls"]//text()',
              "am.wnlex.noun.quantitys": '//tei:extent/tei:measure[@unit="wnlex.noun.quantitys"]//text()',
              "am.wnlex.noun.phenomenons": '//tei:extent/tei:measure[@unit="wnlex.noun.phenomenons"]//text()',
              "am.wnlex.verb.creations": '//tei:extent/tei:measure[@unit="wnlex.verb.creations"]//text()',
              "am.wnlex.adj.perts": '//tei:extent/tei:measure[@unit="wnlex.adj.perts"]//text()',
              "am.wnlex.adv.alls": '//tei:extent/tei:measure[@unit="wnlex.adv.alls"]//text()',
              "am.wnlex.noun.processs": '//tei:extent/tei:measure[@unit="wnlex.noun.processs"]//text()',
              "am.wnlex.noun.artifacts": '//tei:extent/tei:measure[@unit="wnlex.noun.artifacts"]//text()',
              "am.wnlex.verb.perceptions": '//tei:extent/tei:measure[@unit="wnlex.verb.perceptions"]//text()',
              "am.wnlex.noun.feelings": '//tei:extent/tei:measure[@unit="wnlex.noun.feelings"]//text()',
              "am.wnlex.verb.weathers": '//tei:extent/tei:measure[@unit="wnlex.verb.weathers"]//text()',
              "am.wnlex.noun.substances": '//tei:extent/tei:measure[@unit="wnlex.noun.substances"]//text()',
              "am.wnlex.noun.shapes": '//tei:extent/tei:measure[@unit="wnlex.noun.shapes"]//text()',
              "am.wnlex.verb.competitions": '//tei:extent/tei:measure[@unit="wnlex.verb.competitions"]//text()',
              "am.wnlex.verb.motions": '//tei:extent/tei:measure[@unit="wnlex.verb.motions"]//text()',
              "am.wnlex.noun.animals": '//tei:extent/tei:measure[@unit="wnlex.noun.animals"]//text()',
              "am.wnlex.noun.acts": '//tei:extent/tei:measure[@unit="wnlex.noun.acts"]//text()',
              "am.wnlex.noun.bodys": '//tei:extent/tei:measure[@unit="wnlex.noun.bodys"]//text()',
              "am.wnlex.noun.objects": '//tei:extent/tei:measure[@unit="wnlex.noun.objects"]//text()',
              "am.wnlex.noun.motives": '//tei:extent/tei:measure[@unit="wnlex.noun.motives"]//text()',
              "am.wnlex.verb.socials": '//tei:extent/tei:measure[@unit="wnlex.verb.socials"]//text()',
              "am.wnlex.noun.groups": '//tei:extent/tei:measure[@unit="wnlex.noun.groups"]//text()',
              "am.wnlex.verb.consumptions": '//tei:extent/tei:measure[@unit="wnlex.verb.consumptions"]//text()',
              "am.wnlex.noun.possessions": '//tei:extent/tei:measure[@unit="wnlex.noun.possessions"]//text()',
              "am.wnlex.noun.Topss": '//tei:extent/tei:measure[@unit="wnlex.noun.Topss"]//text()',
              "am.wnlex.noun.relations": '//tei:extent/tei:measure[@unit="wnlex.noun.relations"]//text()',
              "am.wnlex.noun.attributes": '//tei:extent/tei:measure[@unit="wnlex.noun.attributes"]//text()',
              "am.wnlex.verb.emotions": '//tei:extent/tei:measure[@unit="wnlex.verb.emotions"]//text()',
              "am.wnlex.noun.locations": '//tei:extent/tei:measure[@unit="wnlex.noun.locations"]//text()',
              "am.wnlex.noun.events": '//tei:extent/tei:measure[@unit="wnlex.noun.events"]//text()',
              "am.wnlex.verb.contacts": '//tei:extent/tei:measure[@unit="wnlex.verb.contacts"]//text()',
              "am.wnlex.xxxs": '//tei:extent/tei:measure[@unit="wnlex.xxxs"]//text()',
              "am.wnlex.verb.changes": '//tei:extent/tei:measure[@unit="wnlex.verb.changes"]//text()',



              "subsubgenre":'//tei:term[@type="text.subsubgenre"]//text()',
              "subgenre_hist":'//tei:term[@type="text.subgenre_hist"]//text()',
              "subgenre_x":'//tei:term[@type="text.subgenre_x"]//text()',
          
              }

    # Mode is selected: obligatory, optional or beta
    if mode =="obl":
        labels = labels_obl
    elif mode =="opt-obl":
        labels = labels_obl+labels_opt
    elif mode =="beta-opt-obl":
        labels = labels_obl+labels_opt+labels_beta
    elif mode =="opt-obl-structure":
        labels = labels_obl+labels_opt+labels_structure
        labels = labels_obl+labels_opt
    elif mode =="beta-opt-obl-structure":
        labels = labels_obl+labels_opt+labels_beta+labels_structure
    elif mode == "beta-opt-obl-subgenre-structure":
        print("ERROR: subgenre is deprecated")
            
    namespaces = {'tei':'http://www.tei-c.org/ns/1.0', 'cligs':"https://cligs.hypotheses.org/ns/cligs"}
    idnos = []
    
    ## Get list of file idnos and create empty dataframe
    for file in glob.glob(wdir + inpath):
        idno_file = os.path.basename(file)[0:6]
        idnos.append(idno_file)
    metadata = pd.DataFrame(columns=labels, index=idnos)
    #print(metadata)

    ## For each file, get the results of each xpath
    for file in glob.glob(wdir + inpath):
        #print(file)
        xml = etree.parse(file)
        ## Before starting, verify that file idno and header idno are identical.
        idno_file = os.path.basename(file)[0:6]
        idno_header = xml.xpath(xpaths["idno"], namespaces=namespaces)[0]
        if idno_file != idno_header: 
            print("Error: "+ idno_file+ " = "+idno_header)
        for label in labels:
            xpath = xpaths[label]
            result = xml.xpath(xpath, namespaces=namespaces)
            
            ## Check whether something was found; if not, let the result be "n.av."
            if len(result) == 1: 
                result = result[0]
            elif ( label == "subgenre.cligs" or label == "digital.source.text" or label == "subgenre.cluster"):
                result = ",".join(result)
            else: 
                result = ""

            if label == "title.viaf":
                result = str(result)
                #print(result)

            ## Write the result to the corresponding cell in the dataframe
            metadata.loc[idno_file,label] = result
        
    ## Add decade column based on pub_year
    metadata["decade"] = metadata["year"].map(lambda x: str(x)[:-1]+"0")

    metadata['title.viaf'] = metadata['title.viaf'].astype(str)
    
    ## Check result and write CSV file to disk.
    #print(metadata.head())
    #metadata['title_viaf'] = metadata['title_viaf'].astype(str)
    #metadata[['author_viaf','title_viaf']] = metadata[['author_viaf','title_viaf']].apply(pd.to_string)
    metadata = metadata.sort_values(by="idno",ascending=True)
    metadatafile=metadatafile+"_"+mode+".csv"
    metadata.to_csv(wdir+metadatafile, sep=",", encoding="utf-8")#, dtype={'title_viaf': str})
    print("Metadata extracted. Number of documents and metadata columns:", metadata.shape)
    return metadata
    

def main(teiFolder, txtFolder, metadataFile, mode):
    from_TEIP5(txtFolder, metadataFile, mode) #The last value choose between the three modes: only obligatory, only optional (the normal mode) and beta
    from_TEIP4(teiFolder, metadataFile) 
"""
if __name__ == "__main__":
    import sys
    from_TEIP5(int(sys.argv[1]))
    from_TEIP4(int(sys.argv[1]))
"""
#from_TEIP5("/home/jose/cligs/ne/","master/*.xml","metadata","obl")
#from_TEIP5("/home/jose/cligs/ne/","master/*.xml","metadata","beta-opt-obl-structure")
#from_TEIP5("/home/jose/cligs/textbox/spanish/novela-espanola/","tei/*.xml","metadata","beta-opt-obl-structure")
