# -*- coding: utf-8 -*-
# @date: April 27, 2016.
# @author: christof

import os
import glob
from lxml import etree
import sys
from xml.sax.handler import ContentHandler
from xml.sax import make_parser

def main(teipath, rngfile):
    """
    Arguments:
    teipath (str): path to the TEI files, e.g. /home/ulrike/Dokumente/Git/textbox/es/novela-espanola/tei/*.xml
    rngfile (str): path to the schema file, e.g. /home/ulrike/Schreibtisch/basisformat.rng
    
    Example:
    from toolbox.check_quality import validate_tei
    validate_tei.main("/home/jose/cligs/ne/master/*.xml", "/home/jose/cligs/reference/tei/standard/cligs.rng")
    """
    mistakes = 0
    for teifile in glob.glob(teipath): 
        idno = os.path.basename(teifile)
        print(idno)
        rngparsed = etree.parse(rngfile)
        rngvalidator = etree.RelaxNG(rngparsed)
        parser = etree.XMLParser(recover=True)
        teiparsed = etree.parse(teifile, parser)
        #teiparsed = etree.parse(teifile)
        validation = rngvalidator.validate(teiparsed)
        log = rngvalidator.error_log
        if validation == True: 
            pass
            #print(idno, "valid!")
        else:
            print(idno, "sorry, not valid!")
            print(log)
            #print(log.last_error)
            #print(log.last_error.domain_name)
            #print(log.last_error.type_name)
            mistakes += 1
    print("number of problematics file: ", mistakes)

"""
if __name__ == "__main__":
    main(int(sys.argv[1])) 
"""

def wellform_xml(teipath):
    mistakes = 0
    for teifile in glob.glob(teipath): 
        #print(teifile)
        idno = os.path.basename(teifile)
        #print(idno)
        try:
            parser = make_parser()
            parser.setContentHandler(ContentHandler())
            parser.parse(teifile)
        except:
            print(idno)
            mistakes += 1
    print("number of problematics file: ", mistakes)

#wellform_xml("/home/jose/cligs/ne/master2/*.xml")