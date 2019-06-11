import re
from unidecode import unidecode
import xml.etree.ElementTree as ET

class XMLDoc:
    def __init__(self, filename, tables=False):
        #import pdb; pdb.set_trace()
        #self.tables = None
        self.nplcit_table = []
        self.references = ""

        self.tree = ET.parse(filename)
        self.root = self.tree.getroot()
        self.abstract_section = self.root.find('abstract')
        self.citations = self.root.find('us-bibliographic-data-grant').find('us-references-cited')
        self.data = self.root.find('description')

        # look into text ../patents/US08685992-20140401.XML 
        title = self.root.find('us-bibliographic-data-grant').find('invention-title').text
        if title:
            self.title = unidecode(title)
        else:
            self.title = ''

        self.abstract = self.parse_abstract()

        background = self.parse_section("background")
        summary = self.parse_section("summary")
        desc = self.parse_section("description")

        self.intro = f'{background} {summary} {desc}'
        self.whole_desc = self.parse_section()
        self.whole = '\n'.join(self.parse_doc())
        self.keywords, self.whole_invention = self.keyword_section('invention', 2)
        self.keypatent, self.whole_patent = self.keyword_section('patent', 2)

        #if tables:
            #self.tables = self.parse_all_tables()
        if self.citations:
            self.nplcit_table = self.parse_citations()
            self.references = '\n'.join(self.nplcit_table)


    def parse_doc(self):
        text = []
        for elem in self.root.iter():
            if elem.tag != 'p':
                continue

            # don't include tables for now
            table = False
            for c in elem.iter():
                if c.tag == 'entry':
                    table = True
            if table:
                continue

            text.append(''.join([unidecode(i).strip('\n') for i in elem.itertext() if i is not None]))
        return text

    def parse_abstract(self):
        abstract = ''
        for elem in self.abstract_section:
            if elem.tag == 'p' and elem.text != None:
                abstract += elem.text + ' '

        return abstract

    #Read XML file and collect introduction (Background)
    def parse_section(self, header=None):
        section = ''

        #Flag indicating if introduction found
        start = False #Search for 'p' section with introduction
        for elem in self.data:

            if elem.text == None:
                continue

            if header:
                #If introduction already found, exit loop
                if elem.tag == 'heading' and start:
                    break

                #Start of introduction
                elif elem.tag == 'heading' and header in elem.text.lower():
                    start = True

            #Body of introduction
            elif elem.tag == 'p' and (start or header is None) and re.search('[a-zA-Z0-9]',elem.text):
                for child in elem.iter():
                    if (child.tag in ['sub', 'sup']) and child.text != None and re.search('[a-zA-Z0-9]',child.text):
                        section += '^'+child.text+'*'
                    elif child.text != None and re.search('[a-zA-Z0-9]',child.text):
                        section += child.text + ' ' 
                        if child.text[-1] == '.':
                            section += ' '

                    if child.tail != None and re.search('[a-zA-Z0-9]',child.tail):
                        section += child.tail + ' '
            
        return unidecode(section)

    #Read XML file and organizes all citations into two table based on type
    def parse_citations(self):
        #Gets citations from xml file
        nplcit_table = []

        for cit in self.citations.iter('us-citation'):
            if cit[0].tag == 'nplcit' and cit[0][0].text:
                cit_name = cit[0][0].text
                if '“' in cit_name:
                    cit_name = cit_name.split('“')[1]
                if '”' in cit_name:
                    cit_name = cit_name.split('”')[0]
                    c = unidecode(cit_name)
                    nplcit_table.append(c)
                
        return nplcit_table

    #function looks for keyword in sections. If keyword encountered, collects the next n sentences
    # defined by num_sentences 
    def keyword_section(self, keyword, num_sentences):
        def extract_keywords(text):
            sentences = []
            text = text.split('.')
            for s in text:
                if keyword.lower() not in s.lower() or s in sentences:
                    continue

                start = max(0, text.index(s) - num_sentences)
                end = min(len(text), text.index(s) + num_sentences)
                for si in text[start:end]:
                    if si not in sentences:
                        sentences.append(si)

            return '\n'.join(sentences)

        desc = whole = ""
        if self.whole_desc:
            desc = extract_keywords(self.whole_desc)

        if self.whole:
            whole = extract_keywords(self.whole)
        return (desc, whole)
