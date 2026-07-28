"""
//******************************************************************************
// FILENAME:           sections.py
// DESCRIPTION:        This Natural Language Processing Python code is the
//                     MTERMS service for the NLP Project.
// CREATION DATE:      February 3, 2016
// INITIAL AUTHOR:     Kenneth Lai (KL549)
// LAST MODIFIED DATE: 1/25/2017 5:40 PM
// LAST MODIFIED BY:   KL549
// Copyright (C) 2017  Partners Information Systems - Clinical and Quality
//                     Analysis (IS - CQA)
//                     Partners HealthCare System Confidential
//******************************************************************************
"""

import re
from dbops import load_table_from_csv
from finder import find
from tokenizer import tokenize
from builtins import str


class Sections:

    def __init__(self, lexicon):

        # using [\r\n] for unix compatibility
        # using [a-np-z] to exclude o as bullet point
        self.boundaries_regex = re.compile(r'''(?x)(?=[^\.])([\r\n])+(?![a-np-z])
                                               |\.+([\r\n])+
                                               |\.+\s+(?=[a-zA-Z])
                                               |\s\s+''')
        self.whitespace_regex = re.compile(r'([a-zA-Z0-9,]{2,})[ \t]{2,}([a-z]+)')
        # load table
        self.sections_table = load_table_from_csv(lexicon)

        # init section char set for section headers that look like:
        # "=============Medications=============" or "ALLERGIES:"
        self.chars = set(['=']) # Removed ":" from here

    def find_sections(self, note):
        "Segment a note into sections."
        sentences = self.find_sentences(note.raw)
        section_id = 1
        section_text = 'Unknown Section Name'
        sentence_id = 1
        start_char_pos = 0
        original_section_text = ''
        for sentence in sentences:
            tokens = tokenize(sentence)
            attrib = {'SectionID': str(section_id),
                      'TermType': section_text,
                      'OriginalText': original_section_text}
            found, entry, offset = find(tokens,
                                        self.sections_table,
                                        0,      # indexer = 0 for sections
                                        [],     # found = none yet
                                        True)   # forward = True
            if found:
                # retain section text in lexicon as part of sentence text
                original_section_text += sentence[:sum(len(f) + 1 for f in found)]
                if original_section_text[-1] in self.chars:
                    original_section_text = original_section_text[:-1]
                if original_section_text[-1] != ' ':
                    original_section_text += ' '
            # if found whole sentence, or found followed by section char
            # if (found and
            #     (offset == len(tokens) or
            #      (offset < len(tokens) and tokens[offset] in self.chars))):
            # if found:
                if sentence_id > 1: # not first section
                    section_id += 1
                column_names = self.sections_table['ColumnNames']
                # get attributes from table
                # attrib = dict(zip((unicode(i, 'cp1252') for i in column_names),
                #                   (unicode(j, 'cp1252') for j in entry)))
                attrib = dict(zip((str(i) for i in column_names),
                                  (str(j) for j in entry)))
                attrib['SectionID'] = str(section_id)
                # attrib['DisplayName'] = section_text
                attrib['DisplayName'] = ' '.join(found)
                attrib['OriginalText'] = original_section_text
                # TermType required in table
                section_text = attrib['TermType']
                # sent = ' '.join(sentence.split()[:offset])
            # else:
            if not found:
                offset = 0
                # get whole sentence
                sent = sentence
                tok = tokens
            note.sections[section_id] = note.sections.get(section_id,
                # create section if section not already there
                {'attrib': attrib,
                 'sentences': []})
            if offset < len(tokens): # sentence continues after found
            # elif offset < len(tokens): # sentence continues after found
                # get rest of sentence
                sent = sentence[sentence.lower().find(tokens[offset],
                                                        # start after found
                                                        sum(len(f) + 1
                                                            for f in found)):]
                # sent = sent + ' ' + sentence[sentence.lower().find(tokens[offset],
                #                                                     # start after found
                #                                                     sum(len(f) + 1
                #                                                         for f in found)):]
                tok = tokens[offset:]
            # original code below
            # # #
            # elif offset + 1 < len(tokens): # sentence continues after found
            #     # get rest of sentence
            #     sent = sentence[sentence.find(tokens[offset + 1],
            #                                  # start after found
            #                                  sum(len(f) + 1
            #                                      for f in found)):]
            #     tok = tokens[offset + 1:]
            # # #
            else: # found whole sentence
                continue # go to next sentence
            # get character positions
            character_positions = []
            for token in tok:
                pos = note.raw.find(token, start_char_pos)
                # if token found and position is "reasonable" (not more than 2
                # sentence lengths away)
                if pos >= 0 and pos - start_char_pos <= 2 * len(sent):
                    start_char_pos = pos + len(token)
                    character_positions.append((pos, start_char_pos))
                else:
                    character_positions.append((start_char_pos, start_char_pos))
            # add sentence to section
            note.sections[section_id]['sentences'].append(
                {'attrib': {'SentenceID': str(sentence_id),
                            'SectionID': str(section_id),
                            'SectionText': section_text},
                 'text': original_section_text + sent,
                 'tokens': tok,
                 'character_positions': character_positions,
                 'terms': {}})
            original_section_text = ''
            sentence_id += 1
                
    def find_sentences(self, raw):
        "Segment a note into sentences, adapted from parser and regexops."
        # remove excess whitespace
        raw = re.sub(self.whitespace_regex, r'\1 \2', raw)
        raw = re.sub(self.whitespace_regex, r'\1 \2', raw)
        # mark sentence boundaries
        raw = raw.replace('."', '".')
        raw = re.sub(self.boundaries_regex, '<><>', raw)
        return [sentence.strip() for sentence in raw.split('<><>')]
