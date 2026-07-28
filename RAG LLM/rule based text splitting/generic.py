"""
//******************************************************************************
// FILENAME:           generic.py
// DESCRIPTION:        This Natural Language Processing Python code is the
//                     MTERMS service for the NLP Project.
// CREATION DATE:      February 5, 2016
// INITIAL AUTHOR:     Kenneth Lai (KL549)
// LAST MODIFIED DATE: 5/10/2016 6:00 PM
// LAST MODIFIED BY:   KL549
// Copyright (C) 2016  Partners Information Systems - Clinical and Quality
//                     Analysis (IS - CQA)
//                     Partners HealthCare System Confidential
//******************************************************************************
"""

from dbops import load_table_from_csv
from finder import find

class Generic():

    def __init__(self, lexicon, name, meds_flag=False):

        # load table
        self.table = load_table_from_csv(lexicon, meds_flag)
        self.table['TableName'] = name

    def find_generic(self, sentence):
        "Find generic terms in a sentence."
        tokens = sentence['tokens']
        indexer = 0
        found_id = 1
        name = self.table['TableName']
        while indexer < len(tokens):
            found, entry, offset = find(tokens,
                                        self.table,
                                        indexer,
                                        [],  # found = none yet
                                        True)  # forward = True
            if found:
                column_names = self.table['ColumnNames']
                # get attributes from table
                attrib = dict(zip((unicode(i, 'cp1252') for i in column_names),
                                  (unicode(j, 'cp1252') for j in entry)))
                # and from parent sentence
                attrib.update(sentence['attrib'])
                # and a few more
                attrib['WordCount'] = str(indexer + 1)
                attrib[name + 'FoundID'] = str(found_id)
                attrib['PhraseLength'] = str(offset)
                generic = {'attrib': attrib,
                           'text': ' '.join(found)}
                sentence['terms'][name] = sentence['terms'].get(name, [])
                # add generic term to sentence
                sentence['terms'][name].append(generic)
                found_id += 1
            indexer += offset
