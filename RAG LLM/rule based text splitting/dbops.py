"""
//******************************************************************************
// FILENAME:           dbops.py
// DESCRIPTION:        This Natural Language Processing Python code is the
//                     MTERMS service for the NLP Project.
// CREATION DATE:      January 22, 2016
// INITIAL AUTHOR:     Kenneth Lai (KL549)
// LAST MODIFIED DATE: 5/11/2016 1:45 PM
// LAST MODIFIED BY:   KL549
// Copyright (C) 2016  Partners Information Systems - Clinical and Quality
//                     Analysis (IS - CQA)
//                     Partners HealthCare System Confidential
//******************************************************************************
"""

from io import StringIO
from csv import reader
from tokenizer import tokenize

def load_table_from_csv(csv_string, meds_flag=False):

    table = {}
    csv_file = StringIO(csv_string)
    rdr = reader(csv_file)
    table['ColumnNames'] = next(rdr)[1:] 
    #print(table['ColumnNames'])
    
    for row in rdr:
        try:
            key = tokenize(row[0])
            value = tuple(row[1:])
            nest(table, key, value, meds_flag)
        except AttributeError: # except search term is null
            pass
    return table

def nest(table, key, value, meds_flag):

    token = key[0].lower()
    table[token] = table.get(token, {}) # check if token already there
    if len(key) > 1:
        nest(table[token], key[1:], value, meds_flag) # recursively nest table
        if len(key) == 2 and not meds_flag: # plurals for non-medications
            next_token = key[1]
            if next_token.endswith('ch') or next_token.endswith('sh'):
                # True to avoid double plurals
                nest(table[token], [next_token + 'es'], value, True)
            else:
                nest(table[token], [next_token + 's'], value, True)
    else:
        table[token]['TerminalTableEntry'] = value # end
        if not meds_flag: # plurals for non-medications
            if token.endswith('ch') or token.endswith('sh'):
                plural = token + 'es'
            else:
                plural = token + 's'
            table[plural] = table.get(plural, {})
            table[plural]['TerminalTableEntry'] = value
