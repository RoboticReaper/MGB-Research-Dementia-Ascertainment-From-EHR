"""
//******************************************************************************
// FILENAME:           tokenizer.py
// DESCRIPTION:        This Natural Language Processing Python code is the
//                     MTERMS service for the NLP Project.
// CREATION DATE:      January 27, 2016
// INITIAL AUTHOR:     Kenneth Lai (KL549)
// LAST MODIFIED DATE: 6/2/2016 2:40 PM
// LAST MODIFIED BY:   CSR18
// Copyright (C) 2016  Partners Information Systems - Clinical and Quality
//                     Analysis (IS - CQA)
//                     Partners HealthCare System Confidential
//******************************************************************************
"""

from re import sub
import re

def tokenize(string):
    string = string.lower()
    string = sub(r'[^\x01-\x7f]',r'', string) # remove non-ascii-string 
    string = string.replace('/', ' / ')
    string = string.replace('•', ' ')
    string = string.replace('·', ' ')
    string = string.replace('‘', '\'')
    string = string.replace('’', '\'')
    string = string.replace('\'', ' \' ')
    string = string.replace('mg ', ' mg ')
    string = string.replace('ml ', ' ml ')
    string = string.replace('unit/ ml ', ' unit / ml ')
    string = string.replace('MG ', ' MG ')
    string = string.replace('ML ', ' ML ')
    string = string.replace('UNIT/ ML ', ' UNIT / ML')
    string = string.replace('|', ' ')
    string = string.replace('.', ' ')
    string = string.replace(':', ' ')
    string = string.replace(';', ' ')
    string = string.replace(',', ' ')
    string = sub(r'(\d)/(\d)', r'\1 \2', string)
    string = string.replace('(', ' ')
    string = string.replace(')', ' ')
    string = string.replace('[', ' ')
    string = string.replace(']', ' ')
    string = string.replace('{', ' ')
    string = string.replace('}', ' ')
    string = string.replace('?', ' ')
    string = string.replace('!', ' ')
    string = string.replace('>', ' ') 
    string = string.replace('<', ' ') 
    string = string.replace('=', ' ') 
    string = string.replace('\\', ' ')
    string = string.replace('&', ' ')
    string = string.replace('%', ' ')
    string = string.replace('+', ' ')
    string = string.replace('-', ' ')
    string = string.replace('*', ' ')
    string = string.replace('=', ' ')
    string = string.replace('<', ' ')
    string = string.replace('>', ' ')
    string = string.replace('~', ' ')
    string = string.replace('@', ' ')
    string = string.replace('#', ' ')
    string = string.replace('_', ' ')
    string = string.replace('`', '')
    string = string.replace('\"', '')
    string = string.replace('“', '')
    string = string.replace('”', '')
    string = string.replace('‘', '\'')
    string = string.replace('’', '\'')
    string = string.replace('\'\'', '')
    string = sub(r'(\D)(\d)', r'\1 \2', string)
    string = sub(r'(\d)(\D)', r'\1 \2', string)
    string = string.strip()
    return string.split()
