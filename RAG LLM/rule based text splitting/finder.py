"""
//******************************************************************************
// FILENAME:           finder.py
// DESCRIPTION:        This Natural Language Processing Python code is the
//                     MTERMS service for the NLP Project.
// CREATION DATE:      May 5, 2015
// INITIAL AUTHOR:     Kenneth Lai (KL549)
// LAST MODIFIED DATE: 1/27/2016 6:15 PM
// LAST MODIFIED BY:   kl549
// Copyright (C) 2016  Partners Information Systems - Clinical and Quality
//                     Analysis (IS - CQA)
//                     Partners HealthCare System Confidential
//******************************************************************************
"""

def find(sentence, table, indexer, found, forward):
    """
    Returns matched (multi-word) term from trie if match exists.
    Args:
        sentence/tokens: List
        table: Dictionary
        indexer: Int
        found: List
        forward: Boolean
    """
    nested_table = table
    offset = len(found)  # words found
    for i in range(offset):
        # follow words into nested table
        nested_table = nested_table[found[i].lower()]
    if indexer + offset < len(sentence):
        token = sentence[indexer + offset]  # current word
    else:
        token = ''  # end of sentence
    lower = token.lower()
    # if current word in nested table and looking forward
    if lower in nested_table and forward:
        found.append(token)  # add word to found
        return find(sentence, table, indexer, found, True)
    elif 'TerminalTableEntry' in nested_table:
        return found, nested_table['TerminalTableEntry'], offset
    elif found:  # we went too far
        found.pop()  # start backtracking
        return find(sentence, table, indexer, found, False)  # look backward
    else:
        return [], (), 1  # none found
