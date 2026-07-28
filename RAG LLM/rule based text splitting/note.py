"""
//******************************************************************************
// FILENAME:           note.py
// DESCRIPTION:        This Natural Language Processing Python code is the
//                     MTERMS service for the NLP Project.
// CREATION DATE:      January 27, 2016
// INITIAL AUTHOR:     Kenneth Lai (KL549)
// LAST MODIFIED DATE: 1/27/2016 6:20 PM
// LAST MODIFIED BY:   KL549
// Copyright (C) 2016  Partners Information Systems - Clinical and Quality
//                     Analysis (IS - CQA)
//                     Partners HealthCare System Confidential
//******************************************************************************
"""

class Note:

    def __init__(self, text):

        self.raw = text
        self.sections = {}
