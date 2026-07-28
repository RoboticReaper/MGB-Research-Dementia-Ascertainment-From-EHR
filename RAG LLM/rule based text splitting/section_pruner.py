"""
//******************************************************************************
// FILENAME:           mainprogram_section_notes_pg.py
// DESCRIPTION:        To remove specific sections that are not to be included for text analysis
// CREATION DATE:      Feb 13, 2018
// INITIAL AUTHOR:     Liqin Wang
// LAST MODIFIED DATE: 
// LAST MODIFIED BY:   lw592
// Copyright (C) 2017  Partners Information Systems - Clinical and Quality
//                     Analysis (IS - CQA)
//                     Partners HealthCare System Confidential
//******************************************************************************
"""

import os
import re
from collections import defaultdict
from sections import Sections
from note import Note

from time import time

sections_to_keep = ['History of Present Illness Section',
                    'Hospital Consultations',
                    'Objective',
                    'Assessment and Plan',
                    'Assessment Section',
                    'Chief Complaint Section',
                    'Chief Complaint Section/ Reason for Visit',
                    'Clinical Presentation',
                    'Complications',
                    'Conclusions',
                    'Physical Exam Section',
                    'Problem Section',
                    'Reason for Referral Section',
                    'Reason for Visit Section',
                    'Social History',
                    'Subjective',
                    'Unknown Section Name']
sections_maybe_keep = ['Addendum',
                       'Advance Directives Section',
                       'Diagnosis Section',
                       'Discharge Summary',
                       'Family History Section',
                       'Findings Section',
                       'General Status Section',
                       'History of Past Illness Section',
                       'Hospital Admission Diagnosis Section',
                       'Hospital Course Section',
                       'Interventions',
                       'Number One Section',
                       'Number Two Section',
                       'Number Three Section',
                       'Review of Systems Section',
                       'Medical (General) History',
                       'Section Header']


class Section_Pruner:

    def __init__(self):
        # define what sections to be excluded.
        # look the lexicon/section_2.csv to find out the complete list of sections
        self.sections_to_remove = ['Operative Note Surgical Procedure Section',
                                   'Overall prognosis for recovery from this episode OASIS',
                                   'Postoperative Diagnosis',
                                   'Postprocedure Diagnosis Section',
                                   'Preoperative Diagnosis',
                                   'Procedure Findings',
                                   'Procedures Section',
                                   'Admission Date',
                                   'Allergies Section',
                                   'Anesthesia Section',
                                   'Attending Physician Name',
                                   'Attending physician name',
                                   'Birth date',
                                   'Current Imaging Procedure Descriptions',
                                   'Date last menstrual period',
                                   'Demographic information section',
                                   'DICOM Object Catalog Section',
                                   'Discharge Diet Section',
                                   'Discharge plan',
                                   'Document Summary',
                                   'Drug-Nutrient Interactions',
                                   'Emergency contact information',
                                   'Encounters',
                                   'Follow-up contact supplemental information',
                                   'Health insurance card',
                                   'Hospital Discharge Date',
                                   'Hospital Discharge Diagnosis Section',
                                   'Hospital Discharge Disposition',
                                   'Hospital Discharge Instructions',
                                   'Hospital Discharge Medication Section',
                                   'Hospital Discharge Physical',
                                   'Hospital Discharge Studies Summary',
                                   'Illness or injury onset date and time',
                                   'Immunizations',
                                   'Interpreter needed',
                                   'Key Images',
                                   'Laboratory Test Section',
                                   'Medical Equipment',
                                   'Medication Section',
                                   'Medications Administered Section',
                                   'MRN',
                                   'Operative Note Fluid Section',
                                   'Oral or dental status section',
                                   'Patient information - acute-CARE',
                                   'Payers',
                                   'Plan of Care Section',
                                   'Planned Procedure Section',
                                   'Prior Imaging Procedure Description',
                                   'Procedure Description Section',
                                   'Procedure Disposition Section',
                                   'Procedure Estimated Blood Loss',
                                   'Procedure Implants',
                                   'Procedure Indications Section',
                                   'Procedure Specimens Taken',
                                   'Radiology Study -Recommendations',
                                   'Results Section',
                                   'RN name',
                                   'Specimen-related information panel',
                                   'Surgical Drains Section',
                                   'Vital Signs Section',
                                   'Addendum',                              # ? section labels
                                   'Advance Directives Section',
                                   'Diagnosis Section',
                                   'Discharge Summary',
                                   'Family History Section',
                                   'Findings Section',
                                   'General Status Section',
                                   'History of Past Illness Section',
                                   # 'History of Present Illness Section',
                                   'Hospital Admission Diagnosis Section',
                                   # 'Hospital Consultations',
                                   'Hospital Course Section',
                                   'Interventions',
                                   'Number One Section',
                                   'Number Two Section',
                                   'Number Three Section',
                                   'Review of Systems Section',
                                   'Medical (General) History',
                                   'Section Header']
        # self.sections_to_keep_map = {label: idx for idx, label in enumerate(sections_to_keep)}
        #print( 'section file path: ', os.path.join(os.path.dirname(__file__), 'lexicon', 'section_2.csv'))
        self.s = Sections(open(os.path.join(os.path.dirname(__file__), 'lexicon', 'section_2.csv')).read())

    def process_note(self, text):
        note = Note(text)
        self.s.find_sections(note)
        for section in note.sections.itervalues():
            if section['attrib']['TermType'] == 'Medication Section':
                for sentence in section['sentences']:
                    print(sentence['text'])
                # print section['sentences']['attrib']['SectionText']

    def prune_sections(self, text):
        new_text = ''
        note = Note(text)
        self.s.find_sections(note)
        for section in note.sections.values():
            #if section['attrib']['TermType'] == 'Medication Section':
            if not any([section['attrib']['TermType'].strip() == x for x in self.sections_to_remove]):
                # print filenm, section['attrib']['TermType']
                # processedstring += ' ' + section['attrib']['TermType'] + ': '
                
                for sentence in section['sentences']:
                    if sentence['text'].strip() != '':
                        new_text += ' ' + sentence['text'] + '.'
                        
##                    if not(section['attrib']['TermType']== 'Unknown Section Name'
##                             and any(string in sentence['text'].strip() for string in ['Address:', 'Brigham and Women\'s Hospital', 'Boston Ma', 'Boston, MA', '75 Francis St', 'Phone:' , 'Fax:', 'Patient:'] )):
##                        

                    #print section['attrib']['TermType'], sentence['text']
                # for each section, add a new line
                new_text += '  '

        return new_text

    def check_sectionizer(self, text):
        new_text = ''
        section_counts = defaultdict(int)       # keeps count of unique section labels inserted in note
        section_list = []                       # ordered list of section labels inserted and original text
        note = Note(text)
        self.s.find_sections(note)
        for section in note.sections.values():
            # if section['attrib']['TermType'].strip() in self.sections_to_remove:    # want to analyze contents of excluded sections
            new_text += ' [' + section['attrib']['TermType'].strip() + ']:'
            for sentence in section['sentences']:
                if sentence['text'].strip() != '':
                    new_text += ' ' + sentence['text'] + '.'

            # new_text += '  \n   '

            section_counts[section['attrib']['TermType']] += 1
            section_list.append((section['attrib']['TermType'], section['attrib']['OriginalText']))

        return new_text, section_counts, section_list

    def get_scd_sections(self, text):
        new_text = {label: '' for label in sections_to_keep + sections_maybe_keep + self.sections_to_remove}
        # new_text = {label: '' for label in self.sections_to_remove}
        section_counts = defaultdict(int)       # keeps count of unique section labels inserted in note
        section_list = []                       # ordered list of section labels inserted and original text
        note = Note(text)
        self.s.find_sections(note)

        for section in note.sections.values():
            section_label = section['attrib']['TermType'].strip()
            # if section_label in sections_to_keep:                         # want contents of keep sections
            # if section_label not in sections_to_keep:                         # want contents of maybe and remove sections
            # if section_label in sections_to_keep + sections_maybe_keep:     # want contents of keep and maybe sections
            # if section_label not in sections_to_keep + sections_maybe_keep:     # want contents of remove sections
            # new_text.update({
            #     section_label: new_text.get(section_label, '') + f' [{section_label}]:'
            # })
            if section_label not in new_text:
                new_text[section_label] = ''
            # new_text[section_label] += f' [{section_label}]:'
            for sentence in section['sentences']:
                sent_text = sentence['text'].strip()
                if sent_text != '':
                    # new_text.update({
                    #     section_label: new_text.get(section_label, '') + f' {sent_text}'
                    # })
                    new_text[section_label] += f' {sent_text}.'
            # new_text.update({
            #     section_label: new_text.get(section_label, '') + f'\n'
            # })
            # new_text[section_label] += '\n'

            section_counts[section['attrib']['TermType']] += 1
            section_list.append((section['attrib']['TermType'], section['attrib']['OriginalText']))


        return new_text, section_counts, section_list
