# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14, 2020

"""

import os, inspect, sys, argparse
import re
import time
import json
from pprint import pprint
import traceback
from section_pruner import Section_Pruner, sections_to_keep
#from tk import tk2 as tk
#from corpus import MyCorpus
# import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from tokenizer import tokenize
# from seperate_big_file import file_seperator
# import spacy
import csv
#import datetime  
import psycopg2, psycopg2.extras
#import pyodbc

# from note import Note
# from sections2 import Sections
# from generic import Generic
# from negation_2 import Negation

keyword_string = 'memory delirium dementia psych neuro mental Parkinson alzheimer confus mood cognit forget agitat neuro moca montreal mmse remember'\
                  + ' difficult recall function word evaluat score drive attention mild impairment speech tremor question disorientation orientation sleep alter exam decline worse loss'\
                  + ' daughter wife husband son family'

def connect_db(host, database, username, password):
    
    # try: 
    connection = psycopg2.connect(host = host, database = database, user= username, password = password)
    cursor = connection.cursor()
    cursor.execute('SELECT version()')
    version = cursor.fetchone()[0]
    print(version)
    print("connected to the database successfully")
        
    # except psycopg2.DatabaseError as e:
        
    #     print(f'Error {e}')
    #     sys.exit(1)
        
    return cursor
    
#def connect_db(server, database, username, password, trust_connect=False):
#    if trust_connect:
#        connection = pyodbc.connect('Driver={SQL Server};SERVER='+server+';DATABASE='+database+';Trusted_Connection=yes;')
#    else:
#        pyodbc.connect('Driver={SQL Server};SERVER='+server+';DATABASE='+database+';UID='+username+';PWD='+ password)
#    cursor = connection.cursor()
#    return cursor

# '''get allergy entries from db'''
# def extract_allergy_dict(cursor, table_name):
#     query_string = 'SELECT * from %s'%(table_name)
#     print('Extract allergies: %s')%(query_string)
#     cursor.execute(query_string)
#     row = cursor.fetchone()
#     columns = [column[0] for column in cursor.description]
#     allergy_list = []
#     for row in cursor.fetchall():
#         allergy_list.append(dict(zip(columns, row)))
#     return allergy_list

# def load_mterms_modules():
#     s = Sections(open('lexicon/section.csv').read())
#     p = Generic(open('lexicon/problem_3.csv').read(), 'problems')
#     n = Negation(open('lexicon/negation_2.csv').read(),
#                  open('lexicon/terminator.csv').read())
#     return s, p, n

def write_to_database(cursor, target_database_table, output_column_names, table_content_list):
    # cursor.execute("drop table {0};".format(target_database_table))
    # cursor.execute("create table if not exists {} (noteid varchar(25), sentence text, negation varchar(50), experiencer varchar(50), term varchar(100), problem_snomedid varchar(100), problem_pref_term varchar(100));".format(target_database_table))

    start_time = time.time()
    ## option 1: slightly slower
    # insert_query = 'insert into %s (%s) '%(target_database_table, ", ".join(output_column_names)) 
    # insert_query =  insert_query +  ''' values %s'''
    # print(insert_query)
    # psycopg2.extras.execute_values (cursor,insert_query, table_content_list, page_size = 100 )
    # cursor.execute('COMMIT')

    # option 2: faster
    records_list_template = ','.join(['%s'] * len(table_content_list))
    # print(records_list_template)
    insert_query_1 = 'insert into %s (%s) '%(target_database_table, ", ".join(output_column_names)) 
    insert_query =  insert_query_1 + ' values {}'.format(records_list_template)
    #print(insert_query)
    cursor.execute(insert_query, table_content_list)
    cursor.execute('COMMIT')

  #  q_list = ', '.join( ['%s'] * len(table_content_list) )
    # cursor.fast_executemany = True
    # q_list = ['?']*len(output_column_names)
    # q_list = ['?']*len(table_content_list)

   # print(q_list)

    # sql = "INSERT INTO %s (%s) VALUES (%s);"%(target_database_table, ", ".join(output_column_names), ", ".join(q_list))
    #sql = "INSERT INTO %s (%s) VALUES (%s);"%(target_database_table, ", ".join(output_column_names), q_list)

    #print(sql)
    #cursor.execute(sql, )
    #chunk_write_database(cursor, sql, table_content_list)
    # cursor.executemany(sql, write_values)
    # cursor.commit()
    # cost_time = time.time() - start_time
    # print("Write to database cost time: %.2fs"%cost_time)
    return 0

def chunk_write_database(cursor, sql, big_list, chunk_size=100):
    whole_size = len(big_list)
    start_id = 0 
    end_id = chunk_size
    batch_number = whole_size//chunk_size+1 
    start_time = time.time()
    for batch_id in range(batch_number):
        start_id = batch_id*chunk_size 
        end_id = start_id+chunk_size
        if end_id > whole_size:
            end_id = whole_size
        if start_id == end_id:
            continue
        cursor.execute(sql, big_list[start_id:end_id])
        cursor.commit()
        if end_id%1000==0:
            end_time = time.time()
            print("     Writing... at: %s, time cost: %s"%(end_id, end_time-start_time))
            start_time = end_time
    print("     Writing... at: %s, time cost: %s"%(end_id, time.time()-start_time))
    print("     Database writing finished! %s rows inserted."%(whole_size))


'''extract reactions from free-text comment field'''
def extract_sections_and_write_database(cursor, source_table_name, column_name_list, target_database_table):
    output_column_names = ["noteid", "section_lnb", "section_class", "sectionsnm", "sectiontxt", "text_length", "keywords"]

	## ----Check if the target table exists or not 
	
#	if 
	
#	else: 
		# create table 
#		sql_string_create_table = ""
	
	## -----------
	
    #sql_string = "SELECT %s from %s where noteid not in (select distinct noteid from %s);"%(", ".join(column_name_list), source_table_name, target_database_table) limit 1000 ## identify noteid that have not been processed (or not in the target database table. 
    sql_string = "select noteid, notetxt from %s limit 1000 ;"%(source_table_name) ## for testing, change the limit 1000 to any other number
    # print('Extract from SQL: ', sql_string)
    text_result_pair = []
    cursor.execute(sql_string)

    i=1
    row = cursor.fetchone()
    #for i in range(total_case_num):
    print('beginning note processing...')
    #wf = open('covid_cohort_notes_no_problems.csv','w+')


    #output_column_no_problem = ["noteid","sentence"]
    #writer = csv.DictWriter(wf, fieldnames=output_column_no_problem, lineterminator='\n')
    #writer.writeheader()
    output_rows = []
    output_rows_noproblem = []
    
    while row!=None:
    
        # print out every 10000
        if i%10000 == 0 :
            print('processed {} notes...'.format(i))
        
        # contain noteid and sourceTxt the noteid is empi_noteid
        noteid, sourceText = row

        sectionized_text, dict_topic_wordlist, dict_topic_uniquewordlist  = process(sourceText)
        # sectionid, sectionized_text, dict_topic_wordlist, dict_topic_uniquewordlist  = process(sourceText)

        if noteid != '' :
            section_id = 0
            sentence_added_flg = False
            for label, text in sectionized_text.items():
                keep_or_remove = 'keep' if label in sections_to_keep else 'remove'
                individual_sections = text.split(f'[{label}]: ')  # split initial newline, separate sections
                for section in individual_sections[1:]:
                    keyword_matches = keyword_match(section.lower())
                    keyword_match_str = ' '.join(keyword_matches)
                    unique_keywords = set(keyword_matches)
                    unique_keyword_str = ' '.join(unique_keywords)
                    # keyword_matches_1, keyword_matches_2, keyword_matches_3 = keyword_match(section.lower())
                    # keyword_match_str_1 = ' '.join(keyword_matches_1)
                    # unique_keywords_1 = set(keyword_matches_1)
                    # unique_keyword_str_1 = ' '.join(unique_keywords_1)
                    # keyword_match_str_2 = ' '.join(keyword_matches_2)
                    # unique_keywords_2 = set(keyword_matches_2)
                    # unique_keyword_str_2 = ' '.join(unique_keywords_2)
                    # keyword_match_str_3 = ' '.join(keyword_matches_3)
                    # unique_keywords_3 = set(keyword_matches_3)
                    # unique_keyword_str_3 = ' '.join(unique_keywords_3)
                    output_rows.append((noteid,
                                        section_id,
                                        keep_or_remove,
                                        label,
                                        section[:-1],
                                        len(section[:-1]),
                                        unique_keyword_str,
                                        # keyword_match_str_1,
                                        # unique_keyword_str_1, len(unique_keywords_1), keyword_match_str_2,
                                        # unique_keyword_str_2, len(unique_keywords_2), keyword_match_str_3,
                                        # unique_keyword_str_3, len(unique_keywords_3)
                                       ))
                    sentence_added_flg = True
                    section_id += 1
                        #output_rows.append({"noteid":noteid, "sentence":sentence_text, "negation":Problem_neg, "term":Problem_text, "problem_snomedid":Problem_snomedid, "problem_pref_term":Problem_prefterm})
                
            if not sentence_added_flg:
                ## even there is no problem detected, we will still print it out , for the purpose of chart review
                output_rows.append((noteid, None, None, None, None, None, None))
                # output_rows.append((noteid, None, None, None, None, None, None, None, None, None))

        if len(output_rows) > 0 and i%10000 == 0 :
            print('write 10000 notes into database...')
            print('Total number of rows to write: ')
            print( len(output_rows))
            # for outrow in output_rows:
            #     writer.writerow(outrow)
            write_to_database(cursor, target_database_table, output_column_names, output_rows)
            output_rows = []
            print('query again...')
            # as the cursor were switchted to the insert mode, now query the database again to processing the remainning ones. 
            #sql_string = "SELECT %s from %s where empi_noteid not in (select distinct empi_noteid from %s);"%(", ".join(column_name_list), table_name, target_database_table)
            # print(sql_string)
            #cursor.execute(sql_string)

        i+=1
        row = cursor.fetchone()

    ## write whatever remained 
    if len(output_rows) > 0 :
        print('write the rest into database...')
        print('Total number of rows to write: ')
        print( len(output_rows))       
        # for outrow in output_rows:
        #     writer.writerow(outrow)
        write_to_database(cursor, target_database_table, output_column_names, output_rows)
        # write_to_database(cursor, target_database_table, output_column_names, output_rows)
        
    # wf.close()

def process(doc):
    
    # w_count = 0

    text = doc
    # text = ' '.join(doc[2:]) ## for processed notes with the format of noteid wordsize words
    #text= line.replace(doc[0], '').strip()
    #print text
    ''' if the document have multiple tab then, it should be concatenated'''

    # remove some redundant wordings: 
    # e.g., ***This text report has been converted from the report, '1476469992.pdf'. Content may not appear exactly as it appears in the original .pdf. For a download of original content and format (pdf), please contact the RPDR Team at RPDRHelp@partners.org. ***
    
    #text = text.replace('***This text report has been converted from the report', '')
    #text = text.replace('Content may not appear exactly as it appears in the original .pdf. For a download of original content and format (pdf), please contact the RPDR Team at RPDRHelp@partners.org. ***', '')
    
    # prune unwanted sections
    #text = spruner.prune_sections(text)

    # # normalize the text
    # words = split_line(text)
    
    #words = text.strip().split()
    # wordlist = []
    #words = [ word for word in words if word not in stoplist and word not in names and word.isalpha()]
    # words = [ word for word in words if word.isalpha()]
    # for word in words:
    #     if word not in ['left']:
    #         word = lmtzr.lemmatize( lmtzr.lemmatize(word), 'v')
    #     else:
    #         word = word
    #     wordlist.append(word)

    ## lemmatize may result some words in the stop list. For example 'times' will be converted to 'time'. 
    ## we remove the stop word list after the lemmatize. 
    # wordlist2 = [ word for word in wordlist if word not in stoplist and word not in names]
    dict_topic_wordlist = {}
    dict_topic_uniquewordlist = {}
    # for topic in topics:
    #
    #     topic_words = topics[topic].split(' ')
    #     topic_wordlist = []
    #     # to check if the words were mentioned in the notes
    #     for word in words:
    #         for topic_word in topic_words:
    #             if len(topic_word)>4 and word.startswith(topic_word):
    #                 topic_wordlist.append(word)
    #             elif len(topic_word) <=4 and topic_word == word:
    #                 topic_wordlist.append(word)
    #         # topic_wordlist = [topic_word if topic_word in word for topic_word in topic_words]
    #     dict_topic_wordlist[topic] = ' '.join(topic_wordlist)
    #     dict_topic_uniquewordlist[topic] = set(topic_wordlist)
    
    #words = [word for word in words if word in includewordlist]
    # text = ' '.join(wordlist2)
    # w_count = len(wordlist2)
    # text = text.strip()

    # sectionized_text, section_counts, section_list = spruner.get_scd_sections(' '.join(words))
    text = re.sub(r'[^\x01-\x7f]', r'', text)  # remove non-ascii-string
    spruner = Section_Pruner()
    sectionized_text, section_counts, section_list = spruner.get_scd_sections(text)

    # return noteid, sectionized_text, dict_topic_wordlist, dict_topic_uniquewordlist
    return sectionized_text, dict_topic_wordlist, dict_topic_uniquewordlist

def keyword_match(text):
    tokenized_text = tokenize(text)
    token_matches = []
    # token_matches_1 = []
    for keyword in keyword_string.split():
        token_matches.extend([token for token in tokenized_text if token.startswith(keyword)])
    # for keyword in topics['topic1'].split():
    #     token_matches_1.extend([token for token in tokenized_text if token.startswith(keyword)])
    # token_matches_2 = []
    # for keyword in topics['topic2'].split():
    #     token_matches_2.extend([token for token in tokenized_text if token.startswith(keyword)])
    # token_matches_3 = []
    # for keyword in topics['topic3'].split():
    #     token_matches_3.extend([token for token in tokenized_text if token.startswith(keyword)])

    return token_matches
    # return token_matches_1, token_matches_2, token_matches_3


def listener(q, output_file):
    '''listens for messages on the q, writes to file. '''

    f = open(output_file, 'a+', newline='')
    f_writer = csv.writer(f)
    while 1:
        print(q.qsize())
        lines = q.get()
        
        if lines == 'kill':
            print('Kill the listener')
            break
        print('writing {} of lines to the output...\n'.format(len(lines)))
        for line in lines:
            # f.write("%s\n" % line)
            for section_data in line:
                f_writer.writerow(section_data)
                f.flush()
    f.close()

    
def word_count(text):
    count = 0
    
    if len(text) < 20:
        noteid = ''
        notetxt = ''
        count = 0
    else:
        noteid = text[0:20]
        notetxt = text[21:].strip()

    if notetxt != '':
        count = len(notetxt.split())

    return noteid, count

            


def split_line( text ):
    '''remove non-ascii'''
    words = tokenize(text) 
    #print(words)
    return words


if __name__ == '__main__':
    #print('PID: ', os.getpid())
    username = ''
    password = ''
    source_database_table = "palliative.scd.mci_2019_cohort_edw_notes_latest_by_noteid_final"
    target_database_table = "palliative.scd.mci_2019_cohort_edw_notes_sections"
    ## if run in voice, use localhost, otherwise, set the server as "VOICE"
    cursor = connect_db('VOICE', 'palliative', username, password)
    extract_sections_and_write_database(cursor, source_database_table, ["noteid", "notetxt"], target_database_table)

    exit(0)
