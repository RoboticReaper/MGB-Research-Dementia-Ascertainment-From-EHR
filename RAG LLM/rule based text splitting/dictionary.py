'''
Author: Liqin Wang
Implementing the dictionary with multiprocessing. We will first chunk the files according the number of CPU cores.
Several pipeline of loading the dictionary. Finally, merging all the dictionary into one. 

'''

from multiprocessing import Pool

import gensim
import os
from gensim import corpora
from tokenizer import tokenize
import pandas as pd
#from normalizer import normalize
from collections import defaultdict
from gensim.corpora import TextCorpus, MmCorpus, Dictionary
import csv
import re
import sys
from time import time
from smart_open import smart_open 

#final_dict = Dictionary()

class MyDictionary:

    def __init__(self):

        self.corpus_file = '''P://Liqin//mortality_prediction_sq_2year//data//notes//full_palliative_cohort_notes_by_date_sample_o.txt'''
        self.frequency = defaultdict(int)
        __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
        # the stop word list in the current folder were modified from mallet original 
        stoplist_string = open(os.path.join(__location__, 'stoplist.txt')).read()
        self.stoplist = set(stoplist_string.split())
        print('Read in stop words list!')

    ''' generator of text line by line'''
    def get_text(self, corpus_file):
        with open(corpus_file) as f:
            for i, line in enumerate(f):
                if i == 0 and 'noteid' in line:
                    continue 
                doc = line.split('\t')
                if len(doc) >= 2:
                    text = ' '.join(doc[1:])
                    #text = text.decode('utf-8')
                    words = text.lower().split()
                    yield words



    ''' get a chunck of texts '''
    # def get_texts(self, corpus_file, chunkSize): 
    #     words = []
    #     with open(corpus_file, 'rb') as f:
    #         for i, line in enumerate(f):
    #             if i == 0 and b'noteid' in line:
    #                 continue 
    #             doc = line.split(b'\t')
    #             if len(doc) >= 2:
    #                 text = b' '.join(doc[1:])
    #                 text = text.decode('utf-8')
    #                 words = text.lower().split()
    #                 yield words


    def chunklines(self, input_file, size=100):
        file_total_lines = sum(1 for line in open(input_file, 'rb'))
        print('Total lines of the file is: ', file_total_lines)
        chunkEnd = 0
        chunkStart = 0
        chunks = []
        with open(input_file,'rb') as f:
            for i, line in enumerate(f):
                chunkStart = chunkEnd
                if i%size == 0 and i!=0:
                    chunkEnd = i
                    chunks.append((chunkStart, chunkEnd-chunkStart))                

                #yield chunkStart, chunkEnd - chunkStart                
                if i >= file_total_lines:
                    chunks.append((chunkStart, chunkEnd-chunkStart))
                    print('Chunks: -------- ')
                    print(chunks)
                    break
        return chunks

    ''' get words of a line of a corpus'''
    def get_words(self, line):
        words = []

        # if decode:
        #     try:
        #         line = line.decode('utf-8')
        #     except UnicodeDecodeError:
        #         try:
        #             line = line.decode('utf-16')
        #         except UnicodeDecodeError:
        #             try:
        #                 line = line.decode('windows-1252')
        #             except UnicodeDecodeError:
        #                 raise UnicodeDecodeError

        doc = line.split('\t')
        if len(doc) >= 2:
            text = ' '.join(doc[1:])
            words = text.lower().split()
        return words

    def get_corpus_chunk(self, corpus_file, chunkStart, chunkSize, stopwords):
        corpus = []
        with open(corpus_file, encoding='utf-8', errors='ignore') as f:
            for i, line in enumerate(f):
                if i>=chunkStart and i < chunkStart + chunkSize:
                    words = self.get_words(line, False, stopwords)
                    # to remove stop words 
                    words = [word for word in words if word not in self.stoplist] 
                    corpus.append(words)
        return corpus


    def split_line(self, text, decode=True):
        if decode:
            try:
                text = text.decode('utf-8')
            except UnicodeDecodeError:
                try:
                    text = text.decode('utf-16')
                except UnicodeDecodeError:
                    try:
                        text = text.decode('windows-1252')
                    except UnicodeDecodeError:
                        raise UnicodeDecodeError
        
        '''remove non-ascii'''
        words = tokenize(text) 
        return words

# def process_wrapper(chunkStart, chunkSize):
#     with open(input_file, 'rb') as f:
#         # f.seek(chunkStart)
#         # #lines = f.read(chunkSize).splitlines()
#         # lines = f.read(chunkSize).splitlines()
#         # print (lines)
#         # print ("load sentences")
#         corpus = []
#         for i, line in enumerate(f):
#             if i>=chunkStart and i < chunkStart + chunkSize:
#                 words = self.get_words(line, decode = True)
#                 corpus.append(words)
#         tem_dict = Dictionary(corpus)
#         final_dict.merge_with(tem_dict)


if __name__=='__main__':

    print ('Building dictionary...')
    start=time()

    ''' Note: the schema of the file path matters, using backslash'''
    # corpus_file = '''F:\\PalliativeCare\\Liqin\\rnn_mortality_prediction\\data\\dementia_note_merged_by_date_3\\dementia_note_by_date_all_prune_norm.txt'''
    # dict_file_name = '''F:\\PalliativeCare\\Liqin\\rnn_mortality_prediction\\data\\dementia_note_merged_by_date_3\\dementia_note_by_date_all_prune_norm_dict'''
    # dict_file_text_name = '''F:\\PalliativeCare\\Liqin\\rnn_mortality_prediction\\data\\dementia_note_merged_by_date_3\\dementia_note_by_date_all_prune_norm_txt_dict'''


    corpus_file = '''P:\\Liqin\\mortality_prediction_sq_2year\\data\\notes\\full_palliative_cohort_notes_by_date_sample_o.txt'''
    dict_file_name = '''P:\\Liqin\\mortality_prediction_sq_2year\\data\\notes\\full_palliative_cohort_notes_by_date_sample_dict'''
    dict_file_text_name = '''P:\\Liqin\\mortality_prediction_sq_2year\\data\\notes\\full_palliative_cohort_notes_by_date_sample_dict_text'''

    iDict = MyDictionary()
    '''the first 20 character are the document id'''

    dct = Dictionary()

    doc_counts = sum(1 for line in open(corpus_file, 'rb'))
    print ("Overall ", doc_counts, " to process")

    print (corpus_file)
    
    '''Single processor '''
    chunkStart = 0
    chunkSize = 10
    for chunkStart, chunkSize in iDict.chunklines(corpus_file, chunkSize):
        corpus = iDict.get_corpus_chunk(corpus_file, chunkStart, chunkSize, True) # remove stopwords = True
        dct.add_documents(corpus)
    
    # cores = 2
    # pool = mp.Pool(cores)

    # #create jobs
    # jobs = []
    
    # chunkStart = 0
    # chunkSize = 10
    # for chunkStart, chunkSize in chunklines(input_file, chunkSize):
    #     print ('loading one chunk')
    #     print('chunkStart: -----', chunkStart)
    #     print('chunkEnd: ', chunkStart+chunkSize)
    #     #mp.Process(process_wrapper,(chunkStart,chunkSize))
    #     jobs.append( pool.apply_async(process_wrapper,(chunkStart,chunkSize)) )

    # #wait for all jobs to finish
    # for job in jobs:
    #     job.get()

    # #clean up
    # pool.close()


    dct.save_as_text(dict_file_text_name)
    dct.save(dict_file_name)

    dict_final = Dictionary()

    corpus_final
    for i in cores:
        if i ==1:
            dict_final = corpora.Dictionary.load('dict_{}.dict'.format(i))
            corpus_final = corpora.MnCorpus('corpus_{}.mm'.format(i))
        else:
            new_dictionary = corpora.Dictionary.load('dict_{}.dict'.format(i))
            new_corpus = corpora.MnCorpus('corpus_{}.mm'.format(i))
            dict_final = dict_final.merge_with(new_dictionary)
            corpus_final = itertools.chaim(corpus_final, dict_final[new_corpus])


    ## filter_extremes(no_below=5, no_above=0.5, keep_n=100000, keep_tokens=None) 

    '''    
    Parameters:	
    no_below (int, optional) – Keep tokens which are contained in at least no_below documents.
    no_above (float, optional) – Keep tokens which are contained in no more than no_above documents (fraction of total corpus size, not an absolute number).
    keep_n (int, optional) – Keep only the first keep_n most frequent tokens.
    keep_tokens (iterable of str) – Iterable of tokens that must stay in dictionary after filtering.
    '''
    dct.filter_extremes( no_below = 50, no_above = 0.5 )
    print( 'length of the dictionary after filtering: ', len(dct) )


    
