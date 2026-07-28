from multiprocessing import Pool
import os, inspect, sys, argparse
#from section_pruner import Section_Pruner
from time import time
import json
from pprint import pprint
import traceback
from section_pruner import Section_Pruner, sections_to_keep
#from tk import tk2 as tk
import multiprocessing as mp
#from corpus import MyCorpus
# import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from tokenizer import tokenize
from seperate_big_file import file_seperator
# import spacy
import csv
import re
## (1) remove sections; (2) remove stop words; (3) stemming; (4) remove words of frequency less than 1. 

'''
https://www.blopig.com/blog/2016/08/processing-large-files-using-python/
'''

spruner = Section_Pruner()

lmtzr = WordNetLemmatizer()
#print('Loading stop words...')

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
# the stop word list in the current folder were modified from mallet original 
# stoplist_string = open(os.path.join(__location__, 'curated_stopwords_ckd.txt')).read()
# stoplist = set(stoplist_string.split())
# names_string = open(os.path.join(__location__, 'lexicon', 'human_names_lexicon_02242020.csv')).read()
# names = set(names_string.split())
# topics = {'topic1': 'memory cognitive difficulty average recall function word evaluation score impairment drive attention executive mild find', 
#             'topic2': 'memory delirium disease mental mild cognitive agitation tremor dementia parkinson psych due difficulty admission impairment'}
# topics = {'topic1': 
# 'memory cognitive difficulty recall function word evaluation score impairment drive attention mild speech delirium mental agitation tremor dementia Parkinson psych alzheimer confusion question disorientation orientation mood sleep alter   moca montreal exam cognition mmse remember decline worse'
# }
topics = {'topic1': 'memory delirium dementia psych neuro mental Parkinson alzheimer confus mood cognit forget agitat neuro moca montreal mmse remember',
'topic2': 'difficult recall function word evaluat score drive attention mild impairment speech tremor question disorientation orientation sleep alter exam decline worse loss',
'topic3': 'duaghter wife husband son family'
}

# print('remove words from following files: ')
# print(' {} stopwords'.format(len(stoplist_string)))
# print('{} names'.format(len(names_string)))
# print('.......')
# includewords_string = open(os.path.join(__location__, 'included_words.txt')).read()
# includewordlist = set(includewords_string.split())

#print("Read in {} included words!".format(len(includewordlist)))

def main():
    ''' need to add the entry file for stop words as sometimes, you want to specify the stop word list...'''
    parser = argparse.ArgumentParser(description='Seperate large note file into multiple files')
    parser.add_argument("input_file", help = "the file path of the input file", type=str )
    parser.add_argument("output_file", help = "the file path of the output file", type=str)
    parser.add_argument("--worker", help="specify the number of cup to run the program", type=int, default = 1) ## there might be some issue
    parser.add_argument("--chunk_size", help="the chunk size assigned to each CUP", type=int, default= 10000)

    print('Loading objects...')

    args = parser.parse_args()
    print(args.input_file)
    print(args.output_file)
    print('number of worker: ', args.worker)
    
    start = time()
    doc_counts = sum(1 for line in open(args.input_file, 'rb'))
    manager = mp.Manager()
    q = manager.Queue()
    
    #init objects
    cores = args.worker
    pool = mp.Pool(cores + 2)

    # put listener to work first
    watcher = pool.apply_async(listener, (q, args.output_file))
    
    #create jobs
    jobs = []
    print (args.input_file)
    
    chunkStart = 0
    chunkSize = args.chunk_size
    
    
    ''' if the input file is too large, seperate them into multiple files to process'''

    if os.stat(args.input_file).st_size /1073741824.0 > 10.0:
        ## get the filename and remove the suffix (.txt)
        parts = os.path.split(args.input_file)
        output_file_path = parts[-1].split('.')[0]
        output_dir = os.path.join(parts[0], output_file_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        fs = file_seperator()
        files = fs.seperator(args.input_file, output_dir, 2000000)
        print('Seperate the large file into {} files'.format(len(files)))
        for file in files: 
            print ('loading file: ' + file)
            for chunkStart, chunkEnd in chunklines(file, chunkSize):
                
                #print('chunkStart: -----', chunkStart)
                #print('chunkEnd: ', chunkStart+chunkSize)
                #mp.Process(process_wrapper,(chunkStart,chunkSize))
                jobs.append( pool.apply_async(process_wrapper,(file, chunkStart,chunkEnd, q)) )
    else:
        print ('loading one file')
        for chunkStart, chunkEnd in chunklines(args.input_file, chunkSize):           
            #print('chunkStart: -----', chunkStart)
            #print('chunkEnd: ', chunkStart+chunkSize)
            #mp.Process(process_wrapper,(chunkStart,chunkSize))
            jobs.append( pool.apply_async(process_wrapper,(args.input_file, chunkStart,chunkEnd, q)) )

    #wait for all jobs to finish
    for job in jobs:
        job.get()

    #now we are done, kill the listener
    q.put('kill')
    pool.close()
    pool.join()

    elapsed = time()-start
    print("finished. Processed {} documents in {} seconds, on average {} docs/hour".format(doc_counts, elapsed, doc_counts*3600.0/elapsed))


def chunkify(input_file, size=100):
    fileEnd = os.path.getsize(input_file)
    with open(input_file, 'rb') as f:
        chunkEnd = f.tell()
        while True:
            chunkStart = chunkEnd
            f.seek(size, 1)
            f.readline()
            chunkEnd = f.tell()
            yield chunkStart, chunkEnd - chunkStart
            if chunkEnd > fileEnd:
                break


def chunklines(input_file, size):
    file_total_lines = sum(1 for line in open(input_file, 'rb'))
    print('Total lines of the file is: ', file_total_lines)
    chunkEnd = 0
    chunkStart = 0
    chunks = []
    while True: 
        if chunkStart + size <= file_total_lines:
            chunkEnd = chunkStart + size -1 
            chunks.append((chunkStart, chunkEnd))
            print('add one chunk: {} - {}'.format(chunkStart, chunkEnd))
            chunkStart = chunkEnd + 1
        elif chunkStart + size > file_total_lines and chunkStart < file_total_lines: 
            chunkEnd = file_total_lines - 1
            chunks.append((chunkStart, chunkEnd))
            print('add one chunk: {} - {}'.format(chunkStart, chunkEnd))
            break
        else:
            break

    # with open(input_file,'rb') as f:
    #     for i, line in enumerate(f):
    #         chunkStart = chunkEnd
    #         if i%size == 0 and i!=0:
    #             chunkEnd = i
    #             chunks.append((chunkStart, chunkEnd-chunkStart))                

    #         #yield chunkStart, chunkEnd - chunkStart                
    #         if i >= file_total_lines:
    #             chunks.append((chunkStart, chunkEnd-chunkStart))
    #             break
    return chunks


def process_wrapper(input_file, chunkStart, chunkEnd, q):
    with open(input_file, encoding='utf-8', errors='replace') as f:
        print('start processing {} - {} lines'.format(chunkStart, chunkEnd))

        lines = []
        for i, line in enumerate(f):
            if i == 0:
                pass
            if i>=chunkStart and i <= chunkEnd:
                noteid, sectionized_text, dict_topic_wordlist, dict_topic_uniquewordlist  = process(line)
                if noteid != '' :
                    section_data = []
                    for label, text in sectionized_text.items():
                        keep_or_remove = 'keep' if label in sections_to_keep else 'remove'
                        individual_sections = text.split(f'[{label}]: ')  # split initial newline, separate sections
                        for section in individual_sections[1:]:
                            keyword_matches_1, keyword_matches_2, keyword_matches_3 = keyword_match(section.lower())
                            keyword_match_str_1 = ' '.join(keyword_matches_1)
                            unique_keywords_1 = set(keyword_matches_1)
                            unique_keyword_str_1 = ' '.join(unique_keywords_1)
                            keyword_match_str_2 = ' '.join(keyword_matches_2)
                            unique_keywords_2 = set(keyword_matches_2)
                            unique_keyword_str_2 = ' '.join(unique_keywords_2)
                            keyword_match_str_3 = ' '.join(keyword_matches_3)
                            unique_keywords_3 = set(keyword_matches_3)
                            unique_keyword_str_3 = ' '.join(unique_keywords_3)
                            section_data.append([noteid, keep_or_remove, label, section[:-1], keyword_match_str_1,
                                                 unique_keyword_str_1, len(unique_keywords_1), keyword_match_str_2,
                                                 unique_keyword_str_2, len(unique_keywords_2), keyword_match_str_3,
                                                 unique_keyword_str_3, len(unique_keywords_3)])
                    lines.append(section_data)
                    #print( 'noteid:{}, word_count:{}'.format(noteid, count))
                        # lines.append([noteid, keep_or_remove, ])
                # else:
                #     print('drop off the line with noteid:'  + noteid)
                #     for topic in sorted(topics.keys()):
                #         wordlist = dict_topic_wordlist[topic]
                #         text = text + topic + '\t' + wordlist + '\t' + ' '.join(dict_topic_uniquewordlist[topic]) + '\t' + str(len(dict_topic_uniquewordlist[topic])) + '\t'
                #
                # #print( 'noteid:{}, word_count:{}'.format(noteid, count))
                #     lines.append(noteid + '\t' + text.strip() )
                # # else:
                # #     print('drop off the line with noteid:'  + noteid)
        q.put(lines)
        return lines


def keyword_match(text):
    tokenized_text = tokenize(text)
    token_matches_1 = []
    for keyword in topics['topic1'].split():
        token_matches_1.extend([token for token in tokenized_text if token.startswith(keyword)])
    token_matches_2 = []
    for keyword in topics['topic2'].split():
        token_matches_2.extend([token for token in tokenized_text if token.startswith(keyword)])
    token_matches_3 = []
    for keyword in topics['topic3'].split():
        token_matches_3.extend([token for token in tokenized_text if token.startswith(keyword)])

    return token_matches_1, token_matches_2, token_matches_3


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

            
def process(line):
    
    doc = line.split('\t')
    w_count = 0
    noteid = ''
    text = ''
    if len(doc) >= 2:
        #print len(doc)
        noteid = doc[0]
        text = ' '.join(doc[1:])
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

        # normalize the text 
        words = split_line(text)
        
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
        sectionized_text, section_counts, section_list = spruner.get_scd_sections(text)

    return noteid, sectionized_text, dict_topic_wordlist, dict_topic_uniquewordlist


def split_line( text ):
    '''remove non-ascii'''
    words = tokenize(text) 
    #print(words)
    return words


if __name__ == "__main__":

    main()
