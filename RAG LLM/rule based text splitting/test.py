import os, inspect, sys, argparse
from collections import defaultdict
import operator
import csv
#from section_pruner import Section_Pruner
from time import time
import json
from pprint import pprint
import time
import traceback
import pandas as pd
import csv
from section_pruner import Section_Pruner, sections_to_keep, sections_maybe_keep
#from tk import tk2 as tk
import multiprocessing as mp
#from corpus import MyCorpus
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from tokenizer import tokenize
# from re import sub
import re

## Comments: python test.py --input_file ##\\Voice\scd\study_note_sectionization\data\mci_2019_cohort_edw_notes_04202020.txt ##--output_dir \\Voice\scd\study_note_sectionization\annotation\corpus --file_size 5000

section_dist_dict = defaultdict(int)
# sections_to_keep_map = {label: idx for idx, label in enumerate(sections_to_keep)}
parser = argparse.ArgumentParser(description='Seperate large note file into multiple files')
parser.add_argument("--input_file", help="the file path of the input file", type=str)
parser.add_argument("--output_file", help="the file path of the output file", type=str, default=None)
parser.add_argument("--output_dir", help="the path of the output directory - prints indiviual note sections in separate subfolders and .txt files",
                    type=str, default=None)
parser.add_argument("--file_size", help="the chunk size assigned to each CUP", type=int, default=100000000)
parser.add_argument("--csv_output", help="indicates output is csv file", action="store_true")
parser.add_argument("--track_section_distribution", help="write section counts at the end of output",
                    action="store_true")
parser.add_argument("--individual_counts", help="write section counts for individual notes", action="store_true")
parser.add_argument("--section_lengths", help="produce file with sections and their word counts; requires --csv_output flag",
                    action="store_true")
parser.add_argument("--keyword_only", help="include only sections containing at least one keyword", action="store_true")
args = parser.parse_args()

keywords_1 = ['difficult',        # cognitive decline
              'recall',
              'remember',
              'function',
              'word',
              'evaluate',
              'score',
              'drive',
              'attention',
              'mild',
              'impairment',
              'speech',
              'tremor',
              'question',
              'disorientation',
              'orientation',
              'sleep',
              'alter',
              'exam',
              'decline',
              'worse',
              'loss']

keywords_2 = ['memory',           # mental disorder
              'delirium',
              'dementia',
              'psych',
              'mental',
              'parkinson',
              'alzheimer',
              'confuse',
              'mood',
              'cognit',
              'forget',
              'agitate',
              'neuro',
              'moca',
              'montreal',
              'mmse']

keywords_3 = ['daughter',         # family member
              'son',
              'wife',
              'husband',
              'family']


def main():
    ''' need to add the entry file for stop words as sometimes, you want to specify the stop word list...'''

    print('Loading objects...')

    # print(args.input_file)
    # print(args.output_file)
    # print(args.file_size)

    spruner = Section_Pruner()
    if args.output_file:
        if args.csv_output:
            if args.output_file.endswith('.csv'):
                w = open(args.output_file, 'w', newline='')
            else:
                out_file = args.output_file + '.csv'
                w = open(out_file, 'w', newline='')
            note_writer = csv.writer(w)
            # note_writer.writerow(['note_id', 'sectionized_text', 'token_count'])        # column headers
        else:
            w = open(args.output_file, 'a+', encoding='utf-8')
            # w = open(os.path.join(args.output_file, 'dummy.txt'), 'a+', encoding='utf-8')

    with open(args.input_file, encoding='utf-8', errors='replace') as f:
        for i, line in enumerate(f):
            if i < args.file_size:
                doc = line.split('\t')
                if len(doc) >= 2:
                    noteid = doc[0]
                    if i % 1000 == 0:
                        print(f'processed through note {i} (ID {noteid})...')
                    # print(f'processing note {i + 1} (ID {noteid})...')
                    # text = ' '.join(doc[2:])
                    text = ' '.join(doc[1:])
                    text = re.sub(r'[^\x01-\x7f]', r'', text)  # remove non-ascii-string
                    if text.strip() != '':
                        sectionized_text, section_counts, section_list = spruner.get_scd_sections(text)

                        df = pd.DataFrame([
                            {'noteid': noteid, 'section': key, 'text': sectionized_text[key].strip()}
                            for key in sectionized_text if sectionized_text[key] != ''
                        ])

                        df.to_csv('output.csv', mode='a', index=False, header=False, quoting=csv.QUOTE_ALL)
                        # with open('output.txt', 'w') as f:
                        #     f.write(f'{noteid},{sectionized_text},')

                        if args.output_dir:
                            output_sectionized_data(sectionized_text, args.output_dir, noteid, args.input_file)
                        elif args.section_lengths:
                            try:
                                output_section_lengths(sectionized_text, note_writer, noteid)
                            except Exception as e:
                                print('--csv_output flag was not specified')
                                print(type(e), str(e))
                                break
                        else:
                            text2, section_counts, section_list = spruner.check_sectionizer(text)
                            towrite = noteid + '\t' + text + '\t' + text2
                            if args.csv_output:
                                note_writer.writerow([noteid, text2])
                                for section, original_text in section_list:
                                    note_writer.writerow([noteid, original_text, section])
                                note_writer.writerow([text[:-1], text2[2:-4]])
                            else:
                                try:
                                    w.write("%s\n" % towrite)
                                    w.flush()
                                except Exception as e:
                                    print('no output_file provided')
                                    print(type(e), str(e))
                                    break

                            if args.track_section_distribution:
                                update_section_distribution(section_counts, w, noteid)
            else:
                break

    if args.track_section_distribution:
        total_count = sum(section_dist_dict.values())
        sorted_dict = dict(sorted(section_dist_dict.items(), key=operator.itemgetter(1), reverse=True))
        if args.csv_output:
            note_writer.writerow([])
            for section, s_count in sorted_dict.items():
                note_writer.writerow([section, s_count])
            note_writer.writerow(['Grand Total', total_count])
        else:
            for section, s_count in sorted_dict.items():
                w.write(f'{section}: {s_count}\n')
            w.write(f'Grand Total: {total_count}')

    try:
        w.close()
    except NameError:
        pass


def update_section_distribution(section_counts, output_file, note_id):
    """
    updates section counts stored in section_dist_dict (for all notes in dataset).
    if section counts for individual notes are desired (i.e. --individual_counts flag is present),
    writes counts for the current note to file.
    """
    for section, count in section_counts.items():
        section_dist_dict[section] += count
        if args.individual_counts:
            if args.csv_output:
                note_writer = csv.writer(output_file)
                note_writer.writerow([note_id, None, None, section, count])
            else:
                output_file.write(f'{section}: {count}\n')


def output_section_lengths(sectionized_text, csv_writer, note_id):
    """
    writes sections to .csv file along with word (token) count.
    each line in the file consists of one section for an individual note.
    """
    for label, label_section_text in sectionized_text.items():
        if label_section_text:
            separate_sections = label_section_text.split('\n')[:-1]
            for separate_section_text in separate_sections:
                start_idx = separate_section_text.find(']: ') + 3       # used to exclude section label from word count
                csv_writer.writerow([note_id,
                                     separate_section_text,
                                     len(tokenize(separate_section_text[start_idx:]))])


def output_sectionized_data(sectionized_text, output_dir, note_id, input_file_name, individual_files=False):
    """
    writes section text samples to file(s) in output_dir.
    if individual_files is True, grouped in separate files by label.
    if False, all sections will go to a single file (args.output_file).
    """
    if individual_files:
        input_file = input_file_name.split('\\')[-1]        # remove input file directory path
        for label, text in sectionized_text.items():
            if text:
                if args.keyword_only:
                    keyword_matches_1, keyword_matches_2, keyword_matches_3 = keyword_match(text.lower())
                    if not keyword_matches_1 + keyword_matches_2 + keyword_matches_3:
                        # print(f'{note_id} - {label} - no keywords')
                        continue
                if args.csv_output:
                    if args.output_file:
                        if label in sections_to_keep or true:
                            file_path = os.path.join(output_dir, f'keep_{args.output_file}')
                        else:
                            file_path = os.path.join(output_dir, f'remove_{args.output_file}')
                    else:
                        input_file = input_file.split('.')[0] + '.csv'       # replace .txt extension with .csv
                        if label in sections_to_keep or true:
                            file_path = os.path.join(output_dir, 'keep', f'{label.replace("/", "_")}_{input_file}')
                        elif label in sections_maybe_keep:
                            file_path = os.path.join(output_dir, 'question', f'{label.replace("/", "_")}_{input_file}')
                        else:
                            file_path = os.path.join(output_dir, 'remove', f'{label.replace("/", "_")}_{input_file}')
                    with open(file_path, 'a', newline='') as f:
                        individual_sections = text.split(f'[{label}]: ')     # split initial newline, separate sections
                        print(individual_sections)
                        section_writer = csv.writer(f)
                        for section in individual_sections[1:]:
                            section_writer.writerow([note_id, label, section])
                else:
                    if label in sections_to_keep or true:
                        file_path = os.path.join(output_dir, 'keep', f'{label.replace("/", "_")}_{input_file}')
                    elif label in sections_maybe_keep:
                        file_path = os.path.join(output_dir, 'question', f'{label.replace("/", "_")}_{input_file}')
                    else:
                        file_path = os.path.join(output_dir, 'remove', f'{label.replace("/", "_")}_{input_file}')
                    with open(file_path, 'a+', encoding='utf-8') as f:
                        f.write(f'{note_id}\t{text}')
    else:
        for label, text in sectionized_text.items():
            if text:
                keep_or_remove = 'keep' if label in sections_to_keep or true else 'remove'
                individual_sections = text.split(f'[{label}]: ')  # split initial newline, separate sections
                with open(args.output_file, 'a', newline='') as f:
                    section_writer = csv.writer(f)
                    for section in individual_sections[1:]:
                        keyword_matches_1, keyword_matches_2, keyword_matches_3 = keyword_match(section.lower())
                        section_writer.writerow([note_id, keep_or_remove, label, section[:-1],
                                                 keyword_matches_1, keyword_matches_2, keyword_matches_3])


def keyword_match(text):
    tokenized_text = tokenize(text)
    token_matches_1 = []
    for keyword in keywords_1:
        token_matches_1.extend([token for token in tokenized_text if keyword in token])
    token_matches_2 = []
    for keyword in keywords_2:
        token_matches_2.extend([token for token in tokenized_text if keyword in token])
    token_matches_3 = []
    for keyword in keywords_3:
        token_matches_3.extend([token for token in tokenized_text if keyword in token])
    # for token in tokenized_text:
    #     token_matches_1.extend([word for word in keywords_1 if word in token])
    #     token_matches_2.extend([word for word in keywords_2 if word in token])
    #     token_matches_3.extend([word for word in keywords_3 if word in token])

    return token_matches_1, token_matches_2, token_matches_3
    #     if token_matches:
    #         return True
    # return False


if __name__ == "__main__":
    start = time.perf_counter()
    main()
    end = time.perf_counter()
    print(f"The whole splitting process took {end - start} seconds.")
