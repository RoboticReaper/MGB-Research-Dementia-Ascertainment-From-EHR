# -*- coding: utf-8 -*-
# @Author: Jie Yang
# @Date:   2018-05-05 14:05:49
# @Last Modified by:   Jie Yang,     Contact: jieynlp@gmail.com
# @Last Modified time: 2019-05-01 10:32:29

from __future__ import division
import numpy as np
import csv
import sys
import random
import time
import pickle
import xgboost as xgb

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from file_io import *
from utils import *
from metric_visual import *

seed_num = 42
random.seed(seed_num)
np.random.seed(seed_num)

def extract_original_annotation(original, annotate, output_file):
    ## should run under Python 3
    result = load_text_classification_data_csv(annotate, [5],False, True, True)
    des = result[0]
    des_num = len(des)
    print(des_num)
    original_result = xlsx_extract_column(original, ["B","C","D","E"], True)
    # a, text = load_text_classification_data_xlsx(original, "D", "E")
    clean_original_text = clinic_text_list_processing(original_result[-1])
    original_num = len(original_result[0])
    match = 0
    with open(output_file, 'w') as cfile:
        csvwriter = csv.writer(cfile) 
        csvwriter.writerow(["ADE","ADR", "HSR","Reasoning", "Notes", "Description"])
        for idx in range(des_num):
            des_string = result[0][idx]
            if des_string in clean_original_text:
                match+= 1
                the_id = clean_original_text.index(des_string)
                csvwriter.writerow([original_result[0][the_id], original_result[1][the_id], original_result[2][the_id],"", "", des_string])

    print(match)



def load_allergy_data(input_file, target="HSR", shuffle_instance =True ):
    print("Load allergy data from file: %s, target label: %s, shuffle: %s"%(input_file, target, shuffle_instance))
    Description, ADE, ADR, HSR, MRN = load_text_classification_data_csv(input_file, [4, 1, 2,3,0])
    print ("Positive num: ADE:%s; ADR:%s; HSR%s"%(sum(ADE), sum(ADR), sum(HSR)))
    if shuffle_instance:
        combined = list(zip(Description, ADE, ADR, HSR))
        random.shuffle(combined)
        Description[:], ADE[:], ADR[:], HSR[:] = zip(*combined)
        print("Instance shuffled.")
    input_x = Description
    if target.upper() == "ADE":
        output_y = ADE 
    elif target.upper() == "ADR":
        output_y = ADR
    elif target.upper() == "HSR":
        output_y = HSR
    else:
        print("Invaild target choice, must by one of ADE/ADR/HSR, given %s"%(target))
        exit(1)
    return input_x, output_y, MRN


def load_txt_allergy_data(input_file,  shuffle_instance =True):
    print("Load allergy data from file: %s,  shuffle: %s"%(input_file, shuffle_instance))
    Description, HSR = load_txt_data(input_file)
    
    print ("Positive num: HSR%s"%(sum(HSR)))
    if shuffle_instance:
        combined = list(zip(Description, HSR))
        random.shuffle(combined)
        Description[:], HSR[:] = zip(*combined)
        print("Instance shuffled.")
    input_x = Description
    output_y = HSR
    MRN = list(range(len(output_y)))
    return input_x, output_y, MRN




def random_select_to_annotate(input_file, output_file):
    random.seed(42)
    input_x, output_y, MRN = load_allergy_data(input_file, "HSR")
    pos_x = []
    neg_x = []
    for x, y in zip(input_x, output_y):
        if y == 1:
            pos_x.append(x)
        else:
            neg_x.append(x)
    new_list = pos_x[:50] + neg_x[:100]
    new_label = [1]*50 + [0]*100
    combined = list(zip(new_list,new_label))
    random.shuffle(combined)
    new_list,new_label= zip(*combined)
    fout = open(output_file, 'wb')
    csvwriter = csv.writer(fout)
    csvwriter.writerow(["ADE", "ADR", "HSR","Description"])
    for each in new_list:
        csvwriter.writerow(["","","",each])



def random_select_raw_to_annotate(input_file,output_file):
    random.seed(42)
    fout = open(output_file, 'wb')
    csvwriter = csv.writer(fout)
    csvwriter.writerow(["ADE", "ADR", "HSR","Description"])
    sentence_dict = {}
    with open(input_file,'rb') as csvfile:
        thereader = csv.reader(csvfile)
        first_line = True 
        for row in thereader:
            if first_line:
                first_line = False
                continue 
            if len(row) !=9:
                print(row)
                exit(0)
            sent = row[-1]
            if sent not in sentence_dict:
                sentence_dict[sent] = 1 
        sent_list = sentence_dict.keys()
        random.shuffle(sent_list)
        # print(sent_list[:10])
        # exit(0)
        for idx in range(400):
            csvwriter.writerow(["","","",sent_list[idx]])


def annotate_agreement(original_file, validation_file):
    hsr_x, hsr_y, MRN = load_allergy_data(original_file, "HSR")
    adr_x, adr_y, MRN = load_allergy_data(original_file, "ADR")
    ade_x, ade_y, MRN = load_allergy_data(original_file, "ADE")
    origin_dict = {}
    for x, y in zip(ade_x, ade_y):
        origin_dict[x] = [y]
    for x, y in zip(adr_x, adr_y):
        origin_dict[x].append(y)
    for x, y in zip(hsr_x, hsr_y):
        origin_dict[x].append(y)
    print("Original num:",len(origin_dict))
    validation_dict = {}
    with open(validation_file,'rb') as csvfile:
        thereader = csv.reader(csvfile)
        first_line = True 
        for row in thereader:
            if first_line:
                first_line = False
                continue 
            sent = row[-1]
            ade = int(row[0])
            adr = int(row[1])
            hsr = int(row[2])
            validation_dict[sent] = [ade, adr, hsr]
    print("Validation num:", len(validation_dict))
    pred_hsr = []
    gold_hsr = []
    pred_adr = []
    gold_adr = []
    pred_ade = []
    gold_ade = []
    for key, value in validation_dict.iteritems():
        pred_ade.append(value[0])
        pred_adr.append(value[1])
        pred_hsr.append(value[2])
        gold_value = origin_dict[key]
        gold_ade.append(gold_value[0])
        gold_adr.append(gold_value[1])
        gold_hsr.append(gold_value[2])
    pred_hsr = np.asarray(pred_hsr)
    gold_hsr = np.asarray(gold_hsr)
    pred_adr = np.asarray(pred_adr)
    gold_adr = np.asarray(gold_adr)
    pred_ade = np.asarray(pred_ade)
    gold_ade = np.asarray(gold_ade)
    p,r,f,acc, kappa = calculate_p_r_f_acc_kappa(gold_ade, pred_ade)
    print("ADE: P=%s, R=%s, F=%s, acc=%s, kappa=%s"%(p,r,f,acc, kappa))
    p,r,f,acc, kappa = calculate_p_r_f_acc_kappa(gold_adr, pred_adr)
    print("ADR: P=%s, R=%s, F=%s, acc=%s, kappa=%s"%(p,r,f,acc, kappa))
    p,r,f,acc, kappa = calculate_p_r_f_acc_kappa(gold_hsr, pred_hsr)
    print("HSR: P=%s, R=%s, F=%s, acc=%s, kappa=%s"%(p,r,f,acc, kappa))

def annotate_single_agreement(a_list, b_list):
    a_list = np.asarray(a_list)
    b_list = np.asarray(b_list)
    p,r,f,acc, kappa = calculate_p_r_f_acc_kappa(a_list, b_list)
    print("P=%s, R=%s, F=%s, acc=%s, kappa=%s"%(p,r,f,acc, kappa))
    return p,r,f,acc, kappa



def load_csv(input_file, idx, idy):
    print("Start loading csv file from %s."%input_file)
    x_list = []
    y_list = []
    with open(input_file,'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader, None)
        for row in csv_reader:
            the_x = row[idx]
            the_y = row[idy]
            if len(the_x) >=1:
                x_list.append(int(the_x))
                y_list.append(the_y)
    print(len(x_list))
    print(x_list)
    return x_list, y_list

def calculate_agreement_csv(file_a, file_b):
    # ade_a, _ = load_csv(file_a, 0, 0)
    adr_a, _ = load_csv(file_a, 1, 0)
    hsr_a, _ = load_csv(file_a, 2, 0)
    # ade_b, _ = load_csv(file_b, 0, 0)
    adr_b, _ = load_csv(file_b, 1, 0)
    hsr_b, _ = load_csv(file_b, 2, 0)
    # print("ADE:")
    # annotate_single_agreement(ade_a, ade_b)
    print("ADR:")
    annotate_single_agreement(adr_a, adr_b)
    print("HSR:")
    annotate_single_agreement(hsr_a, hsr_b)

def calculate_agreement_xlsx_pair(file_a, file_b):
    print("A:", file_a)
    print("B:", file_b)
    label_a1, label_a2 = xlsx_extract_column(file_a,['D', "E"], True)
    label_b1, label_b2 = xlsx_extract_column(file_b,['D', "E"], True)
    print(len(label_a1), len(label_a2))
    print(len(label_b1), len(label_b2))
    # print(label_b1)
    p,r,f,acc, kappa = annotate_single_agreement(label_a1, label_b1)
    print("Allergy related:", kappa)
    p,r,f,acc, kappa = annotate_single_agreement(label_a2, label_b2)
    print("EHR related:", kappa)


def calculate_agreement_xlsx(all_file):
    Kim, Neelam, Chris, Paige = xlsx_extract_column(all_file,['B', "D", 'G', "J"], True)
    print(Kim)
    print(Paige)
    print("Kim, Neelam")
    annotate_single_agreement(Kim, Neelam)
    print("Kim, Chris")
    annotate_single_agreement(Kim, Chris)
    print("Neelam, Chris")
    annotate_single_agreement(Chris, Neelam)
    print("Paige, Kim")
    annotate_single_agreement(Paige, Kim)
    print("Paige, Neelam")
    annotate_single_agreement(Paige, Neelam)
    print("Paige, Chris")
    annotate_single_agreement(Paige, Chris)

if __name__ == '__main__':
 
    # input_file = r"../Data/9K_reports_20181228.xlsx"
    # original = "annotate_agreement/annotation_evaluation_150cases_original.csv"
    CM = "annotate_agreement/Cases_ehr_allergy_related_keyword_search_random_100_third_round_CM.xlsx"
    NP = "annotate_agreement/Cases_ehr_allergy_related_keyword_search_random_100_third_round_NP_04302019.xlsx"
    PW = "annotate_agreement/Cases_ehr_allergy_related_keyword_search_random_100_third_round_PW.xlsx"
    Kim = "annotate_agreement/Cases_ehr_allergy_related_keyword_search_random_100_third_round_KB.xlsx"
    # all_file = "annotate_agreement/Cases_and_labels_2.xlsx"
    # calculate_agreement_xlsx(all_file)
    # exit(0)
    # load_csv(validation_file, 0, 5)
    calculate_agreement_xlsx_pair(NP, PW)
    exit(0)
    extract_original_annotation(input_file, NP, "annotate_agreement/out.csv")
    # exit(0)
    annotate_agreement(input_file, validation_file)







