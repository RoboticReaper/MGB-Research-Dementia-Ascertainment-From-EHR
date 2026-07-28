# -*- coding: utf-8 -*-
# @Author: Jie Yang
# @Date:   2018-05-05 14:05:49
# @Last Modified by:   Jie Yang,     Contact: jieynlp@gmail.com
# @Last Modified time: 2019-05-01 12:19:36

from __future__ import division
import numpy as np
import csv
import random
from file_io import *


seed_num = 42
random.seed(seed_num)
np.random.seed(seed_num)

def select_all_cases(mgh_sorted_withoutkeyword_xlsx, bwh_sorted_all_xlsx, output_file):
    mgh_ids, mgh_prob, mgh_descriptions = xlsx_extract_column(mgh_sorted_withoutkeyword_xlsx, ["A","C", "D"], True)
    bwh_ids, bwh_prob, bwh_descriptions = xlsx_extract_column(bwh_sorted_all_xlsx, ["A","C", "D"], True)
    index_list = list(range(0, 1000, 5))
    print(index_list)
    print(len(index_list))
    to_evaluate_id = 0
    with open(output_file,'w') as cfile:
        csv_writer = csv.writer(cfile)
        csv_writer.writerow(["EvaluateID","Tag", "POSI", "Case ID", "HSR", "Description", "Prob"])
        for idx in index_list:
            to_evaluate_id += 1
            csv_writer.writerow([to_evaluate_id, "M", idx, mgh_ids[idx], "", mgh_descriptions[idx], mgh_prob[idx]])
        for idx in index_list:
            to_evaluate_id += 1
            csv_writer.writerow([to_evaluate_id, "B", idx, bwh_ids[idx], "", bwh_descriptions[idx], bwh_prob[idx]])



def select_single_cases(sorted_xlsx, output_file):
    the_ids, the_prob, the_descriptions = xlsx_extract_column(sorted_xlsx, ["A","C", "D"], True)
    index_list = list(range(0, 5000, 5))
    with open(output_file,'w') as cfile:
        csv_writer = csv.writer(cfile)
        csv_writer.writerow(["POSI", "Case ID", "HSR", "Description", "Prob"])
        for idx in index_list:
            csv_writer.writerow([idx, the_ids[idx], "", the_descriptions[idx], the_prob[idx]])





if __name__ == '__main__':
    mgh = "decode/HSR.MGH-withoutKeywords.att.trainall.decode.xlsx"
    bwh = "decode/HSR.BWH.att.trainall.decode.xlsx"
    # output_file = "decode/HSR.mgh_bwh.to.evaluate.sample.csv"
    # select_all_cases(mgh, bwh, output_file)
    output_file = "decode/HSR.bwh.to.evaluate.sample.top5000.csv"
    select_single_cases(bwh, output_file)

    






