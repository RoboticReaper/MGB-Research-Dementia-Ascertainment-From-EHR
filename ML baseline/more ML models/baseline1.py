# -*- coding: utf-8 -*-
# @Author: Jie Yang
# @Date:   2018-05-05 14:05:49
# @Last Modified by:   Jie Yang,     Contact: jieynlp@gmail.com
# @Last Modified time: 2019-01-24 12:14:35

from __future__ import division
import csv
import os
import sys
import random
import time
import pickle
# import cPickle as pickle

import numpy as np
import xgboost as xgb
import sklearn
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
import pandas as pd
import matplotlib.pyplot as plt

from utils import filter_duplicate, build_alphabet, data_split, unigram2bigram, plot_multiple_classification_results_pp
from metric_visual import calculate_p_r_f_acc_kappa, calculate_roc_list, calculate_precision_recall_list, plot_multi_curve
from file_io import load_pair_txt_file
import stat_util

seed_num = 42
random.seed(seed_num)
np.random.seed(seed_num)


def load_csv_data(input_file):
    with open(input_file, encoding='utf-8', newline='') as f:
        csv_reader = csv.reader(f)
        csv_data = [(row[4], int(row[1]), int(row[2]), int(row[3]), row[0]) for row in csv_reader]
    Description, ADE, ADR, HSR, MRN = zip(*csv_data)

    return Description, ADE, ADR, HSR, MRN


def load_allergy_data(input_file, target="HSR", shuffle_instance=True):
    print("Load allergy data from file: %s, target label: %s, shuffle: %s"%(input_file, target, shuffle_instance))
    Description, ADE, ADR, HSR, MRN = load_csv_data(input_file)
    Description, ADE, ADR, HSR, MRN = filter_duplicate([Description, ADE, ADR, HSR, MRN])
    print("Positive num: ADE:%s; ADR:%s; HSR%s"%(sum(ADE), sum(ADR), sum(HSR)))
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


def load_scd_data(input_file, shuffle_instance=True):
    print("Loading merged data from: %s" % (input_file))
    # Using pandas for easier string label handling
    df = pd.read_csv(input_file)

    # Map "yes"/"no" to 1/0
    label_map = {"yes": 1, "no": 0}

    # Assuming columns: 'empi', 'label', 'notes'
    # Adjust names if your CSV headers differ
    descriptions = df['notes'].astype(str).tolist()
    labels = df['label'].map(label_map).tolist()
    empi_ids = df['empi'].astype(str).tolist()

    # Check for NaNs in labels
    if df['label'].isnull().any():
        print("Warning: Found null labels. Dropping those rows.")
        valid_idx = df['label'].notnull()
        descriptions = df.loc[valid_idx, 'notes'].astype(str).tolist()
        labels = df.loc[valid_idx, 'label'].map(label_map).tolist()
        empi_ids = df.loc[valid_idx, 'empi'].astype(str).tolist()

    print("Total records: %d | Positive (Yes): %d" % (len(labels), sum(labels)))

    if shuffle_instance:
        combined = list(zip(descriptions, labels, empi_ids))
        random.shuffle(combined)
        descriptions, labels, empi_ids = zip(*combined)

    # Note: note2annotator is kept for compatibility with your existing model call,
    # but mapped to EMPI if no specific annotator exists.
    note2annotator = {eid: "N/A" for eid in empi_ids}

    return list(descriptions), list(labels), list(empi_ids), note2annotator


# def load_txt_allergy_data(input_file, shuffle_instance=True):
#     print("Load allergy data from file: %s,  shuffle: %s"%(input_file, shuffle_instance))
#     Description, HSR = load_txt_data(input_file)
#
#     print ("Positive num: HSR%s"%(sum(HSR)))
#     if shuffle_instance:
#         combined = list(zip(Description, HSR))
#         random.shuffle(combined)
#         Description[:], HSR[:] = zip(*combined)
#         print("Instance shuffled.")
#     input_x = Description
#     output_y = HSR
#     MRN = list(range(len(output_y)))
#     return input_x, output_y, MRN


def run_nfold_baseline(input_file, output_dir, baseline_type="svm", partition_num=10, write_decoded=False,
                       headers=False, feature_importances=False):
    # target = "HSR"
    input_x, output_y, note_id, note2annotator = load_scd_data(input_file, shuffle_instance=True)
    # input_x, output_y, note_id, note2annotator = load_scd_data(input_file, 6, 12, shuffle_instance=False)
    # input_x, output_y, MRN = load_allergy_data(input_file, target)
    # input_x, output_y, MRN = load_txt_allergy_data(input_file, True)
    # label2id, id2label = build_alphabet(output_y, True)
    # positive_label = 1
    # positive_id = label2id[positive_label]
    label2id = {0: 0, 1: 1}
    positive_id = 1
    # if write_decoded:
    #     fout = open("decode.csv", 'wb')
    #     csvwriter = csv.writer(fout)
    #     csvwriter.writerow(["MRN",target, "Pred_"+target, "Description"])
    # else:
    #     csvwriter = None
    csvwriter = None
    dev_noteid_list = []
    dev_decode_list = []
    dev_gold_list = []
    test_decode_list = []
    test_gold_list = []
    calibration_sig_list = []
    calibration_iso_list = []
    for idx in range(partition_num):
        start_time = time.time()
        print("Proceesing: ", idx, "/", partition_num, "..........................................")
        ## split into train/dev/test
        X = data_split(input_x, partition_num, idx)
        Y = data_split(output_y, partition_num, idx)
        the_note_id = data_split(note_id, partition_num, idx)
        dev_noteid_list.extend(the_note_id[1])
        dev_preds, dev_golds, test_preds, test_golds, \
            feature_names, feature_importance_data, tree_representation,\
            calibration_data_sig, calibration_data_iso, clf = model(X, Y, the_note_id, label2id, baseline_type,
                                                               csvwriter, feature_importances)
        dev_decode_list.append(dev_preds)
        dev_gold_list.append(dev_golds)
        test_decode_list.append(test_preds)
        test_gold_list.append(test_golds)
        if calibration_data_sig:
            calibration_sig_list.extend(calibration_data_sig)
        if calibration_data_iso:
            calibration_iso_list.extend(calibration_data_iso)

        # # write tree representation (randforest, xgboost)
        # if baseline_type == 'randforest':
        #     with open(os.path.join(output_dir, f"{baseline_type.upper()}_{idx}_tree.txt"), 'w', encoding='utf-8') as f:
        #         f.write(tree_representation)
        # elif baseline_type == 'xgboost':
        #     pass

        # write features and feature importances
        with open(os.path.join(output_dir, f"{baseline_type.upper()}_{idx}_features.csv"), 'w', newline='') as f:
            feature_writer = csv.writer(f)
            feature_writer.writerow([baseline_type])
            for jdx, feat in enumerate(feature_names):
                feature_writer.writerow([jdx, feat])

        if feature_importances and feature_importance_data is not None:
            if baseline_type == 'xgboost':
                with open(os.path.join(output_dir, f"{baseline_type.upper()}_{idx}_fi.csv"), 'w', newline='') as f:
                    feature_writer = csv.writer(f)
                    feature_writer.writerow([baseline_type])
                    for feat, feat_im in feature_importance_data.items():
                        feature_writer.writerow([feat, feat_im])
            else:
                with open(os.path.join(output_dir, f"{baseline_type.upper()}_{idx}_fi.csv"), 'w', newline='') as f:
                    feature_writer = csv.writer(f)
                    feature_writer.writerow([baseline_type])
                    for feat_im in feature_importance_data:
                        feature_writer.writerow([feat_im])

        cost_time = time.time() - start_time
        print("     Time cost: %.2f s" % cost_time)
    dev_all_decode = np.concatenate(dev_decode_list, axis=0)
    dev_all_gold = np.concatenate(dev_gold_list, axis=0)
    # test_all_decode = np.concatenate(test_decode_list, axis=0)
    # test_all_gold = np.concatenate(test_gold_list, axis=0)
    fpr, tpr, roc_thresholds, auc = calculate_roc_list(dev_all_gold, dev_all_decode, positive_id)
    # precision, recall, pr_thresholds = calculate_precision_recall_list(test_all_gold, test_all_decode, positive_id)
    precision, recall, pr_thresholds, auprc = calculate_precision_recall_list(dev_all_gold, dev_all_decode, positive_id)
    # fpr, tpr, roc_thresholds, auc, precision, recall, pr_thresholds = calculate_roc_precision_recall(test_all_gold, test_all_decode, positive_id)
    with open(os.path.join(output_dir, baseline_type.upper()+".original.pkl"), 'wb') as f:
        # pickle.dump([fpr, tpr, roc_thresholds, auc, precision, recall, pr_thresholds, baseline_type.upper()], f)
        pickle.dump([fpr, tpr, roc_thresholds, auc, precision, recall, pr_thresholds, auprc, baseline_type.upper()], f)
    # plot_classification_results(fpr, tpr, auc, precision, recall, baseline_type.upper())

    dev_pred_labels = [0 if prob < 0.5 else 1 for _, prob in dev_all_decode]
    conf_matrix = confusion_matrix(dev_all_gold, dev_pred_labels)
    print(f'\nconfusion matrix:\n{conf_matrix}')

    # # calibration plots
    # calibration_model_plot(dev_all_gold, dev_all_decode, calibration_sig_list, calibration_iso_list)

    # auc 95% confidence interval
    score, ci_lower, ci_upper, scores = stat_util.score_ci(
        dev_all_gold, dev_all_decode[:, 1], score_fun=roc_auc_score
    )
    print('\n95% confidence interval:')
    print(f'auroc: {score}')
    print(f'lower: {ci_lower}')
    print(f'upper: {ci_upper}\n')

    score, ci_lower, ci_upper, scores = stat_util.score_ci(
        dev_all_gold, dev_all_decode[:, 1], score_fun=average_precision_score
    )
    print('\n95% confidence interval:')
    print(f'auprc: {score}')
    print(f'lower: {ci_lower}')
    print(f'upper: {ci_upper}\n')

    # np.savetxt(f'results/{baseline_type}_dev_preds.txt', dev_all_decode)
    # np.savetxt(f'results/{baseline_type}_dev_golds.txt', dev_all_gold.astype('int32'))
    dev_all = np.concatenate((np.array([dev_noteid_list]).T, dev_all_decode, np.array([dev_pred_labels]).T,
                              np.array([dev_all_gold]).T), axis=1)
    output_header = ['note_id', 'pred_prob_0', 'pred_prob_1', 'pred_label', 'gold_label']
    dev_dataframe = pd.DataFrame(dev_all, columns=output_header)
    dev_dataframe['annotator'] = dev_dataframe['note_id'].map(note2annotator)
    dev_dataframe.to_csv(os.path.join(output_dir, f'{baseline_type}_dev_output.csv'), index=None)

    # print("auc:", auc)

    return fpr, tpr, roc_thresholds, auc, precision, recall, pr_thresholds


def run_baseline(train_file, test_file, output_dir, baseline_type="svm", partition_num=10, write_decoded=False,
                 headers=False, feature_importances=False):
    # target = "HSR"

    train_x, train_y, train_note_id, train_note2annotator = load_scd_data(train_file, 6, 12, shuffle_instance=False)
    test_x, test_y, test_note_id, test_note2annotator = load_scd_data(test_file, 6, 12, shuffle_instance=False)
    # input_x, output_y, note_id, note2annotator = load_scd_data(input_file, shuffle_instance=False)
    # input_x, output_y, MRN = load_allergy_data(input_file, target)
    # input_x, output_y, MRN = load_txt_allergy_data(input_file, True)
    label2id, id2label = build_alphabet(train_y + test_y, True)
    # label2id, id2label = build_alphabet(output_y, True)
    positive_label = 1
    positive_id = label2id[positive_label]
    # if write_decoded:
    #     fout = open("decode.csv", 'wb')
    #     csvwriter = csv.writer(fout)
    #     csvwriter.writerow(["MRN",target, "Pred_"+target, "Description"])
    # else:
    #     csvwriter = None
    csvwriter = None
    dev_noteid_list = []
    dev_decode_list = []
    dev_gold_list = []
    test_noteid_list = []
    test_decode_list = []
    test_gold_list = []
    calibration_sig_list = []
    calibration_iso_list = []
    # for idx in range(partition_num):
    start_time = time.time()
    print("Proceesing..........................................")
    # print("Proceesing: ", idx, "/", partition_num, "..........................................")
    ## split into train/dev/test
    X = (train_x, test_x, test_x)
    Y = (train_y, test_y, test_y)
    the_note_id = (train_note_id, test_note_id, test_note_id)
    # X = data_split(input_x, partition_num, idx)
    # Y = data_split(output_y, partition_num, idx)
    # the_note_id = data_split(note_id, partition_num, idx)
    test_noteid_list.extend(the_note_id[2])
    # dev_noteid_list.extend(the_note_id[1])
    dev_preds, dev_golds, test_preds, test_golds, \
        feature_names, feature_importance_data, tree_representation,\
        calibration_data_sig, calibration_data_iso, model_object = model(X, Y, the_note_id, label2id, baseline_type,
                                                                         csvwriter, feature_importances)
    dev_decode_list.append(dev_preds)
    dev_gold_list.append(dev_golds)
    test_decode_list.append(test_preds)
    test_gold_list.append(test_golds)
    if calibration_data_sig:
        calibration_sig_list.extend(calibration_data_sig)
    if calibration_data_iso:
        calibration_iso_list.extend(calibration_data_iso)

    # # write tree representation (randforest, xgboost)
    # if baseline_type == 'randforest':
    #     with open(os.path.join(output_dir, f"{baseline_type.upper()}_{idx}_tree.txt"), 'w', encoding='utf-8') as f:
    #         f.write(tree_representation)
    # elif baseline_type == 'xgboost':
    #     pass

    # write features and feature importances
    with open(os.path.join(output_dir, f"{baseline_type.upper()}_features.csv"), 'w', newline='') as f:
        feature_writer = csv.writer(f)
        feature_writer.writerow([baseline_type])
        for jdx, feat in enumerate(feature_names):
            feature_writer.writerow([jdx, feat])

    if feature_importances and feature_importance_data is not None:
        if baseline_type == 'xgboost':
            with open(os.path.join(output_dir, f"{baseline_type.upper()}_fi.csv"), 'w', newline='') as f:
                feature_writer = csv.writer(f)
                feature_writer.writerow([baseline_type])
                for feat, feat_im in feature_importance_data.items():
                    feature_writer.writerow([feat, feat_im])
        else:
            with open(os.path.join(output_dir, f"{baseline_type.upper()}_fi.csv"), 'w', newline='') as f:
                feature_writer = csv.writer(f)
                feature_writer.writerow([baseline_type])
                for feat_im in feature_importance_data:
                    feature_writer.writerow([feat_im])

    cost_time = time.time() - start_time
    print("     Time cost: %.2f s" % cost_time)

    dev_all_decode = np.concatenate(dev_decode_list, axis=0)
    dev_all_gold = np.concatenate(dev_gold_list, axis=0)
    test_all_decode = np.concatenate(test_decode_list, axis=0)
    test_all_gold = np.concatenate(test_gold_list, axis=0)
    fpr, tpr, roc_thresholds, auc = calculate_roc_list(test_all_gold, test_all_decode, positive_id)
    # precision, recall, pr_thresholds = calculate_precision_recall_list(test_all_gold, test_all_decode, positive_id)
    precision, recall, pr_thresholds, auprc = calculate_precision_recall_list(test_all_gold, test_all_decode, positive_id)
    # fpr, tpr, roc_thresholds, auc, precision, recall, pr_thresholds = calculate_roc_precision_recall(test_all_gold, test_all_decode, positive_id)
    with open(os.path.join(output_dir, baseline_type.upper()+".original.pkl"), 'wb') as f:
        # pickle.dump([fpr, tpr, roc_thresholds, auc, precision, recall, pr_thresholds, baseline_type.upper()], f)
        pickle.dump([fpr, tpr, roc_thresholds, auc, precision, recall, pr_thresholds, auprc, baseline_type.upper()], f)
    with open(os.path.join(output_dir, baseline_type.upper()+".model.pkl"), 'wb') as f:
        pickle.dump(model_object, f)

    # plot_classification_results(fpr, tpr, auc, precision, recall, baseline_type.upper())

    # # calibration plots
    # calibration_model_plot(dev_all_gold, dev_all_decode, calibration_sig_list, calibration_iso_list)

    # auc 95% confidence interval
    score, ci_lower, ci_upper, scores = stat_util.score_ci(
        test_all_gold, test_all_decode[:, 1], score_fun=roc_auc_score
    )
    print('\n95% confidence interval:')
    print(f'auroc: {score}')
    print(f'lower: {ci_lower}')
    print(f'upper: {ci_upper}\n')

    score, ci_lower, ci_upper, scores = stat_util.score_ci(
        test_all_gold, test_all_decode[:, 1], score_fun=average_precision_score
    )
    print('\n95% confidence interval:')
    print(f'auprc: {score}')
    print(f'lower: {ci_lower}')
    print(f'upper: {ci_upper}\n')

    # np.savetxt(f'results/{baseline_type}_dev_preds.txt', dev_all_decode)
    # np.savetxt(f'results/{baseline_type}_dev_golds.txt', dev_all_gold.astype('int32'))
    test_all = np.concatenate((np.array([test_noteid_list]).T, test_all_decode, np.array([test_all_gold]).T), axis=1)
    output_header = ['note_id', 'pred_prob_0', 'pred_prob_1', 'gold_label']
    test_dataframe = pd.DataFrame(test_all, columns=output_header)
    test_dataframe['annotator'] = test_dataframe['note_id'].map(test_note2annotator)
    test_dataframe.to_csv(os.path.join(output_dir, f'{baseline_type}_dev_output.csv'), index=None)

    # print("auc:", auc)

    return fpr, tpr, roc_thresholds, auc, precision, recall, pr_thresholds


def model_comparison(model_dir, model_list, model_titles, partition_num=10, rerun=True, write_metrics_to_file=False, add_dl=False):
    result_list = []
    print("Comparison %s models: %s"%(len(model_list), " vs ".join(model_list)))
    if rerun:
        print("Rerun all models.")
        for each_model in model_list:
            fpr, tpr, roc_thresholds, auc, precision, recall, pr_thresholds = run_nfold_baseline(input_file, each_model, partition_num, False)
            result_list.append([fpr, tpr, auc, precision, recall, each_model.upper()])
    else:
        print("Load model results.")
        for each_model in model_list:
            with open(os.path.join(model_dir, each_model.upper()+".original.pkl"), 'rb') as pickle_file:
                # fpr, tpr, roc_thresholds, auc, precision, recall, pr_thresholds, baseline_type = pickle.load(pickle_file)
                fpr, tpr, roc_thresholds, auc, precision, recall, pr_thresholds, auprc, baseline_type = pickle.load(pickle_file)
            result_list.append([fpr, tpr, auc, precision, recall, each_model.upper()])
            # print(recall)
            if write_metrics_to_file:
                with open(os.path.join(model_dir, f'{each_model}_auc.txt'), 'w', encoding='utf-8') as f:
                    # f.write(str(auc))
                    f.write(f'auroc: {auc}\nauprc: {auprc}')
                with open(os.path.join(model_dir, f'{each_model}_metrics.csv'), 'w', newline='') as f:
                    metric_writer = csv.writer(f)
                    metric_writer.writerow(['threshold', 'precision', 'recall', 'f1'])
                    for t, p, r in zip(pr_thresholds, precision, recall):
                        f = 2 * p * r / (p + r)
                        metric_writer.writerow([t, p, r, f])

    prec_data = [result[3] for result in result_list]
    rec_data = [result[4] for result in result_list]

    if add_dl:
        # get AUPRC data
        dl_metrics_df = pd.read_csv(os.path.join(model_dir, 'dl_metrics.csv'))
        dl_prec_data = list(dl_metrics_df['Precision'])
        dl_rec_data = list(dl_metrics_df['Recall'])
        prec_data.append(dl_prec_data)
        rec_data.append(dl_rec_data)
        model_titles.append('Deep Learning')

        # get AUROC data
        dl_roc_data_df = pd.read_csv(os.path.join(model_dir, 'dl_roc_data.csv'))
        dl_fp_data = list(dl_roc_data_df['FPR'])
        dl_tp_data = list(dl_roc_data_df['TPR'])
        result_list.append([dl_fp_data, dl_tp_data, [], [], [], 'DL'])
        # for fp, tp in zip(dl_fp_data, dl_tp_data):
        #     result_list.append([fp, tp, 0, 0, 0, 'DL'])

    # plot_multi_curve(rec_data, prec_data, model_titles, x_name='Recall', y_name='Precision')
    # plot_multiple_classification_results_pp(result_list, model_titles)
    plot_multi_curve(rec_data, prec_data, model_titles, x_name='Recall', y_name='Precision', save_dir=model_dir)
    plot_multiple_classification_results_pp(result_list, model_titles, save_dir=model_dir)


def model(X, Y, the_MRN, label2id, baseline_type, csvwriter, feature_importances):
    train_x, dev_x, test_x = X
    train_y, dev_y, test_y = Y
    train_mrn, dev_mrn, test_mrn = the_MRN
    print("Train/dev/test num: %s/%s/%s" % (len(train_x), len(dev_x), len(test_x)))

    ## extract feature
    print("Extracting feature...")
    use_unigram = True
    if use_unigram:
        x_vectorizer = CountVectorizer()
        x_tfidftransformer = TfidfTransformer()
        train_weight = x_tfidftransformer.fit_transform(x_vectorizer.fit_transform(train_x)).toarray()
        dev_weight = x_tfidftransformer.transform(x_vectorizer.transform(dev_x)).toarray()
        test_weight = x_tfidftransformer.transform(x_vectorizer.transform(test_x)).toarray()
        feature_names = x_vectorizer.get_feature_names_out()
    num_round = None
    # num_round = 20  # loop
    ## add bigram
    use_bigram = False
    if use_bigram:
        print("Add bigram feature")
        train_x_bi = unigram2bigram(train_x)
        dev_x_bi = unigram2bigram(dev_x)
        test_x_bi = unigram2bigram(test_x)
        bix_vectorizer = CountVectorizer()
        bix_tfidftransformer = TfidfTransformer()
        train_weight_bi = bix_tfidftransformer.fit_transform(bix_vectorizer.fit_transform(train_x_bi)).toarray()
        dev_weight_bi = bix_tfidftransformer.transform(bix_vectorizer.transform(dev_x_bi)).toarray()
        test_weight_bi = bix_tfidftransformer.transform(bix_vectorizer.transform(test_x_bi)).toarray()
        if use_unigram:
            feature_names += bix_vectorizer.get_feature_names_out()
            train_weight = np.concatenate((train_weight, train_weight_bi), axis=1)
            dev_weight = np.concatenate((dev_weight, dev_weight_bi), axis=1)
            test_weight = np.concatenate((test_weight, test_weight_bi), axis=1)
        else:
            feature_names = bix_vectorizer.get_feature_names_out()
            train_weight = train_weight_bi
            dev_weight = dev_weight_bi
            test_weight = test_weight_bi
    print("Use bigram: ", use_bigram)
    print("Iteration: ", num_round)
    print(train_weight.shape, dev_weight.shape, test_weight.shape)

    ## prepare label -> id 
    train_label_id = []
    for label in train_y:
        train_label_id.append(label2id[label])
    train_label_array = np.array(train_label_id)

    dev_label_id = []
    for label in dev_y:
        dev_label_id.append(label2id[label])
    dev_label_array = np.array(dev_label_id)

    test_label_id = []
    for label in test_y:
        test_label_id.append(label2id[label])
    test_label_array = np.array(test_label_id)
    tree_representation = None
    if baseline_type.upper() == 'XGBOOST':
        clf = xgboost(train_weight, train_label_array, label2id, feature_names, num_round)
        # feat_map_file = 'feat_map.txt'
        # with open(feat_map_file, 'w', encoding='utf-8') as f:
        #     clf.dump_model('model_dump.txt', feat_map_file)
        # clf.dump_model('model_dump.txt', feat_map_file)
        # xgb.plot_tree(clf, fmap='feat_map.txt')
        # xgb.plot_tree(clf)
        # plt.show()
    elif baseline_type.upper() == 'SVM':
        clf = linearSVM(train_weight, train_label_array, num_round)
    elif baseline_type.upper() == 'LOGISTIC':
        clf = logistic(train_weight, train_label_array, num_round)
    elif baseline_type.upper() == 'RANDFOREST':
        clf = random_forest(train_weight, train_label_array)

        # print('Writing random forest paths for dev data to file...')
        # sample_id = 0
        # with open('results/test/randforest_treepath_testing.txt', 'w', encoding='utf-8') as f:
        #     for est_idx, estimator in enumerate(clf.estimators_):
        #         feature = estimator.tree_.feature
        #         threshold = estimator.tree_.threshold
        #         node_indicator = estimator.decision_path(dev_weight)
        #         leaf_id = estimator.apply(dev_weight)
        #         # obtain ids of the nodes `sample_id` goes through, i.e., row `sample_id`
        #         node_index = node_indicator.indices[node_indicator.indptr[sample_id]:
        #                                             node_indicator.indptr[sample_id + 1]]
        #         # print('Rules used to predict sample {id}:\n'.format(id=sample_id))
        #         f.write('Rules used by estimator {estimator} to predict sample {id}:\n'.format(estimator=est_idx,
        #                                                                                        id=sample_id))
        #         for node_id in node_index:
        #             # continue to the next node if it is a leaf node
        #             if leaf_id[sample_id] == node_id:
        #                 continue
        #             # check if value of the split feature for sample 0 is below threshold
        #             if (dev_weight[sample_id, feature[node_id]] <= threshold[node_id]):
        #                 threshold_sign = "<="
        #             else:
        #                 threshold_sign = ">"
        #             f.write("decision node {node} : (X_dev[sample {sample}, feature '{feature}'] = {value}) "
        #                     "{inequality} {threshold})\n".format(node=node_id,
        #                                                          sample=sample_id,
        #                                                          feature=feature_names[feature[node_id]],
        #                                                          # feature=feature[node_id],
        #                                                          value=dev_weight[sample_id, feature[node_id]],
        #                                                          inequality=threshold_sign,
        #                                                          threshold=threshold[node_id]))
        #
        #         f.write('\n')

        # leaf_id = clf.apply(test_weight[0].reshape(1, -1))
        # indicator_matrix, tree_split_idx = clf.decision_path(test_weight[0].reshape(1, -1))
        # indicator_matrix_dense = indicator_matrix.toarray()
        # print('first estimator paths:')
        # print(indicator_matrix_dense[:, tree_split_idx[0]:tree_split_idx[1]])
        tree_representation = sklearn.tree.export_text(clf.estimators_[0], feature_names=feature_names)
    else:
        print("Error: no model founded, should be among XGBOOST/SVM/LOGISTIC/RANDFOREST, input: ", baseline_type)
        exit(0)

    cal_sig_data = None
    cal_iso_data = None
    if baseline_type.upper() == 'XGBOOST':
        ddev = xgb.DMatrix(dev_weight) 
        dtest = xgb.DMatrix(test_weight) 
        dev_preds = clf.predict(ddev)
        test_preds = clf.predict(dtest)
    else:
        dev_preds = clf.predict_proba(dev_weight)
        test_preds = clf.predict_proba(test_weight)

        # cal_sig_data, cal_iso_data = calibration_model(clf, train_weight, train_label_array, dev_weight)

    # save feature importances
    if feature_importances:
        importances = get_feature_importances(clf, baseline_type)
        return dev_preds, dev_y, test_preds, test_y, feature_names, importances, tree_representation, cal_sig_data, cal_iso_data, clf

    return dev_preds, dev_y, test_preds, test_y, feature_names, None, tree_representation, cal_sig_data, cal_iso_data, clf

    ## prepare decode file to write
    # decode_x = test_x 
    # decode_y = test_preds.tolist()
    # gold_y = test_label_array.tolist()
    # dev_test_mrn = test_mrn
    # print(len(dev_test_mrn),len(decode_x), len(decode_y), len(gold_y))
    # ins_num = len(decode_x)
    # if csvwriter:
    #     for msr,g_y, p_y, des in zip(dev_test_mrn, gold_y, decode_y, decode_x):
    #         csvwriter.writerow([msr,g_y, p_y, des])

    # output_results = []
    # p,r,f, acc = calculate_auc(dev_label_array, dev_preds)
    # print("Dev  P: %s, R: %s, F: %s, acc: %s"%(p,r,f, acc))
    # output_results += [p,r,f, acc]
    # p,r,f, acc = calculate_p_r_f(test_label_array, test_preds)
    # print("Test P: %s, R: %s, F: %s, acc: %s"%(p,r,f, acc))
    # output_results += [p,r,f, acc]
    # return dev_preds, dev_y, test_preds,  test_y


def calibration_model(clf, train_weight, train_label_array, dev_weight):
    # calibration
    calibrated_sigmoid_model = CalibratedClassifierCV(clf, method='sigmoid')
    calibrated_sigmoid_model.fit(train_weight, train_label_array)
    prob_cal_sig = calibrated_sigmoid_model.predict_proba(dev_weight)[:, 1]

    calibrated_isotonic_model = CalibratedClassifierCV(clf, method='isotonic')
    calibrated_isotonic_model.fit(train_weight, train_label_array)
    prob_cal_iso = calibrated_sigmoid_model.predict_proba(dev_weight)[:, 1]

    return prob_cal_sig, prob_cal_iso


def calibration_model_plot(dev_y, dev_preds, sig_probs, iso_probs):
    prob_uncal = dev_preds[:, 1]

    # reliability plots
    uncalibrated_pred_positive, actual_positive = calibration_curve(dev_y, prob_uncal, n_bins=10)
    plt.plot([0, 1], [0, 1])
    plt.plot(actual_positive, uncalibrated_pred_positive)
    plt.grid()
    plt.xlabel("Average probability")
    plt.ylabel("Proportion of positives")
    plt.title("Reliability Plot - Uncalibrated")
    # plt.legend(loc='best')
    plt.show()

    sigmoid_calibrated_pred_positive, actual_positive = calibration_curve(dev_y, sig_probs, n_bins=10)
    plt.plot([0, 1], [0, 1])
    plt.plot(actual_positive, sigmoid_calibrated_pred_positive)
    plt.grid()
    plt.xlabel("Average probability")
    plt.ylabel("Proportion of positives")
    plt.title("Reliability Plot - Sigmoid Calibration")
    # plt.legend(loc='best')
    plt.show()

    isotonic_calibrated_pred_positive, actual_positive = calibration_curve(dev_y, iso_probs, n_bins=10)
    plt.plot([0, 1], [0, 1])
    plt.plot(actual_positive, isotonic_calibrated_pred_positive)
    plt.grid()
    plt.xlabel("Average probability")
    plt.ylabel("Proportion of positives")
    plt.title("Reliability Plot - Isotonic Calibration")
    # plt.legend(loc='best')
    plt.show()


def random_forest(train_weight, train_label_array):
    ## return array of shape = [n_samples, n_classes]
    print("Running random forest Classifier...")
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=100,  random_state=0)
    clf.fit(train_weight, train_label_array) 

    return clf


def logistic(train_weight, train_label_array, num_round=None):
    print("Running logistic Classifier...")
    from sklearn.linear_model import SGDClassifier
    if num_round:
        clf = SGDClassifier(loss='log_loss', max_iter=num_round)
    else:
        clf = SGDClassifier(loss='log_loss')
    clf.fit(train_weight, train_label_array) 

    return clf


def linearSVM(train_weight, train_label_array, num_round=None):
    from sklearn.svm import LinearSVC
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.linear_model import SGDClassifier
    # clf = CalibratedClassifierCV(LinearSVC())
    if num_round:
        clf = CalibratedClassifierCV(SGDClassifier(loss='hinge', max_iter=num_round))
    else:
        clf = CalibratedClassifierCV(SGDClassifier(loss='hinge'))
    clf.fit(train_weight, train_label_array)

    return clf


def xgboost(train_weight, train_label_array, label2id, feat_names, num_round=None):
    ## return array of shape = [n_samples, n_classes], return probabilities
    print("Running XGBoost model...")

    # feat_types = ['q' for _ in feat_names]
    # dtrain = xgb.DMatrix(train_weight, label=train_label_array, feature_names=feat_names, feature_types=feat_types)
    dtrain = xgb.DMatrix(train_weight, label=train_label_array)
    # ddev = xgb.DMatrix(dev_weight, label=dev_label_array)
    # dtest = xgb.DMatrix(test_weight) 
    param = {'max_depth':100, 'eta':0.1, 'eval_metric':'merror', 'silent':1, 'objective':'multi:softprob', 'num_class':len(label2id)}  # 参数
    # evallist  = [(dtrain,'train'), (ddev,'dev')] 
    
    # clf = xgb.train(param, dtrain, num_round, evallist)
    if num_round:
        clf = xgb.train(param, dtrain, num_round)
    else:
        clf = xgb.train(param, dtrain)

    return clf


def run_all_model(input_annotated_file, decode_file, output_file, baseline_type="svm", rerun = True):
    target = "HSR"
    input_x, output_y, MRN = load_allergy_data(input_file, target)
    label2id, id2label = build_alphabet(output_y, True)
    positive_label = 1
    positive_id = label2id[positive_label]
    # input_x_dict = {}
    # for input_sent in input_x:
    #     input_x_dict[input_sent] = 1
    record_id_list, record_list = load_pair_txt_file(decode_file)
    ## find duplicated
    # dup_count = 0
    # for each_record in record_list:
    #     if each_record in input_x_dict:
    #         dup_count += 1 
    # print("Dup count: ", dup_count)
    # exit(0)

    x_vectorizer, x_tfidftransformer, train_weight = feature_extraction(input_x)
    test_weight = x_tfidftransformer.transform(x_vectorizer.transform(record_list)).toarray()
    use_bigram = False 
    num_round = 20
    if use_bigram:
        print("Add bigram feature")
        train_x_bi = unigram2bigram(input_x)
        bix_vectorizer, bix_tfidftransformer, train_weight_bi = feature_extraction(train_x_bi)
        test_x_bi = unigram2bigram(record_list)
        test_weight_bi = bix_tfidftransformer.transform(bix_vectorizer.transform(test_x_bi)).toarray()
        train_weight = np.concatenate((train_weight, train_weight_bi), axis=1)
        test_weight = np.concatenate((test_weight, test_weight_bi), axis=1)
    print("Use bigram: ", use_bigram)
    print("Iteration: ", num_round)

    train_label_id = []
    for label in output_y:
        train_label_id.append(label2id[label])
    train_label_array = np.array(train_label_id)

    test_label_id = []
    for idx in range(len(record_list)):
        test_label_id.append(0)
    test_label_array = np.array(test_label_id)
    if rerun:
        if baseline_type.upper() == 'XGBOOST':
            clf = xgboost(train_weight, train_label_array, label2id, num_round)
        elif baseline_type.upper() == 'SVM':
            clf = linearSVM(train_weight, train_label_array, num_round)
        elif baseline_type.upper() == 'LOGISTIC':
            clf = logistic(train_weight, train_label_array, num_round)
        elif baseline_type.upper() == 'RANDFOREST':
            clf = random_forest(train_weight, train_label_array)
    else:
        print("Reload model:", baseline_type.upper()+".original.model")
        if baseline_type.upper() == 'XGBOOST':
            clf = xgb.Booster({'nthread': 4})
            clf.load_model(baseline_type.upper()+".original.model")
        else:
            with open(baseline_type.upper()+".original.model", 'rb') as pickle_file:
                clf = pickle.load(pickle_file)
    if baseline_type.upper() == 'XGBOOST':
        dtest = xgb.DMatrix(test_weight) 
        test_preds = clf.predict(dtest)
    else:
        test_preds = clf.predict_proba(test_weight)
    print("decode shape:",test_preds.shape)
    print("Origin size:", len(record_list))
    
    if baseline_type.upper() == 'XGBOOST':
        clf.save_model(baseline_type.upper()+".original.model")
    else:
        with open(baseline_type.upper()+".original.model", 'wb') as f:
            pickle.dump(clf, f)
    print("Model saved to file:", baseline_type.upper()+".original.model")
    print(test_preds[:10,:])


def feature_extraction(train_x):
    print("Extracting feature...")
    x_vectorizer = CountVectorizer()
    x_tfidftransformer = TfidfTransformer()
    train_weight = x_tfidftransformer.fit_transform(x_vectorizer.fit_transform(train_x)).toarray()
    return x_vectorizer, x_tfidftransformer, train_weight

# def model(X, Y,  the_MRN, label2id, baseline_type, csvwriter):
#     train_x, dev_x, test_x = X
#     train_y, dev_y, test_y = Y
#     train_mrn, dev_mrn, test_mrn = the_MRN
#     print("Train/dev/test num: %s/%s/%s"%(len(train_x), len(dev_x), len(test_x)))
    
#     ## extract feature
#     print("Extracting feature...")
#     x_vectorizer = CountVectorizer()
#     x_tfidftransformer = TfidfTransformer()
#     train_weight = x_tfidftransformer.fit_transform(x_vectorizer.fit_transform(train_x)).toarray()
#     dev_weight = x_tfidftransformer.transform(x_vectorizer.transform(dev_x)).toarray()
#     test_weight = x_tfidftransformer.transform(x_vectorizer.transform(test_x)).toarray()
#     num_round = 20  # loop
#     ## add bigram
#     use_bigram = False
#     if use_bigram:
#         print("Add bigram feature")
#         train_x_bi = unigram2bigram(train_x)
#         dev_x_bi = unigram2bigram(dev_x)
#         test_x_bi = unigram2bigram(test_x)
#         bix_vectorizer = CountVectorizer()
#         bix_tfidftransformer = TfidfTransformer()
#         train_weight_bi = bix_tfidftransformer.fit_transform(bix_vectorizer.fit_transform(train_x_bi)).toarray()
#         dev_weight_bi = bix_tfidftransformer.transform(bix_vectorizer.transform(dev_x_bi)).toarray()
#         test_weight_bi = bix_tfidftransformer.transform(bix_vectorizer.transform(test_x_bi)).toarray()
#         train_weight = np.concatenate((train_weight, train_weight_bi), axis=1)
#         dev_weight = np.concatenate((dev_weight, dev_weight_bi), axis=1)
#         test_weight = np.concatenate((test_weight, test_weight_bi), axis=1)
#     print "Use bigram: ", use_bigram
#     print "Iteration: ", num_round
#     print train_weight.shape, dev_weight.shape, test_weight.shape

#     ## prepare label -> id 
#     train_label_id = []
#     for label in train_y:
#         train_label_id.append(label2id[label])
#     train_label_array = np.array(train_label_id)

#     dev_label_id = []
#     for label in dev_y:
#         dev_label_id.append(label2id[label])
#     dev_label_array = np.array(dev_label_id)

#     test_label_id = []
#     for label in test_y:
#         test_label_id.append(label2id[label])
#     test_label_array = np.array(test_label_id)
#     if baseline_type.upper() == 'XGBOOST':
#         clf, dev_preds, test_preds = xgboost(train_weight, dev_weight, test_weight, train_label_array, None, test_label_array,label2id, num_round)
#     elif baseline_type.upper() == 'SVM':
#         clf, dev_preds, test_preds = linearSVM(train_weight, dev_weight, test_weight, train_label_array, num_round)
#     elif baseline_type.upper() == 'LOGISTIC':
#         clf, dev_preds, test_preds = logistic(train_weight, dev_weight, test_weight, train_label_array, num_round)
#     elif baseline_type.upper() == 'RANDFOREST':
#         clf, dev_preds, test_preds = random_forest(train_weight, dev_weight, test_weight, train_label_array)
#     else:
#         print("Error: no model founded, should be among XGBOOST/SVM/LOGISTIC/RANDFOREST, input: ", baseline_type)
#         exit(0)
    
#     return dev_preds, dev_y, test_preds,  test_y


def run_all_model_by_load_model(input_annotated_file, decode_file, output_file):
    target = "HSR"
    input_x, output_y, MRN = load_allergy_data(input_file, target)
    label2id, id2label = build_alphabet(output_y, True)
    positive_label = 1
    positive_id = label2id[positive_label]
    record_id_list, record_list = load_pair_txt_file(decode_file)

    x_vectorizer, x_tfidftransformer, train_weight = feature_extraction(input_x)
    test_weight = x_tfidftransformer.transform(x_vectorizer.transform(record_list)).toarray()
    use_bigram = False 
    num_round = 20
    if use_bigram:
        print("Add bigram feature")
        train_x_bi = unigram2bigram(input_x)
        bix_vectorizer, bix_tfidftransformer, train_weight_bi = feature_extraction(train_x_bi)
        test_x_bi = unigram2bigram(record_list)
        test_weight_bi = bix_tfidftransformer.transform(bix_vectorizer.transform(test_x_bi)).toarray()
        train_weight = np.concatenate((train_weight, train_weight_bi), axis=1)
        test_weight = np.concatenate((test_weight, test_weight_bi), axis=1)
    print("Use bigram: ", use_bigram)
    print("Iteration: ", num_round)

    train_label_id = []
    for label in output_y:
        train_label_id.append(label2id[label])
    train_label_array = np.array(train_label_id)

    test_label_id = []
    for idx in range(len(record_list)):
        test_label_id.append(0)
    test_label_array = np.array(test_label_id)
    xg_model = xgb.Booster({'nthread': 4})
    xg_model.load_model("XGBOOST.original.model")
    dtest = xgb.DMatrix(test_weight) 
    xg_test = xg_model.predict(dtest)[:, positive_id]
    with open("RANDFOREST.original.model", 'rb') as pickle_file:
        rf_model = pickle.load(pickle_file)
        rf_test = rf_model.predict_proba(test_weight)[:, positive_id]
    with open("SVM.original.model", 'rb') as pickle_file:
        svm_model = pickle.load(pickle_file)
        svm_test = svm_model.predict_proba(test_weight)[:, positive_id]
    with open("LOGISTIC.original.model", 'rb') as pickle_file:
        log_model = pickle.load(pickle_file)
        log_test = log_model.predict_proba(test_weight)[:, positive_id]

    exit(0)
    fout = open(output_file, 'wb')
    csvwriter = csv.writer(fout)
    csvwriter.writerow(["RecordID","XGBOOST", "RANDFOREST", "SVM", "LOGISTIC", "Description"])
    for idx in range(len(record_list)):
        csvwriter.writerow([record_id_list[idx], xg_test[idx], rf_test[idx], svm_test[idx], log_test[idx], record_list[idx]])
    print("All results are written in file:", output_file)


def csv2txt(input_csv, output_txt, tab=None):
    input_x, output_y, MRN = load_allergy_data(input_csv, tab, False )
    for idn, (x, y) in enumerate(zip(input_x, output_y)):
        if idn == 0:
            fout = open(output_txt+str(0),'wb')
        elif idn%500 == 0:
            the_part = idn/500
            fout = open(output_txt+str(int(the_part)),'wb')
        fout.write(x.replace('\n'," ")+" ||| "+str(y)+"\n")


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
        sent_list = list(sentence_dict.keys())
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
    for key, value in validation_dict.items():
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


def get_feature_importances(model, model_type):
    """
    returns values for insight into the most informative features.

    sklearn tree classifiers (ada, rf, extra) return Gini Importance (a.k.a. Mean Decrease in Impurity) - see
    https://stackoverflow.com/questions/15810339/how-are-feature-importances-in-randomforestclassifier-determined.

    log and svm return the coefficients (weights) assigned to features.

    xgb returns 'gain' - see
    https://towardsdatascience.com/be-careful-when-interpreting-your-features-importance-in-xgboost-6e16132588e7.
    """
    if model_type == 'xgboost':
        return model.get_score(importance_type='gain')
    elif model_type == 'randforest':
        return model.feature_importances_
    elif model_type == 'logistic':
        return model.coef_[0]
    elif model_type == 'svm':
        return None
    else:
        print('!!! invalid model type, no feature importances !!!\n')

    return None


if __name__ == '__main__':
    input_file = r"../data/RAG_retrieved_notes_section_large.csv"
    # input_file = r"../data/scd_annotation_dataset_final_filtered.csv"
    # train_file = r"../data/scd_annotation_dataset_final_filtered_corrected_errors_corrected_false_labels.csv"
    # train_file = r"../../case_finding/data/aav_random_5000_sections_with_keywords_updated_labels_rm_dupl_excel.csv"
    # test_file = r"../../case_finding/data/aav_datasetII_generalizability_labeled_rm_dupl_excel.csv"
    model_output_dir = '../results/adrd_baseline_section_large'
    # model_output_dir = 'results/combined_datasets'
    # model_output_dir = 'results/generalizability_output/updated_v2'

    model_list = ['logistic', 'svm', 'randforest', 'xgboost']
    for md in model_list:
        # run_baseline(train_file, test_file, model_output_dir, md, write_decoded=False, feature_importances=False)
        # run_nfold_baseline(input_file, model_output_dir, md, partition_num=5, write_decoded=False,
        #                    feature_importances=False)
        run_nfold_baseline(input_file, model_output_dir, md, partition_num=5)

    model_titles = ['RAG-LR', 'RAG-SVM', 'RAG-RF', 'RAG-XGBoost']
    rerun = False
    model_comparison(model_output_dir, model_list, model_titles, partition_num=5, rerun=rerun, write_metrics_to_file=True, add_dl=False)
