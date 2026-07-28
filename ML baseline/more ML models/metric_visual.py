# -*- coding: utf-8 -*-
# @Author: Jie Yang
# @Date:   2019-01-28 11:40:37
# @Last Modified by:   Baoren Liu,     Contact: liubaoren2006@gmail.com
# @Last Modified time: 2026-07-28
# -*- coding: utf-8 -*-
# import matplotlib
# matplotlib.use("TkAgg")
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics

# -*- coding: utf-8 -*-
# @Author: Jie Yang
# @Date:   2019-01-28 11:40:37
# @Last Modified by:   Jie Yang,    Contact: jieynlp@gmail.com
# @Last Modified time: 2019-04-03 09:43:54
# -*- coding: utf-8 -*-
# import matplotlib
# matplotlib.use("TkAgg")
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics

# -*- coding: utf-8 -*-
# @Author: Jie Yang
# @Date:   2019-01-28 11:40:37
# @Last Modified by:   Jie Yang,    Contact: jieynlp@gmail.com
# @Last Modified time: 2019-04-03 09:43:54
# -*- coding: utf-8 -*-
# import matplotlib
# matplotlib.use("TkAgg")
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics


def calculate_clf_p_r_f_acc_kappa(gold_label, pred_label_prob, positive_id=1, threshold=0.5):
    """
    Calculates metrics. If a 2D probability array is passed, it applies the
    specified threshold to the positive class instead of defaulting to 0.5.
    """
    assert (gold_label.ndim == 1)

    if pred_label_prob.ndim == 2:
        pred_label = (pred_label_prob[:, positive_id] >= threshold).astype(int)
    elif pred_label_prob.ndim > 2:
        print("PRF calculation error: dimension of pred_label should <= 2")
        return 0.0, 0.0, 0.0, 0.0, 0.0
    else:
        pred_label = pred_label_prob

    gold_true = gold_label == positive_id
    pred_true = pred_label == positive_id
    gold_true_num = np.count_nonzero(gold_true)
    pred_true_num = np.count_nonzero(pred_true)
    right_pred_num = np.count_nonzero(gold_true & pred_true)

    p = (right_pred_num + 0.) / pred_true_num if pred_true_num > 0 else 0.0
    r = (right_pred_num + 0.) / gold_true_num if gold_true_num > 0 else 0.0
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    acc = (np.sum(gold_label == pred_label) + 0.) / gold_label.size

    print("Gold: %s; Pred: %s; Right: %s" % (gold_true_num, pred_true_num, right_pred_num))

    all_num = gold_label.size
    gold_false_num = all_num - gold_true_num
    pred_false_num = all_num - pred_true_num

    pe = (gold_true_num * pred_true_num + gold_false_num * pred_false_num + 0.) / (all_num * all_num)
    kappa = (acc - pe) / (1 - pe) if (1 - pe) > 0 else 0.0

    return p, r, f, acc, kappa


def calculate_p_r_f_acc_kappa(gold_label, pred_label):
    ## gold_label: numpy array, binary. size: (instance_num)
    ## pred_label: numpy array, binary, predict result. size: (instance_num,)
    gold_true = np.sum(gold_label)
    pred_true = np.sum(pred_label)
    right_pred = np.sum(gold_label * pred_label)
    p = (right_pred + 0.) / pred_true if pred_true > 0 else 0.0
    r = (right_pred + 0.) / gold_true if gold_true > 0 else 0.0
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    acc = (np.sum(gold_label == pred_label) + 0.) / gold_label.shape[0]
    print("Gold: %s; Pred: %s; Right: %s" % (gold_true, pred_true, right_pred))
    all_num = gold_label.size
    gold_false = all_num - gold_true
    pred_false = all_num - pred_true
    pe = (gold_true * pred_true + gold_false * pred_false + 0.) / (all_num * all_num)
    kappa = (acc - pe) / (1 - pe) if (1 - pe) > 0 else 0.0
    return p, r, f, acc, kappa


def calculate_roc_list(gold_label, pred_label_prob, positive_id):
    ## gold_label: numpy array, binary. size: (instance_num,)
    ## pred_label: numpy array, float probability, predict result. size: (instance_num, label_num)
    pred = pred_label_prob[:, positive_id]
    fpr, tpr, roc_thresholds = metrics.roc_curve(gold_label, pred, pos_label=positive_id)
    auc = metrics.auc(fpr, tpr)
    return fpr, tpr, roc_thresholds, auc


def calculate_precision_recall_list(gold_label, pred_label_prob, positive_id):
    ## gold_label: numpy array, binary. size: (instance_num,)
    ## pred_label: numpy array, float probability, predict result. size: (instance_num, label_num)
    pred = pred_label_prob[:, positive_id]
    precision, recall, pr_thresholds = metrics.precision_recall_curve(gold_label, pred, pos_label=positive_id)
    auprc = metrics.auc(recall, precision)
    return precision, recall, pr_thresholds, auprc


def plot_roc(fpr, tpr, auc, model_name="model", save_dir=None):
    plt.title('ROC Curve')
    plt.plot(fpr, tpr, 'b', label='%s: AUC=%0.3f' % (model_name, auc))
    plt.plot([0, 1], [0, 1], color='gray', marker='.', linestyle='dashed', alpha=0.5)
    plt.legend(loc='lower right')
    plt.xlim([0, 1.01])
    plt.ylim([0, 1.01])
    plt.ylabel('True Positive Rate (TPR)')
    plt.xlabel('False Positive Rate (FPR)')
    xtick = [(x + 0.) / 10 for x in range(0, 11)]
    ytick = xtick
    plt.xticks(xtick)
    plt.yticks(ytick)
    plt.grid()
    if save_dir:
        plt.savefig(save_dir, dpi=400, bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def plot_multi_roc(fpr_list, tpr_list, auc_list, model_name_list, save_dir=None):
    plt.title('ROC Curve')
    model_num = len(fpr_list)
    color_list = ['tab:blue', 'tab:red', 'tab:orange', 'tab:green', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:olive',
                  'tab:cyan']
    for idx in range(model_num):
        plt.plot(fpr_list[idx], tpr_list[idx], color_list[idx],
                 label='%s: AUC=%0.3f' % (model_name_list[idx], auc_list[idx]))
    plt.plot([0, 1], [0, 1], color='gray', marker='.', linestyle='dashed', alpha=0.5)
    plt.legend(loc='lower right')
    plt.xlim([0, 1.01])
    plt.ylim([0, 1.01])
    plt.ylabel('True Positive Rate (TPR)')
    plt.xlabel('False Positive Rate (FPR)')
    xtick = [(x + 0.) / 10 for x in range(0, 11)]
    ytick = xtick
    plt.xticks(xtick)
    plt.yticks(ytick)
    plt.grid()
    if save_dir:
        plt.savefig(save_dir, dpi=400, bbox_inches='tight')
    else:
        plt.show()


def add_hardcoded_prc_points():
    """Helper function to add the specific 4 points to PRC graphs."""
    # (Recall/Sensitivity, Precision/PPV)
    plt.plot(0.811, 0.835, marker='o', markersize=5, color='magenta', linestyle='None', label='Rule-based Baseline (F1=0.823)')
    # plt.plot(0.586, 0.876, marker='s', markersize=5, color='black', linestyle='None', label='Rule 2 (F1=0.702)')
    plt.plot(0.911, 0.957, marker='D', markersize=5, color='red', linestyle='None', label='RAG-LLM (F1=0.933)')
    plt.plot(0.917, 0.886, marker='^', markersize=5, color='lime', linestyle='None', label='Keyword-LLM (F1=0.901)')


def plot_precision_recall(precision, recall, model_name="model", save_dir=None):
    plt.title('Precision-Recall Curve')

    # Calculate AUC and Best F1
    auc_val = metrics.auc(recall, precision)
    denominator = precision + recall
    f1s = np.divide(2 * precision * recall, denominator, out=np.zeros_like(precision), where=denominator != 0)
    best_f1 = np.max(f1s)
    add_hardcoded_prc_points()
    plt.plot(recall, precision, 'b', label='%s (AUC=%0.3f, Best F1=%0.3f)' % (model_name, auc_val, best_f1))

    f_list = [0.2, 0.4, 0.6, 0.8]
    for f in f_list:
        all_rec = [f / (2 - f) + idx * 0.01 for idx in range(100)]
        rec = [a for a in all_rec if a <= 1]
        pre_f = [f * r / (2 * r - f) for r in rec]
        plt.plot(rec, pre_f, color='gray', linestyle='dashed')
        plt.text(0.9, (f / (2 - f) + 0.01 + f / 30), "F1=%s" % (f))


    plt.legend(loc='lower left')
    plt.xlim([0, 1.01])
    plt.ylim([0, 1.01])
    plt.ylabel('Precision')
    plt.xlabel('Recall')
    xtick = [(x + 0.) / 10 for x in range(0, 11)]
    ytick = xtick
    plt.xticks(xtick)
    plt.yticks(ytick)
    plt.grid()
    if save_dir:
        if save_dir.lower().endswith('.pdf'):
            save_dir = save_dir[:-4] + '.png'
        plt.savefig(save_dir, dpi=400, bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def plot_multi_precision_recall(precision_list, recall_list, model_name_list, save_dir=None):
    plt.title('Precision-Recall Curve')
    model_num = len(precision_list)
    color_list = ['tab:blue', 'tab:red', 'tab:orange', 'tab:green', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:olive',
                  'tab:cyan']

    add_hardcoded_prc_points()
    for idx in range(model_num):
        # Calculate AUC and Best F1 for each model
        auc_val = metrics.auc(recall_list[idx], precision_list[idx])
        denominator = precision_list[idx] + recall_list[idx]
        f1s = np.divide(2 * precision_list[idx] * recall_list[idx], denominator, out=np.zeros_like(precision_list[idx]),
                        where=denominator != 0)
        best_f1 = np.max(f1s)

        plt.plot(recall_list[idx], precision_list[idx], color_list[idx],
                 label='%s (AUC=%0.3f, Best F1=%0.3f)' % (model_name_list[idx], auc_val, best_f1))

    f_list = [0.2, 0.4, 0.6, 0.8]
    for f in f_list:
        all_rec = [f / (2 - f) + idx * 0.01 for idx in range(100)]
        rec = [a for a in all_rec if a <= 1]
        pre_f = [f * r / (2 * r - f) for r in rec]
        plt.plot(rec, pre_f, color='gray', linestyle='dashed')
        plt.text(0.9, (f / (2 - f) + 0.01 + f / 30), "F1=%s" % (f))



    plt.legend(loc='lower left')
    plt.xlim([0, 1.01])
    plt.ylim([0, 1.01])
    plt.ylabel('Precision')
    plt.xlabel('Recall')
    xtick = [(x + 0.) / 10 for x in range(0, 11)]
    ytick = xtick
    plt.xticks(xtick)
    plt.yticks(ytick)
    plt.grid()
    if save_dir:
        if save_dir.lower().endswith('.pdf'):
            save_dir = save_dir[:-4] + '.png'
        plt.savefig(save_dir, dpi=400, bbox_inches='tight')
    else:
        plt.show()


def plot_multi_curve(x_list, y_list, model_name_list, f1_ci_list=None, x_name="X_Name", y_name="Y_name", scale_one=True, log_y=False,
                     save_dir=None):
    plt.clf()
    plt.title('Dataset I: Precision-Recall Performance')
    model_num = len(x_list)
    color_list = ['tab:blue', 'tab:red', 'tab:orange', 'tab:green', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:olive',
                  'tab:cyan']
    # Check if this generic function is plotting PRC, if so add the points
    if 'recall' in x_name.lower() and 'precision' in y_name.lower():
        add_hardcoded_prc_points()
    for idx in range(model_num):
        # Calculate AUC and Best F1 assuming generic x/y are Recall/Precision for PRCs
        try:
            auc_val = metrics.auc(x_list[idx], y_list[idx])
        except ValueError:
            # Handle cases where x_list might not be perfectly monotonic
            sort_idx = np.argsort(x_list[idx])
            auc_val = metrics.auc(x_list[idx][sort_idx], y_list[idx][sort_idx])

        denominator = y_list[idx] + x_list[idx]
        f1s = np.divide(2 * y_list[idx] * x_list[idx], denominator, out=np.zeros_like(y_list[idx]),
                        where=denominator != 0)
        best_f1 = np.max(f1s)

        # UPDATE: Check if CI data was provided and format the label accordingly
        if f1_ci_list is not None and idx < len(f1_ci_list) and f1_ci_list[idx] is not None:
            lower, upper = f1_ci_list[idx]
            # label_str = '%s (AUC=%0.3f, F1=%0.3f [%0.3f-%0.3f])' % (model_name_list[idx], auc_val, best_f1, lower,
            #                                                              upper)
            label_str = '%s (F1=%0.3f)' % (model_name_list[idx], best_f1)
        else:
            # label_str = '%s (AUC=%0.3f, F1=%0.3f)' % (model_name_list[idx], auc_val, best_f1)
            label_str = '%s (F1=%0.3f)' % (model_name_list[idx], best_f1)

        plt.plot(x_list[idx], y_list[idx], color_list[idx],
                 label=label_str)

    f_list = [0.2, 0.4, 0.6, 0.8]
    for f in f_list:
        all_rec = [f / (2 - f) + idx * 0.01 for idx in range(100)]
        rec = [a for a in all_rec if a <= 1]
        pre_f = [f * r / (2 * r - f) for r in rec]
        plt.plot(rec, pre_f, color='gray', linestyle='dashed')
        plt.text(0.9, (f / (2 - f) + 0.01 + f / 30), "F1=%s" % (f))

    # 1. Grab the current handles and labels
    handles, labels = plt.gca().get_legend_handles_labels()

    # 2. Define exact desired order (using substrings that uniquely identify each label)
    desired_order = [
        'RAG-LLM',
        'Keyword-LLM',
        'RAG-XGBoost',
        'Rule-based Baseline',
        # 'SVM',
        # 'LR',
        # 'Rule 2',
        # 'RF'
    ]

    # 3. Create new lists to hold the sorted handles and labels
    sorted_handles = []
    sorted_labels = []

    # 4. Match the current labels to desired order
    for item in desired_order:
        for h, l in zip(handles, labels):
            if item in l:  # If the substring is in the label text
                sorted_handles.append(h)
                sorted_labels.append(l)
                break  # Found it, move to the next item in desired_order

    # 5. Apply the newly sorted legend
    plt.legend(sorted_handles, sorted_labels, loc='lower left')

    if scale_one:
        plt.xlim([0, 1.001])
        plt.ylim([0, 1.001])
        plt.plot([0, 1], [1, 0], color='gray', marker='.', linestyle='dashed', alpha=0.5)
        xtick = [(x + 0.) / 10 for x in range(0, 11)]
        ytick = xtick
        plt.xticks(xtick)
        plt.yticks(ytick)
    if log_y:
        plt.yscale('log')
    plt.ylabel(y_name)
    plt.xlabel(x_name)
    plt.grid()
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'prc.png'), dpi=400, bbox_inches='tight')
    else:
        plt.show()

def add_hardcoded_test_prc_points():
    """
    Replace the x (recall) and y (precision) coordinates here with the actual
    performance values from your test dataset for the standalone models.
    """
    # Example Test Data Points: plt.plot(Recall, Precision, ...)
    plt.plot(0.946, 0.909, marker='D', markersize=5, color='red', linestyle='None', label='RAG-LLM (F1=0.927)')
    plt.plot(0.946, 0.886, marker='^', markersize=5, color='lime', linestyle='None', label='Keyword-LLM (F1=0.915)')
    plt.plot(0.703, 0.693, marker='o', markersize=5, color='magenta', linestyle='None', label='Rule-based Baseline (F1=0.698)')
    # plt.plot(0.60, 0.65, marker='s', markersize=10, color='black', linestyle='None', label='Rule 2 (F1=placeholder)')


def plot_multi_curve_test(x_list, y_list, model_name_list, f1_ci_list=None, precalculated_f1_list=None, x_name="Recall", y_name="Precision",
                          scale_one=True, log_y=False, save_dir=None):
    plt.clf()
    plt.title('Dataset II: Precision-Recall Performance')

    model_num = len(x_list)
    color_list = ['tab:blue', 'tab:red', 'tab:orange', 'tab:green', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:olive',
                  'tab:cyan']

    if 'recall' in x_name.lower() and 'precision' in y_name.lower():
        add_hardcoded_test_prc_points()

    for idx in range(model_num):
        try:
            auc_val = metrics.auc(x_list[idx], y_list[idx])
        except ValueError:
            sort_idx = np.argsort(x_list[idx])
            auc_val = metrics.auc(x_list[idx][sort_idx], y_list[idx][sort_idx])

        # FIX: Check if a precalculated F1 score was provided for this model
        if precalculated_f1_list is not None and idx < len(precalculated_f1_list) and precalculated_f1_list[idx] is not None:
            display_f1 = precalculated_f1_list[idx]
        else:
            # Fallback to max F1 if no predefined list is passed
            denominator = y_list[idx] + x_list[idx]
            f1s = np.divide(2 * y_list[idx] * x_list[idx], denominator, out=np.zeros_like(y_list[idx]),
                            where=denominator != 0)
            display_f1 = np.max(f1s)

        # Apply the display_f1 to the label string
        if f1_ci_list is not None and idx < len(f1_ci_list) and f1_ci_list[idx] is not None:
            lower, upper = f1_ci_list[idx]
            label_str = '%s (F1=%0.3f)' % (model_name_list[idx], display_f1)
        else:
            label_str = '%s (F1=%0.3f)' % (model_name_list[idx], display_f1)

        plt.plot(x_list[idx], y_list[idx], color_list[idx], label=label_str)

    f_list = [0.2, 0.4, 0.6, 0.8]
    for f in f_list:
        all_rec = [f / (2 - f) + idx * 0.01 for idx in range(100)]
        rec = [a for a in all_rec if a <= 1]
        pre_f = [f * r / (2 * r - f) for r in rec]
        plt.plot(rec, pre_f, color='gray', linestyle='dashed')
        plt.text(0.9, (f / (2 - f) + 0.01 + f / 30), "F1=%s" % (f))

    handles, labels = plt.gca().get_legend_handles_labels()

    desired_test_order = [
        'RAG-LLM',
        'Keyword-LLM',
        'RAG-XGBoost',
        'Rule-based Baseline',
    ]

    sorted_handles = []
    sorted_labels = []

    for item in desired_test_order:
        for h, l in zip(handles, labels):
            if item in l:
                sorted_handles.append(h)
                sorted_labels.append(l)
                break

    plt.legend(sorted_handles, sorted_labels, loc='lower left')

    if scale_one:
        plt.xlim([0, 1.001])
        plt.ylim([0, 1.001])
        plt.plot([0, 1], [1, 0], color='gray', marker='.', linestyle='dashed', alpha=0.5)
        xtick = [(x + 0.) / 10 for x in range(0, 11)]
        ytick = xtick
        plt.xticks(xtick)
        plt.yticks(ytick)

    if log_y:
        plt.yscale('log')

    plt.ylabel(y_name)
    plt.xlabel(x_name)
    plt.grid()

    if save_dir:
        plt.savefig(os.path.join(save_dir, 'test_prc.png'), dpi=400, bbox_inches='tight')
    else:
        plt.show()

# def calculate_clf_p_r_f_acc_kappa(gold_label, pred_label, positive_id=1):
#     ## gold_label: numpy array, binary. size: (instance_num,)
#     ## pred_label: numpy array, binary, predict result. size: (instance_num,)
#     assert(gold_label.ndim==1)
#     if pred_label.ndim == 2:
#         pred_label = np.argmax(pred_label, axis=1)
#     elif pred_label.ndim > 2:
#         print("PRF calculation error: dimension of pred_label should <= 2")
#     gold_true = gold_label == positive_id
#     pred_true = pred_label == positive_id
#     gold_true_num = np.count_nonzero(gold_true)
#     pred_true_num = np.count_nonzero(pred_true)
#     right_pred_num = np.count_nonzero(gold_true == pred_true)
#     p = (right_pred_num+0.)/pred_true_num
#     r = (right_pred_num+0.)/gold_true_num
#     f = 2*p*r/(p+r)
#     acc = (np.sum(gold_label==pred_label)+0.)/gold_label.size
#     print("Gold: %s; Pred: %s; Right: %s"%(gold_true_num, pred_true_num, right_pred_num))
#     all_num = gold_label.size
#     gold_false_num = all_num-gold_true_num
#     pred_false_num = all_num - pred_true_num
#     pe = (gold_true_num*pred_true_num + gold_false_num*pred_false_num+0.)/(all_num*all_num)
#     kappa = (acc-pe)/(1-pe)
#     return p,r,f, acc, kappa
#
#
#
# def calculate_p_r_f_acc_kappa(gold_label, pred_label):
#     ## gold_label: numpy array, binary. size: (instance_numfe
#     ## pred_label: numpy array, binary, predict result. size: (instance_num,)
#     # print(gold_label)
#     # exit(0)
#     gold_true = np.sum(gold_label)
#     pred_true = np.sum(pred_label)
#     right_pred = np.sum(gold_label*pred_label)
#     p = (right_pred+0.)/pred_true
#     r = (right_pred+0.)/gold_true
#     f = 2*p*r/(p+r)
#     acc = (np.sum(gold_label==pred_label)+0.)/gold_label.shape[0]
#     print("Gold: %s; Pred: %s; Right: %s"%(gold_true, pred_true, right_pred))
#     all_num = gold_label.size
#     gold_false = all_num-gold_true
#     pred_false = all_num - pred_true
#     pe = (gold_true*pred_true + gold_false*pred_false+0.)/(all_num*all_num)
#     kappa = (acc-pe)/(1-pe)
#     return p,r,f, acc, kappa
#
#
# def calculate_roc_list(gold_label, pred_label_prob, positive_id):
#     ## gold_label: numpy array, binary. size: (instance_num,)
#     ## pred_label: numpy array, float probability, predict result. size: (instance_num, label_num)
#
#     pred = pred_label_prob[:, positive_id]
#     fpr, tpr, roc_thresholds = metrics.roc_curve(gold_label, pred, pos_label=positive_id)
#     auc = metrics.auc(fpr, tpr)
#     return fpr, tpr, roc_thresholds, auc
#
#
# def calculate_precision_recall_list(gold_label, pred_label_prob, positive_id):
#     ## gold_label: numpy array, binary. size: (instance_num,)
#     ## pred_label: numpy array, float probability, predict result. size: (instance_num, label_num)
#     pred = pred_label_prob[:, positive_id]
#     precision, recall, pr_thresholds = metrics.precision_recall_curve(gold_label, pred, pos_label=positive_id)
#     auprc = metrics.auc(recall, precision)
#     return precision, recall, pr_thresholds, auprc
#     # return precision, recall, pr_thresholds
#
#
# def plot_roc(fpr, tpr, auc, model_name="model", save_dir=None):
#     r''' plot roc curve for one model, based on different cut-off probabilities
#         Args:
#             fpr (numpy array): fpr value array
#             tpr (numpy array): tpr value array
#             auc (float): auc value
#             model_name (string): name of the model
#             save_dir (string): file directoy to be saved
#     '''
#     plt.title('ROC Curve')
#     plt.plot(fpr, tpr, 'b', label = '%s: AUC=%0.4f' % (model_name,auc))
#     plt.plot([0, 1], [0, 1],color='gray', marker='.', linestyle='dashed',alpha=0.5)
#     plt.legend(loc='best')
#     plt.xlim([0, 1.01])
#     plt.ylim([0, 1.01])
#     plt.ylabel('True Positive Rate (TPR)')
#     plt.xlabel('False Positive Rate (FPR)')
#     xtick = [(x+0.)/10 for x in range(0, 11)]
#     ytick = xtick
#     plt.xticks(xtick)
#     plt.yticks(ytick)
#     plt.grid()
#     if save_dir:
#         plt.savefig(save_dir)
#     else:
#         plt.show()
#     plt.close()
#
# def plot_multi_roc(fpr_list, tpr_list, auc_list, model_name_list, save_dir=None):
#     r''' plot roc curves for multiple models, based on different cut-off probabilities
#         Args:
#             fpr_list (list of numpy array): fpr value array
#             tpr_list (list of numpy array): tpr value array
#             auc_list (list of float): auc value
#             model_name_list (list of string): name of the model
#             save_dir (list of string): file directoy to be saved
#     '''
#     plt.title('ROC Curve')
#     model_num = len(fpr_list)
#     color_list = ['tab:blue', 'tab:red', 'tab:orange', 'tab:green', 'tab:purple', 'tab:brown', 'tab:pink',  'tab:olive', 'tab:cyan']
#     for idx in range(model_num):
#         plt.plot(fpr_list[idx], tpr_list[idx], color_list[idx], label = '%s: AUC=%0.4f' % (model_name_list[idx], auc_list[idx]))
#     plt.plot([0, 1], [0, 1],color='gray', marker='.', linestyle='dashed', alpha=0.5)
#     plt.legend(loc='best')
#     plt.xlim([0, 1.01])
#     plt.ylim([0, 1.01])
#     plt.ylabel('True Positive Rate (TPR)')
#     plt.xlabel('False Positive Rate (FPR)')
#     xtick = [(x+0.)/10 for x in range(0, 11)]
#     ytick = xtick
#     plt.xticks(xtick)
#     plt.yticks(ytick)
#     plt.grid()
#     if save_dir:
#         plt.savefig(save_dir)
#     else:
#         plt.show()
#
#
# def plot_precision_recall(precision, recall, model_name="model", save_dir=None):
#     r''' plot precision-recall for one model, based on different cut-off probabilities
#         Args:
#             precision (numpy array): precision value array
#             recall (numpy array): recall value array
#             model_name (string): name of the model
#             save_dir (string): file directoy to be saved
#     '''
#     plt.title('Precision-Recall Curve')
#     plt.plot(recall, precision, 'b', label = '%s' % (model_name))
#
#     f_list = [0.2, 0.4, 0.6, 0.8]
#     for f in f_list:
#         all_rec = [f/(2-f)+idx*0.01 for idx in range(100)]
#         rec = [a for a in all_rec if a <=1]
#         pre_f = [f*r/(2*r-f) for r in rec]
#         plt.plot(rec, pre_f,color='gray', linestyle='dashed')
#         plt.text(0.9,(f/(2-f)+0.01+f/30),"F1=%s"%(f))
#
#
#     plt.legend(loc='best')
#     plt.xlim([0, 1.01])
#     plt.ylim([0, 1.01])
#     plt.ylabel('Precision')
#     plt.xlabel('Recall')
#     xtick = [(x+0.)/10 for x in range(0, 11)]
#     ytick = xtick
#     plt.xticks(xtick)
#     plt.yticks(ytick)
#     plt.grid()
#     if save_dir:
#         plt.savefig(save_dir)
#     else:
#         plt.show()
#     plt.close()
#
#
# def plot_multi_precision_recall(precision_list, recall_list, model_name_list, save_dir=None):
#     r''' plot precision-recall for multiple models, based on different cut-off probabilities
#         Args:
#             precision_list (list of numpy array): precision value array
#             recall_list (list of numpy array): recall value array
#             model_name_list (list of string): name of the model
#             save_dir (string): file directoy to be saved
#     '''
#     plt.title('Precision-Recall Curve')
#     model_num = len(precision_list)
#     color_list = ['tab:blue', 'tab:red', 'tab:orange', 'tab:green', 'tab:purple', 'tab:brown', 'tab:pink',  'tab:olive', 'tab:cyan']
#     for idx in range(model_num):
#         plt.plot(recall_list[idx], precision_list[idx], color_list[idx], label = '%s' % (model_name_list[idx]))
#     f_list = [0.2, 0.4, 0.6, 0.8]
#     for f in f_list:
#         all_rec = [f/(2-f)+idx*0.01 for idx in range(100)]
#         rec = [a for a in all_rec if a <=1]
#         pre_f = [f*r/(2*r-f) for r in rec]
#         plt.plot(rec, pre_f,color='gray', linestyle='dashed')
#         plt.text(0.9,(f/(2-f)+0.01+f/30),"F1=%s"%(f))
#     plt.legend(loc='lower left')
#     plt.xlim([0, 1.01])
#     plt.ylim([0, 1.01])
#     plt.ylabel('Precision')
#     plt.xlabel('Recall')
#     xtick = [(x+0.)/10 for x in range(0, 11)]
#     ytick = xtick
#     plt.xticks(xtick)
#     plt.yticks(ytick)
#     plt.grid()
#     if save_dir:
#         plt.savefig(save_dir)
#     else:
#         plt.show()
#
#
# def plot_multi_curve(x_list, y_list, model_name_list, x_name="X_Name", y_name="Y_name", scale_one=True, log_y=False, save_dir=None):
#     r''' plot precision-recall for multiple models, based on different cut-off probabilities
#         Args:
#             x_list (list of numpy array): list of x label array
#             y_list (list of numpy array): list of y label array
#             model_name_list (list of string): name of the model
#             x_name (string): name of x axis
#             y_name (string): name of y axis
#             save_dir (string): file directoy to be saved
#     '''
#     plt.clf()
#     plt.title('%s-%s Curve'%(y_name, x_name))
#     model_num = len(x_list)
#     color_list = ['tab:blue', 'tab:red', 'tab:orange', 'tab:green', 'tab:purple', 'tab:brown', 'tab:pink',  'tab:olive', 'tab:cyan']
#     for idx in range(model_num):
#         plt.plot(x_list[idx], y_list[idx], color_list[idx], label = '%s' % (model_name_list[idx]))
#
#     f_list = [0.2, 0.4, 0.6, 0.8]
#     for f in f_list:
#         all_rec = [f/(2-f)+idx*0.01 for idx in range(100)]
#         rec = [a for a in all_rec if a <=1]
#         pre_f = [f*r/(2*r-f) for r in rec]
#         plt.plot(rec, pre_f,color='gray', linestyle='dashed')
#         plt.text(0.9,(f/(2-f)+0.01+f/30),"F1=%s"%(f))
#
#     plt.legend(loc='best')
#     if scale_one:
#         plt.xlim([0, 1.001])
#         plt.ylim([0, 1.001])
#         plt.plot([0, 1], [1, 0],color='gray', marker='.', linestyle='dashed',alpha=0.5)
#         xtick = [(x+0.)/10 for x in range(0, 11)]
#         ytick = xtick
#         plt.xticks(xtick)
#         plt.yticks(ytick)
#     if log_y:
#         plt.yscale('log')
#     plt.ylabel(y_name)
#     plt.xlabel(x_name)
#     plt.grid()
#     if save_dir:
#         plt.savefig(os.path.join(save_dir, 'prc.pdf'))
#     else:
#         plt.show()



if __name__ == '__main__':
    prec = [1, 0.9,0.8,0.4,0.2]
    rec = [0,0.3,0.7,0.9,1]
    plot_precision_recall(prec, rec)
