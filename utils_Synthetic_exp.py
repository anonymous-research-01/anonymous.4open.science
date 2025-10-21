#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math
import sys
import os

import numpy as np

from meata import meata_f1, auc_pr_meata, meata_f1_first,meata_auc_pr
from config.meata_config import parameter_dict

try:
    # This will work when the script is run as a file
    current_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # Use an alternative method to set the current_dir when __file__ is not defined
    current_dir = os.getcwd()  # Gets the current working directory

project_root = os.path.dirname(os.path.dirname(current_dir))  # Two levels up
sys.path.append(project_root)


from sklearn.metrics import f1_score
from metrics.f1_score_f1_pa import * # used for PA-F1 
from metrics.AUC import * # Range AUC
from metrics.affiliation.generics import convert_vector_to_events  #Affiliation
from metrics.affiliation.metrics import pr_from_events #Affiliation
from metrics.vus.models.feature import Window #VUS
from metrics.vus.metrics import get_range_vus_roc #VUS
from metrics.eTaPR_pkg import *
from metrics.eTaPR_pkg import f1_score_etapr
from metrics.AUCs_Compute import *
from metrics.Range_Based_PR import *
from metrics.metrics_pa import PointAdjustKPercent

from pate.PATE_metric import PATE
from pate.PATE_utils import convert_events_to_array_PATE, categorize_predicted_ranges_with_ids, \
    convert_vector_to_events_PATE

from evaluation.metrics import basic_metricor, generate_curve


def evaluate_all_metrics(pred, labels, vus_zone_size = 20, e_buffer=20, d_buffer=20,pred_case_id=None):
    # pred, labels
    # predicted_array, label_array
    grader = basic_metricor()

    #Affliation
    events_pred = convert_vector_to_events(pred)
    events_gt = convert_vector_to_events(labels)
    Trange = (0, len(pred))
    affiliation = pr_from_events(events_pred, events_gt, Trange)

    # Vus
    # print("pred",pred)
    # print("labels",labels)
    # print("vus_zone_size",vus_zone_size)
    vus_results = get_range_vus_roc(pred, labels, vus_zone_size,version="v1")
    vus_results_new = get_range_vus_roc(pred, labels, vus_zone_size,version="v2")
    # print("vus_results",vus_results)
    # print("vus_results_new",vus_results_new)


    # Auc
    # print("pred",pred)
    # print("labels",labels)
    auc_result = compute_auc(labels, pred)
    auc_pr = compute_auprc(labels, pred)
    # print("auc_result",auc_result)
    # print("auc_pr",auc_pr)

    AUC_ROC = grader.metric_ROC(labels, score=pred)
    AUC_PR = grader.metric_PR(labels, score=pred)

    # print("AUC_ROC",AUC_ROC)
    # print("AUC_PR",AUC_PR)

    # Ours
    pate = PATE(labels, pred, e_buffer, d_buffer, n_jobs=1, num_splits_MaxBuffer=1, include_zero=False,
                binary_scores=False)
    pate_f1 = PATE(labels, pred, e_buffer, d_buffer, Big_Data=False, n_jobs=1, num_splits_MaxBuffer=1,
                   include_zero=False, binary_scores=True)



    

    #R-based
    print("pred",pred)
    print("labels",labels)
    if events_pred== []:
        Rbased_precision, Rbased_recall, Rbased_f1_score = 0,0,0
    else:
        Rbased_precision, Rbased_recall, Rbased_f1_score = get_F1Score_RangeBased(labels, pred)
    print("Rbased_f1_score",Rbased_f1_score)
    RF1 = grader.metric_RF1(labels, score=pred, preds=pred)
    print("RF1",RF1)

    #eTaPR
    if events_pred== []:
        eTaPR_precision, eTaPR_recall, eTaPR_f1_score = 0,0,0
    else:
        eTaPR_precision, eTaPR_recall, eTaPR_f1_score = f1_score_etapr.get_eTaPR_fscore(labels, pred, theta_p = 0.5, theta_r = 0.01, delta=0) #Default Settings from the original paper

    #Standard Original F1-Score
    original_F1Score = f1_score(labels, pred)


    # meata_f1
    meata_f1_proportion,meata_f1_proportion_no_random,\
        meata_f1_w0,coefficient_meata_f1_w0,meata_f1_w1,coefficient_meata_f1_w1,_ = meata_f1(labels, pred, binary_preds=True,parameter_dict=parameter_dict, cal_mode="proportion")
    import copy
    # parameter_dict_copy = copy.deepcopy(parameter_dict)
    # parameter_dict_copy1 = copy.deepcopy(parameter_dict)
    # parameter_dict_copy["parameter_eta"] = 0
    # meata_f1_proportion_w0,meata_f1_proportion_no_random_w0,meata_f1_w0,coefficient_meata_f1_w0,meata_f1_w1,coefficient_meata_f1_w1 = meata_f1(labels, pred, parameter_dict=parameter_dict_copy, cal_mode="proportion")
    # parameter_dict_copy1["parameter_eta"] = 1
    # meata_f1_proportion_w1,meata_f1_proportion_no_random_w1,meata_f1_w0,coefficient_meata_f1_w0,meata_f1_w1,coefficient_meata_f1_w1 = meata_f1(labels, pred, parameter_dict=parameter_dict_copy1, cal_mode="proportion")
    parameter_dict_copy6 = copy.deepcopy(parameter_dict)
    parameter_dict_copy6["parameter_eta"] = 1/2
    meata_f1_proportion_w1_2,meata_f1_proportion_no_random_w1_2,_,_,_,_,_ = meata_f1(labels, pred, binary_preds=True,parameter_dict=parameter_dict_copy6, cal_mode="proportion")

    #
    meata_f1_detection,meata_f1_detection_no_random,_,_,_,_,_ = meata_f1(labels, pred, binary_preds=True,parameter_dict=parameter_dict, cal_mode="detection")

    # meata_f1_score_first
    parameter_dict_copy2 = copy.deepcopy(parameter_dict)
    # parameter_dict_copy2["parameter_eta"] = 2*(2**(1/2) -1)
    # parameter_dict_copy2["parameter_eta"] = 2*(2**(1/2) -1)
    parameter_dict_copy2["parameter_eta"] = 1/2
    meata_f1_score_first,meata_f1_score_first_no_random,_,_,_,_ = meata_f1_first(labels, pred, parameter_dict=parameter_dict_copy2, pred_case_id=pred_case_id,cal_mode="proportion")
    parameter_dict_copy3 = copy.deepcopy(parameter_dict)
    parameter_dict_copy3["parameter_eta"] = 0
    meata_f1_score_first_w0,meata_f1_score_first_no_random_w0,_,_,_,_ = meata_f1_first(labels, pred, parameter_dict=parameter_dict_copy3, pred_case_id=pred_case_id,cal_mode="proportion")
    parameter_dict_copy4 = copy.deepcopy(parameter_dict)
    parameter_dict_copy4["parameter_eta"] = 1
    meata_f1_score_first_w1,meata_f1_score_first_no_random_w1,_,_,_,_ = meata_f1_first(labels, pred, parameter_dict=parameter_dict_copy4, pred_case_id=pred_case_id,cal_mode="proportion")

    # meata_f1_score_first_d
    parameter_dict_copy5 = copy.deepcopy(parameter_dict)
    parameter_dict_copy5["parameter_eta"] = 2*(2**(1/2) -1)
    meata_f1_score_first_d,meata_f1_score_first_d_no_random,_,_,_,_ = meata_f1_first(labels, pred, parameter_dict=parameter_dict_copy5, pred_case_id=pred_case_id,cal_mode="detection")


    # meata

    # meata_proportion,meata_proportion_no_random = auc_pr_meata(labels, pred,Big_Data=True,cal_mode="proportion")

    # meata_detection,meata_detection_no_random = auc_pr_meata(labels, pred,Big_Data=False,cal_mode="detection")




    #Point-Adj
    # 会改变pred的值
    pa_accuracy, pa_precision, pa_recall, pa_f_score = get_adjust_F1PA(pred, labels) #Effect on others!!

    score_list_simple = {
                   "original_F1Score":original_F1Score, 

                   # "pa_precision":pa_precision,
                   # "pa_recall":pa_recall,
                   "pa_f_score":pa_f_score,
                  
                   # "Rbased_precision":Rbased_precision,
                   # "Rbased_recall":Rbased_recall,
                   "Rbased_f1score":Rbased_f1_score, 
                  
                  # "eTaPR_precision":eTaPR_precision,
                  # "eTaPR_recall":eTaPR_recall,
                  "eTaPR_f1_score":eTaPR_f1_score,  
                  
                  # "Affiliation precision": affiliation['precision'],
                  # "Affiliation recall": affiliation['recall'],
                  "Affliation F1score":  get_f_score(affiliation['precision'],affiliation['recall'] ),
                  
                  "VUS_ROC": vus_results["VUS_ROC"],
                  "VUS_PR": vus_results["VUS_PR"],
                  
                  "AUC":auc_result, 
                   "AUC_PR":auc_pr,
                  
                  "PATE":pate,
                  "PATE-F1":pate_f1,

                "meata_f1_proportion": meata_f1_proportion,
                "meata_f1_detection": meata_f1_detection,
                # "meata_proportion":meata_proportion,
                # "meata_detection":meata_detection,


                # "meata_proportion_no_random": meata_proportion_no_random,
                # "meata_detection_no_random": meata_detection_no_random,
                "meata_f1_proportion_no_random": meata_f1_proportion_no_random,
                "meata_f1_proportion_no_random_w0": meata_f1_w0,
                "meata_f1_proportion_no_random_w1": meata_f1_w1,
                "meata_f1_proportion_no_random_w1_2": meata_f1_proportion_no_random_w1_2,

                "meata_f1_detection_no_random": meata_f1_detection_no_random,


                "meata_f1_score_first": meata_f1_score_first,
                "meata_f1_score_first_no_random": meata_f1_score_first_no_random,
                "meata_f1_score_first_no_random_w0": meata_f1_score_first_no_random_w0,
                "meata_f1_score_first_no_random_w1": meata_f1_score_first_no_random_w1,

                "meata_f1_score_first_d": meata_f1_score_first_d,
                "meata_f1_score_first_d_no_random": meata_f1_score_first_d_no_random,
                  }

    for key in score_list_simple:
        score_list_simple[key] = round(score_list_simple[key], 2)
    
    # return score_list, score_list_simple
    return score_list_simple


def evaluate_all_metrics_v2(pred, labels, vus_zone_size=20, e_buffer=20, d_buffer=20, pred_case_id=None,\
                            parameter_near_single_side_range=None,parameter_dict_new=None,max_ia_distant_length=-1):
    # pred, labels
    # predicted_array, label_array

    window_length = len(labels)
    grader = basic_metricor()


    # Affliation
    # events_pred = convert_vector_to_events(pred)
    # events_gt = convert_vector_to_events(labels)
    # Trange = (0, len(pred))
    # affiliation = pr_from_events(events_pred, events_gt, Trange)

    Affiliation_F = grader.metric_Affiliation(labels, score=pred, preds=pred)

    # Vus
    # 0.24 0.57
    # 0.18 0.50
    # 0.18 0.50
    # 0.18 0.47
    # 0.18 0.47
    # 0.18 0.47
    # 0.18 0.47
    # 0.18 0.41
    # vus_results = get_range_vus_roc(pred, labels, vus_zone_size)
    # VUS_ROC,VUS_PR = vus_results["VUS_ROC"],vus_results["VUS_PR"]
    _, _, _, _, _, _, VUS_ROC, VUS_PR = generate_curve(label=labels, score=pred, \
                                                       slidingWindow=vus_zone_size)

    # Auc
    # auc_result = compute_auc(labels, pred)
    # auc_pr = compute_auprc(labels, pred)

    AUC_ROC = grader.metric_ROC(labels, score=pred)
    AUC_PR = grader.metric_PR(labels, score=pred)

    # Ours
    pate = PATE(labels, pred, e_buffer, d_buffer, n_jobs=1, num_splits_MaxBuffer=1, include_zero=False,
                binary_scores=False)
    pate_f1 = PATE(labels, pred, e_buffer, d_buffer, Big_Data=False, n_jobs=1, num_splits_MaxBuffer=1,
                   include_zero=False, binary_scores=True)

    # R-based
    # if events_pred == []:
    #     Rbased_precision, Rbased_recall, Rbased_f1_score = 0, 0, 0
    # else:
    #     Rbased_precision, Rbased_recall, Rbased_f1_score = get_F1Score_RangeBased(labels, pred)
    RF1 = grader.metric_RF1(labels, score=pred, preds=pred)


    # eTaPR
    # if events_pred == []:
    #     eTaPR_precision, eTaPR_recall, eTaPR_f1_score = 0, 0, 0
    # else:
    #     eTaPR_precision, eTaPR_recall, eTaPR_f1_score = f1_score_etapr.get_eTaPR_fscore(labels, pred, theta_p=0.5,
    #                                                                                     theta_r=0.01,
    #                                                                                     delta=0)  # Default Settings from the original paper
    eTaPR_F1 = grader.metric_eTaPR_F1(labels, score=pred, preds=pred)


    # Standard Original F1-Score
    # original_F1Score = f1_score(labels, pred)
    PointF1 = grader.metric_PointF1(labels, score=pred, preds=pred)
    # parameter_dict_new = parameter_dict
    # parameter_dict_new["parameter_near_single_side_range"] = parameter_near_single_side_range
    # meata_f1
    # final_meata_f1,meata_f1, \
    #             meata_f1_w_gt,meata_f1_w_near_ngt,meata_f1_w_distant_ngt,meata_f1_w_ngt
    final_meata_f1, meata_f1_score,\
        meata_f1_w_gt, meata_f1_w_near_ngt, meata_f1_w_distant_ngt,meata_f1_w_ngt = meata_f1(labels,
                                                                                             pred,
                                                                                             binary_preds=True,
                                                                                             parameter_dict=parameter_dict_new,
                                                                                             pred_case_id=pred_case_id,
                                                                                             # max_ia_distant_length=max_ia_distant_length,
                                                                                             f1_type="row_add",
                                                                                             cal_mode="proportion")

    final_meata, meata, meata_w_gt, meata_w_near_ngt, meata_w_distant_ngt, meata_w_ngt = meata_auc_pr(labels,
                                                                                                      pred,
                                                                                                      output=pred,
                                                                                                      parameter_dict=parameter_dict,
                                                                                                      cal_mode="proportion")
    d = 1
    import copy
    # parameter_dict_copy = copy.deepcopy(parameter_dict)
    # parameter_dict_copy1 = copy.deepcopy(parameter_dict)
    # parameter_dict_copy["parameter_eta"] = 0
    # meata_f1_proportion_w0,meata_f1_proportion_no_random_w0,meata_f1_w0,coefficient_meata_f1_w0,meata_f1_w1,coefficient_meata_f1_w1 = meata_f1(labels, pred, parameter_dict=parameter_dict_copy, cal_mode="proportion")
    # parameter_dict_copy1["parameter_eta"] = 1
    # meata_f1_proportion_w1,meata_f1_proportion_no_random_w1,meata_f1_w0,coefficient_meata_f1_w0,meata_f1_w1,coefficient_meata_f1_w1 = meata_f1(labels, pred, parameter_dict=parameter_dict_copy1, cal_mode="proportion")
    # parameter_dict_copy6 = copy.deepcopy(parameter_dict)
    # parameter_dict_copy6["parameter_eta"] = 1 / 2
    # meata_f1_proportion_w1_2, meata_f1_proportion_no_random_w1_2, _, _, _, _, _ = meata_f1(labels, pred,
    #                                                                                        binary_preds=True,
    #                                                                                        parameter_dict=parameter_dict_copy6,
    #                                                                                        cal_mode="proportion")

    #
    # meata_f1_detection, meata_f1_detection_no_random, _, _, _, _, _ = meata_f1(labels, pred, binary_preds=True,
    #                                                                            parameter_dict=parameter_dict,
    #                                                                            cal_mode="detection")

    # meata_f1_score_first
    # parameter_dict_copy2 = copy.deepcopy(parameter_dict)
    # # parameter_dict_copy2["parameter_eta"] = 2*(2**(1/2) -1)
    # # parameter_dict_copy2["parameter_eta"] = 2*(2**(1/2) -1)
    # parameter_dict_copy2["parameter_eta"] = 1 / 2
    # meata_f1_score_first, meata_f1_score_first_no_random, _, _, _, _ = meata_f1_first(labels, pred,
    #                                                                                   parameter_dict=parameter_dict_copy2,
    #                                                                                   pred_case_id=pred_case_id,
    #                                                                                   cal_mode="proportion")
    # parameter_dict_copy3 = copy.deepcopy(parameter_dict)
    # parameter_dict_copy3["parameter_eta"] = 0
    # meata_f1_score_first_w0, meata_f1_score_first_no_random_w0, _, _, _, _ = meata_f1_first(labels, pred,
    #                                                                                         parameter_dict=parameter_dict_copy3,
    #                                                                                         pred_case_id=pred_case_id,
    #                                                                                         cal_mode="proportion")
    # parameter_dict_copy4 = copy.deepcopy(parameter_dict)
    # parameter_dict_copy4["parameter_eta"] = 1
    # meata_f1_score_first_w1, meata_f1_score_first_no_random_w1, _, _, _, _ = meata_f1_first(labels, pred,
    #                                                                                         parameter_dict=parameter_dict_copy4,
    #                                                                                         pred_case_id=pred_case_id,
    #                                                                                         cal_mode="proportion")

    # meata_f1_score_first_d
    # parameter_dict_copy5 = copy.deepcopy(parameter_dict)
    # parameter_dict_copy5["parameter_eta"] = 2 * (2 ** (1 / 2) - 1)
    # meata_f1_score_first_d, meata_f1_score_first_d_no_random, _, _, _, _ = meata_f1_first(labels, pred,
    #                                                                                       parameter_dict=parameter_dict_copy5,
    #                                                                                       pred_case_id=pred_case_id,
    #                                                                                       cal_mode="detection")

    # meata

    # meata_proportion,meata_proportion_no_random = auc_pr_meata(labels, pred,Big_Data=True,cal_mode="proportion")

    # meata_detection,meata_detection_no_random = auc_pr_meata(labels, pred,Big_Data=False,cal_mode="detection")

    # Point-Adj
    # 会改变pred的值
    # pa_accuracy, pa_precision, pa_recall, pa_f_score = get_adjust_F1PA(pred, labels)  # Effect on others!!
    labels_ranges = convert_vector_to_events_PATE(labels)
    pred_ranges = convert_vector_to_events_PATE(pred)
    try:
        pa_k = PointAdjustKPercent(window_length, labels_ranges, pred_ranges)
        pa_k_score = pa_k.get_score()
    except:
        # d= 1
        pa_k_score = np.NAN
    # metric(
    #     length, anomalies[0], predicted_anomalies
    # )
    PointF1PA = grader.metric_PointF1PA(labels, score=pred, preds=pred)

    score_list_simple = {
        "original_F1Score": PointF1,

        # "pa_precision":pa_precision,
        # "pa_recall":pa_recall,
        "pa_f_score": PointF1PA,
        "pa_k_score": pa_k_score,

        # "Rbased_precision":Rbased_precision,
        # "Rbased_recall":Rbased_recall,
        "Rbased_f1score": RF1,

        # "eTaPR_precision":eTaPR_precision,
        # "eTaPR_recall":eTaPR_recall,
        "eTaPR_f1_score": eTaPR_F1,

        # "Affiliation precision": affiliation['precision'],
        # "Affiliation recall": affiliation['recall'],
        "Affliation F1score": Affiliation_F,

        # "VUS_ROC": vus_results["VUS_ROC"],
        # "VUS_PR": vus_results["VUS_PR"],

        "VUS_ROC": VUS_ROC,
        "VUS_PR": VUS_PR,

        "AUC": AUC_ROC,
        "AUC_PR": AUC_PR,

        "PATE": pate,
        "PATE-F1": pate_f1,
        # "meata_f1_proportion": meata_f1_score,
        # "meata_f1_detection": 0,
        # "meata_proportion":meata_proportion,
        # "meata_detection":meata_detection,

        # "meata_proportion_no_random": meata_proportion_no_random,
        # "meata_detection_no_random": meata_detection_no_random,
        # "final_meata_f1": final_meata_f1,
        # "meata_f1_w_gt": meata_f1_w_gt,
        # "meata_f1_w_near_ngt": meata_f1_w_near_ngt,
        # "meata_f1_w_distant_ngt": meata_f1_w_distant_ngt,

        # "meata_f1_detection_no_random": 0,
        #
        # "meata_f1_score_first": 0,
        # "meata_f1_score_first_no_random": 0,
        # "meata_f1_score_first_no_random_w0": 0,
        # "meata_f1_score_first_no_random_w1": 0,
        #
        # "meata_f1_score_first_d": 0,
        # "meata_f1_score_first_d_no_random": 0,

        # final_meata, meata, meata_w_gt, meata_w_near_ngt, meata_w_distant_ngt, meata_w_ngt

        "meata_f1_proportion": meata_f1_score,
        "meata_f1_detection": 0,

        "final_meata": final_meata,
        "meata": meata,
        "meata_w_gt": meata_w_gt,
        "meata_w_near_ngt": meata_w_near_ngt,
        "meata_w_distant_ngt": meata_w_distant_ngt,
        "meata_w_ngt": meata_w_ngt,
    }

    for key in score_list_simple:
        score_list_simple[key] = round(score_list_simple[key], 2)

    # return score_list, score_list_simple
    return score_list_simple


def synthetic_generator(label_anomaly_ranges, predicted_ranges, vus_zone_size = 20, e_buffer = 20, d_buffer = 20, time_series_length = 500):
    """
    Runs a synthetic data experiment given label and prediction ranges.
    
    Parameters:
    - label_anomaly_ranges: List of [start, end] ranges for actual anomalies.
    - predicted_ranges: List of [start, end] ranges for detected anomalies.
    - time_series_length: Total length of the time series.
    - vus_zone_size: Size of the VUS method buffer zone.
    - e_buffer: Eaely prebuffer size.
    - d_buffer: Delayed postbuffer size.
    
    Returns:
    - A dictionary containing the categorized ranges with IDs, predicted array, and label array.
    """
    categorized_ranges_with_ids = categorize_predicted_ranges_with_ids(
        predicted_ranges, label_anomaly_ranges, e_buffer, d_buffer, time_series_length)
    
    predicted_array = convert_events_to_array_PATE(predicted_ranges, time_series_length)
    label_array = convert_events_to_array_PATE(label_anomaly_ranges, time_series_length)
    
    return {
        "categorized_ranges_with_ids": categorized_ranges_with_ids,
        "predicted_array": predicted_array,
        "label_array": label_array
    }




