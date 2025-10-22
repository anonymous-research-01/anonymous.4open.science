#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import os

from DQE import DQE
from config.meata_config import parameter_dict

try:
    # This will work when the script is run as a file
    current_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # Use an alternative method to set the current_dir when __file__ is not defined
    current_dir = os.getcwd()  # Gets the current working directory

project_root = os.path.dirname(os.path.dirname(current_dir))  # Two levels up
sys.path.append(project_root)


from metrics.metrics_pa import PointAdjustKPercent

from pate.PATE_metric import PATE
from pate.PATE_utils import convert_events_to_array_PATE, categorize_predicted_ranges_with_ids, \
    convert_vector_to_events_PATE

from evaluation.metrics import basic_metricor, generate_curve




def evaluate_all_metrics(pred, labels, vus_zone_size=20, e_buffer=20, d_buffer=20, pred_case_id=None, \
                         parameter_near_single_side_range=None, parameter_dict_new=None, max_ia_distant_length=-1):
    window_length = len(labels)
    grader = basic_metricor()


    # Affliation

    Affiliation_F = grader.metric_Affiliation(labels, score=pred, preds=pred)

    # Vus
    _, _, _, _, _, _, VUS_ROC, VUS_PR = generate_curve(label=labels, score=pred, \
                                                       slidingWindow=vus_zone_size)

    # Auc
    AUC_ROC = grader.metric_ROC(labels, score=pred)
    AUC_PR = grader.metric_PR(labels, score=pred)

    # Ours
    pate = PATE(labels, pred, e_buffer, d_buffer, n_jobs=1, num_splits_MaxBuffer=1, include_zero=False,
                binary_scores=False)
    pate_f1 = PATE(labels, pred, e_buffer, d_buffer, Big_Data=False, n_jobs=1, num_splits_MaxBuffer=1,
                   include_zero=False, binary_scores=True)

    # R-based
    RF1 = grader.metric_RF1(labels, score=pred, preds=pred)


    # eTaPR
    eTaPR_F1 = grader.metric_eTaPR_F1(labels, score=pred, preds=pred)


    # Standard Original F1-Score
    PointF1 = grader.metric_PointF1(labels, score=pred, preds=pred)
    meata, meata_w_gt, meata_w_near_ngt, meata_w_distant_ngt, meata_w_ngt = DQE(labels,
                                                                                             pred,
                                                                                             output=pred,
                                                                                             parameter_dict=parameter_dict,
                                                                                             cal_mode="proportion")

    labels_ranges = convert_vector_to_events_PATE(labels)
    pred_ranges = convert_vector_to_events_PATE(pred)
    pa_k = PointAdjustKPercent(window_length, labels_ranges, pred_ranges)
    pa_k_score = pa_k.get_score()

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


        "VUS_ROC": VUS_ROC,
        "VUS_PR": VUS_PR,

        "AUC": AUC_ROC,
        "AUC_PR": AUC_PR,

        "PATE": pate,
        "PATE-F1": pate_f1,

        "final_meata": meata,
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




