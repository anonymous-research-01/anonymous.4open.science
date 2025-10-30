#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dqe.dqe_metric import DQE
from config.dqe_config import parameter_dict
from metrics.metrics_pa import PointAdjustKPercent, DelayThresholdedPointAdjust, LatencySparsityAware
from metrics.pate.PATE_metric import PATE
from metrics.pate.PATE_utils import convert_events_to_array_PATE, categorize_predicted_ranges_with_ids, \
    convert_vector_to_events_PATE
from evaluation.metrics import basic_metricor, generate_curve


def evaluate_all_metrics(pred, labels, vus_zone_size=20, e_buffer=20, d_buffer=20, parameter_dict=parameter_dict):
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

    # PATE
    pate = PATE(labels, pred, e_buffer, d_buffer, n_jobs=1, num_splits_MaxBuffer=1, include_zero=False,
                binary_scores=False)
    # pate_f1 = PATE(labels, pred, e_buffer, d_buffer, Big_Data=False, n_jobs=1, num_splits_MaxBuffer=1,
    #                include_zero=False, binary_scores=True)

    # R-based
    RF1 = grader.metric_RF1(labels, score=pred, preds=pred)

    # eTaPR
    eTaPR_F1 = grader.metric_eTaPR_F1(labels, score=pred, preds=pred)

    # Standard Original F1-Score
    PointF1 = grader.metric_PointF1(labels, score=pred, preds=pred)

    # DQE
    # dqe, dqe_w_tq, dqe_w_fq_near, dqe_w_distant_ngt, dqe_w_fq = DQE(labels,pred,parameter_dict=parameter_dict)
    dqe = DQE(labels, pred, parameter_dict=parameter_dict)

    labels_ranges = convert_vector_to_events_PATE(labels)
    pred_ranges = convert_vector_to_events_PATE(pred)

    # %K-PA-F
    pa_k = PointAdjustKPercent(window_length, labels_ranges, pred_ranges)
    pa_k_score = pa_k.get_score()

    # DTPA-F
    dtpa_f = DelayThresholdedPointAdjust(window_length, labels_ranges, pred_ranges)
    dtpa_f_score = dtpa_f.get_score()

    # LS-F
    ls_f = LatencySparsityAware(window_length, labels_ranges, pred_ranges)
    ls_f_score = ls_f.get_score()

    # PA-F
    # PointF1PA = grader.metric_PointF1PA(labels, score=pred, preds=pred)

    score_list_simple = {
        "original_F1Score": PointF1,

        # "pa_precision":pa_precision,
        # "pa_recall":pa_recall,
        # "pa_f_score": PointF1PA,
        "pa_k_score": pa_k_score,
        "dtpa_f_score": dtpa_f_score,
        "ls_f_score": ls_f_score,

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
        # "PATE-F1": pate_f1,

        "dqe": dqe,
        # "dqe_w_tq": dqe_w_tq,
        # "dqe_w_fq_near": dqe_w_fq_near,
        # "dqe_w_fq_distant": dqe_w_distant_ngt,
        # "dqe_w_fq": dqe_w_fq,
    }

    for key in score_list_simple:
        score_list_simple[key] = round(score_list_simple[key], 2)

    return score_list_simple


def synthetic_generator(label_anomaly_ranges, predicted_ranges, vus_zone_size=20, e_buffer=20, d_buffer=20,
                        time_series_length=500):
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




