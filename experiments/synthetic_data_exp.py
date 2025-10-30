#!/usr/bin/env python3
# -*- coding: utf-8 -*-
###########################
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import pandas as pd
import json
import argparse

from metrics.pate.PATE_utils import convert_events_to_array_PATE
from experiments.utils_synthetic_exp import evaluate_all_metrics
from config.dqe_config import parameter_dict


class CustomEncoder(json.JSONEncoder):
    def encode(self, obj):
        if isinstance(obj, list):
            items = [self.encode(item) for item in obj]
            return "[\n" + ",\n".join(items) + "\n]"
        elif isinstance(obj, dict):
            items = [f'"{key}":{self.encode(value)}' for key, value in obj.items()]
            return "{" + ",".join(items) + "}"
        return super().encode(obj)

if __name__ == '__main__':

    ## ArgumentParser
    parser = argparse.ArgumentParser(description='Running DQE synthetic data experiments')
    parser.add_argument('--exp_name', type=str, default='over-counting tp')
    args = parser.parse_args()
    
    metric_name_list = [
        'VUS_ROC',
        'VUS_PR',
        'AUC',
        'AUC_PR',
        'PATE',
    
        "dqe",
        # "dqe_w_tq",
        # "dqe_w_fq_near",
        # "dqe_w_fq_distant",

        'Affliation F1score',
        'eTaPR_f1_score',
        'original_F1Score',
        'pa_k_score',
        'dtpa_f_score',
        'ls_f_score',
        'Rbased_f1score',
    
    ]
    
    name_dict = {
        "original_F1Score": "Original-F",
        "pa_k_score": "\%K-PA-F",
        "dtpa_f_score":"DTPA-F",
        "ls_f_score":"LS-F",
        "Rbased_f1score": "RF",
        "eTaPR_f1_score": "eTaF",
        "Affliation F1score": "AF",
    
        "dqe": "DQE",
        # "dqe_w_tq": "DQE_w_tq",
        # "dqe_w_fq_near": "DQE_w_fq_near",
        # "dqe_w_fq_distant": "DQE_w_fq_distant",

        "VUS_ROC": "VUS-ROC",
        "VUS_PR": "VUS-PR",
        "AUC": "AUC-ROC",
        "AUC_PR": "AUC-PR",
        "PATE": "PATE",
    }
    
    
    choose_metric_name_order_list = [
        "VUS-ROC", "VUS-PR", "AUC-ROC", "AUC-PR", "PATE",
        "Original-F",
        "\%K-PA-F",
        "RF", "eTaF", "AF",

        "DQE",
        "DQE_w_tq",
        "DQE_w_fq_near",
        "DQE_w_fq_distant",
        "DQE_w_fq",
    ]
    
    # over-counting tp
    if args.exp_name == "over-counting tp":
        json_file_name = "over-counting tp"
        vus_zone_size = e_buffer = d_buffer = near_single_side_range = 3
        adjust_dis = 20
        label_ranges = [[[23, 24], [31, 32], [39, 48]], [[39, 48]], [[23, 24], [31, 32]]] # The first is GT, and the following are detections.
        ts_len = 80
        choose_metric_name_order_list = [
            "Original-F", "AUC-ROC", "AUC-PR",
            "DTPA-F",
            "LS-F",
            "\%K-PA-F",
            "VUS-ROC", "VUS-PR", "PATE",
            "RF", "eTaF", "AF",
            "DQE",
        ]

    # tp timeliness
    if args.exp_name == "tp timeliness":
        ts_len = 300
        label_ranges = [[[140, 159]], [[143, 145]], [[154, 156]], [[143, 145], [148, 150]], [[143, 145], [154, 156]]]
        json_file_name = "tp timeliness"
        choose_metric_name_order_list = [
            "AF",
            "Original-F", "AUC-ROC", "AUC-PR",
            "\%K-PA-F",
            "VUS-ROC", "VUS-PR", "PATE",
            "RF", "eTaF",
            "DQE",
        ]
        vus_zone_size = e_buffer = d_buffer = near_single_side_range = 20
    

    # fp proximity
    if args.exp_name == "fp proximity":
        ts_len = 110

        label_ranges = [[[30, 39]], [[42, 43]], [[46, 47]], [[50, 51]], [[70, 71]], [[80, 81]], [[90, 91]]]
        json_file_name = "fp proximity"
        vus_zone_size = e_buffer = d_buffer = near_single_side_range = 20
    
        choose_metric_name_order_list = [
            "Original-F",
            "\%K-PA-F",
            "RF", "eTaF",
            "AUC-PR", "AUC-ROC",
            "VUS-ROC", "VUS-PR", "PATE",
            "AF",
            "DQE",
        ]
    
    # fp insensitivity or overevaluation of duration
    if args.exp_name == "fp insensitivity or overevaluation of duration":
        ts_len = 300
        label_ranges = [[[140, 159]], [[162, 164], [178, 180]], [[162, 164], [166, 168], [170, 172], [174, 176], [178, 180]]]
        json_file_name = "fp insensitivity or overevaluation of duration"

        choose_metric_name_order_list = [
            "PATE",
            "AUC-PR",
            "VUS-PR",
            "AF",
            "Original-F",
            "\%K-PA-F",
            "RF", "eTaF",
            "AUC-ROC", "VUS-ROC",
            "DQE",
        ]

        vus_zone_size = e_buffer = d_buffer = near_single_side_range = 20
    
    # fp duration overevaluation (af)
    if args.exp_name == "fp duration overevaluation (af)":
        ts_len = 38
        label_ranges = [[[29, 30], [35, 36]], [[25, 26], [35, 36]], [[29, 30], [34, 35]]]
        json_file_name = "fp duration overevaluation (af)"
        vus_zone_size = e_buffer = d_buffer = near_single_side_range = 3
    
        choose_metric_name_order_list = [
            "AF",
            "Original-F", "AUC-ROC", "AUC-PR",
            "\%K-PA-F",
            "VUS-ROC", "VUS-PR", "PATE",
            "RF", "eTaF",
            "DQE",
        ]

    ordered_num = len(label_ranges)

    label_array_list = []
    res_data = []
    
    parameter_dict_new = parameter_dict
    parameter_dict_new["near_single_side_range"] = near_single_side_range
    
    for i, single_range in enumerate(label_ranges):
        label_array = convert_events_to_array_PATE(single_range, time_series_length=ts_len)
        label_array_list.append(label_array)
        score_list_simple = evaluate_all_metrics(label_array,
                                                 label_array_list[0],
                                                 vus_zone_size,
                                                 e_buffer,
                                                 d_buffer,
                                                 parameter_dict=parameter_dict_new,
                                                 near_single_side_range=near_single_side_range)

        selected_dict = {}
        for j, paint_name in enumerate(metric_name_list):
            value = score_list_simple[paint_name]
            selected_dict[paint_name] = value
        res_data.append(selected_dict)
    
    file_path = "dataset/metric_cal_res_windows/synthetic_data_res/" + json_file_name + ".json"
    

    res_data_dict = {}
    
    new_res_data = []
    for i, pred_score_dict in enumerate(res_data):
        new_score_dict = {}
        for j, (metric_name, metric_score) in enumerate(pred_score_dict.items()):
            if name_dict[metric_name] in choose_metric_name_order_list:
                new_score_dict[name_dict[metric_name]] = metric_score
        new_res_data.append(new_score_dict)

    directory = os.path.dirname(file_path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    with open(file_path, "w", encoding="utf-8") as file:
        encoder = CustomEncoder()
        json_str = encoder.encode(new_res_data)
        file.write(json_str)
    
    
    reorder_new_res_data = []
    for i, pred_score_dict in enumerate(new_res_data):
        reorder_single_dict = {}
        for j, choose_metric_name in enumerate(choose_metric_name_order_list):
            find_pred = pred_score_dict[choose_metric_name]
    
            reorder_single_dict[choose_metric_name] = find_pred
        reorder_new_res_data.append(reorder_single_dict)
    
    new_res_data = reorder_new_res_data
    
    new_res_data = new_res_data[1:]
    
    df = pd.DataFrame(new_res_data)
    
    print("res_data", new_res_data)
