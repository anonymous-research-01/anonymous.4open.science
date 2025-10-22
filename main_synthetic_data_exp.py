#!/usr/bin/env python3
# -*- coding: utf-8 -*-
###########################
import matplotlib.pyplot as plt

from pate.PATE_utils import convert_events_to_array_PATE, convert_vector_to_events_PATE
from utils_Synthetic_exp import evaluate_all_metrics, synthetic_generator, evaluate_all_metrics
from config.meata_config import parameter_dict


paint_name_list = [
    'VUS_ROC',
    'VUS_PR',
    'AUC',
    'AUC_PR',
    'PATE',

    "final_meata",
    "meata",
    "meata_w_gt",
    "meata_w_near_ngt",
    "meata_w_distant_ngt",
    "meata_w_ngt",

    'Affliation F1score',
    'eTaPR_f1_score',
    'original_F1Score',
    'pa_f_score',
    'pa_k_score',
    'dtpa_f_score',
    'ls_f_score',
    'Rbased_f1score',

]

test_flag = "no test"
test_flag1 = "test"


name_dict = {
    "original_F1Score": "Original-F",
    "pa_f_score": "PA-F",
    "pa_k_score": "\%K-PA-F",
    "dtpa_f_score":"DTPA-F",
    "ls_f_score":"LS-F",
    "Rbased_f1score": "RF",
    "eTaPR_f1_score": "eTaF",
    "Affliation F1score": "AF",

    "final_meata": "DQE",
    "meata": "RQ_DQE",
    "meata_w_gt": "DQE_w_gt",
    "meata_w_near_ngt": "DQE_w_near_ngt",
    "meata_w_distant_ngt": "DQE_w_distant_ngt",
    "meata_w_ngt": "DQE_w_ngt",

    "VUS_ROC": "VUS-ROC",
    "VUS_PR": "VUS-PR",
    "AUC": "AUC-ROC",
    "AUC_PR": "AUC-PR",
    "PATE": "PATE",
}


choose_metric_name_order_list = [
    "VUS-ROC", "VUS-PR", "AUC-ROC", "AUC-PR", "PATE",
    "Original-F", "PA-F", "\%K-PA-F", "RF", "eTaF", "AF",
    "DLBE",
    "DLBE_w_gt",
    "DLBE_w_near_ngt",
    "DLBE_w_distant_ngt",

    "final_meata",
    "meata",
    "meata_w_gt",
    "meata_w_near_ngt",
    "meata_w_distant_ngt",
    "meata_w_ngt",

    "mf1_f",
    "mf1_fnr",
    "mf1_fnr_w0",
    "mf1_fnr_w1",
    "mf1_first_d",
    "mf1_first_d_nr",
]

# over-counting tp
if test_flag1 == "test":
    pass
    json_file_name = "figure (Long-length Bias 3)"
    vus_zone_size = e_buffer = d_buffer = parameter_near_single_side_range = 3
    adjust_dis = 20
    label_ranges = [
        [[3 + adjust_dis, 4 + adjust_dis], [11 + adjust_dis, 12 + adjust_dis], [19 + adjust_dis, 28 + adjust_dis]],
        [[19 + adjust_dis, 28 + adjust_dis]],
        [[3 + adjust_dis, 4 + adjust_dis], [11 + adjust_dis, 12 + adjust_dis]]]
    window_length = 80
    choose_metric_name_order_list = [
        "Original-F", "AUC-ROC", "AUC-PR",
        "PA-F",
        "DTPA-F",
        "LS-F",
        "\%K-PA-F",
        "VUS-ROC", "VUS-PR", "PATE",
        "RF", "eTaF", "AF",
        "DQE",
    ]
    caption_str = "Comparison of the metrics about the issue of unreasonable evaluation on TP predictions(L1) using synthetic data."

# tp timeliness
if test_flag == "test":
    pass
    case_type = 2
    if case_type == 1:
        window_length = 300
        label_ranges = [
            [
                [140, 159],
            ],
            # case1
            [
                [(140 + 159 + 1) // 2 - 6 - 1, (140 + 159 + 1) // 2 - 6 + 1],
            ],
            [
                [(140 + 159 + 1) // 2 + 5 - 1, (140 + 159 + 1) // 2 + 5 + 1],
            ],
            # case2
            [
                [(140 + 159 + 1) // 2 - 6 - 1, (140 + 159 + 1) // 2 - 6 + 1],
                [(140 + 159 + 1) // 2 - 1 - 1, (140 + 159 + 1) // 2 - 1 + 1],
            ],
            [
                [(140 + 159 + 1) // 2 - 6 - 1, (140 + 159 + 1) // 2 - 6 + 1],
                [(140 + 159 + 1) // 2 + 5 - 1, (140 + 159 + 1) // 2 + 5 + 1],
            ]
        ]
        json_file_name = "gt_onset_case1"
        choose_metric_name_order_list = [
            "AF",
            "Original-F", "AUC-PR", "AUC-ROC",
            "\%K-PA-F",
            "VUS-ROC", "VUS-PR", "PATE",
            "RF", "eTaF",
            "DQE",
        ]
    elif case_type == 2:
        window_length = 300
        label_ranges = [
            [
                [140, 159],
            ],
            # case1
            [
                [(140 + 159 + 1) // 2 - 6 - 1, (140 + 159 + 1) // 2 - 6 + 1],
            ],
            [
                [(140 + 159 + 1) // 2 + 5 - 1, (140 + 159 + 1) // 2 + 5 + 1],
            ],
            # case2
            [
                [(140 + 159 + 1) // 2 - 6 - 1, (140 + 159 + 1) // 2 - 6 + 1],
                [(140 + 159 + 1) // 2 - 1 - 1, (140 + 159 + 1) // 2 - 1 + 1],
            ],
            [
                [(140 + 159 + 1) // 2 - 6 - 1, (140 + 159 + 1) // 2 - 6 + 1],
                [(140 + 159 + 1) // 2 + 5 - 1, (140 + 159 + 1) // 2 + 5 + 1],
            ]
        ]
        json_file_name = "gt_onset_case2"
        choose_metric_name_order_list = [
            "AF",
            "Original-F", "AUC-ROC", "AUC-PR",
            "\%K-PA-F",
            "VUS-ROC", "VUS-PR", "PATE",
            "RF", "eTaF",
            "DQE",
        ]
    vus_zone_size = e_buffer = d_buffer = parameter_near_single_side_range = 20

    caption_str = "Comparison of the metrics about the issue of overrating for random predictions(L5) using synthetic data."

# fp proximity
if test_flag == "test":
    window_length = 110

    # after
    label_ranges = [[[30, 39]],

                    [[43 - 1, 46 - 3]],
                    [[49 - 1 - 2, 52 - 3 - 2]],
                    [[55 - 1 - 4, 58 - 3 - 4]],

                    [[70, 71]],
                    [[80, 81]],
                    [[90, 91]],
                    ]
    label_ranges1 = [[[30, 39]],

                     [[46, 55]],
                     [[47, 47]],
                     ]
    json_file_name = "Non-GT area pred case"
    vus_zone_size = e_buffer = d_buffer = parameter_near_single_side_range = 20

    # print("label_ranges", label_ranges)

    caption_str = "Comparison of the metrics about the issue of unreasonable evaluation on FP predictions(L2) using synthetic data."
    choose_metric_name_order_list = [
        "Original-F",
        # "PA-F",
        "\%K-PA-F",
        "RF", "eTaF",
        "AUC-PR", "AUC-ROC",
        "VUS-ROC", "VUS-PR", "PATE",  # insensitivity distance and length
        "AF",
        "DQE",
    ]


# fp insensitivity or overevaluation of duration
if test_flag == "test":
    pass

    case_type = 1
    if case_type == 1:
        window_length = 300
        label_ranges = [
            [
                [140, 159],
            ],
            # case1
            [
                [159 + 1 + 11 - 8 - 1, 159 + 1 + 11 - 8 + 1],
                [159 + 1 + 11 + 8 - 1, 159 + 1 + 11 + 8 + 1],
            ],
            [
                [159 + 1 + 11 - 8 - 1, 159 + 1 + 11 - 8 + 1],
                [159 + 1 + 11 - 4 - 1, 159 + 1 + 11 - 4 + 1],
                [159 + 1 + 11 - 0 - 1, 159 + 1 + 11 - 0 + 1],
                [159 + 1 + 11 + 4 - 1, 159 + 1 + 11 + 4 + 1],
                [159 + 1 + 11 + 8 - 1, 159 + 1 + 11 + 8 + 1],
            ],
        ]
        json_file_name = "ngt_duration_case1"

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
    if case_type == 2:
        window_length = 300
        label_ranges = [
            [
                [140, 159],
            ],
            [
                [159 + 1 + 50 - 11 - 1, 159 + 1 + 50 - 11 + 1],
                [159 + 1 + 50 + 11 - 1, 159 + 1 + 50 + 11 + 1],
            ],
            [
                [159 + 1 + 50 - 11 - 1, 159 + 1 + 50 - 11 + 8],
                [159 + 1 + 50 + 11 - 8, 159 + 1 + 50 + 11 + 1],
            ],
        ]
        json_file_name = "ngt_duration_case2"
        choose_metric_name_order_list = [
            "AUC-PR",
            "VUS-PR",
            "PATE",
            "AF",
            "Original-F",
            "\%K-PA-F",
            "RF", "eTaF",
            "AUC-ROC",
            "VUS-ROC",
            "DQE",
        ]

    vus_zone_size = e_buffer = d_buffer = parameter_near_single_side_range = 20

    caption_str = "Comparison of the metrics about the issue of overrating for random predictions(L5) using synthetic data."

# fp duration overevaluation (af)

if test_flag == "test":
    pass
    window_length = 38
    label_ranges = [[[29, 30], [35, 36]],
                    [[25, 26], [35, 36]],
                    [[29, 30], [34, 35]]]
    json_file_name = "af problem"
    vus_zone_size = e_buffer = d_buffer = parameter_near_single_side_range = 3

    caption_str = "Comparison of the metrics about the issue of unreasonable evaluation on hybrid TP and FP predictions(L3) using synthetic data."
    choose_metric_name_order_list = [
        "AF",
        "Original-F", "AUC-ROC", "AUC-PR",
        "\%K-PA-F",
        "VUS-ROC", "VUS-PR", "PATE",
        "RF", "eTaF",
        "DQE",
    ]

# print("label_ranges", label_ranges)
# print("window_length", window_length)

ordered_num = len(label_ranges)
# print("ordered_num", ordered_num)

label_array_list = []
res_data = []

parameter_dict_new = parameter_dict
parameter_dict_new["parameter_near_single_side_range"] = parameter_near_single_side_range

for i, single_range in enumerate(label_ranges):
    if i >= 0 and i < ordered_num:
        label_array = convert_events_to_array_PATE(single_range, time_series_length=window_length)
        import copy

        label_array_copy = copy.deepcopy(label_array)
        label_array_list.append(label_array_copy)
        # print("============== pred" + str(i))

        score_list_simple = evaluate_all_metrics(label_array,
                                                 label_array_list[0],
                                                 vus_zone_size, e_buffer,
                                                 d_buffer,
                                                 pred_case_id=i,
                                                 parameter_near_single_side_range=parameter_near_single_side_range,
                                                 parameter_dict_new=parameter_dict_new,
                                                 )
        # print(score_list_simple)

        selected_dict = {}

        for j, paint_name in enumerate(paint_name_list):
            value = score_list_simple[paint_name]
            selected_dict[paint_name] = value
        res_data.append(selected_dict)


import json


class CustomEncoder(json.JSONEncoder):
    def encode(self, obj):
        if isinstance(obj, list):
            items = [self.encode(item) for item in obj]
            return "[\n" + ",\n".join(items) + "\n]"
        elif isinstance(obj, dict):
            items = [f'"{key}":{self.encode(value)}' for key, value in obj.items()]
            return "{" + ",".join(items) + "}"
        return super().encode(obj)

file_path = "dataset/metric_cal_res_windows/synthetic_data_res" + json_file_name + ".json"

import pandas as pd


# print("res_data", res_data)
res_data_dict = {}

new_res_data = []
for i, pred_score_dict in enumerate(res_data):
    new_score_dict = {}
    for j, (metric_name, metric_score) in enumerate(pred_score_dict.items()):
        if name_dict[metric_name] in choose_metric_name_order_list:
            new_score_dict[name_dict[metric_name]] = metric_score
    new_res_data.append(new_score_dict)

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


insert_data = []
figure_name = json_file_name
figure_str = "\\includegraphics{figures/single_prediction_figures/" + figure_name
for i in range(1, len(label_ranges)):
    insert_data.append(figure_str + " pred" + i.__str__() + "}")

df.insert(0, figure_str + " GT" + "}", insert_data)

print("new_res_data", new_res_data)
