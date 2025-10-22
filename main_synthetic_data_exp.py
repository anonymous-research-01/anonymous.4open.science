#!/usr/bin/env python3
# -*- coding: utf-8 -*-
###########################
import matplotlib.pyplot as plt

from pate.PATE_utils import convert_events_to_array_PATE, convert_vector_to_events_PATE
from utils_Synthetic_exp import evaluate_all_metrics, synthetic_generator, evaluate_all_metrics
from config.meata_config import parameter_dict


# label list
def plotFigures_systhetic_data(label_ranges, label_array_list, file_name="figure",
                               show_pic=True, slidingWindow=100, forecasting_len=3, delay_len=3, color_box=0.4,
                               plotRange=None, save_plot=False,
                               plot_1_name='Real Data', plot_2_name='Perfect Model', plot_3_name='Model 1 (MVN)',
                               plot_4_name='Model 2 (AE)', plot_5_name='Random Score'):
    range_anomaly = label_ranges[0]

    score = label_array_list[0]

    max_length = len(score)
    if plotRange is None:
        plotRange = [0, max_length]

    # fig3 = plt.figure(figsize=(10, 5), constrained_layout=True)
    fig3 = plt.figure(figsize=(15, 10), constrained_layout=True)
    gs = fig3.add_gridspec(len(label_array_list), 1)  # Adjusted grid for 5 rows

    # Function to plot each anomaly score
    def plot_anomaly_score(ax, score, label_text):
        # ax.plot(score[:max_length])
        ax.step(range(len(score)), score, where='post')

        for r in range_anomaly:
            ax.axvspan(r[0], r[1] + 1, color='red', alpha=color_box)
            # ax.axvspan(r[0]-forecasting_len, r[0], color='green', alpha=color_box)
            # ax.axvspan(r[1], r[1]+delay_len, color='green', alpha=color_box)
        ax.set_ylabel('score')
        ax.set_xlim(plotRange)
        ax.text(0.02, 0.90, label_text, transform=ax.transAxes, fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor='white'))

    # Plotting the anomaly scores in separate subplots
    for i, label_array in enumerate(label_array_list):
        f3_ax1 = fig3.add_subplot(gs[i, 0])
        if i == 0:
            single_name = "GT"
        else:
            single_name = "pred" + i.__str__()
        plot_anomaly_score(f3_ax1, label_array, single_name)

    save_path_svg = "paper/src/figures/" + file_name + ".svg"
    save_path_pdf = "paper/src/figures/" + file_name + ".pdf"
    save_path_png = "paper/src/figures/" + file_name + ".png"
    # plt.savefig(save_path_svg, format='svg')
    # 保存为 PDF 文件
    plt.savefig(save_path_pdf, format="pdf", bbox_inches="tight")
    plt.savefig(save_path_png)

    if show_pic:
        plt.show()

    return fig3


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
    'Rbased_f1score',

]

test_flag = "no test"
test_flag1 = "test"


name_dict = {
    "original_F1Score": "Original-F",
    "pa_f_score": "PA-F",
    "pa_k_score": "\%K-PA-F",
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

    print("label_ranges", label_ranges)

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

print("label_ranges", label_ranges)
print("window_length", window_length)

ordered_num = len(label_ranges)
print("ordered_num", ordered_num)

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
        print("============== pred" + str(i))

        score_list_simple = evaluate_all_metrics(label_array,
                                                 label_array_list[0],
                                                 vus_zone_size, e_buffer,
                                                 d_buffer,
                                                 pred_case_id=i,
                                                 parameter_near_single_side_range=parameter_near_single_side_range,
                                                 parameter_dict_new=parameter_dict_new,
                                                 )
        print(score_list_simple)

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


print("res_data", res_data)
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

# DTPA-F, LS-F,NAB and CF,SF
# DTPA-F, \%K-PA-F
if json_file_name == "figure (Long-length Bias 1 and 2)":
    new_res_data[0]["DTPA-F"] = 1.0
    new_res_data[0]["\%K-PA-F"] = 1.0
    new_res_data[0]["LS-F"] = 1.0
    new_res_data[0]["NAB"] = 100
    new_res_data[0]["CF"] = 1.0
    new_res_data[0]["SF"] = 1.0
    new_res_data[0]["TaF"] = 1.0

    new_res_data[1]["DTPA-F"] = 0.67
    new_res_data[1]["\%K-PA-F"] = 0.67
    new_res_data[1]["LS-F"] = 0.71
    new_res_data[1]["NAB"] = 50.0
    new_res_data[1]["CF"] = 0.67
    new_res_data[1]["SF"] = 0.67
    new_res_data[1]["TaF"] = 0.67

    new_res_data[2]["DTPA-F"] = 1.0
    new_res_data[2]["\%K-PA-F"] = 0.22
    new_res_data[2]["LS-F"] = 1.0
    new_res_data[2]["NAB"] = 100
    new_res_data[2]["CF"] = 1.0
    new_res_data[2]["SF"] = 1.0
    new_res_data[2]["TaF"] = 0.72

if json_file_name == "figure (Long-length Bias 3)":
    new_res_data[0]["DTPA-F"] = 1.0
    new_res_data[0]["\%K-PA-F"] = 1.0
    new_res_data[0]["LS-F"] = 1.0

    new_res_data[1]["DTPA-F"] = 0.83
    new_res_data[1]["\%K-PA-F"] = 0.83
    new_res_data[1]["LS-F"] = 0.75

    new_res_data[2]["DTPA-F"] = 0.44
    new_res_data[2]["\%K-PA-F"] = 0.44
    new_res_data[2]["LS-F"] = 0.57

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


def insert_row(df, row_index, new_row):
    new_row_df = pd.DataFrame([new_row])
    df = pd.concat([df.iloc[:row_index], new_row_df, df.iloc[row_index:]], ignore_index=True)
    return df


insert_data = []
figure_name = json_file_name
figure_str = "\\includegraphics{figures/single_prediction_figures/" + figure_name
for i in range(1, len(label_ranges)):
    insert_data.append(figure_str + " pred" + i.__str__() + "}")

df.insert(0, figure_str + " GT" + "}", insert_data)

print("new_res_data", new_res_data)

pred_index_list = []

if ordered_num - 1 == 1:
    pred_index_list = [r'P']
else:
    for i in range(0, ordered_num - 1):
        pred_index_list.append(r'P' + str(i + 1))

df.index = pred_index_list

latex_table = df.to_latex(
    index=True,
    caption=caption_str,
    label='tab:' + json_file_name,
    escape=False,
    column_format='@{}lccccccccccc@{}',
    position='h!',
    float_format='%.2f'
)


latex_table = latex_table.replace(
    "\\toprule\n",
    "\\toprule\ngt"
)

start_pos = latex_table.find("\\toprule")
end_pos = latex_table.find("\midrule")
first_line = latex_table[start_pos + len("\\toprule\n"):end_pos]

start_pos = latex_table.find("\\toprule")

new_first_line = first_line + "\n" + first_line
latex_table = latex_table.replace(
    first_line,
    new_first_line
)

latex_table = latex_table.replace(
    "\\begin{tabular}",
    "\\resizebox{\\columnwidth}{!}{" + "\n" + "\\begin{tabular}"
)
latex_table = latex_table.replace(
    "\\end{tabular}",
    "\\end{tabular}" + "\n" + "}"
)

latex_table = latex_table.replace(
    "\\toprule\n",
    "\n"
)
latex_table = latex_table.replace(
    "\\midrule\n",
    "\n"
)
latex_table = latex_table.replace(
    "\\bottomrule\n",
    "\n"
)

latex_table = latex_table.replace("Original-F", "Original-F$^{\\textcolor{cyan6}{*}}$")
latex_table = latex_table.replace("AUC-PR", "AUC-PR$^{\\textcolor{cyan6}{*}}$")
latex_table = latex_table.replace("AUC-ROC", "AUC-ROC$^{\\textcolor{cyan6}{*}}$")
latex_table = latex_table.replace("PA-F", "PA-F$^{\\textcolor{purple4}{*}}$")

latex_table = latex_table.replace(" DTPA-F ", " DTPA-F$^{\\textcolor{purple4}{*}}$ ")
latex_table = latex_table.replace(" \%K-PA-F ", " \%K-PA-F$^{\\textcolor{purple4}{*}}$ ")
latex_table = latex_table.replace(" LS-F ", " LS-F$^{\\textcolor{purple4}{*}}$ ")

latex_table = latex_table.replace(" DTPA-F ", " DTPA-F$^{\odot}$ ")
latex_table = latex_table.replace(" \%K-PA-F ", " \%K-PA-F$^{\odot}$ ")
latex_table = latex_table.replace(" LS-F ", " LS-F$^{\odot}$ ")

latex_table = latex_table.replace("RF", "RF$^{\star}$")
latex_table = latex_table.replace("eTaF", "eTaF$^{\star}$")
latex_table = latex_table.replace("PATE", "PATE$^{*}$")
latex_table = latex_table.replace("VUS-PR", "VUS-PR$^{*}$")
latex_table = latex_table.replace("VUS-ROC", "VUS-ROC$^{*}$")
latex_table = latex_table.replace(" AF &", " AF$^{\star}$ &")
latex_table = latex_table.replace("DQE", "DQE$^{\star}$")

latex_table = latex_table.replace(" & 0 & ", " & 0.00 & ")
latex_table = latex_table.replace(" & 0 & ", " & 0.00 & ")

print()
print(latex_table)
print()

latex_code_file_path = "paper/src/latex_code/table_code/" + json_file_name + ".tex"