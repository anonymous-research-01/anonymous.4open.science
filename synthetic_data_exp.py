#!/usr/bin/env python3
# -*- coding: utf-8 -*-
###########################
import os
import pandas as pd
import json
import argparse

from metrics.pate.PATE_utils import convert_events_to_array_PATE
from utils_synthetic_exp import evaluate_all_metrics
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
    # parser.add_argument('--exp_name', type=str, default='tp timeliness')
    # parser.add_argument('--exp_name', type=str, default='fp proximity')
    # parser.add_argument('--exp_name', type=str, default='fp insensitivity or overevaluation of duration')
    # parser.add_argument('--exp_name', type=str, default='fp duration overevaluation (af)')
    parser.add_argument('--print', type=bool, default=True)
    parser.add_argument('--test_time', type=bool, default=True)
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
        # "dqe_w_fq",
    
        'Affliation F1score',
        'eTaPR_f1_score',
        'original_F1Score',
        # 'pa_f_score',
        'pa_k_score',
        'dtpa_f_score',
        'ls_f_score',
        'Rbased_f1score',
    
    ]
    
    name_dict = {
        "original_F1Score": "Original-F",
        # "pa_f_score": "PA-F",
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
        # "dqe_w_fq": "DQE_w_fq",
    
        "VUS_ROC": "VUS-ROC",
        "VUS_PR": "VUS-PR",
        "AUC": "AUC-ROC",
        "AUC_PR": "AUC-PR",
        "PATE": "PATE",
    }
    
    
    choose_metric_name_order_list = [
        "VUS-ROC", "VUS-PR", "AUC-ROC", "AUC-PR", "PATE",
        "Original-F",
        # "PA-F",
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
        label_ranges = [
            [[3 + adjust_dis, 4 + adjust_dis], [11 + adjust_dis, 12 + adjust_dis], [19 + adjust_dis, 28 + adjust_dis]],
            [[19 + adjust_dis, 28 + adjust_dis]],
            [[3 + adjust_dis, 4 + adjust_dis], [11 + adjust_dis, 12 + adjust_dis]]]
        ts_len = 80
        choose_metric_name_order_list = [
            "Original-F", "AUC-ROC", "AUC-PR",
            # "PA-F",
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

        label_ranges = [[[30, 39]],
    
                        [[43 - 1, 46 - 3]],
                        [[49 - 1 - 2, 52 - 3 - 2]],
                        [[55 - 1 - 4, 58 - 3 - 4]],
    
                        [[70, 71]],
                        [[80, 81]],
                        [[90, 91]],
                        ]
        json_file_name = "fp proximity"
        vus_zone_size = e_buffer = d_buffer = near_single_side_range = 20
    
        choose_metric_name_order_list = [
            "Original-F",
            # "PA-F",
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
        label_ranges = [
            [
                [140, 159],
            ],
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
        label_ranges = [[[29, 30], [35, 36]],
                        [[25, 26], [35, 36]],
                        [[29, 30], [34, 35]]]
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
        # print("============== pred" + str(i))
        label_array = convert_events_to_array_PATE(single_range, time_series_length=ts_len)
        label_array_list.append(label_array)
        score_list_simple = evaluate_all_metrics(label_array,
                                                 label_array_list[0],
                                                 vus_zone_size,
                                                 e_buffer,
                                                 d_buffer,
                                                 parameter_dict=parameter_dict_new,
                                                 )

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
    if directory:  # 防止路径就是文件名，目录为空
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
    
    print("new_res_data", new_res_data)
    
    
    
    
    
    # for i, pred_score_dict in enumerate(res_data):
    #     for j, (metric_name,metric_score) in enumerate(pred_score_dict.items()):
    #         if metric_name not in res_data_dict.keys():
    #             res_data_dict[metric_name] = [metric_score]
    #         else:
    #             res_data_dict[metric_name].append(metric_score)
    
    # print("res_data_dict",res_data_dict)
    
    
    # 添加索引列，分别命名为 pred1 和 pred2，并加粗
    # 手动改数据行名
    
    pred_index_list = []
    
    
    if ordered_num-1 == 1:
        # pred_index_list = [r'\textbf{pred}']
        pred_index_list = [r'P']
    else:
        # for i in range(0,ordered_num):
        # remove gt line
        for i in range(0, ordered_num - 1):
            # if i == 0:
            #     pred_index_list.append(r'\textbf{gt}')
            # else:
            #     pred_index_list.append(r'\textbf{pred' + str(i) +  '}')
    
            # remove gt line
            # pred_index_list.append(r'\textbf{pred' + str(i + 1) + '}')
            pred_index_list.append(r'P' + str(i + 1))
    
    df.index = pred_index_list


    # 转换为 LaTeX 表格
    latex_table = df.to_latex(
        index=True,  # 包含索引列
        caption=json_file_name,  # 表格标题
        label='tab:'+json_file_name,  # 表格标签
        escape=False,  # 不转义特殊字符
        column_format='@{}lccccccccccc@{}',  # 列格式
        position='h!',  # 表格位置
        float_format='%.2f'  # 浮点数格式化为两位小数
    )
    
    # 手动修改表头，去掉多余的行，并添加“Metric”
    # 手动改第一行数据行名`
    latex_table = latex_table.replace(
        "\\toprule\n",
        # "\\toprule\n\\textbf{Metric}"
        "\\toprule\ngt"
    )
    
    start_pos = latex_table.find("\\toprule")
    end_pos = latex_table.find("\midrule")
    # first_line = latex_table[start_pos:end_pos]
    first_line = latex_table[start_pos+len("\\toprule\n"):end_pos]
    d=1
    # new_first_line = first_line
    
    # for i, metric_name in enumerate(new_res_data[0].keys()):
    #     if metric_name in first_line:
    #         new_first_line = new_first_line.replace(" "+metric_name+" ",
    #                                                 " "+"\\textbf{"+ metric_name + "}"+" ")
    
    
    # 添加表宽调整
    # latex_table = latex_table.replace(
    #     first_line,
    #     new_first_line
    # )
    
    start_pos = latex_table.find("\\toprule")
    
    # first_line_category = first_line.replace(
    #     first_line,
    #     new_first_line
    # )
    new_first_line = first_line+"\n"+first_line
    latex_table = latex_table.replace(
        first_line,
        new_first_line
    )
    
    latex_table = latex_table.replace(
        "\\begin{tabular}",
        "\\resizebox{\\columnwidth}{!}{"+"\n"+"\\begin{tabular}"
    )
    latex_table = latex_table.replace(
        "\\end{tabular}",
        "\\end{tabular}"+"\n"+"}"
    )
    
    # latex_table = latex_table.replace(
    #     "\\bottomrule\n",
    #     # "\\toprule\n\\textbf{Metric}"
    #     "\n"
    # )
    
    
    latex_table = latex_table.replace(
        "\\toprule\n",
        # "\\toprule\n\\textbf{Metric}"
        "\n"
    )
    latex_table = latex_table.replace(
        "\\midrule\n",
        # "\\toprule\n\\textbf{Metric}"
        "\n"
    )
    latex_table = latex_table.replace(
        "\\bottomrule\n",
        # "\\toprule\n\\textbf{Metric}"
        "\n"
    )
    
    # gt & \includegraphics{figures/single_prediction_figures/Non-GT area pred case gt} &
    # Original-F$^{\dagger}
    # PA-F$^{\odot}
    # RF$^{\star}
    # eTaF$^{\star}
    # AUC-PR$^{*}
    # VUS-PR$^{*}
    # PATE$^{*}
    # AUC-ROC$^{*}
    # VUS-ROC$^{*}
    # AF$^{\star}
    # DLBE$^{\star}
    # gt & \includegraphics{figures/single_prediction_figures/Non-GT area pred case gt} & Original-F$^{\textcolor{cyan6}{*}}$ & PA-F$^{\textcolor{purple4}{*}}$ & RF$^{\star}$ & eTaF$^{\star}$ & AUC-PR$^{*}$ & VUS-PR$^{*}$ & PATE$^{*}$ & AUC-ROC$^{*}$ & VUS-ROC$^{*}$ & AF$^{\star}$ & DLBE$^{\star}$ \\
    
    
    latex_table = latex_table.replace("Original-F","Original-F$^{\\textcolor{cyan6}{*}}$")
    latex_table = latex_table.replace("AUC-PR","AUC-PR$^{\\textcolor{cyan6}{*}}$")
    latex_table = latex_table.replace("AUC-ROC","AUC-ROC$^{\\textcolor{cyan6}{*}}$")
    latex_table = latex_table.replace("PA-F","PA-F$^{\\textcolor{purple4}{*}}$")
    # "DTPA-F",
    # "\%K-PA-F",
    # "LS-F"
    latex_table = latex_table.replace(" DTPA-F "," DTPA-F$^{\\textcolor{purple4}{*}}$ ")
    latex_table = latex_table.replace(" \%K-PA-F "," \%K-PA-F$^{\\textcolor{purple4}{*}}$ ")
    latex_table = latex_table.replace(" LS-F "," LS-F$^{\\textcolor{purple4}{*}}$ ")
    
    # latex_table = latex_table.replace("Original-F","Original-F$^{\dagger}$")
    # latex_table = latex_table.replace("AUC-PR","AUC-PR$^{\dagger}$")
    # latex_table = latex_table.replace("AUC-ROC","AUC-ROC$^{\dagger}$")
    # latex_table = latex_table.replace("PA-F","PA-F$^{\odot}$")
    # "DTPA-F",
    # "\%K-PA-F",
    # "LS-F"
    latex_table = latex_table.replace(" DTPA-F "," DTPA-F$^{\odot}$ ")
    latex_table = latex_table.replace(" \%K-PA-F "," \%K-PA-F$^{\odot}$ ")
    latex_table = latex_table.replace(" LS-F "," LS-F$^{\odot}$ ")
    
    
    latex_table = latex_table.replace("RF","RF$^{\star}$")
    latex_table = latex_table.replace("eTaF","eTaF$^{\star}$")
    latex_table = latex_table.replace("PATE","PATE$^{*}$")
    latex_table = latex_table.replace("VUS-PR","VUS-PR$^{*}$")
    latex_table = latex_table.replace("VUS-ROC","VUS-ROC$^{*}$")
    latex_table = latex_table.replace(" AF &"," AF$^{\star}$ &")
    latex_table = latex_table.replace("DQE","DQE$^{\star}$")
    
    latex_table = latex_table.replace(" & 0 & "," & 0.00 & ")
    latex_table = latex_table.replace(" & 0 & "," & 0.00 & ")
    
    
    print()
    print(latex_table)
    print()
