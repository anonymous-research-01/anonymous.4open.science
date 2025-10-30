import json
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import numpy as np
import pandas as pd
import time
import copy
import math
import argparse



from evaluation.metrics import get_metrics
from evaluation.slidingWindows import find_length_rank


if __name__ == '__main__':
    ## ArgumentParser
    parser = argparse.ArgumentParser(description='Running DQE real-world experiments')
    args = parser.parse_args()

    dataset_dir = "../dataset/"
    file_path = dataset_dir + "all_methods_pred_res_min_max_scale/"

    ori_data_path = dataset_dir + "TSB-AD-U/"

    res_save_dir = dataset_dir + "metric_cal_res_windows/"

    dataset_methods_name_list = ['SR', 'KMeansAD_U', 'Sub_KNN', 'TimesNet', 'CNN', 'Sub_LOF', 'FFT', 'Sub_MCD']

    file_path_dict = {}
    dataset_methods_file_list = os.listdir(file_path)
    for file_name in dataset_methods_file_list:
        file_path_dict[file_name] = file_path + file_name

    choose_method_num = len(dataset_methods_name_list)

    dataset_methods_choose_name_list = dataset_methods_name_list

    data_set_choose_file_list = []

    dataset_name_list = [
                         # 'Exathlon',
                         # 'IOPS',
                         # 'LTDB',
                         # 'SED',
                         # 'SMD',
                         # 'TODS',
                         'WSD',
                         'YAHOO'
    ]

    file_method_metric_dict = {} # save
    for i, data_set_choose_file in enumerate(dataset_methods_file_list):
        dataset_name = data_set_choose_file.split("_")[1]
        if dataset_name not in dataset_name_list:
            continue
        file_method_metric_dict[data_set_choose_file] = {}
        data_set_choose_file_path = file_path_dict[data_set_choose_file]
        with open(data_set_choose_file_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        for j, dataset_methods_choose_name in enumerate(dataset_methods_choose_name_list):

            methods_choose_outputs = data[dataset_methods_choose_name]

            gt_array = data["gt"]
            gt_range = data["gt_range"]

            # cal score for all metric
            ori_data_file_path = ori_data_path + data_set_choose_file.replace(".json",".csv")
            df = pd.read_csv(ori_data_file_path).dropna()

            train_index = data_set_choose_file.split('.')[0].split('_')[-3]

            ori_data = df.iloc[:, 0:-1].values.astype(float)
            label = df['Label'].astype(int).to_numpy()
            slidingWindow = find_length_rank(ori_data, rank=1)

            output = methods_choose_outputs
            output_array = np.array(output)

            metric_score_dict = get_metrics(output_array, label, slidingWindow=slidingWindow,thre=100)
            file_method_metric_dict[data_set_choose_file][dataset_methods_choose_name] = metric_score_dict


    # cal mean res
    # cal number each dataset
    dataset_count_dict = {}
    for id_dataset_name, dataset_name in enumerate(file_method_metric_dict.keys()):
        dataset_name_to_find = dataset_name.split("_")[1]
        if dataset_name_to_find not in dataset_name_list:
            continue
        if dataset_name_to_find not in dataset_count_dict:
            dataset_count_dict[dataset_name_to_find] = 0

        dataset_count_dict[dataset_name_to_find] += 1

    print(dataset_count_dict)

    metric_name_list = [
        'Standard-F1',
        'AUC-ROC',
        'AUC-PR',

        'PA-K',

        'VUS-ROC',
        'VUS-PR',
        'PATE',
        # 'PATE_F1',

        # 'PA-F1',
        'R-based-F1',
        'eTaPR_F1',
        'Affiliation-F',

        "DQE",  # DQE
    ]

    method_name_index_dict = {}
    for i, method_name in enumerate(dataset_methods_choose_name_list):
        method_name_index_dict[method_name] = i

    dataset_name_score_list_dict = {
        # 'Exathlon': {},
        # 'IOPS': {},
        # 'LTDB': {},
        # 'SED': {},
        # 'SMD': {},
        # 'TODS': {},
        'YAHOO': {},
        'WSD': {},
    }


    # create mean score data structure for single dataset(dataset, metric, method)
    for id_dataset_name, dataset_name in enumerate(dataset_name_score_list_dict.keys()):
        metric_score_list_dict = {}

        for id_metric_name, metric_name in enumerate(metric_name_list):

            method_score_list_dict = {}

            for id_method_name, method_name in enumerate(dataset_methods_choose_name_list):
                method_score_list_dict[method_name] = {"score_list": [], "mean_score": None}

            metric_score_list_dict[metric_name] = method_score_list_dict

        dataset_name_score_list_dict[dataset_name] = metric_score_list_dict

    # create mean score data structure for all datasets(metric, method)
    metric_score_list_dict = {}

    for id_metric_name, metric_name in enumerate(metric_name_list):

        method_score_list_dict = {}

        for id_method_name, method_name in enumerate(dataset_methods_choose_name_list):
            method_score_list_dict[method_name] = {"score_list": [], "mean_score": None}

        metric_score_list_dict[metric_name] = method_score_list_dict

    # cal dataset mean score for every metric
    # add to list
    # target order, dataset,metric, method
    for id_dataset_name_need, dataset_name_need in enumerate(dataset_name_list):
        for id_metric_name_need, metric_name_need in enumerate(metric_name_list):
            for id_method_choose_need, method_name_need in enumerate(dataset_methods_choose_name_list):
                # have order, dataset, method, metric
                for id_file_to_find, (dataset_to_find, dataset_dict) in enumerate(file_method_metric_dict.items()):
                    dataset_name_to_find = dataset_to_find.split("_")[1]
                    if dataset_name_need == dataset_name_to_find:
                        for id_method_to_find, (method_name_to_find, method_dict) in enumerate(dataset_dict.items()):
                            if method_name_need == method_name_to_find:
                                for id_metric_to_find, (metric_name_to_find, metric_dict) in enumerate(
                                        method_dict.items()):
                                    if metric_name_need == metric_name_to_find:
                                        # find
                                        find_score = method_dict[metric_name_to_find]
                                        if isinstance(find_score, float) and math.isnan(find_score):
                                            find_score = 0
                                        dataset_name_score_list_dict[dataset_name_need][metric_name_need][
                                            method_name_need]["score_list"].append(find_score)

    for id_metric_name_need, metric_name_need in enumerate(metric_name_list):
        for id_method_choose_need, method_name_need in enumerate(dataset_methods_choose_name_list):
            # have order, dataset, method, metric
            for id_file_to_find, (dataset_file_to_find, dataset_dict) in enumerate(file_method_metric_dict.items()):
                dataset_name_to_find = dataset_file_to_find.split("_")[1]
                if dataset_name_to_find not in dataset_name_list:
                    continue
                for id_method_to_find, (method_name_to_find, method_dict) in enumerate(dataset_dict.items()):
                    if method_name_need == method_name_to_find:
                        for id_metric_to_find, (metric_name_to_find, metric_dict) in enumerate(method_dict.items()):
                            if metric_name_need == metric_name_to_find:
                                # find
                                find_score = method_dict[metric_name_to_find]
                                if isinstance(find_score, float) and math.isnan(find_score):
                                    find_score = 0
                                metric_score_list_dict[metric_name_need][method_name_need]["score_list"].append(
                                    find_score)

    # cal mean score for single dataset
    for id_dateset_name, (dateset_name, dateset_score_dict) in enumerate(dataset_name_score_list_dict.items()):
        for id_metric_name, (metric_name, metric_score_dict) in enumerate(dateset_score_dict.items()):
            for id_method_name, (method_name, method_score_info_dict) in enumerate(metric_score_dict.items()):
                score_list = dataset_name_score_list_dict[dateset_name][metric_name][method_name]["score_list"]

                mean_score = np.array(score_list).mean()
                dataset_name_score_list_dict[dateset_name][metric_name][method_name]["mean_score"] = mean_score

                # cal range
                max_score = np.array(score_list).max()
                min_score = np.array(score_list).min()
                median_score = np.median(score_list)

                dataset_name_score_list_dict[dateset_name][metric_name][method_name]["max_score"] = float(max_score)
                dataset_name_score_list_dict[dateset_name][metric_name][method_name]["min_score"] = float(min_score)
                dataset_name_score_list_dict[dateset_name][metric_name][method_name]["median_score"] = float(
                    median_score)
                dataset_name_score_list_dict[dateset_name][metric_name][method_name]["score_range"] = float(
                    max_score - min_score)

    # cal mean score for all datasets
    for id_metric_name, (metric_name, metric_score_dict) in enumerate(metric_score_list_dict.items()):
        for id_method_name, (method_name, method_score_info_dict) in enumerate(metric_score_dict.items()):
            score_list = metric_score_list_dict[metric_name][method_name]["score_list"]
            mean_score = np.array(score_list).mean()
            metric_score_list_dict[metric_name][method_name]["mean_score"] = mean_score

    # creating rankings by sort for single dataset
    dataset_name_score_list_dict_sort = copy.deepcopy(
        dataset_name_score_list_dict)  # copy structure and change dict item
    for id_dateset_name, (dateset_name, dateset_score_dict) in enumerate(dataset_name_score_list_dict_sort.items()):
        # for a dataset
        for id_metric_name, (metric_name, metric_score_dict) in enumerate(dateset_score_dict.items()):
            # for a metric,sort method
            method_score_dict_list = []
            for id_method_name, (method_name, method_score_res_dict) in enumerate(metric_score_dict.items()):
                method_score_dict_list.append({"method_name": method_name,
                                               "method_index": method_name_index_dict[method_name],
                                               "mean_score": method_score_res_dict["mean_score"],
                                               })
            method_score_dict_list_sorted = sorted(method_score_dict_list, key=lambda x: x["mean_score"], reverse=True)

            # dict->list
            dataset_name_score_list_dict_sort[dateset_name][metric_name] = method_score_dict_list_sorted

    # creating rankings by sort for all datasets
    metric_score_list_dict_sort = copy.deepcopy(metric_score_list_dict)  # copy structure and change dict item
    for id_metric_name, (metric_name, metric_score_dict) in enumerate(metric_score_list_dict_sort.items()):
        # for a metric,sort method
        method_score_dict_list = []
        for id_method_name, (method_name, method_score_res_dict) in enumerate(metric_score_dict.items()):
            method_score_dict_list.append({"method_name": method_name,
                                           "method_index": method_name_index_dict[method_name],
                                           "mean_score": method_score_res_dict["mean_score"],
                                           })
        method_score_dict_list_sorted = sorted(method_score_dict_list, key=lambda x: x["mean_score"], reverse=True)

        # dict->list
        metric_score_list_dict_sort[metric_name] = method_score_dict_list_sorted

    # save rankings
    dataset_name_score_list_dict_copy_index_dict_sort = copy.deepcopy(
        dataset_name_score_list_dict_sort)  # copy structure and simplify ranking info item
    metric_score_list_dict_copy_index_dict_sort = copy.deepcopy(metric_score_list_dict_sort)

    print("single dataset rankings result")

    for id_dateset_name, (dateset_name, dateset_score_dict) in enumerate(dataset_name_score_list_dict_sort.items()):
        # for a dataset
        # find case

        print()
        print("dateset_name", dateset_name)
        for id_metric_name, (metric_name, metric_score_dict_list) in enumerate(dateset_score_dict.items()):
            add_info_list = []
            sort_index_info_list = []
            for idx, metric_score_dict in enumerate(metric_score_dict_list):
                metric_score_dict["sort_id"] = idx
                add_info_list.append(metric_score_dict)

                sort_index_info_list.append(
                    str(metric_score_dict["sort_id"]) \
                    + " " + "m:" + str(metric_score_dict["method_index"]) \
                    + " " + str(metric_score_dict["method_name"]) \
                    + " " + str(round(metric_score_dict["mean_score"], 2)) \
                    )

            dataset_name_score_list_dict_sort[dateset_name][metric_name] = add_info_list

            dataset_name_score_list_dict_copy_index_dict_sort[dateset_name][metric_name] = sort_index_info_list

            print(" " + metric_name, add_info_list)

    print()
    print("all datasets rankings res")

    for id_metric_name, (metric_name, metric_score_dict_list) in enumerate(metric_score_list_dict_sort.items()):
        add_info_list = []
        sort_index_info_list = []

        for idx, metric_score_dict in enumerate(metric_score_dict_list):
            metric_score_dict["sort_id"] = idx
            add_info_list.append(metric_score_dict)

            sort_index_info_list.append(
                str(metric_score_dict["sort_id"]) \
                + " " + "m:" + str(metric_score_dict["method_index"]) \
                + " " + str(metric_score_dict["method_name"]) \
                + " " + str(round(metric_score_dict["mean_score"], 2)) \
                )

        metric_score_list_dict_sort[metric_name] = add_info_list

        metric_score_list_dict_copy_index_dict_sort[metric_name] = sort_index_info_list
        print(" " + metric_name, add_info_list)

    # patch all datasets result
    dataset_name_score_list_dict_sort["all_dataset"] = metric_score_list_dict_sort
    dataset_name_score_list_dict_copy_index_dict_sort["all_dataset"] = metric_score_list_dict_copy_index_dict_sort


    # write file

    def custom_json_formatter(data, indent=4):
        def _format(value, level=0):
            if isinstance(value, list):
                elements = [json.dumps(item, ensure_ascii=False) for item in value]
                max_length = max(len(item) for item in elements)
                formatted_elements = ", ".join(item.ljust(max_length) for item in elements)
                return f"[{formatted_elements}]"
            elif isinstance(value, dict):
                lines = []
                for key, val in value.items():
                    lines.append(" " * (indent * level) + f'"{key}": {(_format(val, level + 1))}')
                return "{\n" + ",\n".join(lines) + "\n" + " " * (indent * (level - 1)) + "}"
            else:
                return json.dumps(value, ensure_ascii=False)

        return _format(data)


    res_save_dir = dataset_dir + "metric_mean_res/"

    res_seve_path = res_save_dir + "metric_mean_res_" + choose_method_num + ".json"

    with open(res_seve_path, "w", encoding="utf-8") as file:
        write_data = custom_json_formatter(dataset_name_score_list_dict_sort)
        file.write(write_data)

    print(f"Results are saved to {res_seve_path}")



