import json
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import numpy as np
import pandas as pd
import time
import argparse


from evaluation.metrics import get_metrics
from evaluation.slidingWindows import find_length_rank


if __name__ == '__main__':

    ## ArgumentParser
    parser = argparse.ArgumentParser(description='Running DQE real-world experiments (case analysis)')
    parser.add_argument('--exp_name', type=str, default='YAHOO case')
    args = parser.parse_args()

    dataset_dir = "../dataset/"
    file_path = dataset_dir + "all_methods_pred_res_min_max_scale/"

    ori_data_path = dataset_dir + "TSB-AD-U/"

    res_save_dir = dataset_dir + "metric_cal_res_windows/"

    if args.exp_name == "WSD case":
        # WSD case
        file_msg = "094_WSD_case_analyze"
        dataset_methods_file_list = ["094_WSD_id_66_WebService_tr_3309_1st_3914.json"]
        # WSD case methods
        dataset_methods_choose_name_list = ['SR', 'CNN', 'Sub_LOF', 'FFT']
    elif args.exp_name == "partition strategy":
        # Impact Analysis of Partition Strategy in Distant FQ Region
        file_msg = "808_YAHOO_partition_strategy"
        dataset_methods_file_list = ["808_YAHOO_id_258_WebService_tr_500_1st_142.json"]
        dataset_methods_choose_name_list = ['Sub_LOF', 'Sub_MCD']
    elif args.exp_name == "detection rate":
        # Impact Analysis of the Detection Rate
        file_msg = "094_WSD_detection_rate"
        dataset_methods_file_list = ["094_WSD_id_66_WebService_tr_3309_1st_3914.json"]
        dataset_methods_choose_name_list = ['CNN', 'FFT']
    elif args.exp_name == "weighting strategy":
        # Impact Analysis of the Triangular Weighting Strategy
        file_msg = "808_YAHOO_weighting_strategy"
        dataset_methods_file_list = ["808_YAHOO_id_258_WebService_tr_500_1st_142.json"]
        dataset_methods_choose_name_list = ['SR', 'KMeansAD_U']
    else:
        # YAHOO case
        file_msg = "808_YAHOO_case_analyze"
        dataset_methods_file_list = ["808_YAHOO_id_258_WebService_tr_500_1st_142.json"]
        # YAHOO case methods
        dataset_methods_choose_name_list = ['SR', 'KMeansAD_U', 'Sub_KNN', 'TimesNet']

    print(file_msg)

    file_path_dict = {}
    for file_name in dataset_methods_file_list:
        file_path_dict[file_name] = file_path+file_name

    file_method_metric_dict = {} # save

    for i, data_set_choose_file in enumerate(dataset_methods_file_list):
        dataset_name= data_set_choose_file.split("_")[1]
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

    res_seve_path = res_save_dir + "metric_calc_res_" + file_msg + ".json"

    with open(res_seve_path, 'w', encoding='utf-8') as json_file:
        json.dump(file_method_metric_dict, json_file, indent=4, ensure_ascii=False)
    print(f"Results are saved to {res_seve_path}")



