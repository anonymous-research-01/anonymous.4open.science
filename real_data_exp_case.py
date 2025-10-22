import json
import os
import numpy as np
import pandas as pd
import time

from evaluation.metrics import get_metrics
from evaluation.slidingWindows import find_length_rank

json_dir = "dataset/"
first_file_path = json_dir + "all_methods_pred_res/"
second_file_path = json_dir +"all_methods_pred_res/without_lstmad/"
third_file_path = json_dir +"all_methods_pred_res/without_further_sub_mcd/"

ori_data_path = json_dir +"TSB-AD-U/"

res_save_dir = json_dir + "metric_cal_res_windows/"


file_path_dict = {}
file_list1 = os.listdir(first_file_path)
# remove
file_list1.remove("without_lstmad")
file_list1.remove("without_further_sub_mcd")


for file_name in file_list1:
    file_path_dict[file_name] = first_file_path+file_name
file_list2 = os.listdir(second_file_path)
for file_name in file_list2:
    file_path_dict[file_name] = second_file_path+file_name



dataset_19_methods_file_list = os.listdir(first_file_path) + os.listdir(second_file_path) + os.listdir(third_file_path)

dataset_20_methods_file_list = os.listdir(first_file_path)+os.listdir(second_file_path)

dataset_21_methods_file_list = os.listdir(first_file_path)


# YAHOO case
choose_method_num = "20_patch_random_add_method_808_YAHOO_case_analyze"
dataset_methods_file_list = ["808_YAHOO_id_258_WebService_tr_500_1st_142_name_list.json"]

# WSD case
# choose_method_num = "20_patch_random_add_method_094_WSD_case_analyze"
# dataset_methods_file_list = ["094_WSD_id_66_WebService_tr_3309_1st_3914_name_list.json"]


# YAHOO case methods
dataset_methods_choose_name_list = ['SR', 'KMeansAD_U', 'Sub_KNN', 'TimesNet']
# WSD case methods
# dataset_methods_choose_name_list = ['SR', 'CNN', 'Sub_LOF', 'FFT']

# Impact Analysis of Partition Strategy in Distant FQ Region
# dataset_methods_choose_name_list = ['Sub_LOF', 'Sub-MCD']

# Impact Analysis of the Detection Rate
# dataset_methods_choose_name_list = ['CNN', 'FFT']

# Impact Analysis of the Triangular Weighting Strategy
# dataset_methods_choose_name_list = ['SR', 'KMeansAD_U']


# remove
if "without_lstmad" in dataset_methods_file_list:
    dataset_methods_file_list.remove("without_lstmad")
if "without_further_sub_mcd" in dataset_methods_file_list:
    dataset_methods_file_list.remove("without_further_sub_mcd")


method_name_index_dict = {}
for i, method_name in enumerate(dataset_methods_choose_name_list):
    method_name_index_dict[method_name] = i

from pprint import pprint
print("method_name_index_dict")
pprint(method_name_index_dict)

data_set_choose_file_list = []
for dataset_method_file in dataset_methods_file_list:
    if ("LTDB" in dataset_method_file
            or "IOPS" in dataset_method_file
            or "Exathlon" in dataset_method_file):
        continue
    if "_name_list" in dataset_method_file:
        data_set_choose_file_list.append(dataset_method_file)

print("len(data_set_choose_file_list)",len(data_set_choose_file_list))


# integral every dataset
dataset_name_list = [
                     'Exathlon',
                     'IOPS',
                     'LTDB',
                     'SED',
                     'SMD',
                     'TODS',
                     'WSD',
                     'YAHOO'
]
dataset_name_dict = {
    'Exathlon': {'file_list': [], 'file_num': 0},
    'IOPS': {'file_list': [], 'file_num': 0},
    'LTDB': {'file_list': [], 'file_num': 0},
    'SED': {'file_list': [], 'file_num': 0},
    'SMD': {'file_list': [], 'file_num': 0},
    'TODS': {'file_list': [], 'file_num': 0},
    'WSD': {'file_list': [], 'file_num': 0},
    'YAHOO': {'file_list': [], 'file_num': 0}
                        }
dataset_name_list.append("all_dataset")
dataset_name_dict["all_dataset"] = {'file_num': 0, 'file_list': []}

for i, data_set_choose_file in enumerate(data_set_choose_file_list):
    dataset_name = data_set_choose_file.split("_")[1]
    dataset_name_dict[dataset_name]["file_num"]+= 1
    dataset_name_dict[dataset_name]["file_list"].append(data_set_choose_file)

    dataset_name_dict["all_dataset"]["file_num"]+= 1
    dataset_name_dict["all_dataset"]["file_list"].append(data_set_choose_file)


# 名字要对应
metric_name_list = ['AUC-PR',
                    'AUC-ROC',
                    'VUS-PR',
                    'VUS-ROC',
                    'PATE',
                    'PATE_F1',
                    'MEATA',

                    'Standard-F1',
                    'PA-F1',
                    'R-based-F1',
                    'eTaPR_F1',
                    'Affiliation-F',

]


dataset_name_score_list_dict = {
    'Exathlon': {},
    'IOPS': {},
    'LTDB': {},
    'SED': {},
    'SMD': {},
    'TODS': {},
    'WSD': {},
    'YAHOO': {}
}

for id_dataset_name, dataset_name in enumerate(dataset_name_score_list_dict.keys()):
    metric_score_list_dict = {}

    for id_metric_name, metric_name in enumerate(metric_name_list):

        method_score_list_dict = {}

        for id_method_name, method_name in enumerate(dataset_methods_choose_name_list):
            method_score_list_dict[method_name] = {"score_list":[],"mean_score":None}

        metric_score_list_dict[metric_name] = method_score_list_dict

    dataset_name_score_list_dict[dataset_name] = metric_score_list_dict



file_method_metric_dict = {} # save

for i, data_set_choose_file in enumerate(data_set_choose_file_list):
    dataset_name= data_set_choose_file.split("_")[1]
    file_method_metric_dict[data_set_choose_file] = {}
    data_set_choose_file_path = file_path_dict[data_set_choose_file]
    with open(data_set_choose_file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    for j, dataset_methods_choose_name in enumerate(dataset_methods_choose_name_list):
        print()

        methods_choose_outputs = data[dataset_methods_choose_name]

        gt_array = data["gt"]
        gt_range = data["gt_range"]

        # cal score for all metric

        ori_data_file_path = ori_data_path + data_set_choose_file.replace("_name_list.json",".csv")
        df = pd.read_csv(ori_data_file_path).dropna()

        train_index = data_set_choose_file.split('.')[0].split('_')[-3]

        ori_data = df.iloc[:, 0:-1].values.astype(float)
        label = df['Label'].astype(int).to_numpy()
        slidingWindow = find_length_rank(ori_data, rank=1)
        print("slidingWindow",slidingWindow)


        output = methods_choose_outputs
        output_array = np.array(output)





        time_start = time.time()
        print("file index,method index",i,j)
        print(" dataset_file",data_set_choose_file)
        print(" methods_name",dataset_methods_choose_name)
        metric_score_dict = get_metrics(output_array, label, slidingWindow=slidingWindow,thre=100)
        print(" metric_score_dict",metric_score_dict)
        time_end = time.time()
        print(" all method time_end - time_start",time_end - time_start)
        file_method_metric_dict[data_set_choose_file][dataset_methods_choose_name] = metric_score_dict

res_seve_path = res_save_dir + "metric_calc_res" + choose_method_num+".json"

with open(res_seve_path, 'w', encoding='utf-8') as json_file:
    json.dump(file_method_metric_dict, json_file, indent=4, ensure_ascii=False)

print(f"write to {res_seve_path}")



