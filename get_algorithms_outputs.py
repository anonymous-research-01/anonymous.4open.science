import time
import os
import pandas as pd
import torch
import random, argparse
import json

from sklearn.preprocessing import MinMaxScaler
from experiments.TSB_AD.model_wrapper import *
from experiments.TSB_AD.HP_list import Optimal_Uni_algo_HP_dict, Optimal_Multi_algo_HP_dict
from experiments.TSB_AD.utils.plot_utils import *

# seeding
seed = 2024
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

print("CUDA Available: ", torch.cuda.is_available())
print("cuDNN Version: ", torch.backends.cudnn.version())

if __name__ == '__main__':
    ## ArgumentParser
    parser = argparse.ArgumentParser(description='Running TSB-AD')
    args = parser.parse_args()

    filename_list = []
    filter_ad_pool = ['SR', 'KMeansAD_U', 'Sub_KNN', 'TimesNet', 'CNN', 'Sub_LOF', 'FFT', 'Sub_MCD']
    # print(filter_ad_pool)

    df_exp_files = pd.read_csv("dataset/File_List/exp_file_list.csv").dropna()
    df_exp_files_array = df_exp_files.iloc[:].values
    df_exp_files_list = df_exp_files_array.reshape(df_exp_files_array.shape[0])
    directory_path = "dataset/TSB-AD-U/"
    file_list_file = "dataset/File_List/TSB-AD-U.csv"
    df = pd.read_csv(file_list_file).dropna()
    file_list_ndarray = df[:].values
    file_list = file_list_ndarray.reshape(file_list_ndarray.shape[0])

    # print(file_list)

    dataset_choose_list = [
        # "TODS",
        # "SMD",
        # "Exathlon",
        # "SED",
        # "IOPS",
        # "LTDB",
        "WSD",
        "YAHOO",
    ]

    cal_model_num = len(filter_ad_pool)
    # print(cal_model_num)

    for i, file_name in enumerate(file_list):
        dataset_name = file_name.split('.')[0].split('_')[1]
        if dataset_name not in dataset_choose_list:
            continue
        if file_name not in df_exp_files_list:
            continue

        df = pd.read_csv(directory_path + file_name).dropna()
        data = df.iloc[:, 0:-1].values.astype(float)
        label = df['Label'].astype(int).to_numpy()
        slidingWindow = find_length_rank(data, rank=1)
        train_index = file_name.split('.')[0].split('_')[-3]
        data_train = data[:int(train_index), :]

        range_anomaly = range_convers_new(label)

        label_ranges = [range_anomaly]
        label_array_list = [label]

        # time_start = time.time()
        for idx, ad_name in enumerate(filter_ad_pool[:cal_model_num]):

            if ad_name in Optimal_Uni_algo_HP_dict.keys():
                Optimal_Det_HP = Optimal_Uni_algo_HP_dict[ad_name]
            else:
                Optimal_Det_HP = Optimal_Multi_algo_HP_dict[ad_name]

            # print("=========",idx,ad_name)
            # time_start_single = time.time()
            if ad_name in Semisupervise_AD_Pool:
                output = run_Semisupervise_AD(ad_name, data_train, data, **Optimal_Det_HP)
            elif ad_name in Unsupervise_AD_Pool:
                output = run_Unsupervise_AD(ad_name, data, **Optimal_Det_HP)
            else:
                raise Exception(f"{ad_name} is not defined")

            # time_end_single = time.time()
            # # print("=========", idx, ad_name)
            # # print("single time cost",time_end_single-time_start_single,(time_end_single-time_start_single)/60)

            output = MinMaxScaler(feature_range=(0, 1)).fit_transform(output.reshape(-1, 1)).ravel()
            label_array_list.append(output)

        json_file_seve_path = "../dataset/all_methods_pred_res_min_max_scale/" + file_name.split('.')[
            0] + ".json"

        label_array_list_to_list = []
        for label_array in label_array_list:
            label_array_list_to_list.append(label_array.tolist())

        data_res_only_list = {}

        method_name_list = ["gt"] + filter_ad_pool

        data_res_only_list["gt_range"] = label_ranges

        # choose num
        choose_num = cal_model_num + 1
        for j, method_name in enumerate(method_name_list[:choose_num]):
            data_res_only_list[method_name] = label_array_list_to_list[j]


        def custom_json_dumps(obj, indent=2):
            """
            自定义 JSON 序列化函数：
            - 如果是列表，只对第一层列表进行格式化，内部子列表保持紧凑格式。
            - 如果是字典，进行格式化。
            """
            if isinstance(obj, list):
                # 对第一层列表进行格式化
                items = []
                for item in obj:
                    if isinstance(item, list):
                        # 内部子列表保持紧凑格式
                        item_str = json.dumps(item, separators=(',', ':'))
                    else:
                        # 其他类型递归处理
                        item_str = custom_json_dumps(item, indent)
                    items.append(item_str)
                indent_str = ' ' * indent
                items_str = ',\n'.join([f'{indent_str}{item}' for item in items])
                return f'[\n{items_str}\n]'
            elif isinstance(obj, dict):
                # 字典内容格式化
                items = []
                for key, value in obj.items():
                    # 递归处理每个值
                    value_str = custom_json_dumps(value, indent)
                    items.append(f'"{key}": {value_str}')
                indent_str = ' ' * indent
                items_str = ',\n'.join([f'{indent_str}{item}' for item in items])
                return f'{{\n{items_str}\n}}'
            else:
                # 其他类型直接返回默认处理结果
                return json.dumps(obj)


        def custom_json_dumps_only_list(obj, indent=2):
            """
            自定义 JSON 序列化函数：
            - 如果是字典，格式化（有缩进和换行）。
            - 如果字典的值是列表，整个列表写在冒号后面，且在一行内。
            - 其他内容保持默认处理。
            """
            if isinstance(obj, dict):
                # 字典内容格式化
                items = []
                for key, value in obj.items():
                    # 递归处理每个值
                    value_str = custom_json_dumps(value, indent)
                    # 如果值是列表，直接写在冒号后面，不换行
                    if isinstance(value, list):
                        value_str = json.dumps(value, separators=(',', ':'))
                    items.append(f'"{key}": {value_str}')
                # 添加缩进
                indent_str = ' ' * indent
                items_str = ',\n'.join([f'{indent_str}{item}' for item in items])
                return f'{{\n{items_str}\n}}'
            else:
                # 其他类型直接返回默认处理结果
                return json.dumps(obj)


        json_str_only_list = custom_json_dumps_only_list(data_res_only_list, indent=2)

        with open(json_file_seve_path, "w") as json_file:
            json_file.write(json_str_only_list)