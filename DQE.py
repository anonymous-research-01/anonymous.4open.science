#!/usr/bin/env python3
# -*- coding: utf-8 -*-
###########################
import copy

import numpy as np
import matplotlib as plt
import scipy.integrate as spi

from sortedcontainers import SortedSet
from copy import deepcopy
import numbers
from sklearn.metrics import precision_recall_curve,auc

from metrics.affiliation.generics import convert_vector_to_events
from pate.PATE_utils import clean_and_compute_auc_pr
from config.meata_config import parameter_dict, parameter_w_gt




def cal_x1_x2(VA_gt_len,parameter_rho):
    if parameter_rho <0:
        x1 = VA_gt_len / 4
        x2 = VA_gt_len * 3/4
    else:
        x1 = parameter_rho * VA_gt_len / 2
        x2 = VA_gt_len - (1 - parameter_rho) * VA_gt_len / 2
    return x1, x2



split_line_set = SortedSet()


def split_intervals(a, b):

    split_points = b

    result = []

    for interval in a:
        start, end = interval

        current_splits = [start] + [point for point in split_points if start < point < end] + [end]

        for i in range(len(current_splits) - 1):
            result.append([current_splits[i], current_splits[i + 1]])

    return result



def pred_in_area(pred,area):
    if area == [] or pred== []:
        return False
    return True if pred[0] >= area[0] and pred[1] <= area[1] else False

def cal_integral_in_range_f_power_func(area, parameter_a, parameter_beta,x_tp,area_len,area_start=None,method=1,greater_than_th=False):
    if not greater_than_th:
        f = lambda x, parameter_a, parameter_beta,x_tp,area_len,area_start: parameter_a * pow(x-area_start, parameter_beta)
    else:
        if method ==1:
            f = lambda x, parameter_a, parameter_beta,x_tp,area_len,area_start: parameter_a * pow(x-area_start, parameter_beta)
        else:
            f = lambda x, parameter_a, parameter_beta,x_tp,area_len,area_start: 1/2-pow((area_len-(x-area_start))/parameter_a, 1/parameter_beta)
    result, error = spi.quad(f, area[0], area[1], args=(parameter_a, parameter_beta,x_tp,area_len,area_start))
    return result

def cal_integral_gt_before_ia_power_func(area, parameter_a, parameter_beta,parameter_w_gt=0,parameter_w_near_ngt=0,
                                         parameter_near_single_side_range=None,area_end=None):
    f = lambda x, parameter_a, parameter_beta,parameter_w_gt,parameter_w_near_ngt,\
               parameter_near_single_side_range,area_end: abs(parameter_a * pow((area_end-x)-parameter_near_single_side_range, parameter_beta))
    result, error = spi.quad(f, area[0], area[1], args=(parameter_a, parameter_beta,parameter_w_gt,parameter_w_near_ngt,
                                                        parameter_near_single_side_range,area_end))
    return result

def cal_integral_gt_after_ia_power_func(area, parameter_a, parameter_beta,parameter_w_gt=0,
                                        parameter_near_single_side_range=None,area_start=None):
    f = lambda x, parameter_a, parameter_beta,parameter_w_gt,\
               parameter_near_single_side_range,area_start: abs(parameter_a * pow((x - area_start) - parameter_near_single_side_range, parameter_beta))

    result, error = spi.quad(f, area[0], area[1], args=(parameter_a, parameter_beta,parameter_w_gt,
                                                        parameter_near_single_side_range,area_start))
    return result

def cal_integral_in_range_d_power_func(area, parameter_a, parameter_beta,x_tp,area_len,area_start=None,method=1,greater_than_th=False):
    if not greater_than_th:
        f = lambda x, parameter_a, parameter_beta, x_tp, area_len, area_start: parameter_a * pow((area_len - (x - area_start)), parameter_beta)
    else:
        if method ==1:
            f = lambda x, parameter_a, parameter_beta,x_tp,area_len,area_start: parameter_a * pow((area_len-(x-area_start)), parameter_beta)
        else:
            f = lambda x, parameter_a, parameter_beta,x_tp,area_len,area_start: 1/2-pow((x-area_start)/parameter_a, 1/parameter_beta)
    result, error = spi.quad(f, area[0], area[1], args=(parameter_a, parameter_beta,x_tp,area_len,area_start))
    return result


def cal_integral_in_range_gt_power_func1(area, parameter_b, parameter_gama,x_tp=None,gt_len=None,area_start=None,parameter_rho=0):
    if parameter_rho < 0:
        f = lambda x, parameter_b, parameter_gama,x_tp,gt_len,area_start:  parameter_b * pow(x - area_start, parameter_gama)
    else:
        f = lambda x, parameter_b, parameter_gama,x_tp,gt_len,area_start: parameter_b * pow(x - area_start, parameter_gama) + 1 / 2
    result, error = spi.quad(f, area[0], area[1], args=(parameter_b, parameter_gama,x_tp,gt_len,area_start))
    return result

def cal_integral_in_range_gt_power_func2(area, parameter_b, parameter_gama,x_tp=None,gt_len=None,area_start=None,parameter_rho=0):
    if parameter_rho < 0:
        f = lambda x, parameter_b, parameter_gama,x_tp,gt_len,area_start:  parameter_b * pow(x - area_start, parameter_gama)
    else:
        f = lambda x, parameter_b, parameter_gama,x_tp,gt_len,area_start: 1/2 - parameter_b * pow((2*x_tp-(x-area_start)), parameter_gama) + 1 / 2
    result, error = spi.quad(f, area[0], area[1], args=(parameter_b, parameter_gama,x_tp,gt_len,area_start))
    return result

def cal_integral_in_range_gt_power_func3(area, parameter_b, parameter_gama,x_tp=None,gt_len=None,area_start=None,parameter_rho=0):
    if parameter_rho < 0:
        f = lambda x, parameter_b, parameter_gama,x_tp,gt_len,area_start:  parameter_b * pow(x - area_start, parameter_gama)
    else:
        f = lambda x, parameter_b, parameter_gama,x_tp,gt_len,area_start: 1/2 - parameter_b * pow(gt_len-2*x_tp+(x-area_start), parameter_gama) + 1 / 2
    result, error = spi.quad(f, area[0], area[1], args=(parameter_b, parameter_gama,x_tp,gt_len,area_start))
    return result

def cal_integral_in_range_gt_power_func4(area, parameter_b, parameter_gama,x_tp=None,gt_len=None,area_start=None,parameter_rho=0):
    if parameter_rho < 0:
        f = lambda x, parameter_b, parameter_gama,x_tp,gt_len,area_start:  parameter_b * pow(x - area_start, parameter_gama)
    else:
        f = lambda x, parameter_b, parameter_gama,x_tp,gt_len,area_start: parameter_b * pow(gt_len-(x-area_start), parameter_gama) + 1 / 2
    result, error = spi.quad(f, area[0], area[1], args=(parameter_b, parameter_gama,x_tp,gt_len,area_start))
    return result

def cal_integral_in_range_gt_power_func(area, parameter_b, parameter_gama,parameter_w_gt=0,area_end=None):
    f = lambda x, parameter_b, parameter_gama,area_end: parameter_b * pow(area_end - x, parameter_gama) + (1 - parameter_w_gt)
    result, error = spi.quad(f, area[0], area[1], args=(parameter_b, parameter_gama,area_end))
    return result


def cal_power_function_coefficient(x_tp,parameter_lamda):
    if x_tp == 0:
        return 1,0
    parameter_gama = 2 / (1 - parameter_lamda)-1
    parameter_b = 1 / (4 * pow(x_tp, parameter_gama))
    return parameter_b, parameter_gama

def cal_power_function_parameter_b(parameter_gama,parameter_w_gt,parameter_w_near_ngt,gt_len):
    parameter_b = parameter_w_gt / pow(gt_len, parameter_gama)
    return parameter_b

def power_function(a, b, x):
    return a * pow(x, b)

def va_gt_fun1(x,parameter_b1,parameter_gama1):
    return parameter_b1 * pow(x, parameter_gama1) + 1 / 2

def compute_f1_score(precision, recall):
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def ddl_func(x,max_area_len):
    parameter_alpha = 2
    parameter_a = 1/max_area_len**parameter_alpha
    score = parameter_a*(max_area_len-x)**parameter_alpha
    return score


def ddl_func1(x,max_area_len):
    score = -1 / max_area_len * x + 1
    return score

def DQE_F1(label_interval_ranges, IA1_VA1_IA2_relative_list, window_length, parameter_dict, pred_label_point_array,
           output=None, pred_case_id=None, max_ia_distant_length=-1, thresh_id=None, cal_mode="proportion", find_type="ts_section"):
    print_msg = False
    # print_msg = True

    parameter_w_gt = parameter_dict["parameter_w_gt"]
    parameter_w_near_ngt = parameter_dict["parameter_w_near_ngt"]
    parameter_w_near_2_ngt_gt_left = parameter_dict["parameter_w_near_2_ngt_gt_left"]
    parameter_w_near_0_ngt_gt_right = parameter_dict["parameter_w_near_0_ngt_gt_right"]

    parameter_near_single_side_range = parameter_dict["parameter_near_single_side_range"]
    parameter_gama = parameter_dict["parameter_gama"]
    parameter_distant_method = parameter_dict["parameter_distant_method"]
    parameter_distant_direction = parameter_dict["parameter_distant_direction"]

    distant_method = parameter_distant_method


    IA_dict = {}
    IA_left_dict = {}
    IA_right_dict = {}
    VA_dict = {}
    VA_f_dict = {}
    VA_gt_dict = {}
    VA_d_dict = {}


    # split
    gt_num = len(label_interval_ranges)
    for i, label_interval_range in enumerate(label_interval_ranges):

        VA_gt_now_start, VA_gt_now_end = label_interval_range

        IA_now_mid = None
        IA_after_mid = None
        if i == 0:
            if gt_num == 1:
                VA_f_now_start = max(VA_gt_now_start - parameter_near_single_side_range, 0)
                VA_f_now_end = VA_gt_now_start
                VA_d_now_start = VA_gt_now_end
                VA_d_now_end = min(VA_gt_now_end + parameter_near_single_side_range, window_length)

                IA_dict["id_" + str(i)] = [0, VA_f_now_start]
                IA_left_dict["id_" + str(i)] = [0, IA_dict["id_" + str(i)][1]]


                VA_f_dict["id_" + str(i)] = [VA_f_now_start, VA_f_now_end]
                VA_gt_dict["id_" + str(i)] = [VA_gt_now_start, VA_gt_now_end]
                VA_d_dict["id_" + str(i + 1)] = [VA_d_now_start, VA_d_now_end]

                # next
                IA_dict["id_" + str(i+1)] = [VA_d_now_end, window_length]
                IA_right_dict["id_" + str(i+1)] = [VA_d_now_end, IA_dict["id_" + str(i+1)][1]]

                if parameter_w_near_2_ngt_gt_left <= 0:
                    IA_dict["id_" + str(i)] = [0, VA_gt_now_start]
                    IA_left_dict["id_" + str(i)] = [0, IA_dict["id_" + str(i)][1]]

                    VA_f_dict["id_" + str(i)] = [VA_gt_now_start, VA_gt_now_start]

                if parameter_w_near_0_ngt_gt_right <= 0:
                    IA_dict["id_" + str(i)] = [0, VA_f_now_start]
                    IA_left_dict["id_" + str(i)] = [0, IA_dict["id_" + str(i)][1]]

                    VA_d_dict["id_" + str(i + 1)] = [VA_gt_now_end, VA_gt_now_end]

                    # next
                    IA_dict["id_" + str(i + 1)] = [VA_gt_now_end, window_length]
                    IA_right_dict["id_" + str(i + 1)] = [VA_gt_now_end, IA_dict["id_" + str(i + 1)][1]]



            else:
                VA_f_now_start = max(VA_gt_now_start - parameter_near_single_side_range, 0)
                VA_f_now_end = VA_gt_now_start
                VA_gt_after_start, VA_gt_after_end = label_interval_ranges[i+1]
                VA_d_now_start = VA_gt_now_end
                VA_d_now_end = min(VA_gt_now_end + parameter_near_single_side_range, VA_gt_after_start)

                IA_dict["id_" + str(i)] = [0, VA_f_now_start]
                IA_left_dict["id_" + str(i)] = [0, IA_dict["id_" + str(i)][1]]


                VA_f_dict["id_" + str(i)] = [VA_f_now_start, VA_f_now_end]
                VA_gt_dict["id_" + str(i)] = [VA_gt_now_start, VA_gt_now_end]
                VA_d_dict["id_" + str(i + 1)] = [VA_d_now_start, VA_d_now_end]

                if parameter_w_near_2_ngt_gt_left <= 0:
                    IA_dict["id_" + str(i)] = [0, VA_gt_now_start]
                    IA_left_dict["id_" + str(i)] = [0, IA_dict["id_" + str(i)][1]]

                    VA_f_dict["id_" + str(i)] = [VA_gt_now_start, VA_gt_now_start]

                if parameter_w_near_0_ngt_gt_right <= 0:
                    IA_dict["id_" + str(i)] = [0, VA_f_now_start]
                    IA_left_dict["id_" + str(i)] = [0, IA_dict["id_" + str(i)][1]]

                    VA_d_dict["id_" + str(i + 1)] = [VA_gt_now_end, VA_gt_now_end]


        elif i == gt_num-1:
            VA_gt_before_start, VA_gt_before_end = label_interval_ranges[i - 1]
            VA_f_now_start = max(VA_gt_now_start - parameter_near_single_side_range, VA_gt_before_end)
            VA_f_now_end = VA_gt_now_start
            VA_d_now_start = VA_gt_now_end
            VA_d_now_end = min(VA_gt_now_end + parameter_near_single_side_range, window_length)

            VA_d_before_end = VA_d_dict["id_" + str(i)][1]
            IA_dict["id_" + str(i)] = [VA_d_before_end, VA_f_now_start]
            IA_now_mid = (IA_dict["id_" + str(i)][0] + IA_dict["id_" + str(i)][1]) / 2
            IA_right_dict["id_" + str(i)] = [IA_now_mid, VA_f_now_start]
            IA_left_dict["id_" + str(i)] = [VA_d_before_end, IA_now_mid]


            VA_f_dict["id_" + str(i)] = [VA_f_now_start, VA_f_now_end]
            VA_gt_dict["id_" + str(i)] = [VA_gt_now_start, VA_gt_now_end]
            VA_d_dict["id_" + str(i + 1)] = [VA_d_now_start, VA_d_now_end]

            # next
            IA_dict["id_" + str(i + 1)] = [VA_d_now_end, window_length]
            IA_right_dict["id_" + str(i + 1)] = [IA_dict["id_" + str(i + 1)][0], window_length]

            if parameter_w_near_2_ngt_gt_left <= 0:
                IA_dict["id_" + str(i)] = [VA_d_before_end, VA_gt_now_start]
                IA_now_mid = (IA_dict["id_" + str(i)][0] + IA_dict["id_" + str(i)][1]) / 2
                IA_right_dict["id_" + str(i)] = [IA_now_mid, IA_dict["id_" + str(i)][1]]
                IA_left_dict["id_" + str(i)] = [IA_dict["id_" + str(i)][0], IA_now_mid]

                VA_f_dict["id_" + str(i)] = [VA_gt_now_start, VA_gt_now_start]

            if parameter_w_near_0_ngt_gt_right <= 0:
                IA_dict["id_" + str(i)] = [VA_gt_before_end, VA_f_now_start]
                IA_now_mid = (IA_dict["id_" + str(i)][0] + IA_dict["id_" + str(i)][1]) / 2
                IA_right_dict["id_" + str(i)] = [IA_now_mid, IA_dict["id_" + str(i)][1]]
                IA_left_dict["id_" + str(i)] = [IA_dict["id_" + str(i)][0], IA_now_mid]

                VA_d_dict["id_" + str(i + 1)] = [VA_gt_now_end, VA_gt_now_end]

                # next
                IA_dict["id_" + str(i + 1)] = [VA_gt_now_end, window_length]
                IA_right_dict["id_" + str(i + 1)] = [IA_dict["id_" + str(i + 1)][0], window_length]



        else:
            VA_gt_before_start, VA_gt_before_end = label_interval_ranges[i-1]
            VA_gt_after_start, VA_gt_after_end = label_interval_ranges[i+1]
            VA_f_now_start = max(VA_gt_now_start - parameter_near_single_side_range, VA_gt_before_end)
            VA_f_now_end = VA_gt_now_start
            VA_d_now_start = VA_gt_now_end
            VA_d_now_end = min(VA_gt_now_end + parameter_near_single_side_range, VA_gt_after_start)

            VA_d_before_end = VA_d_dict["id_" + str(i)][1]
            IA_dict["id_" + str(i)] = [VA_d_before_end, VA_f_now_start]
            IA_now_mid = (IA_dict["id_" + str(i)][0] + IA_dict["id_" + str(i)][1]) / 2
            IA_right_dict["id_" + str(i)] = [IA_now_mid, VA_f_now_start]
            IA_left_dict["id_" + str(i)] = [VA_d_before_end, IA_now_mid]

            VA_f_dict["id_" + str(i)] = [VA_f_now_start, VA_f_now_end]
            VA_gt_dict["id_" + str(i)] = [VA_gt_now_start, VA_gt_now_end]
            VA_d_dict["id_" + str(i + 1)] = [VA_d_now_start, VA_d_now_end]

            if parameter_w_near_2_ngt_gt_left <= 0:
                IA_dict["id_" + str(i)] = [VA_d_before_end, VA_gt_now_start]
                IA_now_mid = (IA_dict["id_" + str(i)][0] + IA_dict["id_" + str(i)][1]) / 2
                IA_right_dict["id_" + str(i)] = [IA_now_mid, IA_dict["id_" + str(i)][1]]
                IA_left_dict["id_" + str(i)] = [IA_dict["id_" + str(i)][0], IA_now_mid]

                VA_f_dict["id_" + str(i)] = [VA_gt_now_start, VA_gt_now_start]

            if parameter_w_near_0_ngt_gt_right <= 0:
                IA_dict["id_" + str(i)] = [VA_gt_before_end, VA_f_now_start]
                IA_now_mid = (IA_dict["id_" + str(i)][0] + IA_dict["id_" + str(i)][1]) / 2
                IA_right_dict["id_" + str(i)] = [IA_now_mid, IA_dict["id_" + str(i)][1]]
                IA_left_dict["id_" + str(i)] = [IA_dict["id_" + str(i)][0], IA_now_mid]

                VA_d_dict["id_" + str(i + 1)] = [VA_gt_now_end, VA_gt_now_end]

        # add by order

        if parameter_w_near_2_ngt_gt_left > 0:
            split_line_set.add(VA_f_now_start)
            split_line_set.add(VA_f_now_end)

        split_line_set.add(VA_gt_now_start)
        split_line_set.add(VA_gt_now_end)

        if parameter_w_near_0_ngt_gt_right > 0:
            split_line_set.add(VA_d_now_start)
            split_line_set.add(VA_d_now_end)

        if distant_method == 2:
            if i > 0 and i < gt_num:
                if IA_now_mid != None:
                    split_line_set.add(IA_now_mid)
                if IA_after_mid != None:
                    split_line_set.add(IA_after_mid)

    if print_msg:
        print("IA_dict", IA_dict)
        print("IA_left_dict", IA_left_dict)
        print("IA_right_dict", IA_right_dict)
        # print("VA_dict",VA_dict)
        print("VA_f_dict", VA_f_dict)
        print("VA_gt_dict", VA_gt_dict)
        print("VA_d_dict", VA_d_dict)


    if print_msg:
        print()
        print("split_line_set", split_line_set)

    # third circle
    split_IA1_VA1_IA2_relative_list = split_intervals(IA1_VA1_IA2_relative_list, split_line_set)
    if print_msg:
        print()
        print("split_IA1_VA1_IA2_relative_list",split_IA1_VA1_IA2_relative_list)


    IA_pred_group_dict = {}
    IA_left_pred_group_dict = {}
    IA_right_pred_group_dict = {}
    VA_pred_group_dict = {}
    VA_f_pred_group_dict = {}
    VA_gt_pred_group_dict = {}
    VA_d_pred_group_dict = {}


    func_dict = {}
    func_dict_copy = {}

    for i in range(gt_num):
        IA_pred_group_dict["id_" + str(i)] = []
        VA_pred_group_dict["id_" + str(i)] = []
        VA_f_pred_group_dict["id_" + str(i)] = []
        VA_gt_pred_group_dict["id_" + str(i)] = []
        VA_d_pred_group_dict["id_" + str(i+1)] = []
        if distant_method == 2:
            IA_left_pred_group_dict["id_" + str(i)] = []
            IA_right_pred_group_dict["id_" + str(i+1)] = []

    IA_pred_group_dict["id_" + str(gt_num)] = []

    for i, basic_interval in enumerate(split_IA1_VA1_IA2_relative_list):
        if distant_method == 1:
            # IA
            for j, (area_id, area) in enumerate(IA_dict.items()):
                if pred_in_area(basic_interval, area):
                    if area_id not in IA_pred_group_dict.keys():
                        IA_pred_group_dict[area_id] = [basic_interval]
                    else:
                        IA_pred_group_dict[area_id].append(basic_interval)
        else:
            # IA left
            for j, (area_id, area) in enumerate(IA_left_dict.items()):
                if pred_in_area(basic_interval, area):
                    if area_id not in IA_left_pred_group_dict.keys():
                        IA_left_pred_group_dict[area_id] = [basic_interval]
                    else:
                        IA_left_pred_group_dict[area_id].append(basic_interval)
                        
            # IA right
            for j, (area_id, area) in enumerate(IA_right_dict.items()):
                if pred_in_area(basic_interval, area):
                    if area_id not in IA_right_pred_group_dict.keys():
                        IA_right_pred_group_dict[area_id] = [basic_interval]
                    else:
                        IA_right_pred_group_dict[area_id].append(basic_interval)
                    
        # VA-f
        for j, (area_id, area) in enumerate(VA_f_dict.items()):
            if pred_in_area(basic_interval, area):
                if area_id not in VA_f_pred_group_dict.keys():
                    VA_f_pred_group_dict[area_id] = [basic_interval]
                else:
                    VA_f_pred_group_dict[area_id].append(basic_interval)

        # VA-gt
        for j, (area_id, area) in enumerate(VA_gt_dict.items()):
            if pred_in_area(basic_interval, area):
                if area_id not in VA_gt_pred_group_dict.keys():
                    VA_gt_pred_group_dict[area_id] = [basic_interval]
                else:
                    VA_gt_pred_group_dict[area_id].append(basic_interval)

        # VA-d
        for j, (area_id, area) in enumerate(VA_d_dict.items()):
            area_id_minus_one = "id_" + j.__str__()
            if pred_in_area(basic_interval, area):
                if area_id not in VA_d_pred_group_dict.keys():
                    VA_d_pred_group_dict[area_id] = [basic_interval]
                else:
                    VA_d_pred_group_dict[area_id].append(basic_interval)

            # VA all
            VA_pred_group_dict[area_id_minus_one] = VA_f_pred_group_dict[area_id_minus_one] + \
                                                    VA_gt_pred_group_dict[area_id_minus_one] + \
                                                    VA_d_pred_group_dict[area_id]

    if print_msg:
        if distant_method == 2:
            print("IA_left_pred_group_dict", IA_left_pred_group_dict)
            print("IA_right_pred_group_dict", IA_right_pred_group_dict)

        print("IA_pred_group_dict", IA_pred_group_dict)
        print("VA_pred_group_dict",VA_pred_group_dict)
        print("VA_f_pred_group_dict", VA_f_pred_group_dict)
        print("VA_gt_pred_group_dict",VA_gt_pred_group_dict)
        print("VA_d_pred_group_dict", VA_d_pred_group_dict)

    gt_precision_integration_dict = {}
    gt_recall_integration_dict = {}

    l_ia_before_set_dict = {}
    d_ia_before_mean_set_dict = {}
    d_ia_before_onset_set_dict = {}
    ddl_score_ia_before_dict = {}

    l_ia_after_set_dict = {}
    d_ia_after_mean_set_dict = {}
    d_ia_after_onset_set_dict = {}
    ddl_score_ia_after_dict = {}

    for i, (area_id, area) in enumerate(VA_gt_dict.items()):
        area_id_back = "id_" + (i+1).__str__()
        VA_gt_len = area[1] - area[0]

        parameter_w_gt_recall = 1
        parameter_b_recall= cal_power_function_parameter_b(parameter_gama,parameter_w_gt_recall,0,VA_gt_len)
        parameter_w_gt_precision = 1/2

        parameter_b_precision= cal_power_function_parameter_b(parameter_gama,parameter_w_gt_precision,0,VA_gt_len)

        s_a_gt_precision = cal_integral_in_range_gt_power_func(area, parameter_b_precision, parameter_gama, \
                                                     parameter_w_gt=parameter_w_gt_precision, \
                                                     area_end=area[1])

        s_a_gt_recall = cal_integral_in_range_gt_power_func(area, parameter_b_recall, parameter_gama, \
                                                     parameter_w_gt=parameter_w_gt_recall, \
                                                     area_end=area[1])

        func_dict.update({
            area_id + "_gt_precision": {"area": area,
                              "area_start": area[0],
                              "area_end": area[1],
                              "area_s": s_a_gt_precision,
                              "parameter_a": parameter_b_precision,
                              "parameter_beta": parameter_gama,
                              "func": "va_gt_func_precision",
                              "x_tp": None,
                              "gt_len": VA_gt_len,
                              "area_len": VA_gt_len,
                              "parameter_rho": None,
                              "parameter_w_gt": parameter_w_gt_precision,
                              "parameter_w_near_ngt": 0
                              },
        })

        func_dict.update({
            area_id + "_gt_recall": {"area": area,
                              "area_start": area[0],
                              "area_end": area[1],
                              "area_s": s_a_gt_recall,
                              "parameter_a": parameter_b_recall,
                              "parameter_beta": parameter_gama,
                              "func": "va_gt_func_recall",
                              "x_tp": None,
                              "gt_len": VA_gt_len,
                              "area_len": VA_gt_len,
                              "parameter_rho": None,
                              "parameter_w_gt": parameter_w_gt_recall,
                              "parameter_w_near_ngt": 0
                              },
        })




        # f
        VA_f_area = VA_f_dict[area_id]
        VA_f_len = VA_f_area[1] - VA_f_area[0]
        VA_f_start = VA_f_area[0]
        VA_f_end = VA_f_area[1]
        parameter_beta_left = parameter_gama
        parameter_a_left = (1 - parameter_w_gt_precision) / pow(parameter_near_single_side_range,parameter_gama)
        parameter_a_right = parameter_a_left
        parameter_beta_right = parameter_beta_left
        s_gt_before = cal_integral_gt_before_ia_power_func(VA_f_area,
                                                                      parameter_a_left,
                                                                      parameter_beta_left,
                                                                      parameter_near_single_side_range=parameter_near_single_side_range,
                                                                      area_end=VA_f_end)
        # d
        VA_d_area = VA_d_dict[area_id_back]
        VA_d_len = VA_d_area[1] - VA_d_area[0]
        VA_d_start = VA_d_area[0]


        # func_dict[area_id + "_f_a"] = {
        func_dict_copy[area_id + "_f_a"] = {
            "area": VA_f_area,
            "area_start": VA_f_area[0],
            "area_end": VA_f_area[1],
            "area_s": s_gt_before,
            "parameter_a": parameter_a_left,
            "parameter_beta": parameter_beta_left,
            "func": "ia_before_func",
            # "func": "va_f_reverse_func",
            "x_tp": None,
            "gt_len": None,
            "area_len": VA_f_len,
            # "greater_than_th": greater_than_th
            "parameter_w_gt": parameter_w_gt_precision,
            "parameter_w_near_ngt": parameter_w_near_ngt,
            "parameter_near_single_side_range": parameter_near_single_side_range,
        }

        # func_dict[area_id + "_d_a"] = {
        func_dict_copy[area_id + "_d_a"] = {
            "area": VA_d_area,
            "area_start": VA_d_area[0],
            "area_end": VA_d_area[1],
            "area_s": s_gt_before,
            "parameter_a": parameter_a_right,
            "parameter_beta": parameter_beta_right,
            "func": "ia_after_func",
            # "func": "va_f_reverse_func",
            "x_tp": None,
            "gt_len": None,
            "area_len": VA_d_len,
            # "greater_than_th": greater_than_th
            "parameter_w_gt": parameter_w_gt_precision,
            "parameter_w_near_ngt": parameter_w_near_ngt,
            "parameter_near_single_side_range": parameter_near_single_side_range,
        }

        # plot_func_multi_paper(func_dict, window_length)

        if i == 2:
            d = 1
        gt_precision_integration_dict[area_id] = s_a_gt_precision
        gt_recall_integration_dict[area_id] = s_a_gt_recall


        va_f_area = VA_f_dict[area_id]
        VA_f_len = va_f_area[1] - va_f_area[0]
        VA_f_start = va_f_area[0]
        VA_f_end = va_f_area[1]

        VA_d_area = VA_d_dict[area_id_back]
        VA_d_len = VA_d_area[1] - VA_d_area[0]
        VA_d_start = VA_d_area[0]
        VA_d_end = VA_d_area[1]

        # va before
        VA_f_i_pred_group = VA_f_pred_group_dict[area_id]
        VA_d_i_after_pred_group = VA_d_pred_group_dict[area_id_back]

        l_ia_before_set_sum = 0
        d_ia_before_mean_set_sum = 0
        d_ia_before_onset_set_sum = 0
        ia_set_num = len(VA_f_i_pred_group)

        VA_f_i_pred_group_len = len(VA_f_i_pred_group)
        for interval_idx,basic_interval in enumerate(VA_f_i_pred_group):
            l_ia_before_set_sum+= basic_interval[1] - basic_interval[0]
            d_ia_before_mean_set_sum+= abs(VA_f_end - (basic_interval[1] + basic_interval[0])/2)

            if interval_idx == VA_f_i_pred_group_len-1:
                d_ia_before_onset_set_sum += abs(VA_f_end - basic_interval[1])


        l_ia_before_set_dict[area_id] = l_ia_before_set_sum
        d_ia_before_mean =  d_ia_before_mean_set_sum/ia_set_num if ia_set_num != 0 else 0
        d_ia_before_mean_set_dict[area_id] = d_ia_before_mean
        d_ia_before_onset =  d_ia_before_onset_set_sum/ia_set_num if ia_set_num != 0 else 0
        d_ia_before_onset_set_dict[area_id] = d_ia_before_onset

        score_ia_near_before_mean_d = ddl_func1(d_ia_before_mean,parameter_near_single_side_range) if parameter_near_single_side_range !=0 else 1
        score_ia_near_before_onset_d = ddl_func1(d_ia_before_onset,parameter_near_single_side_range) if parameter_near_single_side_range !=0 else 1

        score_ia_near_before_l = ddl_func1(l_ia_before_set_sum, parameter_near_single_side_range) if parameter_near_single_side_range !=0 else 1

        dl_score_ia_before = score_ia_near_before_mean_d * \
                                 score_ia_near_before_onset_d * \
                                 score_ia_near_before_l

        if print_msg:
            print()
            print("area_id", area_id)
            print(" score_ia_near_before")
            print(" score_ia_near_before_mean_d", score_ia_near_before_mean_d)
            print(" score_ia_near_before_onset_d", score_ia_near_before_onset_d)
            print(" score_ia_near_before_l", score_ia_near_before_l)
            print(" dl_score_ia_before", dl_score_ia_before)

            
        ddl_score_ia_before_dict[area_id] = dl_score_ia_before


        # va after
        l_ia_after_set_sum = 0
        d_ia_after_mean_set_sum = 0
        d_ia_after_onset_set_sum = 0
        ia_set_num = len(VA_d_i_after_pred_group)


        for interval_idx,basic_interval in enumerate(VA_d_i_after_pred_group):
            l_ia_after_set_sum += basic_interval[1] - basic_interval[0]
            d_ia_after_mean_set_sum += abs((basic_interval[1] + basic_interval[0]) / 2 - VA_d_start)

            if interval_idx == 0:
                d_ia_after_onset_set_sum = abs(VA_d_start - basic_interval[0])

        l_ia_after_set_dict[area_id_back] = l_ia_after_set_sum
        d_ia_after_mean = d_ia_after_mean_set_sum / ia_set_num if ia_set_num != 0 else 0
        d_ia_after_mean_set_dict[area_id_back] = d_ia_after_mean
        d_ia_after_onset = d_ia_after_onset_set_sum / ia_set_num if ia_set_num != 0 else 0
        d_ia_after_onset_set_dict[area_id_back] = d_ia_after_onset


        score_ia_near_after_mean_d = ddl_func1(d_ia_after_mean,parameter_near_single_side_range) if parameter_near_single_side_range !=0 else 1
        score_ia_near_after_onset_d = ddl_func1(d_ia_after_onset,parameter_near_single_side_range) if parameter_near_single_side_range !=0 else 1
        score_ia_near_after_l = ddl_func1(l_ia_after_set_sum, parameter_near_single_side_range) if parameter_near_single_side_range !=0 else 1


        if thresh_id == 50:
            d = 1

        dl_score_ia_after = score_ia_near_after_mean_d*\
                                score_ia_near_after_onset_d*\
                                score_ia_near_after_l

        if print_msg:
            print()
            print("area_id", area_id)
            print(" score_ia_near_after")
            print(" score_ia_near_after_mean_d", score_ia_near_after_mean_d)
            print(" score_ia_near_after_onset_d", score_ia_near_after_onset_d)
            print(" score_ia_near_after_l", score_ia_near_after_l)
            print(" dl_score_ia_after", dl_score_ia_after)

        ddl_score_ia_after_dict[area_id_back] = dl_score_ia_after

    if print_msg:
        print()
        print("ia_near_before")
        print(" d_ia_before_mean_set_dict", d_ia_before_mean_set_dict)
        print(" d_ia_before_onset_set_dict", d_ia_before_onset_set_dict)
        print(" l_ia_before_set_dict", l_ia_before_set_dict)
        print(" ddl_score_ia_before_dict", ddl_score_ia_before_dict)
        print()
        print("ia_near_after")
        print(" d_ia_after_mean_set_dict", d_ia_after_mean_set_dict)
        print(" d_ia_after_onset_set_dict", d_ia_after_onset_set_dict)
        print(" l_ia_after_set_dict", l_ia_after_set_dict)
        print(" ddl_score_ia_after_dict", ddl_score_ia_after_dict)


    ddl_score_ia_distant_dict = {}
    if distant_method ==1:

        if print_msg:
            print()
            print("max_ia_distant_length", max_ia_distant_length)

        l_ia_distant_set_dict = {}
        d_ia_distant_mean_set_dict = {}
        d_ia_distant_onset_set_dict = {}

        for i, (area_id, basic_interval_list) in enumerate(IA_pred_group_dict.items()):

            l_ia_distant_set_sum = 0
            d_ia_distant_mean_set_sum = 0
            d_ia_distant_onset_set_sum = 0
            ia_set_num = len(basic_interval_list)

            IA_mid = (IA_dict[area_id][0] + IA_dict[area_id][1]) / 2
            IA_len = IA_dict[area_id][1] - IA_dict[area_id][0]
            IA_start = IA_dict[area_id][0]
            IA_end = IA_dict[area_id][1]
            
            for interval_idx, basic_interval in enumerate(basic_interval_list):
                basic_interval_mid = (basic_interval[1] + basic_interval[0]) / 2
                if parameter_distant_direction == "both":
                    l_ia_distant_set_sum += basic_interval[1] - basic_interval[0]
                    if i == 0:
                        base_position = IA_dict[area_id][1]
                        d_ia_distant_mean_set_sum += abs(basic_interval_mid - base_position)
                    elif i == gt_num:
                        base_position = IA_dict[area_id][0]
                        d_ia_distant_mean_set_sum += abs(basic_interval_mid - base_position)
                    else:
                        d_ia_distant_mean_set_sum += min(abs(basic_interval_mid - IA_dict[area_id][0]),\
                                                         abs(basic_interval_mid - IA_dict[area_id][1]))

                    if i == 0:
                        if interval_idx == ia_set_num - 1:
                            d_ia_distant_onset_set_sum = abs(IA_end - basic_interval[1])
                    elif i == gt_num:
                        if interval_idx == 0:
                            d_ia_distant_onset_set_sum = abs(basic_interval[0] - IA_start)
                    else:
                        if interval_idx == 0:
                            d_ia_distant_onset_set_sum = min(d_ia_distant_onset_set_sum,abs(basic_interval[0] - IA_start))
                        if interval_idx == ia_set_num - 1:
                            d_ia_distant_onset_set_sum = min(d_ia_distant_onset_set_sum,abs(IA_end - basic_interval[1]))
                elif parameter_distant_direction == "left":
                    l_ia_distant_set_sum += basic_interval[1] - basic_interval[0]
                    d_ia_distant_mean_set_sum += abs(basic_interval_mid - IA_dict[area_id][0])
                    if interval_idx == 0:
                        d_ia_distant_onset_set_sum = abs(basic_interval[0] - IA_start)


            l_ia_distant_set_dict[area_id] = l_ia_distant_set_sum
            d_ia_distant_mean = d_ia_distant_mean_set_sum / ia_set_num if ia_set_num != 0 else 0
            d_ia_distant_mean_set_dict[area_id] = d_ia_distant_mean
            d_ia_distant_onset = d_ia_distant_onset_set_sum
            d_ia_distant_onset_set_dict[area_id] = d_ia_distant_onset

            score_ia_distant_mean_d = ddl_func1(d_ia_distant_mean,max_ia_distant_length) if max_ia_distant_length !=0 else 1
            score_ia_distant_onset_d = ddl_func1(d_ia_distant_onset,max_ia_distant_length) if max_ia_distant_length !=0 else 1

            score_ia_distant_l = ddl_func(l_ia_distant_set_sum, max_ia_distant_length) if max_ia_distant_length !=0 else 1


            dl_score_ia_distant = score_ia_distant_mean_d * \
                                         score_ia_distant_onset_d * \
                                         score_ia_distant_l

            if print_msg:
                print()
                print("area_id", area_id)
                print(" score_ia_distant")
                print(" score_ia_distant_mean_d", score_ia_distant_mean_d)
                print(" score_ia_distant_onset_d", score_ia_distant_onset_d)
                print(" score_ia_distant_l", score_ia_distant_l)
                print(" dl_score_ia_distant", dl_score_ia_distant)


            ddl_score_ia_distant_dict[area_id] = dl_score_ia_distant
    elif distant_method == 3:

        if print_msg:
            print()
            print("max_ia_distant_length", max_ia_distant_length)

        l_ia_distant_set_dict = {}
        d_ia_distant_mean_set_dict = {}
        d_ia_distant_onset_set_dict = {}

        # 5th circle
        for i, (area_id, basic_interval_list) in enumerate(IA_pred_group_dict.items()):

            l_ia_distant_set_sum = 0
            d_ia_distant_mean_set_sum = 0
            d_ia_distant_onset_set_sum = 0
            ia_set_num = len(basic_interval_list)

            IA_mid = (IA_dict[area_id][0] + IA_dict[area_id][1]) / 2
            IA_len = IA_dict[area_id][1] - IA_dict[area_id][0]
            IA_start = IA_dict[area_id][0]
            IA_end = IA_dict[area_id][1]

            for interval_idx, basic_interval in enumerate(basic_interval_list):
                l_ia_distant_set_sum += basic_interval[1] - basic_interval[0]
                if i == 0:
                    base_position = IA_dict[area_id][1]
                    d_ia_distant_mean_set_sum += abs((basic_interval[1] + basic_interval[0]) / 2 - base_position)
                elif i == gt_num:
                    base_position = IA_dict[area_id][0]
                    d_ia_distant_mean_set_sum += abs((basic_interval[1] + basic_interval[0]) / 2 - base_position)
                else:
                    basic_interval_mid = (basic_interval[1] + basic_interval[0]) / 2
                    d_ia_distant_mean_set_sum += min(abs(basic_interval_mid - IA_dict[area_id][0]), \
                                                     abs(basic_interval_mid - IA_dict[area_id][1]))

                if i == 0:
                    if interval_idx == ia_set_num - 1:
                        d_ia_distant_onset_set_sum = abs(IA_end - basic_interval[1])
                elif i == gt_num:
                    if interval_idx == 0:
                        d_ia_distant_onset_set_sum = abs(basic_interval[0] - IA_start)
                else:
                    if interval_idx == 0:
                        d_ia_distant_onset_set_sum = min(d_ia_distant_onset_set_sum, abs(basic_interval[0] - IA_start))
                    if interval_idx == ia_set_num - 1:
                        d_ia_distant_onset_set_sum = min(d_ia_distant_onset_set_sum, abs(IA_end - basic_interval[1]))

            l_ia_distant_set_dict[area_id] = l_ia_distant_set_sum
            d_ia_distant_mean = d_ia_distant_mean_set_sum / ia_set_num if ia_set_num != 0 else 0
            d_ia_distant_mean_set_dict[area_id] = d_ia_distant_mean
            # d_ia_distant_onset = d_ia_distant_onset_set_sum / ia_set_num if ia_set_num != 0 else 0
            d_ia_distant_onset = d_ia_distant_onset_set_sum
            d_ia_distant_onset_set_dict[area_id] = d_ia_distant_onset

            score_ia_distant_mean_d = ddl_func1(d_ia_distant_mean,
                                                max_ia_distant_length) if max_ia_distant_length != 0 else 1
            score_ia_distant_onset_d = ddl_func1(d_ia_distant_onset,
                                                 max_ia_distant_length) if max_ia_distant_length != 0 else 1

            score_ia_distant_l = ddl_func(l_ia_distant_set_sum,
                                          max_ia_distant_length) if max_ia_distant_length != 0 else 1


            dl_score_ia_distant = score_ia_distant_mean_d * \
                                   score_ia_distant_onset_d * \
                                  score_ia_distant_l

            if print_msg:
                print()
                print("area_id", area_id)
                print(" score_ia_distant")
                print(" score_ia_distant_mean_d", score_ia_distant_mean_d)
                print(" score_ia_distant_onset_d", score_ia_distant_onset_d)
                print(" score_ia_distant_l", score_ia_distant_l)
                print(" dl_score_ia_distant", dl_score_ia_distant)

            ddl_score_ia_distant_dict[area_id] = dl_score_ia_distant

    else:
        # max_ia_distant_length_half = max_ia_distant_length/2

        if print_msg:
            print()
            print("max_ia_distant_length", max_ia_distant_length)

        l_ia_distant_before_set_dict = {}
        d_ia_distant_before_mean_set_dict = {}
        d_ia_distant_before_onset_set_dict = {}
        ddl_score_ia_distant_before_dict = {}

        l_ia_distant_after_set_dict = {}
        d_ia_distant_after_mean_set_dict = {}
        d_ia_distant_after_onset_set_dict = {}
        ddl_score_ia_distant_after_dict = {}
        # cal label_function_gt
        for i, (area_id, area) in enumerate(VA_gt_dict.items()):
            area_id_back = "id_" + (i + 1).__str__()


            IA_f_area = IA_left_dict[area_id]
            IA_f_len = IA_f_area[1] - IA_f_area[0]
            IA_f_start = IA_f_area[0]
            IA_f_end = IA_f_area[1]

            IA_d_area = IA_right_dict[area_id_back]
            IA_d_len = IA_d_area[1] - IA_d_area[0]
            IA_d_start = IA_d_area[0]
            IA_d_end = IA_d_area[1]

            # ia distant before
            IA_f_i_pred_group = IA_left_pred_group_dict[area_id]
            IA_d_i_after_pred_group = IA_right_pred_group_dict[area_id_back]

            l_ia_distant_before_set_sum = 0
            d_ia_distant_before_mean_set_sum = 0
            d_ia_distant_before_onset_set_sum = 0
            ia_distant_before_set_num = len(IA_f_i_pred_group)

            for interval_idx, basic_interval in enumerate(IA_f_i_pred_group):
                l_ia_distant_before_set_sum += basic_interval[1] - basic_interval[0]

                d_ia_distant_before_mean_set_sum += abs(IA_f_end - (basic_interval[1] + basic_interval[0]) / 2)

                if interval_idx == ia_distant_before_set_num - 1:
                    d_ia_distant_before_onset_set_sum += abs(IA_f_end - basic_interval[1])

            l_ia_distant_before_set_dict[area_id] = l_ia_distant_before_set_sum
            d_ia_distant_before_mean = d_ia_distant_before_mean_set_sum / ia_distant_before_set_num if ia_distant_before_set_num != 0 else 0
            d_ia_distant_before_mean_set_dict[area_id] = d_ia_distant_before_mean
            d_ia_distant_before_onset = d_ia_distant_before_onset_set_sum
            d_ia_distant_before_onset_set_dict[area_id] = d_ia_distant_before_onset


            score_ia_distant_before_mean_d = ddl_func1(d_ia_distant_before_mean,
                                                                                 max_ia_distant_length) if max_ia_distant_length !=0 else 1
            score_ia_distant_before_onset_d = ddl_func1(d_ia_distant_before_onset,
                                                                                   max_ia_distant_length) if max_ia_distant_length !=0 else 1
            score_ia_distant_before_l = ddl_func1(l_ia_distant_before_set_sum, IA_f_len) if IA_f_len !=0 else 1

            dl_score_ia_distant_before = score_ia_distant_before_mean_d * \
                                         score_ia_distant_before_onset_d * \
                                         score_ia_distant_before_l

            if print_msg:
                print()
                print("area_id", area_id)
                print(" score_ia_distant_before")
                print(" score_ia_distant_before_mean_d", score_ia_distant_before_mean_d)
                print(" score_ia_distant_before_onset_d", score_ia_distant_before_onset_d)
                print(" score_ia_distant_before_l", score_ia_distant_before_l)
                print(" dl_score_ia_distant_before", dl_score_ia_distant_before)

            ddl_score_ia_distant_before_dict[area_id] = dl_score_ia_distant_before

            # va after
            l_ia_distant_after_set_sum = 0
            d_ia_distant_after_mean_set_sum = 0
            d_ia_distant_after_onset_set_sum = 0
            ia_distant_after_set_num = len(IA_d_i_after_pred_group)

            for interval_idx, basic_interval in enumerate(IA_d_i_after_pred_group):
                l_ia_distant_after_set_sum += basic_interval[1] - basic_interval[0]
                d_ia_distant_after_mean_set_sum += abs((basic_interval[1] + basic_interval[0]) / 2 - IA_d_start)

                if interval_idx == 0:
                    d_ia_distant_after_onset_set_sum = abs(IA_d_start - basic_interval[0])

            l_ia_distant_after_set_dict[area_id_back] = l_ia_distant_after_set_sum
            d_ia_distant_after_mean = d_ia_distant_after_mean_set_sum / ia_distant_after_set_num if ia_distant_after_set_num != 0 else 0
            d_ia_distant_after_mean_set_dict[area_id_back] = d_ia_distant_after_mean
            d_ia_distant_after_onset = d_ia_distant_after_onset_set_sum
            d_ia_distant_after_onset_set_dict[area_id_back] = d_ia_distant_after_onset

            score_ia_distant_after_mean_d = ddl_func1(d_ia_distant_after_mean, max_ia_distant_length) if max_ia_distant_length !=0 else 1
            score_ia_distant_after_onset_d = ddl_func1(d_ia_distant_after_onset, max_ia_distant_length) if max_ia_distant_length !=0 else 1
            score_ia_distant_after_l = ddl_func1(l_ia_distant_after_set_sum, IA_d_len) if IA_d_len !=0 else 1

            dl_score_ia_distant_after = score_ia_distant_after_mean_d * \
                                        score_ia_distant_after_onset_d * \
                                        score_ia_distant_after_l

            if print_msg:
                print()
                print("score_ia_distant_after")
                print(" score_ia_distant_after_mean_d",score_ia_distant_after_mean_d)
                print(" score_ia_distant_after_onset_d",score_ia_distant_after_onset_d)
                print(" score_ia_distant_after_l",score_ia_distant_after_l)
                print(" dl_score_ia_distant_after",dl_score_ia_distant_after)

            ddl_score_ia_distant_after_dict[area_id_back] = dl_score_ia_distant_after



    if print_msg:
        if distant_method == 1:
            print()
            print("ia_distant")
            print(" d_ia_distant_mean_set_dict", d_ia_distant_mean_set_dict)
            print(" d_ia_distant_onset_set_dict", d_ia_distant_onset_set_dict)
            print(" l_ia_distant_set_dict", l_ia_distant_set_dict)
            print(" ddl_score_ia_distant_dict", ddl_score_ia_distant_dict)
        else:
            print()
            print("ia_distant_left")
            print(" d_ia_distant_before_mean_set_dict", d_ia_distant_before_mean_set_dict)
            print(" d_ia_distant_before_onset_set_dict", d_ia_distant_before_onset_set_dict)
            print(" l_ia_distant_before_set_dict", l_ia_distant_before_set_dict)
            print(" ddl_score_ia_distant_before_dict", ddl_score_ia_distant_before_dict)
            print()
            print("ia_distant_right")
            print(" d_ia_distant_after_mean_set_dict", d_ia_distant_after_mean_set_dict)
            print(" d_ia_distant_after_onset_set_dict", d_ia_distant_after_onset_set_dict)
            print(" l_ia_distant_after_set_dict", l_ia_distant_after_set_dict)
            print(" ddl_score_ia_distant_after_dict", ddl_score_ia_distant_after_dict)

    # adjust
    # adjust ia


    adjusted_ddl_score_ia_near_left_dict = deepcopy(ddl_score_ia_before_dict)
    adjusted_ddl_score_ia_near_right_dict = deepcopy(ddl_score_ia_after_dict)
    adjusted_ddl_score_ia_near_dict_ratio = {}
    # 默认得分是1
    if distant_method == 1:
        adjusted_ddl_score_ia_dict = deepcopy(ddl_score_ia_distant_dict) # 默认
        adjusted_ddl_score_ia_dict_ratio_contribution = {} # distant
        adjusted_ddl_score_ia_distant_dict_ratio = {}
    else:
        adjusted_ddl_score_ia_distant_left_dict = deepcopy(ddl_score_ia_distant_before_dict) # distant left
        adjusted_ddl_score_ia_distant_right_dict = deepcopy(ddl_score_ia_distant_after_dict) # distant right
        adjusted_ddl_score_ia_distant_both_sides_dict_ratio = {}

        # adjust ia near ratio
    for i in range(gt_num + 1):
        area_id = "id_" + str(i)
        area_id_front = "id_" + str(i - 1)
        area_id_back = "id_" + str(i + 1)


        ia_near_ratio_before = parameter_w_near_0_ngt_gt_right
        ia_near_ratio_after = parameter_w_near_2_ngt_gt_left

        # adjust ia near weight
        # ia near before:0-gt_num-1
        # ia near after:1-gt_num
        if parameter_w_near_2_ngt_gt_left > 0 and parameter_w_near_0_ngt_gt_right > 0:
            if i == 0:
                ia_near_ratio_before = 0
                if VA_f_dict[area_id][0] >= VA_f_dict[area_id][1] and VA_d_dict[area_id_back][0] < VA_d_dict[area_id_back][1]:
                        ia_near_ratio_after = 0

            elif i == gt_num:
                ia_near_ratio_after = 0
                if VA_d_dict[area_id][0] >= VA_d_dict[area_id][1] and VA_f_dict[area_id_front][0] < VA_f_dict[area_id_front][1]:
                        ia_near_ratio_before = 0
            else:
                if i == 1:
                    if VA_d_dict[area_id][0] < VA_d_dict[area_id][1] and VA_f_dict[area_id_front][0] >= VA_f_dict[area_id_front][1]:
                            ia_near_ratio_before = 1
                if i == gt_num-1:
                    if VA_f_dict[area_id][0] < VA_f_dict[area_id][1] and VA_d_dict[area_id_back][0] >= VA_d_dict[area_id_back][1]:
                            ia_near_ratio_after = 1

        adjusted_ddl_score_ia_near_dict_ratio[area_id] = [ia_near_ratio_before,ia_near_ratio_after]

    # adjust ia near score
    for i in range(gt_num):
        area_id = "id_" + str(i)
        area_id_front = "id_" + str(i - 1)
        area_id_back = "id_" + str(i + 1)
        # adjust left
        if parameter_w_near_2_ngt_gt_left > 0:
            if VA_f_pred_group_dict[area_id] == [] \
                    and VA_gt_pred_group_dict[area_id] == [] and VA_d_pred_group_dict[area_id_back] == []:
                adjusted_ddl_score_ia_near_left_dict[area_id] = 0
        else:
            adjusted_ddl_score_ia_near_left_dict[area_id] = 0

        # adjust right
        if parameter_w_near_0_ngt_gt_right > 0:
            if VA_d_pred_group_dict[area_id_back] == [] \
                    and VA_gt_pred_group_dict[area_id] == [] and VA_f_pred_group_dict[area_id] == []:
                adjusted_ddl_score_ia_near_right_dict[area_id_back] = 0
        else:
            adjusted_ddl_score_ia_near_right_dict[area_id_back] = 0
    
    # two logics
    if distant_method == 1:

        for i in range(gt_num + 1):
            area_id = "id_" + str(i)
            area_id_front = "id_" + str(i - 1)
            area_id_back = "id_" + str(i + 1)

            ia_distant_ratio_before = 1 / 2
            ia_distant_ratio_after = 1 / 2

            # adjust ia distant
            if i == 0:
                ia_distant_ratio_before = 0
                if VA_pred_group_dict[area_id] == [] and IA_pred_group_dict[area_id] == []:
                    adjusted_ddl_score_ia_dict[area_id] = 0
                if IA_dict[area_id][0] >= IA_dict[area_id][1] and IA_dict[area_id_back][0] < IA_dict[area_id_back][1]:
                    ia_distant_ratio_after = 0
            elif i == gt_num:
                # 默认
                ia_distant_ratio_after = 0
                if VA_pred_group_dict[area_id_front] == [] and IA_pred_group_dict[area_id] == []:
                    adjusted_ddl_score_ia_dict[area_id] = 0
                if IA_dict[area_id][0] >= IA_dict[area_id][1] and IA_dict[area_id_front][0] < IA_dict[area_id_front][1]:
                    ia_distant_ratio_before = 0
            else:
                if IA_pred_group_dict[area_id] == [] and \
                        VA_pred_group_dict[area_id_front] == [] and VA_pred_group_dict[area_id] == []:
                    adjusted_ddl_score_ia_dict[area_id] = 0

                if i == 1:
                    if IA_dict[area_id][0] < IA_dict[area_id][1] and IA_dict[area_id_front][0] >= IA_dict[area_id_front][1]:
                            ia_distant_ratio_before = 1
                if i == gt_num-1:
                    if IA_dict[area_id][0] < IA_dict[area_id][1] and IA_dict[area_id_back][0] >= IA_dict[area_id_back][1]:
                            ia_distant_ratio_after = 1

            adjusted_ddl_score_ia_distant_dict_ratio[area_id] = [ia_distant_ratio_before,ia_distant_ratio_after]
            ddl_score_ia_i_before_contribution = adjusted_ddl_score_ia_dict[area_id] * ia_distant_ratio_before
            ddl_score_ia_i_after_contribution = adjusted_ddl_score_ia_dict[area_id] * ia_distant_ratio_after
            adjusted_ddl_score_ia_dict_ratio_contribution[area_id] = [ddl_score_ia_i_before_contribution,
                                                                     ddl_score_ia_i_after_contribution]
    else:
        for i in range(gt_num + 1):
            area_id = "id_" + str(i)
            area_id_front = "id_" + str(i - 1)
            area_id_back = "id_" + str(i + 1)


            ia_distant_ratio_before = 1 / 2
            ia_distant_ratio_after = 1 / 2

            # adjust ia distant weight
            # ia distant before:0-gt_num-1 
            # ia distant after:1-gt_num 
            if i == 0:
                ia_distant_ratio_before = 0
                if IA_left_dict[area_id][0] >= IA_left_dict[area_id][1] and IA_right_dict[area_id_back][0] < IA_right_dict[area_id_back][1]:
                        ia_distant_ratio_after = 0

            elif i == gt_num:
                ia_distant_ratio_after = 0
                if IA_right_dict[area_id][0] >= IA_right_dict[area_id][1] and IA_left_dict[area_id_front][0] < IA_left_dict[area_id_front][1]:
                        ia_distant_ratio_before = 0
            else:
                if i == 1:
                    if IA_right_dict[area_id][0] < IA_right_dict[area_id][1] and IA_left_dict[area_id_front][0] >= IA_left_dict[area_id_front][1]:
                            ia_distant_ratio_before = 1
                if i == gt_num-1:
                    if IA_left_dict[area_id][0] < IA_left_dict[area_id][1] and IA_right_dict[area_id_back][0] >= IA_right_dict[area_id_back][1]:
                            ia_distant_ratio_after = 1

            adjusted_ddl_score_ia_distant_both_sides_dict_ratio[area_id] = [ia_distant_ratio_before,ia_distant_ratio_after]

        # adjust ia distant
        for i in range(gt_num):
            area_id = "id_" + str(i)
            area_id_front = "id_" + str(i - 1)
            area_id_back = "id_" + str(i + 1)
            # adjust left
            if IA_left_pred_group_dict[area_id] == [] \
                    and VA_pred_group_dict[area_id] == [] and IA_right_pred_group_dict[area_id_back] == []:
                adjusted_ddl_score_ia_distant_left_dict[area_id] = 0

            # adjust right
            if IA_right_pred_group_dict[area_id_back] == [] \
                    and VA_pred_group_dict[area_id] == [] and IA_left_pred_group_dict[area_id] == []:
                adjusted_ddl_score_ia_distant_right_dict[area_id_back] = 0


    # cal recall,precision

    local_recall_matrix = []
    local_precision_matrix = []
    local_f1_matrix = []
    local_fq_near_matrix = []
    local_fq_distant_matrix = []



    recall_va_dict = {}
    detection_score_ia_i_dict = {}
    precision_va_dict = {}

    pred_func_dict = {}
    basic_interval_precision_integral_dict = {}
    basic_interval_recall_integral_dict = {}

    gt_detected_num = 0
    for i in range(gt_num):
        area_id = "id_" + str(i)
        area_id_back = "id_" + str(i + 1)

        VA_gt_area = VA_gt_dict[area_id]
        VA_gt_len = VA_gt_dict[area_id][1] - VA_gt_dict[area_id][0]
        VA_gt_start = VA_gt_dict[area_id][0]
        VA_gt_end = VA_gt_dict[area_id][1]

        VA_f_area = VA_f_dict[area_id]
        if VA_f_area != []:
            VA_f_len = VA_f_dict[area_id][1] - VA_f_dict[area_id][0]
            VA_f_start = VA_f_dict[area_id][0]
            VA_f_end = VA_f_dict[area_id][1]

        VA_d_area = VA_d_dict[area_id_back]
        if VA_d_area != []:
            VA_d_len = VA_d_dict[area_id_back][1] - VA_d_dict[area_id_back][0]
            VA_d_start = VA_d_dict[area_id_back][0]
            VA_d_end = VA_d_dict[area_id_back][1]

        VA_f_i_pred_group = VA_f_pred_group_dict[area_id]
        VA_gt_i_pred_group = VA_gt_pred_group_dict[area_id]
        VA_d_i_after_pred_group = VA_d_pred_group_dict[area_id_back]


        parameter_a_p = func_dict[area_id+"_gt_precision"]["parameter_a"]
        parameter_beta_p = func_dict[area_id+"_gt_precision"]["parameter_beta"]
        parameter_w_gt_p = func_dict[area_id+"_gt_precision"]["parameter_w_gt"]

        parameter_a_r = func_dict[area_id+"_gt_recall"]["parameter_a"]
        parameter_beta_r = func_dict[area_id+"_gt_recall"]["parameter_beta"]
        parameter_w_gt_r = func_dict[area_id+"_gt_recall"]["parameter_w_gt"]


        pred_precision_integral_all = 0
        pred_precision_integral_all_before = 0
        pred_precision_integral_all_after = 0

        pred_recall_integral_all = 0

        if VA_gt_i_pred_group != []:
            gt_detected_num += 1
        for j, basic_interval in enumerate(VA_gt_i_pred_group):
            if basic_interval != []:
                pred_func_dict[area_id + "_gt_precision" + "_" + str(j)] = {
                    "area": basic_interval,
                    "area_start": VA_gt_start,
                    "area_end": VA_gt_end,
                    "parameter_a": parameter_a_p,
                    "parameter_beta": parameter_beta_p,
                    "func": "va_gt3_func",
                    "x_tp": None,
                    "gt_len": VA_gt_len,
                    "area_len": VA_gt_len,
                   "parameter_rho":None,
                    "parameter_w_gt": parameter_w_gt_p,
                    "parameter_w_near_ngt": 0
                }

                pred_func_dict[area_id + "_gt_recall" + "_" + str(j)] = {
                    "area": basic_interval,
                    "area_start": VA_gt_start,
                    "area_end": VA_gt_end,
                    "parameter_a": parameter_a_r,
                    "parameter_beta": parameter_beta_r,
                    "func": "va_gt3_func",
                    "x_tp": None,
                    "gt_len": VA_gt_len,
                    "area_len": VA_gt_len,
                   "parameter_rho":None,
                    "parameter_w_gt": parameter_w_gt_r,
                    "parameter_w_near_ngt": 0
                }

                cal_integral_basic_interval_gt_precision = cal_integral_in_range_gt_power_func(basic_interval,
                                                                                     parameter_a_p,
                                                                                     parameter_beta_p,
                                                                                     parameter_w_gt_p,
                                                                                     area_end=VA_gt_end)

                cal_integral_basic_interval_gt_recall = cal_integral_in_range_gt_power_func(basic_interval,
                                                                                     parameter_a_r,
                                                                                     parameter_beta_r,
                                                                                     parameter_w_gt_r,
                                                                                     area_end=VA_gt_end)

                pred_precision_integral_all += cal_integral_basic_interval_gt_precision
                pred_recall_integral_all += cal_integral_basic_interval_gt_recall
                basic_interval_precision_integral_dict[area_id+"_gt_precision" + "_"+ str(j)] =cal_integral_basic_interval_gt_precision
                basic_interval_recall_integral_dict[area_id+"_gt_recall" + "_"+ str(j)] =cal_integral_basic_interval_gt_recall



        gt_i_recall_integration = gt_recall_integration_dict[area_id]

        recall_va_i = pred_recall_integral_all / gt_i_recall_integration
        local_recall_matrix.append(recall_va_i)
        recall_va_dict[area_id] = recall_va_i

        detection_score_i = 1 if VA_pred_group_dict[area_id] != [] else 0
        detection_score_ia_i_dict[area_id] = detection_score_i
        # invalid part punish in is not enough
        # detection
        if cal_mode == "detection":
            precision_va_dict[area_id] = detection_score_i
            local_precision_matrix.append(detection_score_i)


        # precision also consider invalid part
        # proportion
        if cal_mode == "proportion":

            cal_precision_method = 3

            if cal_precision_method == 1:
                # method1

                precision_va_i_group = VA_f_i_pred_group + VA_d_i_after_pred_group + VA_gt_i_pred_group

                precision_i_all = 0
                precision_i_valid = 0
                for precision_i_item in precision_va_i_group:
                    precision_i_all += precision_i_item[1] - precision_i_item[0]
                for precision_i_item in VA_gt_i_pred_group:
                    precision_i_valid += precision_i_item[1] - precision_i_item[0]

            else:
                precision_i_all = pred_precision_integral_all
                precision_i_valid = pred_precision_integral_all
                precision_i_group = VA_f_i_pred_group + VA_d_i_after_pred_group

                if cal_precision_method == 2:
                    # method2
                    for precision_i_item in precision_i_group:
                        precision_i_all += (1-parameter_w_gt_p) * (precision_i_item[1] - precision_i_item[0])

                elif cal_precision_method == 3:

                    parameter_a_left = func_dict_copy[area_id + "_f_a"]["parameter_a"]
                    parameter_beta_left = func_dict_copy[area_id + "_f_a"]["parameter_beta"]
                    parameter_a_right = parameter_a_left
                    parameter_beta_right = parameter_beta_left

                    for interval_id, IA_basic_interval in enumerate(VA_f_i_pred_group):

                        func_dict[area_id + "_ia_before" + "_" + str(interval_id)] = {
                            "area": IA_basic_interval,
                            "area_start": VA_f_dict[area_id][0],
                            "area_end": VA_f_dict[area_id][1],
                            "parameter_a": parameter_a_left,
                            "parameter_beta": parameter_beta_left,
                            "func": "ia_before_func",
                            "x_tp": None,
                            "gt_len": None,
                            "ia_len": VA_f_dict[area_id][1]-VA_f_dict[area_id][0],
                            "area_len": None,
                            "parameter_rho": None,
                            "parameter_w_gt": 0,
                            "parameter_w_near_ngt": 0
                        }

                        cal_integral_gt_before = cal_integral_gt_before_ia_power_func(IA_basic_interval,
                                                                                      parameter_a_left,
                                                                                      parameter_beta_left,
                                                                                      parameter_near_single_side_range=parameter_near_single_side_range,
                                                                                      area_end=VA_f_end)
                        pred_precision_integral_all_before += cal_integral_gt_before

                    for interval_id, IA_basic_interval in enumerate(VA_d_i_after_pred_group):
                        func_dict[area_id + "_ia_after" + "_" + str(interval_id)] = {
                            "area": IA_basic_interval,
                            "area_start": VA_d_dict[area_id_back][0],
                            "area_end": VA_d_dict[area_id_back][1],
                            "parameter_a": parameter_a_right,
                            "parameter_beta": parameter_beta_right,
                            "func": "ia_after_func",
                            "x_tp": None,
                            "gt_len": None,
                            "ia_len": VA_d_dict[area_id_back][1]-VA_d_dict[area_id_back][0],
                            "area_len": None,
                            "parameter_rho": None,
                            "parameter_w_gt": 0,
                            "parameter_w_near_ngt": 0
                        }
                        cal_integral_gt_after = cal_integral_gt_after_ia_power_func(IA_basic_interval,
                                                                                    parameter_a_right,
                                                                                    parameter_beta_right,
                                                                                    parameter_near_single_side_range=parameter_near_single_side_range,
                                                                                    area_start=VA_d_start)
                        pred_precision_integral_all_after += cal_integral_gt_after

                    precision_i_all += pred_precision_integral_all_before
                    precision_i_all += pred_precision_integral_all_after

            precision_va_i = precision_i_valid / precision_i_all if precision_i_all != 0 else 0
            precision_va_dict[area_id] = precision_va_i
            local_precision_matrix.append(precision_va_i)

            meata_f1_i = compute_f1_score(precision_va_i, recall_va_i)
            local_f1_matrix.append(meata_f1_i)

        adjusted_ddl_score_ia_near = adjusted_ddl_score_ia_near_left_dict[area_id]*adjusted_ddl_score_ia_near_dict_ratio[area_id][1] + \
                                     adjusted_ddl_score_ia_near_right_dict[area_id_back]*adjusted_ddl_score_ia_near_dict_ratio[area_id_back][0]
        if print_msg:
            print()
            print("adjusted_ddl_score_ia_near_left_dict[area_id]", adjusted_ddl_score_ia_near_left_dict[area_id],"*")
            print("adjusted_ddl_score_ia_near_dict_ratio[area_id][1]", adjusted_ddl_score_ia_near_dict_ratio[area_id][1],"+")
            print("adjusted_ddl_score_ia_near_right_dict[area_id_back]", adjusted_ddl_score_ia_near_right_dict[area_id_back],"*")
            print("adjusted_ddl_score_ia_near_dict_ratio[area_id_back][0]", adjusted_ddl_score_ia_near_dict_ratio[area_id_back][0],"=")
            print("adjusted_ddl_score_ia_near", adjusted_ddl_score_ia_near)

        if distant_method == 1:
            if parameter_distant_direction == "both":
                adjusted_ddl_score_ia_distant = adjusted_ddl_score_ia_dict_ratio_contribution[area_id][1] + \
                                                adjusted_ddl_score_ia_dict_ratio_contribution[area_id_back][0]
            elif parameter_distant_direction == "left":
                adjusted_ddl_score_ia_distant = adjusted_ddl_score_ia_dict[area_id_back]
        else:
            adjusted_ddl_score_ia_distant = adjusted_ddl_score_ia_distant_left_dict[area_id]*adjusted_ddl_score_ia_distant_both_sides_dict_ratio[area_id][1] + \
                                     adjusted_ddl_score_ia_distant_right_dict[area_id_back]*adjusted_ddl_score_ia_distant_both_sides_dict_ratio[area_id_back][0]

        local_fq_near_matrix.append(adjusted_ddl_score_ia_near)
        local_fq_distant_matrix.append(adjusted_ddl_score_ia_distant)




    gt_detected_rate = gt_detected_num/gt_num

    if print_msg:
        print()
        print("basic_interval_integral_dict", basic_interval_precision_integral_dict)


    if print_msg:
        print()
        print("parameter_w_near_0_ngt_gt_right", parameter_w_near_0_ngt_gt_right)
        print("parameter_w_near_2_ngt_gt_left", parameter_w_near_2_ngt_gt_left)
        print("ddl_score_ia_before_dict", ddl_score_ia_before_dict)
        print("ddl_score_ia_after_dict", ddl_score_ia_after_dict)
        print("adjusted_ddl_score_ia_near_dict_ratio", adjusted_ddl_score_ia_near_dict_ratio)
        print("adjusted_ddl_score_ia_near_left_dict", adjusted_ddl_score_ia_near_left_dict)
        print("adjusted_ddl_score_ia_near_right_dict", adjusted_ddl_score_ia_near_right_dict)

        print()
        if distant_method == 1:
            print("adjusted_ddl_score_ia_distant_dict_ratio", adjusted_ddl_score_ia_distant_dict_ratio)
            print("ddl_score_ia_distant_dict", ddl_score_ia_distant_dict)
            print("adjusted_ddl_score_ia_dict", adjusted_ddl_score_ia_dict)
            print("adjusted_ddl_score_ia_dict_ratio_contribution", adjusted_ddl_score_ia_dict_ratio_contribution)
        else:
            print("adjusted_ddl_score_ia_distant_both_sides_dict_ratio", adjusted_ddl_score_ia_distant_both_sides_dict_ratio)
            print("ddl_score_ia_distant_before_dict", ddl_score_ia_distant_before_dict)
            print("ddl_score_ia_distant_after_dict", ddl_score_ia_distant_after_dict)
            print("adjusted_ddl_score_ia_distant_left_dict", adjusted_ddl_score_ia_distant_left_dict)
            print("adjusted_ddl_score_ia_distant_right_dict", adjusted_ddl_score_ia_distant_right_dict)

        print()
        print("local_recall_matrix", local_recall_matrix)
        print("local_precision_matrix", local_precision_matrix)
        print("local_fq_near_matrix", local_fq_near_matrix)
        print("local_fq_distant_matrix", local_fq_distant_matrix)
        print("gt_detected_rate", gt_detected_rate)

    return local_recall_matrix,local_precision_matrix,local_f1_matrix,local_fq_near_matrix,local_fq_distant_matrix,gt_detected_rate


def plotFigures_systhetic_data(label_ranges,label_array_list,slidingWindow=100, forecasting_len=3,delay_len=3,color_box=0.4, plotRange=None, save_plot=False,
                               plot_1_name='Real Data', plot_2_name='Perfect Model', plot_3_name='Model 1 (MVN)',
                               plot_4_name='Model 2 (AE)', plot_5_name='Random Score'):
    range_anomaly = label_ranges[0]

    score = label_array_list[0]

    max_length = len(score)
    if plotRange is None:
        plotRange = [0, max_length]

    fig3 = plt.figure(figsize=(15, 10), constrained_layout=True)
    gs = fig3.add_gridspec(len(label_array_list), 1)  # Adjusted grid for 5 rows

    def plot_anomaly_score(ax, score, label_text):
        ax.step(range(len(score)), score, where='post')

        for r in range_anomaly:
            ax.axvspan(r[0], r[1]+1, color='red', alpha=color_box)
        ax.set_ylabel('score')
        ax.set_xlim(plotRange)
        ax.text(0.02, 0.90, label_text, transform=ax.transAxes, fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor='white'))

    # Plotting the anomaly scores in separate subplots
    for i, label_array in enumerate(label_array_list):
        f3_ax1 = fig3.add_subplot(gs[i, 0])
        plot_anomaly_score(f3_ax1, label_array, "pred"+i.__str__())



    plt.show()

    return fig3


def triangle_weights(n: int):
    if n <= 0:
        raise ValueError("n must be integer")

    x = np.linspace(0, 1, n + 1)

    base = 1.0
    height = 2.0 / base

    weights = np.zeros_like(x)
    mid = 0.5
    for i, val in enumerate(x):
        if val <= mid:
            weights[i] = height * (val / mid)
        else:
            weights[i] = height * ((1 - val) / (1 - mid))

    weights /= weights.sum()
    return x, weights

def triangle_weights_add(n: int,values):
    if n <= 0:
        raise ValueError("n must be integer")

    x = np.linspace(0, 1, n + 1)

    base = 1.0
    height = 2.0 / base

    weights = np.zeros_like(x)
    mid = 0.5
    for i, val in enumerate(x):
        if val <= mid:
            weights[i] = height * (val / mid)
        else:
            weights[i] = height * ((1 - val) / (1 - mid))

    weights /= weights.sum()
    # try:
    weight_sum = weights @ values
    # except:
    #     d=1
    return weight_sum,weights


def triangle_weights_add_v1(x: np.ndarray,values):
    height = 2.0
    weights = np.where(x <= 0.5,
                 height * (x / 0.5),
                 height * ((1 - x) / 0.5))
    weights /= weights.sum()
    
    weight_sum = weights @ values

    return weight_sum,weights

def DQE(y_true, y_score=None, output=None, parameter_dict=parameter_dict, max_ia_distant_length=None, pos_label=1, \
        sample_weight=None, drop_intermediate=True, Big_Data=False, \
        num_desired_thresholds=250, ype="row_auc_add", find_type="ts_section", cal_mode="proportion"):
    y_score = output
    # print_msg = True
    print_msg = False
    window_length = len(y_true)

    thresh_num = 101
    thresholds = np.linspace(1, 0, thresh_num)

    # array -> interval_ranges
    gt_label_interval_ranges = convert_vector_to_events(y_true)

    gt_num = len(gt_label_interval_ranges)
    if ("max_ia_distant_length" in parameter_dict
            and isinstance(parameter_dict["max_ia_distant_length"], numbers.Number)
            and parameter_dict["max_ia_distant_length"] > 0):
        max_ia_distant_length = parameter_dict["max_ia_distant_length"]
    else:
        parameter_near_single_side_range = parameter_dict["parameter_near_single_side_range"]
        parameter_distant_method = parameter_dict["parameter_distant_method"]
        max_ia_distant_length =  find_max_ia_distant_length_in_single_ts(gt_label_interval_ranges,window_length,gt_num,parameter_near_single_side_range,parameter_distant_method,max_ia_distant_length=0,parameter_dict=parameter_dict)

    if print_msg:
        print()
        print("max_ia_distant_length",max_ia_distant_length)

    local_recall_matrix = []
    local_precision_matrix = []
    local_f1_matrix = []
    local_near_fq_matrix = []
    local_distant_fq_matrix = []

    gt_detection_list = []

    for i, threshold in enumerate(thresholds):
        # score->binary array
        binary_predicted = (y_score >= threshold).astype(int)
        pred_label_interval_ranges = convert_vector_to_events(binary_predicted)
        local_recall_list, local_precision_list,local_f1_list, \
            local_near_fq_list, local_distant_fq_list, gt_detected_rate = DQE_F1(gt_label_interval_ranges,
                                                                                 pred_label_interval_ranges,
                                                                                 window_length,
                                                                                 parameter_dict,
                                                                                 binary_predicted,
                                                                                 pred_case_id=i,
                                                                                 max_ia_distant_length=max_ia_distant_length,
                                                                                 output=output,
                                                                                 thresh_id=i,
                                                                                 cal_mode=cal_mode)

        label_ranges = [gt_label_interval_ranges]
        label_ranges.append(pred_label_interval_ranges)
        label_array_list= [y_true.tolist()]

        label_array_list.append(binary_predicted.tolist())

        local_recall_matrix.append(local_recall_list)
        local_precision_matrix.append(local_precision_list)
        local_f1_matrix.append(local_f1_list)
        gt_detection_list.append(gt_detected_rate)


        if threshold > 0:
            local_near_fq_matrix.append(local_near_fq_list)
            local_distant_fq_matrix.append(local_distant_fq_list)

    local_recall_matrix_np = np.array(local_recall_matrix)
    local_precision_matrix_np = np.array(local_precision_matrix)
    local_f1_matrix_np = np.array(local_f1_matrix)
    local_near_fq_matrix_np = np.array(local_near_fq_matrix)
    local_distant_fq_matrix_np = np.array(local_distant_fq_matrix)

    parameter_w_gt = parameter_dict["parameter_w_gt"]
    parameter_w_near_ngt = parameter_dict["parameter_w_near_ngt"]

    row_mean_recall_list = []
    row_mean_precision_list = []

    row_mean_near_fp_list = []
    row_mean_distant_fp_list = []

    row_local_f1_list = []

    # weigh_sum_method = "equal"
    weigh_sum_method = "triangle"

    for i, threshold in enumerate(thresholds):
        if threshold <=0:
            continue
        row_mean_near_fq = np.mean(local_near_fq_matrix_np[i])
        row_mean_distant_fq = np.mean(local_distant_fq_matrix_np[i])

        row_mean_recall = np.mean(local_recall_matrix_np[i])*gt_detection_list[i]
        row_mean_precision = np.mean(local_precision_matrix_np[i])*gt_detection_list[i]

        row_mean_recall_list.append(row_mean_recall)
        row_mean_precision_list.append(row_mean_precision)

        row_mean_near_fp_list.append(row_mean_near_fq)
        row_mean_distant_fp_list.append(row_mean_distant_fq)

        row_local_f1 = compute_f1_score(row_mean_precision,row_mean_recall)

        row_local_f1_list.append(row_local_f1)

    row_local_f1_list_np = np.array(row_local_f1_list)


    local_meata_value_list = []
    local_meata_value_list_w_gt = []
    local_meata_value_list_w_near_ngt = []
    local_meata_value_list_w_distant_ngt = []
    for j, threshold in enumerate(thresholds):
        if threshold<=0:
            continue
        row_mean_f1_value = row_local_f1_list_np[j]
        row_mean_near_fp_value = row_mean_near_fp_list[j]
        row_mean_distant_fp_value = row_mean_distant_fp_list[j]
        local_meata_value = cal_dqe_row(parameter_w_gt, parameter_w_near_ngt, row_mean_distant_fp_value,
                                        row_mean_f1_value, row_mean_near_fp_value)
        local_meata_value_w_gt = cal_dqe_row(1, 0, row_mean_distant_fp_value,row_mean_f1_value, row_mean_near_fp_value)
        local_meata_value_w_near_ngt = cal_dqe_row(0, 1, row_mean_distant_fp_value,row_mean_f1_value, row_mean_near_fp_value)
        local_meata_value_w_distant_ngt = cal_dqe_row(0, 0, row_mean_distant_fp_value,row_mean_f1_value, row_mean_near_fp_value)

        local_meata_value_list.append(local_meata_value)
        local_meata_value_list_w_gt.append(local_meata_value_w_gt)
        local_meata_value_list_w_near_ngt.append(local_meata_value_w_near_ngt)
        local_meata_value_list_w_distant_ngt.append(local_meata_value_w_distant_ngt)


    if weigh_sum_method == "equal":

        meata = np.mean(local_meata_value_list)
        meata_w_gt = np.mean(local_meata_value_list_w_gt)
        meata_w_near_ngt = np.mean(local_meata_value_list_w_near_ngt)
        meata_w_distant_ngt = np.mean(local_meata_value_list_w_distant_ngt)
    else:

        meata, triangle_weights = triangle_weights_add_v1(thresholds[:-1], np.array(local_meata_value_list))
        meata_w_gt, _ = triangle_weights_add_v1(thresholds[:-1], np.array(local_meata_value_list_w_gt))
        meata_w_near_ngt, _ = triangle_weights_add_v1(thresholds[:-1], np.array(local_meata_value_list_w_near_ngt))
        meata_w_distant_ngt, _ = triangle_weights_add_v1(thresholds[:-1],
                                                      np.array(local_meata_value_list_w_distant_ngt))


    final_meata = meata

    if print_msg:
        print()
        print(" meata", meata)
        print(" meata_w_gt", meata_w_gt)
        print(" meata_w_near_ngt", meata_w_near_ngt)
        print(" meata_w_distant_ngt", meata_w_distant_ngt)
        print(" final_meata", final_meata)
    meata_w_ngt = (meata_w_near_ngt+meata_w_distant_ngt) / 2
    return meata,meata_w_gt,meata_w_near_ngt,meata_w_distant_ngt,meata_w_ngt


def cal_local_meata_value(gt_detection_mean, id_gt, id_thresh, local_distant_fq_matrix_np, local_f1_add_fp_matrix,
                          local_f1_matrix_np, local_near_fq_matrix_np, parameter_w_gt, parameter_w_near_ngt):
    local_f1_add_fp_matrix[id_thresh][id_gt] = parameter_w_gt * local_f1_matrix_np[id_thresh][
        id_gt] * gt_detection_mean + \
                                               parameter_w_near_ngt * \
                                               local_near_fq_matrix_np[id_thresh][id_gt] + \
                                               (1 - parameter_w_gt - parameter_w_near_ngt) * \
                                               local_distant_fq_matrix_np[id_thresh][id_gt]


def cal_dqe_row(parameter_w_gt, parameter_w_near_ngt, row_mean_distant_fp_value, row_mean_f1_value,
                row_mean_near_fp_value):
    local_meata_value = parameter_w_gt * row_mean_f1_value + \
                        parameter_w_near_ngt * row_mean_near_fp_value + \
                        (1 - parameter_w_gt - parameter_w_near_ngt) * row_mean_distant_fp_value
    return local_meata_value


def cal_meata_auc_row_add(mean_distant_fq, mean_local_near_fq, parameter_w_gt, parameter_w_near_ngt, tq_auc_pr_final):
    meata = parameter_w_gt * tq_auc_pr_final + \
            parameter_w_near_ngt * mean_local_near_fq + \
            (1 - parameter_w_gt - parameter_w_near_ngt) * mean_distant_fq
    return meata


def cal_meata_auc_add_row(col_mean_distant_fq, col_mean_local_near_fq, local_tq_auc_pr_new, parameter_w_gt,
                          parameter_w_near_ngt):
    local_meata = parameter_w_gt * local_tq_auc_pr_new + \
                  parameter_w_near_ngt * col_mean_local_near_fq + \
                  (1 - parameter_w_gt - parameter_w_near_ngt) * col_mean_distant_fq
    return local_meata


def cal_dqe(mean_distant_fq, mean_local_near_fq, parameter_w_gt, parameter_w_near_ngt, tq_auc_pr):
    meata = parameter_w_gt * tq_auc_pr + \
            parameter_w_near_ngt * mean_local_near_fq + \
            (1 - parameter_w_gt - parameter_w_near_ngt) * mean_distant_fq
    return meata



def row_add_weight_sum(local_distant_fq_mean, local_near_fq_mean, \
                       meata_f1_pr, parameter_w_gt, parameter_w_near_ngt,gt_detected_rate):
    meata_f1 = parameter_w_gt * meata_f1_pr*gt_detected_rate + \
               parameter_w_near_ngt * local_near_fq_mean + \
               (1 - parameter_w_gt - parameter_w_near_ngt) * local_distant_fq_mean
    return meata_f1


def cal_meata_f1(local_distant_fq_i, local_near_fq_i, local_precision_i, local_recall_i, parameter_w_gt,
                 parameter_w_near_ngt,gt_detected_rate):
    # print_msg = True
    print_msg = False
    local_meata_recall_i = parameter_w_gt * local_recall_i*gt_detected_rate + \
                           parameter_w_near_ngt * local_near_fq_i + \
                           (1 - parameter_w_gt - parameter_w_near_ngt) * local_distant_fq_i
    local_meata_precision_i = parameter_w_gt * local_precision_i*gt_detected_rate + \
                              parameter_w_near_ngt * local_near_fq_i + \
                              (1 - parameter_w_gt - parameter_w_near_ngt) * local_distant_fq_i
    meata_f1_i = compute_f1_score(local_meata_precision_i, local_meata_recall_i)
    if print_msg:
        print()
        print(" parameter_w_gt,parameter_w_near_ngt",parameter_w_gt,parameter_w_near_ngt)
        print(" local_meata_recall_i,", local_meata_recall_i)
        print(" local_meata_precision_i,", local_meata_precision_i)
        print(" local_meata_f1_i,", meata_f1_i)

    return local_meata_recall_i,local_meata_precision_i,meata_f1_i


def find_max_ia_distant_length_in_single_ts(label_interval_ranges,ts_length,gt_num,near_single_side_range,parameter_distant_method,max_ia_distant_length,parameter_dict):    # print_msg = False
    print_msg = True
    parameter_w_near_2_ngt_gt_left = parameter_dict["parameter_w_near_2_ngt_gt_left"]
    parameter_w_near_0_ngt_gt_right = parameter_dict["parameter_w_near_0_ngt_gt_right"]

    for i, label_interval_range in enumerate(label_interval_ranges):
        if i == 0:
            VA_gt_now_start, VA_gt_now_end = label_interval_range
            if parameter_w_near_2_ngt_gt_left <= 0:
                ia_distant_length = max(VA_gt_now_start - 0, 0)
            elif parameter_w_near_0_ngt_gt_right <= 0:
                ia_distant_length = max(VA_gt_now_start - 0 -near_single_side_range,0)
            else:
                ia_distant_length = max(VA_gt_now_start - 0 -near_single_side_range,0)
            if gt_num == 1:
                if parameter_w_near_2_ngt_gt_left <= 0:
                    ia_distant_length = max(ia_distant_length, ts_length-VA_gt_now_end-near_single_side_range)
                elif parameter_w_near_0_ngt_gt_right <= 0:
                    ia_distant_length = max(ia_distant_length, ts_length-VA_gt_now_end)
                else:
                    ia_distant_length = max(ia_distant_length, ts_length-VA_gt_now_end-near_single_side_range)
        elif i == gt_num-1:
            VA_gt_now_start, VA_gt_now_end = label_interval_range
            VA_gt_before_start, VA_gt_before_end = label_interval_ranges[i - 1]
            if parameter_distant_method == 1:
                if parameter_w_near_2_ngt_gt_left <= 0:
                    ia_distant_length = max(VA_gt_now_start - VA_gt_before_end - near_single_side_range * 1, 0)
                elif parameter_w_near_0_ngt_gt_right <= 0:
                    ia_distant_length = max(VA_gt_now_start - VA_gt_before_end - near_single_side_range * 1, 0)
                else:
                    ia_distant_length = max(VA_gt_now_start - VA_gt_before_end - near_single_side_range * 2, 0)
            else:
                if parameter_w_near_2_ngt_gt_left <= 0:
                    ia_distant_length = max((VA_gt_now_start - VA_gt_before_end - near_single_side_range * 1)/2, 0)
                elif parameter_w_near_0_ngt_gt_right <= 0:
                    ia_distant_length = max((VA_gt_now_start - VA_gt_before_end - near_single_side_range * 1)/2, 0)
                else:
                    ia_distant_length = max((VA_gt_now_start - VA_gt_before_end - near_single_side_range * 2)/2, 0)

            if parameter_w_near_2_ngt_gt_left <= 0:
                ia_distant_length = max(ia_distant_length, ts_length-VA_gt_now_end-near_single_side_range)
            elif parameter_w_near_0_ngt_gt_right <= 0:
                ia_distant_length = max(ia_distant_length, ts_length-VA_gt_now_end)
            else:
                ia_distant_length = max(ia_distant_length, ts_length-VA_gt_now_end-near_single_side_range)
        else:
            VA_gt_now_start, VA_gt_now_end = label_interval_range
            VA_gt_before_start, VA_gt_before_end = label_interval_ranges[i-1]
            if parameter_distant_method == 1:
                if parameter_w_near_2_ngt_gt_left <= 0:
                    ia_distant_length = max(VA_gt_now_start - VA_gt_before_end-near_single_side_range*1,0)
                elif parameter_w_near_0_ngt_gt_right <= 0:
                    ia_distant_length = max(VA_gt_now_start - VA_gt_before_end-near_single_side_range*1,0)
                else:
                    ia_distant_length = max(VA_gt_now_start - VA_gt_before_end-near_single_side_range*2,0)
            else:
                if parameter_w_near_2_ngt_gt_left <= 0:
                    ia_distant_length = max((VA_gt_now_start - VA_gt_before_end-near_single_side_range*1)/2,0)
                elif parameter_w_near_0_ngt_gt_right <= 0:
                    ia_distant_length = max((VA_gt_now_start - VA_gt_before_end-near_single_side_range*1)/2,0)
                else:
                    ia_distant_length = max((VA_gt_now_start - VA_gt_before_end-near_single_side_range*2)/2,0)


        if ia_distant_length > max_ia_distant_length:
            max_ia_distant_length = ia_distant_length

    return max_ia_distant_length

def find_max_ia_distant_length_in_ts(ts_label_data, parameter_dict=parameter_dict, find_type="ts_section"):
    parameter_distant_method = parameter_dict["parameter_distant_method"]

    if isinstance(ts_label_data, list):
        ts_label_data = np.array(ts_label_data)
    ts_label_data_list = []
    if ts_label_data.ndim == 2:
        ts_label_data_list = ts_label_data.tolist()
    else:
        ts_label_data_list.append(ts_label_data)


    max_ia_distant_length = 0
    near_single_side_range = parameter_dict["parameter_near_single_side_range"]

    for single_ts_label_data in ts_label_data_list:
        ts_length = len(single_ts_label_data)
        label_interval_ranges = convert_vector_to_events(single_ts_label_data)
        gt_num = len(label_interval_ranges)
        max_ia_distant_length =  find_max_ia_distant_length_in_single_ts(label_interval_ranges,
                                                                         ts_length,
                                                                         gt_num,
                                                                         near_single_side_range,
                                                                         parameter_distant_method,
                                                                         max_ia_distant_length,
                                                                         parameter_dict,
                                                                         )
    max_ia_distant_length = max(max_ia_distant_length, 0)
    parameter_dict["max_ia_distant_length"] = max_ia_distant_length
    return max_ia_distant_length