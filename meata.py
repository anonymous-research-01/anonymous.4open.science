#!/usr/bin/env python3
# -*- coding: utf-8 -*-
###########################
import copy
import math

import numpy as np
import scipy.integrate as spi

from sklearn.metrics._ranking import _binary_clf_curve
from sortedcontainers import SortedSet
from copy import deepcopy
import numbers
from sklearn.metrics import precision_recall_curve,auc

from metrics.affiliation.generics import convert_vector_to_events
from pate.PATE_utils import clean_and_compute_auc_pr
from config.meata_config import parameter_dict, parameter_w_gt
from pate.PATE_utils import convert_events_to_array_PATE, convert_interval_to_point_array




def cal_x1_x2(VA_gt_len,parameter_rho):
    if parameter_rho <0:
        x1 = VA_gt_len / 4
        x2 = VA_gt_len * 3/4
    else:
        x1 = parameter_rho * VA_gt_len / 2
        x2 = VA_gt_len - (1 - parameter_rho) * VA_gt_len / 2
    # return int(x1), int(x2)
    return x1, x2



split_line_set = SortedSet()


def split_intervals(a, b):
    # 将集合b转换为排序后的列表
    # Convert the set b to a sorted list
    # split_points = sorted(b)
    split_points = b

    # 初始化结果列表
    # Initialize the result list
    result = []

    # 遍历二维列表a中的每个区间
    # Iterate through each interval in the 2D list a
    for interval in a:
        start, end = interval  # 获取区间的起始点和结束点
        # 获取当前区间内的分割点
        # Get the split points within the current interval
        current_splits = [start] + [point for point in split_points if start < point < end] + [end]

        # 根据分割点生成新的子区间
        # Generate new sub-intervals based on the split points
        for i in range(len(current_splits) - 1):
            result.append([current_splits[i], current_splits[i + 1]])

    return result



def pred_in_area(pred,area):
    if area == [] or pred== []:
        return False
    return True if pred[0] >= area[0] and pred[1] <= area[1] else False


# label_function_va_f

def cal_integral_in_range_f_power_func(area, parameter_a, parameter_beta,x_tp,area_len,area_start=None,method=1,greater_than_th=False):

    # import sympy as sp
    # 
    # # 定义符号变量
    # x = sp.symbols('x')
    # 
    # # 定义函数
    # f = parameter_a * pow(x-area_start, parameter_beta)
    # 
    # 
    # # 计算定积分
    # result = sp.integrate(f, (x, area[0], area[1]))
    # 积分区间要换成矩形的坐标
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

    # import sympy as sp
    #
    # # 定义符号变量
    # x = sp.symbols('x')
    #
    # # 定义函数
    # f = parameter_a * pow(x-area_start, parameter_beta)
    #
    #
    # # 计算定积分
    # result = sp.integrate(f, (x, area[0], area[1]))
    # 积分区间要换成矩形的坐标
    # if not greater_than_th:
    # parameter_b * pow(x - area_start, parameter_gama)
    # parameter_b * pow(x - area_start, parameter_gama) + (1 - parameter_w_gt - parameter_w_near_ngt)
    # f = lambda x, parameter_a, parameter_beta,area_start: parameter_a * pow(x-area_start, parameter_beta)
    #                     parameter_a_left = -(1-parameter_w_gt) / parameter_near_single_side_range
    #                     parameter_beta_left = 1
    # abs(parameter_b * pow((area_end-x)-parameter_near_single_side_range, parameter_gama))
    f = lambda x, parameter_a, parameter_beta,parameter_w_gt,parameter_w_near_ngt,\
               parameter_near_single_side_range,area_end: abs(parameter_a * pow((area_end-x)-parameter_near_single_side_range, parameter_beta))
    # else:
    #     if method ==1:
    #         f = lambda x, parameter_a, parameter_beta,x_tp,area_len,area_start: parameter_a * pow(x-area_start, parameter_beta)
    #     else:
    #         f = lambda x, parameter_a, parameter_beta,x_tp,area_len,area_start: 1/2-pow((area_len-(x-area_start))/parameter_a, 1/parameter_beta)
    result, error = spi.quad(f, area[0], area[1], args=(parameter_a, parameter_beta,parameter_w_gt,parameter_w_near_ngt,
                                                        parameter_near_single_side_range,area_end))
    return result

def cal_integral_gt_after_ia_power_func(area, parameter_a, parameter_beta,parameter_w_gt=0,
                                        parameter_near_single_side_range=None,area_start=None):

    # import sympy as sp
    #
    # # 定义符号变量
    # x = sp.symbols('x')
    #
    # # 定义函数
    # f = parameter_a * pow(x-area_start, parameter_beta)
    #
    #
    # # 计算定积分
    # result = sp.integrate(f, (x, area[0], area[1]))
    # 积分区间要换成矩形的坐标
    # if not greater_than_th:
    # 1 / 2 - parameter_b * pow(x - area_start, parameter_gama)
    # (1 - parameter_w_gt) - parameter_b * pow(x - area_start, parameter_gama)
    # f = lambda x, parameter_a, parameter_beta,area_start: 1/2-parameter_a * pow(x-area_start, parameter_beta)
    # f = lambda x, parameter_a, parameter_beta,parameter_w_gt,area_start: (1 - parameter_w_gt) + parameter_a * pow(x - area_start, parameter_beta)
    # abs(parameter_b * pow((x - area_start) - parameter_near_single_side_range, parameter_gama))
    f = lambda x, parameter_a, parameter_beta,parameter_w_gt,\
               parameter_near_single_side_range,area_start: abs(parameter_a * pow((x - area_start) - parameter_near_single_side_range, parameter_beta))
    # else:
    #     if method ==1:
    #         f = lambda x, parameter_a, parameter_beta,x_tp,area_len,area_start: parameter_a * pow(x-area_start, parameter_beta)
    #     else:
    #         f = lambda x, parameter_a, parameter_beta,x_tp,area_len,area_start: 1/2-pow((area_len-(x-area_start))/parameter_a, 1/parameter_beta)
    result, error = spi.quad(f, area[0], area[1], args=(parameter_a, parameter_beta,parameter_w_gt,
                                                        parameter_near_single_side_range,area_start))
    return result

# def d_power_func(x,parameter_a, parameter_beta,x_tp,area_len,area_start=None):
#     return parameter_a * pow((area_len-(x-area_start)), parameter_beta)

def cal_integral_in_range_d_power_func(area, parameter_a, parameter_beta,x_tp,area_len,area_start=None,method=1,greater_than_th=False):

    # import sympy as sp
    #
    # # 定义符号变量
    # x = sp.symbols('x')
    #
    # # 定义函数
    # f = parameter_a * pow((area_len-(x-area_start)), parameter_beta)
    #
    #
    # # 计算定积分
    # result = sp.integrate(f, (x, area[0], area[1]))
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

    # import sympy as sp
    #
    # # 定义符号变量
    # x = sp.symbols('x')
    #
    # # 定义函数
    # f = parameter_b * pow(x - area_start, parameter_gama) + 1 / 2
    # # f = parameter_a * pow(x, parameter_beta) + 1/2
    #
    #
    # # 计算定积分
    # result = sp.integrate(f, (x, area[0], area[1]))
    if parameter_rho < 0:
        f = lambda x, parameter_b, parameter_gama,x_tp,gt_len,area_start:  parameter_b * pow(x - area_start, parameter_gama)
    else:
        f = lambda x, parameter_b, parameter_gama,x_tp,gt_len,area_start: parameter_b * pow(x - area_start, parameter_gama) + 1 / 2
    result, error = spi.quad(f, area[0], area[1], args=(parameter_b, parameter_gama,x_tp,gt_len,area_start))
    return result

def cal_integral_in_range_gt_power_func2(area, parameter_b, parameter_gama,x_tp=None,gt_len=None,area_start=None,parameter_rho=0):

    # import sympy as sp
    #
    # # 定义符号变量
    # x = sp.symbols('x')
    #
    # # 定义函数
    # f = 1/2 - parameter_b * pow((2*x_tp-(x-area_start)), parameter_gama) + 1 / 2
    #
    #
    # # 计算定积分
    # result = sp.integrate(f, (x, area[0], area[1]))
    if parameter_rho < 0:
        f = lambda x, parameter_b, parameter_gama,x_tp,gt_len,area_start:  parameter_b * pow(x - area_start, parameter_gama)
    else:
        f = lambda x, parameter_b, parameter_gama,x_tp,gt_len,area_start: 1/2 - parameter_b * pow((2*x_tp-(x-area_start)), parameter_gama) + 1 / 2
    result, error = spi.quad(f, area[0], area[1], args=(parameter_b, parameter_gama,x_tp,gt_len,area_start))
    return result

def cal_integral_in_range_gt_power_func3(area, parameter_b, parameter_gama,x_tp=None,gt_len=None,area_start=None,parameter_rho=0):

    # import sympy as sp
    #
    # # 定义符号变量
    # x = sp.symbols('x')
    #
    # # 定义函数
    # f = 1/2 - parameter_b * pow(gt_len-2*x_tp+(x-area_start), parameter_gama) + 1 / 2
    #
    #
    # # 计算定积分
    # result = sp.integrate(f, (x, area[0], area[1]))
    if parameter_rho < 0:
        f = lambda x, parameter_b, parameter_gama,x_tp,gt_len,area_start:  parameter_b * pow(x - area_start, parameter_gama)
    else:
        f = lambda x, parameter_b, parameter_gama,x_tp,gt_len,area_start: 1/2 - parameter_b * pow(gt_len-2*x_tp+(x-area_start), parameter_gama) + 1 / 2
    result, error = spi.quad(f, area[0], area[1], args=(parameter_b, parameter_gama,x_tp,gt_len,area_start))
    return result

def cal_integral_in_range_gt_power_func4(area, parameter_b, parameter_gama,x_tp=None,gt_len=None,area_start=None,parameter_rho=0):

    # import sympy as sp
    #
    # # 定义符号变量
    # x = sp.symbols('x')
    #
    # # 定义函数
    # f = parameter_b * pow(gt_len-(x-area_start), parameter_gama) + 1 / 2
    #
    #
    # # 计算定积分
    # result = sp.integrate(f, (x, area[0], area[1]))
    if parameter_rho < 0:
        f = lambda x, parameter_b, parameter_gama,x_tp,gt_len,area_start:  parameter_b * pow(x - area_start, parameter_gama)
    else:
        f = lambda x, parameter_b, parameter_gama,x_tp,gt_len,area_start: parameter_b * pow(gt_len-(x-area_start), parameter_gama) + 1 / 2
    result, error = spi.quad(f, area[0], area[1], args=(parameter_b, parameter_gama,x_tp,gt_len,area_start))
    return result

def cal_integral_in_range_gt_power_func(area, parameter_b, parameter_gama,parameter_w_gt=0,area_end=None):

    # import sympy as sp
    #
    # # 定义符号变量
    # x = sp.symbols('x')
    #
    # # 定义函数
    # f = 1/2 - parameter_b * pow(gt_len-2*x_tp+(x-area_start), parameter_gama) + 1 / 2
    #
    #
    # # 计算定积分
    # result = sp.integrate(f, (x, area[0], area[1]))
    # if parameter_rho < 0:
    #     f = lambda x, parameter_b, parameter_gama,x_tp,gt_len,area_start:  parameter_b * pow(x - area_start, parameter_gama)
    # else:
    # a * (-x)**alpha + 0.6
    # parameter_b = (1-(parameter_w1+parameter_w2))/pow(gt_len, parameter_gama)
    # f = lambda x, parameter_b, parameter_gama: parameter_b * pow(x, parameter_gama)+(parameter_w1+parameter_w2)
    # parameter_b * pow(area_end - x, parameter_gama) + (1 - parameter_w_gt)
    # parameter_b * pow(area_end-x, parameter_gama)+(1-parameter_w_gt)
    f = lambda x, parameter_b, parameter_gama,area_end: parameter_b * pow(area_end - x, parameter_gama) + (1 - parameter_w_gt)
    result, error = spi.quad(f, area[0], area[1], args=(parameter_b, parameter_gama,area_end))
    return result


def cal_power_function_coefficient(x_tp,parameter_lamda):
    # rho=0/1,one side function not exist
    if x_tp == 0:
        return 1,0
    parameter_gama = 2 / (1 - parameter_lamda)-1
    parameter_b = 1 / (4 * pow(x_tp, parameter_gama))
    return parameter_b, parameter_gama

def cal_power_function_parameter_b(parameter_gama,parameter_w_gt,parameter_w_near_ngt,gt_len):
    # rho=0/1,one side function not exist
    # parameter_b = parameter_w_gt / pow(gt_len, parameter_gama)
    parameter_b = parameter_w_gt / pow(gt_len, parameter_gama)
    return parameter_b

def power_function(a, b, x):
    return a * pow(x, b)

def va_gt_fun1(x,parameter_b1,parameter_gama1):
    return parameter_b1 * pow(x, parameter_gama1) + 1 / 2

def compute_f1_score(precision, recall):
# Calculate the F1 score from precision and recall
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def entropy(array):
    # 计算每个元素的出现次数
    unique, counts = np.unique(array, return_counts=True)
    # 计算每个元素的概率
    probabilities = counts / len(array)
    # 计算熵
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy

def autocorrelation(array, lag=1):
    mean = np.mean(array)
    var = np.var(array)
    n = len(array)
    if var == 0:
        return 0
    return ((array[:n-lag] - mean) * (array[lag:] - mean)).sum() / var / (n - lag)

def get_meate_gt_area(label_interval_ranges,window_length,parameter_dict):
    VA_f_window = parameter_dict["forecasting_window"]
    VA_d_window =parameter_dict["delay_window"]
    parameter_rho =parameter_dict["parameter_rho"]

    print("VA_f_window",VA_f_window)
    print("VA_d_window",VA_d_window)

    f_window_rate = 0
    d_window_rate = 0
    f_d_window_len = VA_f_window + VA_d_window
    if f_d_window_len != 0:
        f_window_rate = VA_f_window/f_d_window_len
        d_window_rate = 1 -f_window_rate

    IA_dict = {}
    VA_dict = {}
    VA_f_dict = {}
    VA_gt_dict = {}
    VA_d_dict = {}

    # pred generated is continuous range a,b(len) -> [a,a+b(len)),range of [a,b] -> [a,b) -> point of [a,b-1],point of [a,b] -> [a,b+1)


    # cal every f/d of gt,split IA and VA


    label_interval_num = len(label_interval_ranges)


    for i, label_interval_range in enumerate(label_interval_ranges):
        # adjust now and before,add before
        VA_gt_now_start,VA_gt_now_end = label_interval_range
        # VA_f_now_start = max(VA_gt_now_start - VA_f_window,0) if parameter_switch_f == True else max(VA_gt_now_start - 3,0)
        VA_f_now_start = max(VA_gt_now_start - VA_f_window,0)
        VA_f_now_end = VA_gt_now_start
        VA_d_now_start = VA_gt_now_end
        # VA_d_now_end = min(VA_gt_now_end + VA_f_window,window_length) if parameter_switch_d == True else min(VA_gt_now_end + 3,window_length)
        VA_d_now_end = min(VA_gt_now_end + VA_f_window,window_length)
        # # this way below is more fast
        # if i == 0:
        #     VA_f_now_start = max(VA_f_now_start,0)
        # if i == label_interval_num:
        #     VA_d_now_end = min(VA_d_now_end,window_length)
        if i >0:
            # adjust now and before
            if VA_d_before_end > VA_f_now_start:
                intersection_range_start = max(VA_f_now_start,VA_gt_before_end)
                intersection_range_end = min(VA_d_before_end,VA_gt_now_start)
                intersection_range = [intersection_range_start,intersection_range_end]
                intersection_len = intersection_range_end - intersection_range_start
        #         if intersection_len == 1 and (parameter_switch_f == True and parameter_switch_d == True) and parameter_rho == 1/2:
        # #             f prior
        #             VA_d_before_end -=1
        #         else:
        #         VA_d_before_end = VA_f_now_start = intersection_range_start + round(parameter_w*intersection_len)
        #         VA_d_before_end = VA_f_now_start = intersection_range_start + round(parameter_w*intersection_len,2)
        #         VA_f_now_start = intersection_range_start + round(f_window_rate*intersection_len,2)
        #         VA_d_before_end = intersection_range_start + round(d_window_rate*intersection_len,2)
                VA_f_now_start = intersection_range_start + round(f_window_rate*intersection_len)
                VA_d_before_end = intersection_range_start + round(d_window_rate*intersection_len)
        #     add before
            IA_dict["id_"+str(i)] = [VA_d_before_end,VA_f_now_start]
            VA_dict["id_"+str(i-1)] = [VA_f_before_start,VA_d_before_end]
            VA_f_dict["id_"+str(i-1)] = [VA_f_before_start,VA_f_before_end]
            VA_gt_dict["id_"+str(i-1)] = [VA_gt_before_start,VA_gt_before_end]
            VA_d_dict["id_"+str(i-1)] = [VA_d_before_start,VA_d_before_end]

            # add by order

            split_line_set.add(VA_f_before_start)
            split_line_set.add(VA_f_before_end)

            split_line_set.add(VA_gt_before_start)
            split_line_set.add(VA_gt_before_end)

            split_line_set.add(VA_d_before_start)
            split_line_set.add(VA_d_before_end)

            split_line_set.add(VA_d_before_end)
            split_line_set.add(VA_f_now_start)

            VA_gt_before_len = VA_gt_before_end - VA_gt_before_start
            x1, x2 = cal_x1_x2(VA_gt_before_len,parameter_rho)

            # # print("area_id,area",area_id,area)
            # # # print("x1,x2",x1, x2)
            # # print("x1_ori,x2_ori",x1+VA_gt_before_start[0], x2+VA_gt_before_start[0])
            x1_ori, x2_ori = x1+VA_gt_before_start, x2+VA_gt_before_start
            x_mid_ori = 2*x1+VA_gt_before_start
            split_line_set.add(x1_ori)
            split_line_set.add(x_mid_ori)
            split_line_set.add(x2_ori)
        else:
            IA_dict["id_"+str(i)] = [0, VA_f_now_start]
            split_line_set.add(VA_f_now_start)

        VA_gt_before_start, VA_gt_before_end = VA_gt_now_start,VA_gt_now_end
        VA_f_before_start = VA_f_now_start
        VA_f_before_end = VA_f_now_end
        VA_d_before_start = VA_d_now_start
        VA_d_before_end = VA_d_now_end

    # last one VA and last two IA
    IA_dict["id_"+str(label_interval_num)] = [VA_d_before_end, window_length]
    VA_dict["id_"+str(label_interval_num-1)] = [VA_f_before_start, VA_d_before_end]
    VA_f_dict["id_"+str(label_interval_num-1)] = [VA_f_before_start, VA_f_before_end]
    VA_gt_dict["id_"+str(label_interval_num-1)] = [VA_gt_before_start, VA_gt_before_end]
    VA_d_dict["id_"+str(label_interval_num-1)] = [VA_d_before_start, VA_d_before_end]

    return VA_f_dict,VA_d_dict,IA_dict,VA_dict,VA_gt_dict

def map_value(x, a, b, c, d):
    """
    将值 x 从区间 [a, b] 等比例映射到区间 [c, d]。

    参数:
    x -- 要映射的值
    a -- 原区间 [a, b] 的下限
    b -- 原区间 [a, b] 的上限
    c -- 目标区间 [c, d] 的下限
    d -- 目标区间 [c, d] 的上限

    返回:
    映射后的值 y
    """
    y = c + (d - c) * (x - a) / (b - a)
    return y

def meata(label_interval_ranges,IA1_VA1_IA2_relative_list,window_length,parameter_dict,pred_label_point_array,pred_case_id=None,cal_mode="proportion"):

    # print_msg = False
    print_msg = True

    # parameter_alpha = parameter_dict["parameter_alpha"]
    parameter_rate = parameter_dict["parameter_rate"]

    parameter_theta =parameter_dict["parameter_theta"]
    # parameter_w = parameter_dict["parameter_w"]
    # parameter_switch_f =parameter_dict["parameter_switch_f"]
    # parameter_switch_d = parameter_dict["parameter_switch_d"]
    VA_f_window = parameter_dict["forecasting_window"]
    VA_d_window =parameter_dict["delay_window"]

    parameter_rho =parameter_dict["parameter_rho"]
    parameter_lamda = parameter_dict["parameter_lamda"]

    parameter_eta = parameter_dict["parameter_eta"]

    if print_msg:
        print("VA_f_window",VA_f_window)
        print("VA_d_window",VA_d_window)

    f_window_rate = 0
    d_window_rate = 0
    f_d_window_len = VA_f_window + VA_d_window
    if f_d_window_len != 0:
        f_window_rate = VA_f_window/f_d_window_len
        d_window_rate = 1 -f_window_rate


    IA_dict = {}
    VA_dict = {}
    VA_f_dict = {}
    VA_gt_dict = {}
    VA_d_dict = {}

    # pred generated is continuous range a,b(len) -> [a,a+b(len)),range of [a,b] -> [a,b) -> point of [a,b-1],point of [a,b] -> [a,b+1)

    # cal every f/d of gt,split IA and VA


    label_interval_num = len(label_interval_ranges)


    s_a_func_dict ={}

    for i, label_interval_range in enumerate(label_interval_ranges):
        # adjust now and before,add before
        VA_gt_now_start,VA_gt_now_end = label_interval_range
        # VA_f_now_start = max(VA_gt_now_start - VA_f_window,0) if parameter_switch_f == True else max(VA_gt_now_start - 3,0)
        VA_f_now_start = max(VA_gt_now_start - VA_f_window,0)
        VA_f_now_end = VA_gt_now_start
        VA_d_now_start = VA_gt_now_end
        # VA_d_now_end = min(VA_gt_now_end + VA_f_window,window_length) if parameter_switch_d == True else min(VA_gt_now_end + 3,window_length)
        VA_d_now_end = min(VA_gt_now_end + VA_d_window,window_length)
        # # this way below is more fast
        # if i == 0:
        #     VA_f_now_start = max(VA_f_now_start,0)
        # if i == label_interval_num:
        #     VA_d_now_end = min(VA_d_now_end,window_length)
        if i >0:
            # adjust now and before
            if VA_d_before_end > VA_f_now_start:
                intersection_range_start = max(VA_f_now_start,VA_gt_before_end)
                intersection_range_end = min(VA_d_before_end,VA_gt_now_start)
                intersection_range = [intersection_range_start,intersection_range_end]
                intersection_len = intersection_range_end - intersection_range_start
        #         if intersection_len == 1 and (parameter_switch_f == True and parameter_switch_d == True) and parameter_rho == 1/2:
        # #             f prior
        #             VA_d_before_end -=1
        #         else:
        #         VA_d_before_end = VA_f_now_start = intersection_range_start + round(parameter_w*intersection_len)
        #         VA_d_before_end = VA_f_now_start = intersection_range_start + round(parameter_w*intersection_len,2)
        #         VA_f_now_start = intersection_range_start + round(f_window_rate*intersection_len,2)
        #         VA_d_before_end = intersection_range_start + round(d_window_rate*intersection_len,2)
                VA_f_now_start = intersection_range_start + round(f_window_rate*intersection_len)
                VA_d_before_end = intersection_range_start + round(d_window_rate*intersection_len)
        #     add before
            IA_dict["id_"+str(i)] = [VA_d_before_end,VA_f_now_start]
            VA_dict["id_"+str(i-1)] = [VA_f_before_start,VA_d_before_end]
            VA_f_dict["id_"+str(i-1)] = [VA_f_before_start,VA_f_before_end]
            VA_gt_dict["id_"+str(i-1)] = [VA_gt_before_start,VA_gt_before_end]
            VA_d_dict["id_"+str(i-1)] = [VA_d_before_start,VA_d_before_end]

            # add by order

            split_line_set.add(VA_f_before_start)
            split_line_set.add(VA_f_before_end)

            split_line_set.add(VA_gt_before_start)
            split_line_set.add(VA_gt_before_end)

            split_line_set.add(VA_d_before_start)
            split_line_set.add(VA_d_before_end)

            split_line_set.add(VA_d_before_end)
            split_line_set.add(VA_f_now_start)

            VA_gt_before_len = VA_gt_before_end - VA_gt_before_start
            x1, x2 = cal_x1_x2(VA_gt_before_len,parameter_rho)

            # # print("area_id,area",area_id,area)
            # # # print("x1,x2",x1, x2)
            # # print("x1_ori,x2_ori",x1+VA_gt_before_start[0], x2+VA_gt_before_start[0])
            x1_ori, x2_ori = x1+VA_gt_before_start, x2+VA_gt_before_start
            x_mid_ori = 2*x1+VA_gt_before_start
            split_line_set.add(x1_ori)
            split_line_set.add(x_mid_ori)
            split_line_set.add(x2_ori)
        else:
            IA_dict["id_"+str(i)] = [0, VA_f_now_start]
            split_line_set.add(VA_f_now_start)

        VA_gt_before_start, VA_gt_before_end = VA_gt_now_start,VA_gt_now_end
        VA_f_before_start = VA_f_now_start
        VA_f_before_end = VA_f_now_end
        VA_d_before_start = VA_d_now_start
        VA_d_before_end = VA_d_now_end

    # last one VA and last two IA
    IA_dict["id_"+str(label_interval_num)] = [VA_d_before_end, window_length]
    VA_dict["id_"+str(label_interval_num-1)] = [VA_f_before_start, VA_d_before_end]
    VA_f_dict["id_"+str(label_interval_num-1)] = [VA_f_before_start, VA_f_before_end]
    VA_gt_dict["id_"+str(label_interval_num-1)] = [VA_gt_before_start, VA_gt_before_end]
    VA_d_dict["id_"+str(label_interval_num-1)] = [VA_d_before_start, VA_d_before_end]





    VA_gt_before_len = VA_gt_before_end - VA_gt_before_start
    x1, x2 = cal_x1_x2(VA_gt_before_len,parameter_rho)

    # # print("area_id,area",area_id,area)
    # # # print("x1,x2",x1, x2)
    # # print("x1_ori,x2_ori",x1+VA_gt_before_start[0], x2+VA_gt_before_start[0])
    x1_ori, x2_ori = x1 + VA_gt_before_start, x2 + VA_gt_before_start
    x_mid_ori = 2 * x1 + VA_gt_before_start
    split_line_set.add(x1_ori)
    split_line_set.add(x_mid_ori)
    split_line_set.add(x2_ori)

    # add by order

    split_line_set.add(VA_f_before_start)
    split_line_set.add(VA_f_before_end)

    split_line_set.add(VA_gt_before_start)
    split_line_set.add(VA_gt_before_end)

    split_line_set.add(VA_d_before_start)
    split_line_set.add(VA_d_before_end)

    if print_msg:
        # print area dict
        # # print("split area")
        print("IA_dict",IA_dict)
        # print("VA_dict",VA_dict)
        print("VA_f_dict",VA_f_dict)
        print("VA_gt_dict",VA_gt_dict)
        print("VA_d_dict",VA_d_dict)

    def dict_filer(dict_data:dict):
        for i, (area_id,area_range) in enumerate(dict_data.items()):
            if area_range != []:
                dict_data[area_id] = [] if area_range[0] == area_range[1] else area_range
        return dict_data

    IA_dict = dict_filer(IA_dict)
    VA_dict = dict_filer(VA_dict)
    VA_f_dict = dict_filer(VA_f_dict)
    VA_gt_dict = dict_filer(VA_gt_dict)
    VA_d_dict = dict_filer(VA_d_dict)

    if print_msg:
        print("dict adjust +++++++++++")
        print("IA_dict",IA_dict)
        # print("VA_dict",VA_dict)
        print("VA_f_dict",VA_f_dict)
        print("VA_gt_dict",VA_gt_dict)
        print("VA_d_dict",VA_d_dict)


        # split pred to basic interval
        print("split_line_set",split_line_set)

    split_IA1_VA1_IA2_relative_list = split_intervals(IA1_VA1_IA2_relative_list, split_line_set)
    # print("split_IA1_VA1_IA2_relative_list",split_IA1_VA1_IA2_relative_list)

    # forecasting_len=10
    # delay_len=10
    IA_pred_group_dict = {}
    VA_pred_group_dict = {}
    VA_f_pred_group_dict = {}
    VA_gt_pred_group_dict = {}
    VA_d_pred_group_dict = {}

    VA_gt1_pred_group_dict = {}
    VA_gt2_pred_group_dict = {}
    VA_gt3_pred_group_dict = {}
    VA_gt4_pred_group_dict = {}
    func_dict = {}


    # "id_"+str(label_interval_num)
    for i in range(label_interval_num):
        IA_pred_group_dict["id_"+str(i)] = []
        VA_pred_group_dict["id_"+str(i)] = []
        VA_f_pred_group_dict["id_"+str(i)] = []
        VA_gt_pred_group_dict["id_"+str(i)] = []
        VA_d_pred_group_dict["id_"+str(i)] = []
        VA_gt1_pred_group_dict["id_"+str(i)] = []
        VA_gt2_pred_group_dict["id_"+str(i)] = []
        VA_gt3_pred_group_dict["id_"+str(i)] = []
        VA_gt4_pred_group_dict["id_"+str(i)] = []
    IA_pred_group_dict["id_"+str(label_interval_num)] = []

    # for i, basic_interval in enumerate(IA1_VA1_IA2_relative_list):
    for i, basic_interval in enumerate(split_IA1_VA1_IA2_relative_list):
        # IA
        for j, (area_id,area) in enumerate(IA_dict.items()):
            if pred_in_area(basic_interval,area):
                if area_id not in IA_pred_group_dict.keys():
                    IA_pred_group_dict[area_id]=[basic_interval]
                else:
                    IA_pred_group_dict[area_id].append(basic_interval)

        # VA-f
        for j, (area_id,area) in enumerate(VA_f_dict.items()):
            if pred_in_area(basic_interval,area):
                if area_id not in VA_f_pred_group_dict.keys():
                    VA_f_pred_group_dict[area_id]=[basic_interval]
                else:
                    VA_f_pred_group_dict[area_id].append(basic_interval)

        # VA-gt
        for j, (area_id,area) in enumerate(VA_gt_dict.items()):
            if pred_in_area(basic_interval,area):
                if area_id not in VA_gt_pred_group_dict.keys():
                    VA_gt_pred_group_dict[area_id]=[basic_interval]
                else:
                    VA_gt_pred_group_dict[area_id].append(basic_interval)

            # group by x1,x2
            VA_gt_len = area[1] - area[0]
            x1, x2 = cal_x1_x2(VA_gt_len,parameter_rho)

            # # print("area_id,area",area_id,area)
            # # # print("x1,x2",x1, x2)
            # # print("x1_ori,x2_ori",x1+area[0], x2+area[0])
            area_gt1 = [0, x1]
            area_gt2 = [x1,2*x1]
            area_gt3 = [2*x1,x2]
            area_gt4 = [x2,area[1]-area[0]]
            area_gt1_ori = [area[0], x1+area[0]]
            area_gt2_ori = [x1+area[0],2*x1+area[0]]
            area_gt3_ori = [2*x1+area[0],x2+area[0]]
            area_gt4_ori = [x2+area[0],area[1]]
            # # print("area_gt1",area_gt1)
            # # print("area_gt2",area_gt2)
            # # print("area_gt3",area_gt3)
            # # print("area_gt4",area_gt4)

            # # print("area_gt1_ori",area_gt1_ori)
            # # print("area_gt2_ori",area_gt2_ori)
            # # print("area_gt3_ori",area_gt3_ori)
            # # print("area_gt4_ori",area_gt4_ori)


            if pred_in_area(basic_interval,area_gt1_ori):
                # if area_id not in VA_gt1_pred_group_dict.keys():
                #     VA_gt1_pred_group_dict[area_id]=[basic_interval]
                # else:
                #     VA_gt1_pred_group_dict[area_id].append(basic_interval)
                VA_gt1_pred_group_dict[area_id].append(basic_interval)
            # if basic_interval == [135, 141] and area == [130, 150]:
            #     a =1
            if pred_in_area(basic_interval,area_gt2_ori):
                # if area_id not in VA_gt2_pred_group_dict.keys():
                #     VA_gt2_pred_group_dict[area_id]=[basic_interval]
                # else:
                #     VA_gt2_pred_group_dict[area_id].append(basic_interval)
                VA_gt2_pred_group_dict[area_id].append(basic_interval)
            if pred_in_area(basic_interval,area_gt3_ori):
                # if area_id not in VA_gt3_pred_group_dict.keys():
                #     VA_gt3_pred_group_dict[area_id]=[basic_interval]
                # else:
                VA_gt3_pred_group_dict[area_id].append(basic_interval)
            if pred_in_area(basic_interval,area_gt4_ori):
                # if area_id not in VA_gt4_pred_group_dict.keys():
                #     VA_gt4_pred_group_dict[area_id]=[basic_interval]
                # else:
                VA_gt4_pred_group_dict[area_id].append(basic_interval)


        # VA-d
        for j, (area_id,area) in enumerate(VA_d_dict.items()):
            if pred_in_area(basic_interval,area):
                if area_id not in VA_d_pred_group_dict.keys():
                    VA_d_pred_group_dict[area_id]=[basic_interval]
                else:
                    VA_d_pred_group_dict[area_id].append(basic_interval)
        # VA
        for j, (area_id,area) in enumerate(VA_dict.items()):
            if pred_in_area(basic_interval,area):
                if area_id not in VA_pred_group_dict.keys():
                    VA_pred_group_dict[area_id]=[basic_interval]
                else:
                    VA_pred_group_dict[area_id].append(basic_interval)
    if print_msg:

        # print pred dict
        print("IA_pred_group_dict",IA_pred_group_dict)
        print("VA_f_pred_group_dict",VA_f_pred_group_dict)
        # print("VA_gt_pred_group_dict",VA_gt_pred_group_dict)
        print("VA_gt1_pred_group_dict",VA_gt1_pred_group_dict)
        print("VA_gt2_pred_group_dict",VA_gt2_pred_group_dict)
        print("VA_gt3_pred_group_dict",VA_gt3_pred_group_dict)
        print("VA_gt4_pred_group_dict",VA_gt4_pred_group_dict)
        print("VA_d_pred_group_dict",VA_d_pred_group_dict)
        # print("VA_pred_group_dict",VA_pred_group_dict)

    d_ia_set_dict = {}
    d_ia_sign_set_dict = {}
    l_ia_set_dict = {}
    dl_score_ia_dict = {}

    dl_part_ia_score_dict = {}

    for i, (area_id, basic_interval_list) in enumerate(IA_pred_group_dict.items()):
        if basic_interval_list == []:
            dl_score_ia_dict[area_id] = 0
            l_ia_set_dict[area_id] = 0
            d_ia_set_dict[area_id] = 0 # only when basic_interval_list ！= [],the value are used,so set value to 0 have boe problem
            d_ia_sign_set_dict[area_id] = 0 # only when basic_interval_list ！= [],the value are used,so set value to 0 have boe problem
            continue
        l_ia = 0
        l_ia_set_sum = 0
        d_ia_set_sum = 0
        d_ia_sign_set_sum = 0
        ia_set_num = len(basic_interval_list)
        IA_mid = (IA_dict[area_id][0] + IA_dict[area_id][1]) / 2
        IA_len = IA_dict[area_id][1] - IA_dict[area_id][0]




        test_group = "group"
        # test_group = "no group"
        if test_group == "group":
            for basic_interval in basic_interval_list:
                l_ia+= basic_interval[1] - basic_interval[0]
                # the length of fitst and last IA could be set a fixed value,min(一个周期的长度,ori len),or min(mean len,ori len)
                if i == 0:
                    base_position = IA_dict[area_id][1]
                    # d_ia_set_sum+= abs((basic_interval[1] + basic_interval[0])/2 -base_position)
                # last id of IA is label_interval_num,not label_interval_num -1
                elif i == label_interval_num:
                    base_position = IA_dict[area_id][0]
                else:
                    base_position = IA_mid
                # d_ia_set_sum+= abs((basic_interval[1] + basic_interval[0])/2 -IA_mid)
                d_ia_set_sum+= abs((basic_interval[1] + basic_interval[0])/2 -base_position)
                # i =0 or i = label_interval_num,only have positive distance

                d_ia_sign_set_sum += (basic_interval[1] + basic_interval[0]) / 2 - base_position

            l_ia_set_dict[area_id] = l_ia
            d_ia =  d_ia_set_sum/ia_set_num if ia_set_num != 0 else 0
            d_ia_set_dict[area_id] = d_ia
            d_ia_sign =  d_ia_sign_set_sum/ia_set_num if ia_set_num != 0 else 0
            d_ia_sign_set_dict[area_id] = d_ia_sign

            dl_score_ia =0
            #
            if l_ia != 0:
                if i == 0:
                    # 一边情况下就没有1了
                    # parameter_alpha1 = parameter_rate/(2+parameter_rate)
                    #
                    # debug1 = (1/2*IA_len-d_ia)
                    # debug2 = 2 / IA_len * (1/2*IA_len-d_ia)
                    # debug3 = parameter_alpha1 * 2 / IA_len * (1/2*IA_len-d_ia)
                    #
                    # debug4 = (1 - l_ia / IA_len)
                    # debug5 = (1 - parameter_alpha1) * (1 - l_ia / IA_len)

                    # dl_score_ia = parameter_alpha1 * 2 / IA_len * (1/2*IA_len-d_ia) + (1 - parameter_alpha1) * (1 - l_ia / IA_len) \
                    #     if IA_len != 0 else 0

                    parameter_alpha = parameter_rate/(1+parameter_rate)
                    debug11 = (IA_len-d_ia)
                    debug21 = 1 / IA_len * (IA_len-d_ia)
                    debug31 = parameter_alpha / IA_len * (IA_len-d_ia)

                    debug41 = (1 - l_ia / IA_len)
                    debug51 = (1 - parameter_alpha) * (1 - l_ia / IA_len)
                    # 得分跟系数也有关系

                    dl_score_ia1 = parameter_alpha / IA_len * (IA_len - d_ia) + (1 - parameter_alpha) * (
                                1 - l_ia / IA_len) \
                        if IA_len != 0 else 0
                    # 对抗边界是2
                    if parameter_rate <=2:
                        min_value = parameter_alpha / IA_len * (IA_len - 1/2*IA_len)
                        # min_value = min_value + 0.0000001
                        dl_score_ia = map_value(dl_score_ia1, min_value, 1, 1e-10, 1)
                    else:
                        min_value = (1 - parameter_alpha) * (1 - 0 / IA_len)
                        dl_score_ia = map_value(dl_score_ia1, min_value, 1, 1e-10, 1)

                        # dl_score_ia = dl_score_ia1
                    # print("0 dl_score_ia",dl_score_ia)
                    # dl_part_ia_score_dict[area_id] = {
                    #     "parameter_alpha":parameter_alpha,
                    #     "d_score":1/IA_len * (IA_len - d_ia),
                    #     "(1 - parameter_alpha)":(1 - parameter_alpha),
                    #     "d_score_parameter":(1 - parameter_alpha) * (
                    #                 1 - l_ia / IA_len)
                    # }
                    if pred_case_id == 2:
                        debug = 1
                # last id of IA is label_interval_num,not label_interval_num -1
                elif i == label_interval_num:

                    # debug1 = (1 / 2 * IA_len - d_ia)
                    # debug2 = 2 / IA_len * (1 / 2 * IA_len - d_ia)
                    # debug3 = parameter_alpha * 2 / IA_len * (1 / 2 * IA_len - d_ia)
                    #
                    # debug4 = (1 - l_ia / IA_len)
                    # debug5 = (1 - parameter_alpha) * (1 - l_ia / IA_len)

                    # dl_score_ia = parameter_alpha * 2 / IA_len * (1/2*IA_len-d_ia) + (1 - parameter_alpha) * (1 - l_ia / IA_len) \
                    #     if IA_len != 0 else 0

                    parameter_alpha = parameter_rate/(1+parameter_rate)
                    debug11 = (IA_len - d_ia)
                    debug21 = 1 / IA_len * (IA_len - d_ia)
                    debug31 = parameter_alpha / IA_len * (IA_len - d_ia)

                    debug41 = (1 - l_ia / IA_len)
                    debug51 = (1 - parameter_alpha) * (1 - l_ia / IA_len)
                    dl_score_ia1 = parameter_alpha / IA_len * (IA_len - d_ia) + (1 - parameter_alpha) * (1 - l_ia / IA_len) \
                        if IA_len != 0 else 0
                    # x, a, b, c, d
                    if parameter_rate <=2:
                        min_value = parameter_alpha / IA_len * (IA_len - 1/2*IA_len)
                        # min_value = min_value + 0.0000001
                        dl_score_ia = map_value(dl_score_ia1, min_value, 1, 1e-10, 1)
                    else:
                        min_value = (1 - parameter_alpha) * (1 - 0 / IA_len)
                        dl_score_ia = map_value(dl_score_ia1, min_value, 1, 1e-10, 1)
                        # dl_score_ia = map_value(dl_score_ia,parameter_alpha,1,0,1)
                    # dl_part_ia_score_dict[area_id] = {
                    #     "parameter_alpha": parameter_alpha,
                    #     "d_score": 1 / IA_len * (IA_len - d_ia),
                    #     "(1 - parameter_alpha)": (1 - parameter_alpha),
                    #     "d_score_parameter": (1 - parameter_alpha) * (
                    #             1 - l_ia / IA_len)
                    # }
                    if pred_case_id == 2:
                        debug = 1
                else:

                    parameter_alpha = parameter_rate/(2+parameter_rate)
                    dl_score_ia = parameter_alpha * 2 /IA_len*d_ia + (1-parameter_alpha)*(1-l_ia/IA_len) \
                        if IA_len !=0 else 0
                    if pred_case_id == 2:
                        print("  parameter_alpha", parameter_alpha)
                        print("  d_ia", d_ia)
                        print("  d_score", 2 / IA_len * d_ia)
                        print("  d_score_parameter", parameter_alpha * 2 / IA_len * d_ia)
                        print("  (1 - parameter_alpha)", (1 - parameter_alpha))
                        print("  l_ia", l_ia)
                        print("  l_score", (1 - l_ia / IA_len))
                        print("  l_score_parameter", (1 - parameter_alpha) * (1 - l_ia / IA_len))
                        print("  IA_len", IA_len)
                        debug = 1
                dl_part_ia_score_dict
        else:
            if print_msg:
                print("basic_interval_list num", len(basic_interval_list))
            dl_score_ia_sum = 0
            for basic_interval in basic_interval_list:
                if i == 0:
                    base_position = IA_dict[area_id][1]
                elif i == label_interval_num:
                    base_position = IA_dict[area_id][0]
                else:
                    base_position = IA_mid

                l_ia = basic_interval[1] - basic_interval[0]
                d_ia = abs((basic_interval[1] + basic_interval[0]) / 2 - base_position)
                if pred_case_id == 2:
                    debug = 1
                dl_score_ia = 0
                #
                if l_ia != 0:
                    if i == 0:
                        parameter_alpha = parameter_rate / (1 + parameter_rate)
                        dl_score_ia1 = parameter_alpha / IA_len * (IA_len - d_ia) + (1 - parameter_alpha) * (
                                1 - l_ia / IA_len) \
                            if IA_len != 0 else 0
                        # 对抗边界是2
                        if parameter_rate <= 2:
                            min_value = parameter_alpha / IA_len * (IA_len - 1 / 2 * IA_len)
                            dl_score_ia = map_value(dl_score_ia1, min_value, 1, 1e-10, 1)
                        else:
                            min_value = (1 - parameter_alpha) * (1 - 0 / IA_len)
                            dl_score_ia = map_value(dl_score_ia1, min_value, 1, 1e-10, 1)
                        if print_msg:
                            print("0 dl_score_ia", dl_score_ia)
                        if pred_case_id == 2:
                            debug = 1
                    elif i == label_interval_num:
                        parameter_alpha = parameter_rate / (1 + parameter_rate)
                        debug11 = (IA_len - d_ia)
                        debug21 = 1 / IA_len * (IA_len - d_ia)
                        debug31 = parameter_alpha / IA_len * (IA_len - d_ia)

                        debug41 = (1 - l_ia / IA_len)
                        debug51 = (1 - parameter_alpha) * (1 - l_ia / IA_len)
                        dl_score_ia1 = parameter_alpha / IA_len * (IA_len - d_ia) + (1 - parameter_alpha) * (
                                    1 - l_ia / IA_len) \
                            if IA_len != 0 else 0
                        # x, a, b, c, d
                        if parameter_rate <= 2:
                            min_value = parameter_alpha / IA_len * (IA_len - 1 / 2 * IA_len)
                            # min_value = min_value + 0.0000001
                            dl_score_ia = map_value(dl_score_ia1, min_value, 1, 1e-10, 1)
                        else:
                            min_value = (1 - parameter_alpha) * (1 - 0 / IA_len)
                            dl_score_ia = map_value(dl_score_ia1, min_value, 1, 1e-10, 1)
                            # dl_score_ia = map_value(dl_score_ia,parameter_alpha,1,0,1)
                        if pred_case_id == 2:
                            debug = 1
                    else:

                        parameter_alpha = parameter_rate / (2 + parameter_rate)
                        dl_score_ia = parameter_alpha * 2 / IA_len * d_ia + (1 - parameter_alpha) * (1 - l_ia / IA_len) \
                            if IA_len != 0 else 0
                        if pred_case_id == 2:
                            if print_msg:
                                print("  parameter_alpha", parameter_alpha)
                                print("  d_ia", d_ia)
                                print("  d_score", 2 / IA_len * d_ia)
                                print("  d_score_parameter", parameter_alpha * 2 / IA_len * d_ia)
                                print("  (1 - parameter_alpha)", (1 - parameter_alpha))
                                print("  l_ia", l_ia)
                                print("  l_score", (1 - l_ia / IA_len))
                                print("  l_score_parameter", (1 - parameter_alpha) * (1 - l_ia / IA_len))
                                print("  IA_len", IA_len)
                                debug = 1

                # sum
                dl_score_ia_sum += dl_score_ia
                l_ia_set_sum += basic_interval[1] - basic_interval[0]
                d_ia_set_sum += abs((basic_interval[1] + basic_interval[0]) / 2 - base_position)
                d_ia_sign_set_sum += (basic_interval[1] + basic_interval[0]) / 2 - base_position


            # sum
            # test_no_group_type = "sum"
            test_no_group_type = "average"
            if test_no_group_type == "sum":
                dl_score_ia = dl_score_ia_sum
                l_ia_set_dict[area_id] = l_ia_set_sum
                d_ia_set_dict[area_id] = d_ia_set_sum
                d_ia_sign_set_dict[area_id] = d_ia_sign_set_sum
            else:
                dl_score_ia = dl_score_ia_sum/len(basic_interval_list) if len(basic_interval_list) !=0 else 0
                l_ia_set_dict[area_id] = l_ia_set_sum/len(basic_interval_list) if len(basic_interval_list) !=0 else 0
                d_ia_set_dict[area_id] = d_ia_set_sum/len(basic_interval_list) if len(basic_interval_list) !=0 else 0
                d_ia_sign_set_dict[area_id] = d_ia_sign_set_sum
        dl_score_ia_dict[area_id] = dl_score_ia
    #     deal empty and no area
    if print_msg:
        print("d_ia_set_dict",d_ia_set_dict)
        print("l_ia_set_dict",l_ia_set_dict)
    # print("dl_score_ia_dict",dl_score_ia_dict)


    # cal label_function_gt

    for i, (area_id, area) in enumerate(VA_gt_dict.items()):

        VA_gt_len = area[1] - area[0]
        x1, x2 = cal_x1_x2(VA_gt_len,parameter_rho)
        x1_ori, x2_ori = x1 + area[0], x2 + area[0]
        x_mid = 2*x1
        x_mid_ori = x_mid + area[0]
        # x1_ori,x2_ori = x1+area[0], x2+area[0]
        area_gt1 = [0, x1]
        area_gt2 = [x1, 2 * x1]
        area_gt3 = [2 * x1, x2]
        area_gt4 = [x2, VA_gt_len]
        area_gt1_ori = [area[0], x1 + area[0]]
        area_gt2_ori = [x1 + area[0], 2 * x1 + area[0]]
        area_gt3_ori = [2 * x1 + area[0], x2 + area[0]]
        area_gt4_ori = [x2 + area[0], area[1]]
        if print_msg:
            print("VA_gt_len",VA_gt_len)

        if parameter_rho < 0:
            parameter_b1, parameter_gama1 = 1/2,0
            parameter_b2, parameter_gama2 = 1/2,0
        else:
            parameter_b1, parameter_gama1 = cal_power_function_coefficient(x1,parameter_lamda)
            parameter_b2, parameter_gama2 = cal_power_function_coefficient(x2-x_mid,parameter_lamda)

        func_dict.update({
            area_id+"_gt1":{"area":area_gt1_ori,
                    "area_start":area_gt1_ori[0],
                    "parameter_a":parameter_b1,
                    "parameter_beta":parameter_gama1,
                    "func":"va_gt1_func",
                    "x_tp":x1,
                    "gt_len":VA_gt_len,
                   "area_len":None,
                   "parameter_rho":parameter_rho
                    },
            area_id+"_gt2": {"area": area_gt2_ori,
                    "area_start": area_gt1_ori[0],
                    "parameter_a": parameter_b1,
                    "parameter_beta": parameter_gama1,
                    "func": "va_gt2_func",
                    "x_tp": x1,
                    "gt_len": VA_gt_len,
                   "area_len":None,
                   "parameter_rho":parameter_rho
                    },
            area_id+"_gt3": {"area": area_gt3_ori,
                    "area_start": area_gt1_ori[0],
                    "parameter_a": parameter_b2,
                    "parameter_beta": parameter_gama2,
                    "func": "va_gt3_func",
                    "x_tp": x2,
                    "gt_len": VA_gt_len,
                   "area_len":None,
                   "parameter_rho":parameter_rho
                    },
            area_id+"_gt4": {"area": area_gt4_ori,
                    "area_start": area_gt1_ori[0],
                    "parameter_a": parameter_b2,
                    "parameter_beta": parameter_gama2,
                    "func": "va_gt4_func",
                    "x_tp": x2,
                    "gt_len": VA_gt_len,
                   "area_len":None,
                   "parameter_rho":parameter_rho
                    },


        })


        s_a_gt1 = cal_integral_in_range_gt_power_func1(area_gt1_ori, parameter_b1, parameter_gama1,x_tp=x1,gt_len=VA_gt_len,area_start=area_gt1_ori[0],parameter_rho=parameter_rho)
        s_a_gt2 = cal_integral_in_range_gt_power_func2(area_gt2_ori, parameter_b1, parameter_gama1,x_tp=x1,gt_len=VA_gt_len,area_start=area_gt1_ori[0],parameter_rho=parameter_rho)
        s_a_gt3 = cal_integral_in_range_gt_power_func3(area_gt3_ori, parameter_b2, parameter_gama2,x_tp=x2,gt_len=VA_gt_len,area_start=area_gt1_ori[0],parameter_rho=parameter_rho)
        s_a_gt4 = cal_integral_in_range_gt_power_func4(area_gt4_ori, parameter_b2, parameter_gama2,x_tp=x2,gt_len=VA_gt_len,area_start=area_gt1_ori[0],parameter_rho=parameter_rho)

        s_a_gt = s_a_gt1+s_a_gt2+s_a_gt3+s_a_gt4

        func_dict[area_id + "_gt1"]["area_s"] = s_a_gt1
        func_dict[area_id + "_gt2"]["area_s"] = s_a_gt2
        func_dict[area_id + "_gt3"]["area_s"] = s_a_gt3
        func_dict[area_id + "_gt4"]["area_s"] = s_a_gt4

        method = 2

        va_f_area = VA_f_dict[area_id]
        s_a_f = 0
        if va_f_area == []:
            pass
        else:
            VA_f_len = va_f_area[1] - va_f_area[0]
            parameter_beta1 = 1
            parameter_a1 = 1 / (2 * VA_f_len)

            greater_than_th = False

            func_dict[area_id+"_f"] = {"area": va_f_area,
                    "area_start": va_f_area[0],
                    "parameter_a": parameter_a1,
                    "parameter_beta": parameter_beta1,
                    "func": "va_f_func",
                    "x_tp": None,
                    "gt_len": None,
                    "area_len": VA_f_len,
                    "greater_than_th": greater_than_th
            }

            # f
            s_max_f = cal_integral_in_range_f_power_func(va_f_area, parameter_a1, parameter_beta1,VA_f_len,VA_f_len,area_start=va_f_area[0],greater_than_th=greater_than_th)
            func_dict[area_id + "_f"]["area_s"] = s_max_f

            s_rate = s_max_f / s_a_gt
            func_dict[area_id + "_f_a"] = func_dict[area_id+"_f"]
            s_a_f = s_max_f
            if s_rate > parameter_theta:
                greater_than_th = True

                # method1
                if method== 1:
                    parameter_beta1_a = VA_f_len / (2 * parameter_theta * s_a_gt) - 1
                    parameter_a1_a = 1 / (2 * pow(VA_f_len, parameter_beta1_a))
                # method2
                else:
                    parameter_beta1_a = VA_f_len / (2 * parameter_theta * s_a_gt) - 1 # same with method1
                    parameter_a1_a = VA_f_len / pow(1/2, parameter_beta1_a)

                s_a_f = cal_integral_in_range_f_power_func(va_f_area, parameter_a1_a, parameter_beta1_a,VA_f_len,VA_f_len,area_start=va_f_area[0],method=method,greater_than_th=greater_than_th)
                func_dict[area_id + "_f_a"] = {
                    "area": va_f_area,
                    "area_start": va_f_area[0],
                    "area_s": s_a_f,
                    "parameter_a": parameter_a1_a,
                    "parameter_beta": parameter_beta1_a,
                    # "func": "va_f_func",
                    # "func": "va_f_reverse_func",
                    "x_tp": None,
                    "gt_len": None,
                    "area_len": VA_f_len,
                    "greater_than_th": greater_than_th
                }
                if method== 1:
                    func_dict[area_id + "_f_a"]["func"] = "va_f_func"
                else:
                    func_dict[area_id + "_f_a"]["func"] = "va_f_reverse_func"





        # d
        va_d_area = VA_d_dict[area_id]
        s_a_d = 0
        if va_d_area == []:
            pass
        else:

            if pred_case_id == 7 and area_id == "id_2":
                debug = 1


            VA_d_len = va_d_area[1] - va_d_area[0]
            parameter_beta2 = 1
            parameter_a2 = 1 / (2 * VA_d_len)

            greater_than_th = False
            s_max_d = cal_integral_in_range_d_power_func(va_d_area, parameter_a2, parameter_beta2, VA_d_len, VA_d_len,
                                                         area_start=va_d_area[0],greater_than_th=greater_than_th)
            func_dict[area_id + "_d"] = {
                "area": va_d_area,
                "area_start": va_d_area[0],
                "parameter_a": parameter_a2,
                "parameter_beta": parameter_beta2,
                "func": "va_d_func",
                "x_tp": None,
                "gt_len": None,
                "area_len": VA_d_len,
                "greater_than_th": greater_than_th
            }

            func_dict[area_id + "_d"]["area_s"] = s_max_d

            func_dict[area_id + "_d_a"] = func_dict[area_id + "_d"]
            s_rate = s_max_d / s_a_gt
            s_a_d = s_max_d



            if s_rate > parameter_theta:
                greater_than_th = True

                # method1
                if method== 1:
                    parameter_beta2_a = VA_d_len / (2 * parameter_theta * s_a_gt) - 1
                    parameter_a2_a = 1 / (2 * pow(VA_d_len, parameter_beta2_a))
                else:
                    # method2
                    parameter_beta2_a = VA_d_len / (2 * parameter_theta * s_a_gt) - 1  # same with method1
                    parameter_a2_a = VA_d_len / pow(1 / 2, parameter_beta2_a)

                s_a_d = cal_integral_in_range_d_power_func(va_d_area, parameter_a2_a, parameter_beta2_a, VA_d_len, VA_d_len,
                                                           area_start=va_d_area[0],method=method,greater_than_th=greater_than_th)
                func_dict[area_id + "_d_a"] = {
                    "area": va_d_area,
                    "area_start": va_d_area[0],
                    "area_s": s_a_d,
                    "parameter_a": parameter_a2_a,
                    "parameter_beta": parameter_beta2_a,
                    # "func": "va_d_func",
                    # "func": "va_d_reverse_func",
                    "x_tp": None,
                    "gt_len": None,
                    "area_len": VA_d_len,
                    "greater_than_th": greater_than_th

                }

                if method== 1:
                    func_dict[area_id + "_d_a"]["func"] = "va_d_func"
                else:
                    func_dict[area_id + "_d_a"]["func"] = "va_d_reverse_func"



        s_a_func_dict[area_id]= {
                                    "s_a_va_gt_len":VA_gt_len,
                                    "s_a_va_gt":s_a_gt,
                                    "s_a_va":s_a_f+s_a_gt+s_a_d,
                                    "x1":x1,
                                    "x2":x2,
                                    "x_mid":x_mid,
                                    "x1_ori": x1_ori,
                                    "x2_ori": x2_ori,
                                    "x_mid_ori":x_mid_ori,
                                    }
    # plot func
    if pred_case_id == 7 and area_id == "id_2":
        debug = 1
        # plot_func_multi(func_dict)
    # plot_func_multi(func_dict)
    # plot_func_multi_paper(func_dict,window_length)

    import pprint

    # print("s_a_func_dict")
    # pprint(s_a_func_dict, indent=4)


    # cal recall_va,detection_score
    recall_va_list = []
    recall_va_dict = {}
    detection_score_ia_i_dict = {}
    precision_va_dict = {}

    gt_num = label_interval_num
    pred_func_dict = {}
    basic_interval_integral_dict = {}

    for i in range(gt_num):
        area_id = "id_"+str(i)
        # print("area_id",area_id)
        area_id_back = "id_"+str(i+1)
        VA_f_i_pred_group = VA_f_pred_group_dict[area_id]
        VA_gt1_i_pred_group = VA_gt1_pred_group_dict[area_id]
        VA_gt2_i_pred_group = VA_gt2_pred_group_dict[area_id]
        VA_gt3_i_pred_group = VA_gt3_pred_group_dict[area_id]
        VA_gt4_i_pred_group = VA_gt4_pred_group_dict[area_id]
        VA_d_i_pred_group = VA_d_pred_group_dict[area_id]

        IA_pred_i_group = IA_pred_group_dict[area_id]
        IA_pred_i_back_group = IA_pred_group_dict[area_id_back]
        VA_pred_i_group = VA_pred_group_dict[area_id]


        pred_integral_all = 0



        # # f
        va_f_area = VA_f_dict[area_id]
        if va_f_area == []:
            pass
        else:
            VA_f_len = va_f_area[1] - va_f_area[0]
            parameter_a1 = func_dict[area_id+"_f_a"]["parameter_a"]
            parameter_beta1 = func_dict[area_id+"_f_a"]["parameter_beta"]
            greater_than_th = func_dict[area_id+"_f_a"]["greater_than_th"]


            for j, basic_interval in enumerate(VA_f_i_pred_group):
                if basic_interval != []:

                    pred_func_dict[area_id+"_f_a" + "_"+ str(j)] = {
                        "area":basic_interval,
                        "area_start":va_f_area[0],
                        "parameter_a":parameter_a1,
                        "parameter_beta":parameter_beta1,
                        # "func":"va_f_func",
                        "x_tp": None,
                        "gt_len": None,
                        "area_len": VA_f_len
                    }
                    if not greater_than_th:
                        pred_func_dict[area_id + "_f_a" + "_" + str(j)]["func"] = "va_d_func"
                    else:
                        if method == 1:
                            pred_func_dict[area_id+"_f_a" + "_"+ str(j)]["func"] = "va_f_func"
                        else:
                            pred_func_dict[area_id+"_f_a" + "_"+ str(j)]["func"] = "va_f_reverse_func"

                    cal_integral_basic_interval_f = cal_integral_in_range_f_power_func(basic_interval, parameter_a1, parameter_beta1,VA_f_len,VA_f_len,area_start=va_f_area[0],method=method,greater_than_th=greater_than_th)
                    pred_integral_all += cal_integral_basic_interval_f
                    basic_interval_integral_dict[area_id+"_f" + "_"+ str(j)] =cal_integral_basic_interval_f


        # gt1-gt4
        s_a_gt_len = s_a_func_dict[area_id]["s_a_va_gt_len"]
        s_a_va = s_a_func_dict[area_id]["s_a_va"]
        parameter_b1 = func_dict[area_id+"_gt1"]["parameter_a"]
        parameter_gama1 = func_dict[area_id+"_gt1"]["parameter_beta"]
        parameter_b2 = func_dict[area_id+"_gt3"]["parameter_a"]
        parameter_gama2 = func_dict[area_id+"_gt3"]["parameter_beta"]
        x1 = s_a_func_dict[area_id]["x1"]
        x2 = s_a_func_dict[area_id]["x2"]
        area_gt = VA_gt_dict[area_id]
        VA_gt_len= area_gt[1] -area_gt[0]

        area_gt1 = func_dict[area_id + "_gt1"]["area"]
        area_gt2 = func_dict[area_id + "_gt2"]["area"]
        area_gt3 = func_dict[area_id + "_gt3"]["area"]
        area_gt4 = func_dict[area_id + "_gt4"]["area"]
        area_start = func_dict[area_id + "_gt1"]["area_start"]

        for j, basic_interval in enumerate(VA_gt1_i_pred_group):
            if basic_interval != []:
                pred_func_dict[area_id + "_gt1" + "_" + str(j)] = {
                    "area": basic_interval,
                    "area_start": area_start,
                    "parameter_a": parameter_b1,
                    "parameter_beta": parameter_gama1,
                    "func": "va_gt1_func",
                    "x_tp": None,
                    "gt_len": VA_gt_len,
                    "area_len": None,
                   "parameter_rho":parameter_rho
                }
                cal_integral_basic_interval_gt1 = cal_integral_in_range_gt_power_func1(basic_interval, parameter_b1, parameter_gama1,x_tp=x1,gt_len=VA_gt_len,area_start=area_start,parameter_rho=parameter_rho)
                pred_integral_all += cal_integral_basic_interval_gt1
                basic_interval_integral_dict[area_id+"_gt1" + "_"+ str(j)] =cal_integral_basic_interval_gt1

        for j, basic_interval in enumerate(VA_gt2_i_pred_group):
            if basic_interval != []:
                pred_func_dict[area_id + "_gt2" + "_" + str(j)] = {
                    "area": basic_interval,
                    "area_start": area_start,
                    "parameter_a": parameter_b1,
                    "parameter_beta": parameter_gama1,
                    "func": "va_gt2_func",
                    "x_tp": x1,
                    "gt_len": VA_gt_len,
                    "area_len": None,
                   "parameter_rho":parameter_rho
                }
                cal_integral_basic_interval_gt2 = cal_integral_in_range_gt_power_func2(basic_interval, parameter_b1, parameter_gama1,x_tp=x1,gt_len=VA_gt_len,area_start=area_start,parameter_rho=parameter_rho)
                pred_integral_all += cal_integral_basic_interval_gt2
                basic_interval_integral_dict[area_id+"_gt2" + "_"+ str(j)] =cal_integral_basic_interval_gt2


        for j, basic_interval in enumerate(VA_gt3_i_pred_group):
            if basic_interval != []:
                pred_func_dict[area_id + "_gt3" + "_" + str(j)] = {
                    "area": basic_interval,
                    "area_start": area_start,
                    "parameter_a": parameter_b2,
                    "parameter_beta": parameter_gama2,
                    "func": "va_gt3_func",
                    "x_tp": x2,
                    "gt_len": VA_gt_len,
                    "area_len": None,
                   "parameter_rho":parameter_rho

                }
                cal_integral_basic_interval_gt3 = cal_integral_in_range_gt_power_func3(basic_interval, parameter_b2, parameter_gama2,x_tp=x2,gt_len=VA_gt_len,area_start=area_start,parameter_rho=parameter_rho)
                pred_integral_all += cal_integral_basic_interval_gt3
                basic_interval_integral_dict[area_id+"_gt3" + "_"+ str(j)] =cal_integral_basic_interval_gt3

        for j, basic_interval in enumerate(VA_gt4_i_pred_group):
            if basic_interval != []:
                pred_func_dict[area_id + "_gt4" + "_" + str(j)] = {
                    "area": basic_interval,
                    "area_start": area_start,
                    "parameter_a": parameter_b2,
                    "parameter_beta": parameter_gama2,
                    "func": "va_gt4_func",
                    "x_tp": x2,
                    "gt_len": VA_gt_len,
                    "area_len": None,
                   "parameter_rho":parameter_rho
                }
                cal_integral_basic_interval_gt4 = cal_integral_in_range_gt_power_func4(basic_interval, parameter_b2, parameter_gama2,x_tp=x2,gt_len=VA_gt_len,area_start=area_start,parameter_rho=parameter_rho)
                pred_integral_all += cal_integral_basic_interval_gt4
                basic_interval_integral_dict[area_id+"_gt4" + "_"+ str(j)] =cal_integral_basic_interval_gt4



        # # d
        va_d_area = VA_d_dict[area_id]
        if va_d_area == []:
            pass
        else:
            VA_d_len = va_d_area[1] - va_d_area[0]
            parameter_a2 = func_dict[area_id+"_d_a"]["parameter_a"]
            parameter_beta2 = func_dict[area_id+"_d_a"]["parameter_beta"]
            greater_than_th = func_dict[area_id+"_d_a"]["greater_than_th"]
            if pred_case_id == 7 and area_id == "id_2":
                debug = 1

            for j, basic_interval in enumerate(VA_d_i_pred_group):
                if basic_interval != []:
                    pred_func_dict[area_id+"_d_a" + "_"+ str(j)] = {
                        "area":basic_interval,
                        "area_start":va_d_area[0],
                        "parameter_a":parameter_a2,
                        "parameter_beta":parameter_beta2,
                        # "func":"va_d_func",
                        "x_tp": None,
                        "gt_len": None,
                        "area_len": VA_d_len
                    }
                    if not greater_than_th:
                        pred_func_dict[area_id + "_d_a" + "_" + str(j)]["func"] = "va_d_func"
                    else:
                        if method == 1:
                            pred_func_dict[area_id+"_d_a" + "_"+ str(j)]["func"] = "va_d_func"
                        else:
                            pred_func_dict[area_id+"_d_a" + "_"+ str(j)]["func"] = "va_d_reverse_func"


                    cal_integral_basic_interval_d = cal_integral_in_range_d_power_func(basic_interval, parameter_a2, parameter_beta2, VA_d_len, VA_d_len,
                                                           area_start=va_d_area[0],method=method,greater_than_th=greater_than_th)
                    pred_integral_all += cal_integral_basic_interval_d
                    basic_interval_integral_dict[area_id+"_d" + "_"+ str(j)] =cal_integral_basic_interval_d

        # print("pred_integral_all",pred_integral_all)
        va_area = VA_dict[area_id]
        recall_va_i = pred_integral_all / s_a_va
        recall_va_list.append(recall_va_i)
        if pred_case_id == 7:
            debug = 1
        recall_va_dict[area_id] = recall_va_i

        detection_score_i = 1 if VA_pred_group_dict[area_id] != [] else 0
        detection_score_ia_i_dict[area_id] = detection_score_i
        # invalid part punish in is not enough
        # detection
        if cal_mode == "detection":
            precision_va_dict[area_id] = detection_score_i


        precision_i_group =IA_pred_i_group+IA_pred_i_back_group+VA_pred_i_group

        precision_i_all = 0
        precision_i_valid = 0
        for precision_i_item in precision_i_group:
            precision_i_all+=precision_i_item[1]-precision_i_item[0]
        for precision_i_item in VA_pred_i_group:
            precision_i_valid+=precision_i_item[1]-precision_i_item[0]
        if pred_case_id == 6:
            debug = 1
        precision_va_i = precision_i_valid/precision_i_all if precision_i_all != 0 else 0

        # precision also consider invalid part
        # proportion
        if cal_mode == "proportion":
            precision_va_dict[area_id] = precision_va_i


    if print_msg:
        # print basic_interval_integral
        print("basic_interval_integral_dict",basic_interval_integral_dict)

    # pot pred func
    # if pred_case_id == 7 and area_id == "id_2":
    #     debug = 1
    #     plot_func_multi(pred_func_dict)
    # plot_func_multi(pred_func_dict)

    if print_msg:
        # print recall_va_dict,precision_va_dict,dl_score_ia_dict
        print("recall_va_dict",recall_va_dict)
        print("precision_va_dict", precision_va_dict)
        print("dl_score_ia_dict", dl_score_ia_dict)



    adjusted_dl_score_ia_dict = {}
    adjusted_dl_score_ia_dict_ratio_contribution = {}

    if pred_case_id == 2:
        debug = 1
    area_id_0 = "id_" + str(0)
    area_id_num = "id_" + str(gt_num)
    # two logics
    logic = 1
    # logic = 3
    if logic == 1:
    # logic 1
    #     one_exist_value = 1/2
        one_exist_value = 1
        for i in range(gt_num+1):
            area_id = "id_" + str(i)
            area_id_front = "id_"+str(i-1)
            area_id_back = "id_"+str(i+1)

            ia_area = IA_dict[area_id]
            l_ia = l_ia_set_dict[area_id]
            d_ia = d_ia_set_dict[area_id]
            d_ia_sign = d_ia_sign_set_dict[area_id]
            ia_len = ia_area[1] - ia_area[0] if ia_area != [] else 0

            if pred_case_id == 22:
                debug = 1

            if i == 0:
                if IA_dict[area_id] == []:
                    # if IA_dict[area_id_back] == [] and area_id_back == area_id_num:
                    if area_id_back == area_id_num:
                        if IA_dict[area_id_back] == []: #自己为空，后面一个为最后一个，后面一个为空
                            ratio_before = 0
                            ratio_after = 1/2
                        else: #自己为空，后面一个为最后一个，后面一个不为空
                            ratio_before = 0
                            ratio_after = 0
                    else:
                        ratio_before = 0
                        ratio_after = 1 / 2

                    if VA_pred_group_dict[area_id] != []:
                        dl_score_ia_i_contribution = 1
                    else:
                        dl_score_ia_i_contribution = 0
                else:
                    # 数据里至少有一个异常，至少有两个ia，如果没有异常，那么IA_dict=[]
                    if IA_pred_group_dict[area_id] != []:
                        if area_id_back == area_id_num:
                            if IA_dict[area_id_back] == []: #自己不为空，后面一个为最后一个，后面一个为空
                                ratio_before = 0
                                ratio_after = 1
                            else: #自己不为空，后面一个为最后一个，后面一个不为空
                                ratio_before = 0
                                ratio_after = 1 / 2
                        else:
                            ratio_before = 0
                            ratio_after = 1 / 2
                        dl_score_ia_i_contribution = dl_score_ia_dict[area_id]
                    else:
                        if VA_pred_group_dict[area_id] != []:
                            if area_id_back == area_id_num:
                                if IA_dict[area_id_back] == []: #自己不为空，后面一个为最后一个，后面一个为空
                                    ratio_before = 0
                                    ratio_after = 1
                                else: #自己不为空，后面一个为最后一个，后面一个不为空
                                    ratio_before = 0
                                    ratio_after = 1 / 2
                            else:
                                ratio_before = 0
                                ratio_after = 1 / 2
                            dl_score_ia_i_contribution = 1
                        else:
                            if area_id_back == area_id_num:
                                if IA_dict[area_id_back] == []: #自己不为空，后面一个为最后一个，后面一个为空
                                    ratio_before = 0
                                    ratio_after = 1
                                else: #自己不为空，后面一个为最后一个，后面一个不为空
                                    ratio_before = 0
                                    ratio_after = 1 / 2
                            else:
                                ratio_before = 0
                                ratio_after = 1 / 2
                            dl_score_ia_i_contribution = 0
            elif i == gt_num:
                if IA_dict[area_id] == []:
                    if area_id_front == area_id_0:
                        if IA_dict[area_id_front] == []: #自己为空，前面一个为最后一个，前面一个为空
                            ratio_before = 1/2
                            ratio_after = 0
                        else: #自己为空，前面一个为第一个，前面一个不为空
                            ratio_before = 0
                            ratio_after = 0
                    else:
                        ratio_before = 1 / 2
                        ratio_after = 0

                    if VA_pred_group_dict[area_id_front] != []:
                        dl_score_ia_i_contribution = 1
                    else:
                        dl_score_ia_i_contribution = 0
                else:
                    if IA_pred_group_dict[area_id] != []:
                        if area_id_front == area_id_0:
                            if IA_dict[area_id_front] == []:
                                ratio_before = 1
                                ratio_after = 0
                            else:
                                ratio_before = 1 / 2
                                ratio_after = 0
                        else:
                            ratio_before = 1 / 2
                            ratio_after = 0
                        dl_score_ia_i_contribution = dl_score_ia_dict[area_id]
                    else:
                        if VA_pred_group_dict[area_id_front] != []:
                            if area_id_front == area_id_0:
                                if IA_dict[area_id_front] == []:
                                    ratio_before = 1
                                    ratio_after = 0
                                else:
                                    ratio_before = 1 / 2
                                    ratio_after = 0
                            else:
                                ratio_before = 1 / 2
                                ratio_after = 0
                            dl_score_ia_i_contribution = 1
                        else:
                            if IA_dict[area_id_front] == []:
                                if area_id_front == area_id_0:
                                    if IA_dict[area_id_front] == []:
                                        ratio_before = 1
                                        ratio_after = 0
                                    else:
                                        ratio_before = 1 / 2
                                        ratio_after = 0
                                else:
                                    ratio_before = 1 / 2
                                    ratio_after = 0
                            dl_score_ia_i_contribution = 0
            else:
                if IA_pred_group_dict[area_id] != []:
                    ratio_before = 1 / 2
                    ratio_after = 1 / 2
                    # if d_ia_sign > 0:
                    #     ratio_before = (ia_len / 2 - d_ia) / ia_len
                    #     ratio_after = (ia_len / 2 + d_ia) / ia_len
                    # elif d_ia_sign < 0:
                    #     ratio_before = (ia_len / 2 + d_ia) / ia_len
                    #     ratio_after = (ia_len / 2 - d_ia) / ia_len
                    # else:
                    #     ratio_before = 1/2
                    #     ratio_after = 1/2
                    dl_score_ia_i_contribution = dl_score_ia_dict[area_id]
                else:
                    ratio_before = 1/2
                    ratio_after = 1/2
                    if VA_pred_group_dict[area_id_front] != [] and VA_pred_group_dict[area_id] != []:
                        # ratio_before = 1/2
                        # ratio_after = 1/2
                        dl_score_ia_i_contribution = 1
                    elif VA_pred_group_dict[area_id_front] == [] and VA_pred_group_dict[area_id] == []:
                        # ratio_before = 1/2
                        # ratio_after = 1/2
                        dl_score_ia_i_contribution = 0
                    else:
                        # if VA_pred_group_dict[area_id_front] != []:
                        #     ratio_before = 1
                        #     ratio_after = 0
                        # else:
                        #     ratio_before = 0
                        #     ratio_after = 1
                        dl_score_ia_i_contribution = one_exist_value

            adjusted_dl_score_ia_dict[area_id] = dl_score_ia_i_contribution
            dl_score_ia_i_before_contribution = dl_score_ia_i_contribution * ratio_before
            dl_score_ia_i_after_contribution = dl_score_ia_i_contribution * ratio_after
            adjusted_dl_score_ia_dict_ratio_contribution[area_id] = [dl_score_ia_i_before_contribution, dl_score_ia_i_after_contribution]


    debug = 1
    # logic 2
    # one_exist_value = 1/2
    # for i in range(gt_num + 1):
    #     area_id = "id_" + str(i)
    #     area_id_front = "id_" + str(i - 1)
    #     area_id_back = "id_" + str(i + 1)
    #
    #     ia_area = IA_dict[area_id]
    #     l_ia = l_ia_set_dict[area_id]
    #     d_ia = d_ia_set_dict[area_id]
    #     d_ia_sign = d_ia_sign_set_dict[area_id]
    #     ia_len = ia_area[1] - ia_area[0] if ia_area != [] else 0
    #
    #     if i == 0:
    #         ratio_before = 0
    #         ratio_after = 1
    #         if IA_pred_group_dict[area_id] != []:
    #             dl_score_ia_i_contribution = dl_score_ia_dict[area_id]
    #         else:
    #             if VA_pred_group_dict[area_id] != []:
    #                 dl_score_ia_i_contribution = 1 / 2
    #             else:
    #                 dl_score_ia_i_contribution = 0
    #     elif i == gt_num:
    #         ratio_before = 1
    #         ratio_after = 0
    #         if IA_pred_group_dict[area_id] != []:
    #             dl_score_ia_i_contribution = dl_score_ia_dict[area_id]
    #         else:
    #             if VA_pred_group_dict[area_id_front] != []:
    #                 dl_score_ia_i_contribution = 1 / 2
    #             else:
    #                 dl_score_ia_i_contribution = 0
    #     else:
    #         if IA_pred_group_dict[area_id] != []:
    #             ratio_before = 1 / 2
    #             ratio_after = 1 / 2
    #             # if d_ia_sign > 0:
    #             #     ratio_before = (ia_len / 2 - d_ia) / ia_len
    #             #     ratio_after = (ia_len / 2 + d_ia) / ia_len
    #             # elif d_ia_sign < 0:
    #             #     ratio_before = (ia_len / 2 + d_ia) / ia_len
    #             #     ratio_after = (ia_len / 2 - d_ia) / ia_len
    #             # else:
    #             #     ratio_before = 1/2
    #             #     ratio_after = 1/2
    #             dl_score_ia_i_contribution = dl_score_ia_dict[area_id]
    #         else:
    #             # ratio_before = 1 / 2
    #             # ratio_after = 1 / 2
    #             if VA_pred_group_dict[area_id_front] != [] and VA_pred_group_dict[area_id] != []:
    #                 ratio_before = 1/2
    #                 ratio_after = 1/2
    #                 dl_score_ia_i_contribution = 1
    #             elif VA_pred_group_dict[area_id_front] == [] and VA_pred_group_dict[area_id] == []:
    #                 ratio_before = 1/2
    #                 ratio_after = 1/2
    #                 dl_score_ia_i_contribution = 0
    #             else:
    #                 if VA_pred_group_dict[area_id_front] != []:
    #                     ratio_before = 1
    #                     ratio_after = 0
    #                 else:
    #                     ratio_before = 0
    #                     ratio_after = 1
    #                 dl_score_ia_i_contribution = one_exist_value

    # logic 3
    if logic == 3:
        # one_exist_value = 1/2
        one_exist_value = 1
        for i in range(gt_num + 1):
            area_id = "id_" + str(i)
            area_id_front = "id_" + str(i - 1)
            area_id_back = "id_" + str(i + 1)

            ia_area = IA_dict[area_id]
            l_ia = l_ia_set_dict[area_id]
            d_ia = d_ia_set_dict[area_id]
            d_ia_sign = d_ia_sign_set_dict[area_id]
            ia_len = ia_area[1] - ia_area[0] if ia_area != [] else 0

            if i == 0:
                ratio_before = 0
                ratio_after = 1
                if IA_dict[area_id] == []:
                    dl_score_ia_i_contribution = 0
                else:
                    if IA_pred_group_dict[area_id] != []:
                        dl_score_ia_i_contribution = dl_score_ia_dict[area_id]
                    else:
                        if VA_pred_group_dict[area_id] != []:
                            dl_score_ia_i_contribution = 1
                        else:
                            dl_score_ia_i_contribution = 0
            elif i == gt_num:
                ratio_before = 1
                ratio_after = 0
                if IA_dict[area_id] == []:
                    dl_score_ia_i_contribution = 0
                else:
                    if IA_pred_group_dict[area_id] != []:
                        dl_score_ia_i_contribution = dl_score_ia_dict[area_id]
                    else:
                        if VA_pred_group_dict[area_id_front] != []:
                            dl_score_ia_i_contribution = 1
                        else:
                            dl_score_ia_i_contribution = 0
            else:
                if IA_pred_group_dict[area_id] != []:
                    # ratio_before = 1 / 2
                    # ratio_after = 1 / 2
                    if d_ia_sign > 0:
                        ratio_before = (ia_len / 2 - d_ia) / ia_len
                        ratio_after = (ia_len / 2 + d_ia) / ia_len
                    elif d_ia_sign < 0:
                        ratio_before = (ia_len / 2 + d_ia) / ia_len
                        ratio_after = (ia_len / 2 - d_ia) / ia_len
                    else:
                        ratio_before = 1/2
                        ratio_after = 1/2
                    dl_score_ia_i_contribution = dl_score_ia_dict[area_id]
                else:
                    # ratio_before = 1 / 2
                    # ratio_after = 1 / 2
                    if VA_pred_group_dict[area_id_front] != [] and VA_pred_group_dict[area_id] != []:
                        ratio_before = 1/2
                        ratio_after = 1/2
                        dl_score_ia_i_contribution = 2
                    elif VA_pred_group_dict[area_id_front] == [] and VA_pred_group_dict[area_id] == []:
                        ratio_before = 1/2
                        ratio_after = 1/2
                        dl_score_ia_i_contribution = 0
                    else:
                        if VA_pred_group_dict[area_id_front] != []:
                            ratio_before = 1
                            ratio_after = 0
                        else:
                            ratio_before = 0
                            ratio_after = 1
                        dl_score_ia_i_contribution = one_exist_value

            adjusted_dl_score_ia_dict[area_id] = dl_score_ia_i_contribution
            dl_score_ia_i_before_contribution = dl_score_ia_i_contribution * ratio_before
            dl_score_ia_i_after_contribution = dl_score_ia_i_contribution * ratio_after
            adjusted_dl_score_ia_dict_ratio_contribution[area_id] = [dl_score_ia_i_before_contribution, dl_score_ia_i_after_contribution]


    if print_msg:
        print("adjusted_dl_score_ia_dict",adjusted_dl_score_ia_dict)
        print("adjusted_dl_score_ia_dict_ratio_contribution",adjusted_dl_score_ia_dict_ratio_contribution)
    event_recall_i_list = []
    event_recall_i_dict = {}
    event_recall_i_dict_w0 = {}
    event_recall_i_dict_w1 = {}
    event_precision_i_dict = {}
    event_precision_i_dict_w0 = {}
    event_precision_i_dict_w1 = {}
    for i in range(gt_num):
        area_id = "id_"+str(i)
        if print_msg:
            print(area_id)
        area_id_back = "id_"+str(i+1)
        recall_va_i = recall_va_dict[area_id]
        # if VA_pred_group_dict[area_id] != [] and IA_pred_group_dict[area_id] == [] and IA_pred_group_dict[area_id_back] == [] :
        #     dl_score_ia_i = 1
        #     dl_score_ia_i_back = 1
        # else:
        #     dl_score_ia_i = dl_score_ia_dict[area_id]
        #     dl_score_ia_i_back = dl_score_ia_dict[area_id_back]

        dl_score_ia_i_after_contribution = adjusted_dl_score_ia_dict_ratio_contribution[area_id][1]
        dl_score_ia_i_back_before_contribution = adjusted_dl_score_ia_dict_ratio_contribution[area_id_back][0]
        if print_msg:
            print(" dl_score_ia_i_after_contribution", dl_score_ia_i_after_contribution)
            print(" recall_va_i,", recall_va_i)
            print(" dl_score_ia_i_back_before_contribution", dl_score_ia_i_back_before_contribution)

        # event_recall_i = (1-2*parameter_eta)*recall_va_i + parameter_eta*(dl_score_ia_i_after_contribution+ dl_score_ia_i_back_before_contribution)
        if pred_case_id == 26:
            debug = 1

        # logic 1
        if logic ==1:
            event_recall_i = (1-parameter_eta)*recall_va_i + parameter_eta*(dl_score_ia_i_after_contribution+ dl_score_ia_i_back_before_contribution)
            # w=0
            event_recall_i_w0 = (1-0)*recall_va_i + 0*(dl_score_ia_i_after_contribution+ dl_score_ia_i_back_before_contribution)
            # w=1
            event_recall_i_w1 = (1-1)*recall_va_i + 1*(dl_score_ia_i_after_contribution+ dl_score_ia_i_back_before_contribution)
        # logic 3
        if logic ==3:
            if i == 0 and IA_dict[area_id] == []:
                event_recall_i = (1-parameter_eta)*recall_va_i + parameter_eta*1/1*(dl_score_ia_i_back_before_contribution)
            elif i == gt_num-1 and IA_dict[area_id_back] == []:
                event_recall_i = (1-parameter_eta)*recall_va_i + parameter_eta*1/1*(dl_score_ia_i_after_contribution)
            else:
                event_recall_i = (1-parameter_eta)*recall_va_i + parameter_eta*1/2*(dl_score_ia_i_after_contribution+ dl_score_ia_i_back_before_contribution)
        # event_recall_i_list
        event_recall_i_dict[area_id] = event_recall_i
        event_recall_i_dict_w0[area_id] = event_recall_i_w0
        event_recall_i_dict_w1[area_id] = event_recall_i_w1

        precision_va_i = precision_va_dict[area_id]
        if print_msg:
            print(" dl_score_ia_i_after_contribution", dl_score_ia_i_after_contribution)
            print(" precision_va_i,", precision_va_i)
            print(" dl_score_ia_i_back_before_contribution", dl_score_ia_i_back_before_contribution)

        # event_precision_i = (1-2*parameter_eta)*precision_va_i + parameter_eta*(dl_score_ia_i_after_contribution+ dl_score_ia_i_back_before_contribution)

        # logic 1
        if logic ==1:
           event_precision_i = (1-parameter_eta)*precision_va_i + parameter_eta*(dl_score_ia_i_after_contribution+ dl_score_ia_i_back_before_contribution)
           # w=0
           event_precision_i_w0 = (1-0)*precision_va_i + 0*(dl_score_ia_i_after_contribution+ dl_score_ia_i_back_before_contribution)
           # w=1
           event_precision_i_w1 = (1-1)*precision_va_i + 1*(dl_score_ia_i_after_contribution+ dl_score_ia_i_back_before_contribution)

        # logic 3
        if logic ==3:
            if i == 0 and IA_dict[area_id] == []:
                event_precision_i = (1-parameter_eta)*precision_va_i + parameter_eta*1/1*(dl_score_ia_i_back_before_contribution)
            elif i == gt_num - 1 and IA_dict[area_id_back] == []:
                event_precision_i = (1-parameter_eta)*precision_va_i + parameter_eta*1/1*(dl_score_ia_i_after_contribution)
            else:
                event_precision_i = (1-parameter_eta)*precision_va_i + parameter_eta*1/2*(dl_score_ia_i_after_contribution+ dl_score_ia_i_back_before_contribution)
        event_precision_i_dict[area_id] = event_precision_i
        event_precision_i_dict_w0[area_id] = event_precision_i_w0
        event_precision_i_dict_w1[area_id] = event_precision_i_w1

    if print_msg:
        print("event_recall_i_dict",event_recall_i_dict)
        print("event_precision_i_dict",event_precision_i_dict)

    if pred_case_id == 2:
        debug = 1
    event_recall = np.array(list(event_recall_i_dict.values())).mean()
    # print(event_recall)
    event_precision = np.array(list(event_precision_i_dict.values())).mean()
    # print(event_precision)
    
    event_recall_w0 = np.array(list(event_recall_i_dict_w0.values())).mean()
    event_precision_w0 = np.array(list(event_precision_i_dict_w0.values())).mean()
    event_recall_w1 = np.array(list(event_recall_i_dict_w1.values())).mean()
    event_precision_w1 = np.array(list(event_precision_i_dict_w1.values())).mean()

    #
    # print("======== meata_recall")
    # print(" event_recall",event_recall)
    # print(" event_recall_w0",event_recall_w0)
    # print(" event_recall_w1",event_recall_w1)
    #
    # print(" event_recall_i_dict",event_recall_i_dict)
    # print(" event_recall_i_dict_w0",event_recall_i_dict_w0)
    # print(" event_recall_i_dict_w1",event_recall_i_dict_w1)
    # print(" adjusted_dl_score_ia_dict_ratio_contribution",adjusted_dl_score_ia_dict_ratio_contribution)
    
    recall_info = {
    "event_recall": event_recall,
    "event_recall_w0": event_recall_w0,
    "event_recall_w1": event_recall_w1,
    "event_recall_i_dict": event_recall_i_dict,
    "event_recall_i_dict_w0": event_recall_i_dict_w0,
    "event_recall_i_dict_w1": event_recall_i_dict_w1,
    "adjusted_dl_score_ia_dict_ratio_contribution": adjusted_dl_score_ia_dict_ratio_contribution,
    }



    meata_f1 = compute_f1_score(event_precision, event_recall)
    meata_f1_w0 = compute_f1_score(event_precision_w0, event_recall_w0)
    meata_f1_w1 = compute_f1_score(event_precision_w1, event_recall_w1)

    gt_label_point_array = convert_interval_to_point_array(label_interval_ranges, time_series_length=window_length)
    ont_count = np.count_nonzero(gt_label_point_array == 1)
    anomaly_rate = ont_count/window_length
    if anomaly_rate > 0.2:
        print("anomaly_rate exceeds threshold")
        non_random_coefficient = 1
    else:
        non_random_coefficient = cal_no_random_measure_coefficient(pred_label_point_array,window_length)


    coefficient_meata_f1 = non_random_coefficient * meata_f1
    coefficient_event_recall = non_random_coefficient * event_recall
    coefficient_event_precision= non_random_coefficient * event_precision
    print("meata_f1",meata_f1)
    print("coefficient_meata_f1",coefficient_meata_f1)

    coefficient_meata_f1_w0 = non_random_coefficient * meata_f1_w0
    coefficient_meata_f1_w1 = non_random_coefficient * meata_f1_w1




    # return event_recall,event_precision,meata_f1,no_random_meata_f1
    # return event_recall,event_precision,meata_f1,coefficient_meata_f1,VA_f_dict,VA_d_dict,coefficient_event_recall,coefficient_event_precision
    return event_recall,event_precision,meata_f1,coefficient_meata_f1,VA_f_dict,VA_d_dict,\
            coefficient_event_recall,coefficient_event_precision,\
        meata_f1_w0,coefficient_meata_f1_w0,meata_f1_w1,coefficient_meata_f1_w1,\
        recall_info

# no random consecutive method
def meata_v1(label_interval_ranges, IA1_VA1_IA2_relative_list, window_length, parameter_dict, pred_label_point_array,
          output=None,pred_case_id=None, thresh_id=None,cal_mode="proportion"):
    # print_msg = False
    print_msg = True

    # parameter_alpha = parameter_dict["parameter_alpha"]
    parameter_rate = parameter_dict["parameter_rate"]

    parameter_theta = parameter_dict["parameter_theta"]
    # parameter_w = parameter_dict["parameter_w"]
    # parameter_switch_f =parameter_dict["parameter_switch_f"]
    # parameter_switch_d = parameter_dict["parameter_switch_d"]
    VA_f_window = parameter_dict["forecasting_window"]
    VA_d_window = parameter_dict["delay_window"]

    parameter_rho = parameter_dict["parameter_rho"]
    parameter_lamda = parameter_dict["parameter_lamda"]

    parameter_eta = parameter_dict["parameter_eta"]

    # no random
    no_random_coefficient = cal_no_random_measure_coefficient_method2(output,window_length)

    if print_msg:
        print("VA_f_window", VA_f_window)
        print("VA_d_window", VA_d_window)

    f_window_rate = 0
    d_window_rate = 0
    f_d_window_len = VA_f_window + VA_d_window
    if f_d_window_len != 0:
        f_window_rate = VA_f_window / f_d_window_len
        d_window_rate = 1 - f_window_rate

    IA_dict = {}
    VA_dict = {}
    VA_f_dict = {}
    VA_gt_dict = {}
    VA_d_dict = {}

    # pred generated is continuous range a,b(len) -> [a,a+b(len)),range of [a,b] -> [a,b) -> point of [a,b-1],point of [a,b] -> [a,b+1)

    # cal every f/d of gt,split IA and VA

    label_interval_num = len(label_interval_ranges)

    s_a_func_dict = {}

    # first circle
    for i, label_interval_range in enumerate(label_interval_ranges):
        # adjust now and before,add before
        VA_gt_now_start, VA_gt_now_end = label_interval_range
        # VA_f_now_start = max(VA_gt_now_start - VA_f_window,0) if parameter_switch_f == True else max(VA_gt_now_start - 3,0)
        VA_f_now_start = max(VA_gt_now_start - VA_f_window, 0)
        VA_f_now_end = VA_gt_now_start
        VA_d_now_start = VA_gt_now_end
        # VA_d_now_end = min(VA_gt_now_end + VA_f_window,window_length) if parameter_switch_d == True else min(VA_gt_now_end + 3,window_length)
        VA_d_now_end = min(VA_gt_now_end + VA_d_window, window_length)
        # # this way below is more fast
        # if i == 0:
        #     VA_f_now_start = max(VA_f_now_start,0)
        # if i == label_interval_num:
        #     VA_d_now_end = min(VA_d_now_end,window_length)
        if i > 0:
            # adjust now and before
            if VA_d_before_end > VA_f_now_start:
                intersection_range_start = max(VA_f_now_start, VA_gt_before_end)
                intersection_range_end = min(VA_d_before_end, VA_gt_now_start)
                intersection_range = [intersection_range_start, intersection_range_end]
                intersection_len = intersection_range_end - intersection_range_start
                #         if intersection_len == 1 and (parameter_switch_f == True and parameter_switch_d == True) and parameter_rho == 1/2:
                # #             f prior
                #             VA_d_before_end -=1
                #         else:
                #         VA_d_before_end = VA_f_now_start = intersection_range_start + round(parameter_w*intersection_len)
                #         VA_d_before_end = VA_f_now_start = intersection_range_start + round(parameter_w*intersection_len,2)
                #         VA_f_now_start = intersection_range_start + round(f_window_rate*intersection_len,2)
                #         VA_d_before_end = intersection_range_start + round(d_window_rate*intersection_len,2)
                VA_f_now_start = intersection_range_start + round(f_window_rate * intersection_len)
                VA_d_before_end = intersection_range_start + round(d_window_rate * intersection_len)
            #     add before
            IA_dict["id_" + str(i)] = [VA_d_before_end, VA_f_now_start]
            VA_dict["id_" + str(i - 1)] = [VA_f_before_start, VA_d_before_end]
            VA_f_dict["id_" + str(i - 1)] = [VA_f_before_start, VA_f_before_end]
            VA_gt_dict["id_" + str(i - 1)] = [VA_gt_before_start, VA_gt_before_end]
            VA_d_dict["id_" + str(i - 1)] = [VA_d_before_start, VA_d_before_end]

            # add by order

            split_line_set.add(VA_f_before_start)
            split_line_set.add(VA_f_before_end)

            split_line_set.add(VA_gt_before_start)
            split_line_set.add(VA_gt_before_end)

            split_line_set.add(VA_d_before_start)
            split_line_set.add(VA_d_before_end)

            split_line_set.add(VA_d_before_end)
            split_line_set.add(VA_f_now_start)

            VA_gt_before_len = VA_gt_before_end - VA_gt_before_start
            x1, x2 = cal_x1_x2(VA_gt_before_len, parameter_rho)

            # # print("area_id,area",area_id,area)
            # # # print("x1,x2",x1, x2)
            # # print("x1_ori,x2_ori",x1+VA_gt_before_start[0], x2+VA_gt_before_start[0])
            x1_ori, x2_ori = x1 + VA_gt_before_start, x2 + VA_gt_before_start
            x_mid_ori = 2 * x1 + VA_gt_before_start
            split_line_set.add(x1_ori)
            split_line_set.add(x_mid_ori)
            split_line_set.add(x2_ori)
        else:
            IA_dict["id_" + str(i)] = [0, VA_f_now_start]
            split_line_set.add(VA_f_now_start)

        VA_gt_before_start, VA_gt_before_end = VA_gt_now_start, VA_gt_now_end
        VA_f_before_start = VA_f_now_start
        VA_f_before_end = VA_f_now_end
        VA_d_before_start = VA_d_now_start
        VA_d_before_end = VA_d_now_end

    # last one VA and last two IA
    IA_dict["id_" + str(label_interval_num)] = [VA_d_before_end, window_length]
    VA_dict["id_" + str(label_interval_num - 1)] = [VA_f_before_start, VA_d_before_end]
    VA_f_dict["id_" + str(label_interval_num - 1)] = [VA_f_before_start, VA_f_before_end]
    VA_gt_dict["id_" + str(label_interval_num - 1)] = [VA_gt_before_start, VA_gt_before_end]
    VA_d_dict["id_" + str(label_interval_num - 1)] = [VA_d_before_start, VA_d_before_end]

    VA_gt_before_len = VA_gt_before_end - VA_gt_before_start
    x1, x2 = cal_x1_x2(VA_gt_before_len, parameter_rho)

    # # print("area_id,area",area_id,area)
    # # # print("x1,x2",x1, x2)
    # # print("x1_ori,x2_ori",x1+VA_gt_before_start[0], x2+VA_gt_before_start[0])
    x1_ori, x2_ori = x1 + VA_gt_before_start, x2 + VA_gt_before_start
    x_mid_ori = 2 * x1 + VA_gt_before_start
    split_line_set.add(x1_ori)
    split_line_set.add(x_mid_ori)
    split_line_set.add(x2_ori)

    # add by order

    split_line_set.add(VA_f_before_start)
    split_line_set.add(VA_f_before_end)

    split_line_set.add(VA_gt_before_start)
    split_line_set.add(VA_gt_before_end)

    split_line_set.add(VA_d_before_start)
    split_line_set.add(VA_d_before_end)

    if print_msg:
        # print area dict
        # # print("split area")
        print("IA_dict", IA_dict)
        # print("VA_dict",VA_dict)
        print("VA_f_dict", VA_f_dict)
        print("VA_gt_dict", VA_gt_dict)
        print("VA_d_dict", VA_d_dict)

    def dict_filer(dict_data: dict):
        for i, (area_id, area_range) in enumerate(dict_data.items()):
            if area_range != []:
                dict_data[area_id] = [] if area_range[0] == area_range[1] else area_range
        return dict_data

    # second circle
    IA_dict = dict_filer(IA_dict)
    VA_dict = dict_filer(VA_dict)
    VA_f_dict = dict_filer(VA_f_dict)
    VA_gt_dict = dict_filer(VA_gt_dict)
    VA_d_dict = dict_filer(VA_d_dict)

    if print_msg:
        print("dict adjust +++++++++++")
        print("IA_dict", IA_dict)
        # print("VA_dict",VA_dict)
        print("VA_f_dict", VA_f_dict)
        print("VA_gt_dict", VA_gt_dict)
        print("VA_d_dict", VA_d_dict)

        # split pred to basic interval
        print("split_line_set", split_line_set)

    # third circle
    split_IA1_VA1_IA2_relative_list = split_intervals(IA1_VA1_IA2_relative_list, split_line_set)
    # print("split_IA1_VA1_IA2_relative_list",split_IA1_VA1_IA2_relative_list)

    # forecasting_len=10
    # delay_len=10
    IA_pred_group_dict = {}
    VA_pred_group_dict = {}
    VA_f_pred_group_dict = {}
    VA_gt_pred_group_dict = {}
    VA_d_pred_group_dict = {}

    VA_gt1_pred_group_dict = {}
    VA_gt2_pred_group_dict = {}
    VA_gt3_pred_group_dict = {}
    VA_gt4_pred_group_dict = {}
    func_dict = {}

    # "id_"+str(label_interval_num)
    for i in range(label_interval_num):
        IA_pred_group_dict["id_" + str(i)] = []
        VA_pred_group_dict["id_" + str(i)] = []
        VA_f_pred_group_dict["id_" + str(i)] = []
        VA_gt_pred_group_dict["id_" + str(i)] = []
        VA_d_pred_group_dict["id_" + str(i)] = []
        VA_gt1_pred_group_dict["id_" + str(i)] = []
        VA_gt2_pred_group_dict["id_" + str(i)] = []
        VA_gt3_pred_group_dict["id_" + str(i)] = []
        VA_gt4_pred_group_dict["id_" + str(i)] = []
    IA_pred_group_dict["id_" + str(label_interval_num)] = []

    # 4th circle
    # for i, basic_interval in enumerate(IA1_VA1_IA2_relative_list):
    for i, basic_interval in enumerate(split_IA1_VA1_IA2_relative_list):
        # IA
        for j, (area_id, area) in enumerate(IA_dict.items()):
            if pred_in_area(basic_interval, area):
                if area_id not in IA_pred_group_dict.keys():
                    IA_pred_group_dict[area_id] = [basic_interval]
                else:
                    IA_pred_group_dict[area_id].append(basic_interval)

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

            # group by x1,x2
            VA_gt_len = area[1] - area[0]
            x1, x2 = cal_x1_x2(VA_gt_len, parameter_rho)

            # # print("area_id,area",area_id,area)
            # # # print("x1,x2",x1, x2)
            # # print("x1_ori,x2_ori",x1+area[0], x2+area[0])
            area_gt1 = [0, x1]
            area_gt2 = [x1, 2 * x1]
            area_gt3 = [2 * x1, x2]
            area_gt4 = [x2, area[1] - area[0]]
            area_gt1_ori = [area[0], x1 + area[0]]
            area_gt2_ori = [x1 + area[0], 2 * x1 + area[0]]
            area_gt3_ori = [2 * x1 + area[0], x2 + area[0]]
            area_gt4_ori = [x2 + area[0], area[1]]
            # # print("area_gt1",area_gt1)
            # # print("area_gt2",area_gt2)
            # # print("area_gt3",area_gt3)
            # # print("area_gt4",area_gt4)

            # # print("area_gt1_ori",area_gt1_ori)
            # # print("area_gt2_ori",area_gt2_ori)
            # # print("area_gt3_ori",area_gt3_ori)
            # # print("area_gt4_ori",area_gt4_ori)

            if pred_in_area(basic_interval, area_gt1_ori):
                # if area_id not in VA_gt1_pred_group_dict.keys():
                #     VA_gt1_pred_group_dict[area_id]=[basic_interval]
                # else:
                #     VA_gt1_pred_group_dict[area_id].append(basic_interval)
                VA_gt1_pred_group_dict[area_id].append(basic_interval)
            # if basic_interval == [135, 141] and area == [130, 150]:
            #     a =1
            if pred_in_area(basic_interval, area_gt2_ori):
                # if area_id not in VA_gt2_pred_group_dict.keys():
                #     VA_gt2_pred_group_dict[area_id]=[basic_interval]
                # else:
                #     VA_gt2_pred_group_dict[area_id].append(basic_interval)
                VA_gt2_pred_group_dict[area_id].append(basic_interval)
            if pred_in_area(basic_interval, area_gt3_ori):
                # if area_id not in VA_gt3_pred_group_dict.keys():
                #     VA_gt3_pred_group_dict[area_id]=[basic_interval]
                # else:
                VA_gt3_pred_group_dict[area_id].append(basic_interval)
            if pred_in_area(basic_interval, area_gt4_ori):
                # if area_id not in VA_gt4_pred_group_dict.keys():
                #     VA_gt4_pred_group_dict[area_id]=[basic_interval]
                # else:
                VA_gt4_pred_group_dict[area_id].append(basic_interval)

        # VA-d
        for j, (area_id, area) in enumerate(VA_d_dict.items()):
            if pred_in_area(basic_interval, area):
                if area_id not in VA_d_pred_group_dict.keys():
                    VA_d_pred_group_dict[area_id] = [basic_interval]
                else:
                    VA_d_pred_group_dict[area_id].append(basic_interval)
        # VA
        for j, (area_id, area) in enumerate(VA_dict.items()):
            if pred_in_area(basic_interval, area):
                if area_id not in VA_pred_group_dict.keys():
                    VA_pred_group_dict[area_id] = [basic_interval]
                else:
                    VA_pred_group_dict[area_id].append(basic_interval)
    if print_msg:
        # print pred dict
        print("IA_pred_group_dict", IA_pred_group_dict)
        print("VA_f_pred_group_dict", VA_f_pred_group_dict)
        # print("VA_gt_pred_group_dict",VA_gt_pred_group_dict)
        print("VA_gt1_pred_group_dict", VA_gt1_pred_group_dict)
        print("VA_gt2_pred_group_dict", VA_gt2_pred_group_dict)
        print("VA_gt3_pred_group_dict", VA_gt3_pred_group_dict)
        print("VA_gt4_pred_group_dict", VA_gt4_pred_group_dict)
        print("VA_d_pred_group_dict", VA_d_pred_group_dict)
        # print("VA_pred_group_dict",VA_pred_group_dict)

    d_ia_set_dict = {}
    d_ia_sign_set_dict = {}
    l_ia_set_dict = {}
    dl_score_ia_dict = {}

    dl_part_ia_score_dict = {}

    # 5th circle
    for i, (area_id, basic_interval_list) in enumerate(IA_pred_group_dict.items()):
        if basic_interval_list == []:
            dl_score_ia_dict[area_id] = 0
            l_ia_set_dict[area_id] = 0
            d_ia_set_dict[
                area_id] = 0  # only when basic_interval_list ！= [],the value are used,so set value to 0 have boe problem
            d_ia_sign_set_dict[
                area_id] = 0  # only when basic_interval_list ！= [],the value are used,so set value to 0 have boe problem
            continue
        l_ia = 0
        l_ia_set_sum = 0
        d_ia_set_sum = 0
        d_ia_sign_set_sum = 0
        ia_set_num = len(basic_interval_list)
        IA_mid = (IA_dict[area_id][0] + IA_dict[area_id][1]) / 2
        IA_len = IA_dict[area_id][1] - IA_dict[area_id][0]

        test_group = "group"
        # test_group = "no group"
        if test_group == "group":
            for basic_interval in basic_interval_list:
                l_ia += basic_interval[1] - basic_interval[0]
                # the length of fitst and last IA could be set a fixed value,min(一个周期的长度,ori len),or min(mean len,ori len)
                if i == 0:
                    base_position = IA_dict[area_id][1]
                    # d_ia_set_sum+= abs((basic_interval[1] + basic_interval[0])/2 -base_position)
                # last id of IA is label_interval_num,not label_interval_num -1
                elif i == label_interval_num:
                    base_position = IA_dict[area_id][0]
                else:
                    base_position = IA_mid
                # d_ia_set_sum+= abs((basic_interval[1] + basic_interval[0])/2 -IA_mid)
                d_ia_set_sum += abs((basic_interval[1] + basic_interval[0]) / 2 - base_position)
                # i =0 or i = label_interval_num,only have positive distance

                d_ia_sign_set_sum += (basic_interval[1] + basic_interval[0]) / 2 - base_position

            l_ia_set_dict[area_id] = l_ia
            d_ia = d_ia_set_sum / ia_set_num if ia_set_num != 0 else 0
            d_ia_set_dict[area_id] = d_ia
            d_ia_sign = d_ia_sign_set_sum / ia_set_num if ia_set_num != 0 else 0
            d_ia_sign_set_dict[area_id] = d_ia_sign

            dl_score_ia = 0
            #
            if l_ia != 0:
                if i == 0:
                    # 一边情况下就没有1了
                    # parameter_alpha1 = parameter_rate/(2+parameter_rate)
                    #
                    # debug1 = (1/2*IA_len-d_ia)
                    # debug2 = 2 / IA_len * (1/2*IA_len-d_ia)
                    # debug3 = parameter_alpha1 * 2 / IA_len * (1/2*IA_len-d_ia)
                    #
                    # debug4 = (1 - l_ia / IA_len)
                    # debug5 = (1 - parameter_alpha1) * (1 - l_ia / IA_len)

                    # dl_score_ia = parameter_alpha1 * 2 / IA_len * (1/2*IA_len-d_ia) + (1 - parameter_alpha1) * (1 - l_ia / IA_len) \
                    #     if IA_len != 0 else 0

                    parameter_alpha = parameter_rate / (1 + parameter_rate)
                    debug11 = (IA_len - d_ia)
                    debug21 = 1 / IA_len * (IA_len - d_ia)
                    debug31 = parameter_alpha / IA_len * (IA_len - d_ia)

                    debug41 = (1 - l_ia / IA_len)
                    debug51 = (1 - parameter_alpha) * (1 - l_ia / IA_len)
                    # 得分跟系数也有关系

                    dl_score_ia1 = parameter_alpha / IA_len * (IA_len - d_ia) + (1 - parameter_alpha) * (
                            1 - l_ia / IA_len) \
                        if IA_len != 0 else 0
                    # 对抗边界是2
                    if parameter_rate <= 2:
                        min_value = parameter_alpha / IA_len * (IA_len - 1 / 2 * IA_len)
                        # min_value = min_value + 0.0000001
                        dl_score_ia = map_value(dl_score_ia1, min_value, 1, 1e-10, 1)
                    else:
                        min_value = (1 - parameter_alpha) * (1 - 0 / IA_len)
                        dl_score_ia = map_value(dl_score_ia1, min_value, 1, 1e-10, 1)

                        # dl_score_ia = dl_score_ia1
                    # print("0 dl_score_ia",dl_score_ia)
                    # dl_part_ia_score_dict[area_id] = {
                    #     "parameter_alpha":parameter_alpha,
                    #     "d_score":1/IA_len * (IA_len - d_ia),
                    #     "(1 - parameter_alpha)":(1 - parameter_alpha),
                    #     "d_score_parameter":(1 - parameter_alpha) * (
                    #                 1 - l_ia / IA_len)
                    # }
                    if pred_case_id == 2:
                        debug = 1
                # last id of IA is label_interval_num,not label_interval_num -1
                elif i == label_interval_num:

                    # debug1 = (1 / 2 * IA_len - d_ia)
                    # debug2 = 2 / IA_len * (1 / 2 * IA_len - d_ia)
                    # debug3 = parameter_alpha * 2 / IA_len * (1 / 2 * IA_len - d_ia)
                    #
                    # debug4 = (1 - l_ia / IA_len)
                    # debug5 = (1 - parameter_alpha) * (1 - l_ia / IA_len)

                    # dl_score_ia = parameter_alpha * 2 / IA_len * (1/2*IA_len-d_ia) + (1 - parameter_alpha) * (1 - l_ia / IA_len) \
                    #     if IA_len != 0 else 0

                    parameter_alpha = parameter_rate / (1 + parameter_rate)
                    debug11 = (IA_len - d_ia)
                    debug21 = 1 / IA_len * (IA_len - d_ia)
                    debug31 = parameter_alpha / IA_len * (IA_len - d_ia)

                    debug41 = (1 - l_ia / IA_len)
                    debug51 = (1 - parameter_alpha) * (1 - l_ia / IA_len)
                    dl_score_ia1 = parameter_alpha / IA_len * (IA_len - d_ia) + (1 - parameter_alpha) * (
                                1 - l_ia / IA_len) \
                        if IA_len != 0 else 0
                    # x, a, b, c, d
                    if parameter_rate <= 2:
                        min_value = parameter_alpha / IA_len * (IA_len - 1 / 2 * IA_len)
                        # min_value = min_value + 0.0000001
                        dl_score_ia = map_value(dl_score_ia1, min_value, 1, 1e-10, 1)
                    else:
                        min_value = (1 - parameter_alpha) * (1 - 0 / IA_len)
                        dl_score_ia = map_value(dl_score_ia1, min_value, 1, 1e-10, 1)
                        # dl_score_ia = map_value(dl_score_ia,parameter_alpha,1,0,1)
                    # dl_part_ia_score_dict[area_id] = {
                    #     "parameter_alpha": parameter_alpha,
                    #     "d_score": 1 / IA_len * (IA_len - d_ia),
                    #     "(1 - parameter_alpha)": (1 - parameter_alpha),
                    #     "d_score_parameter": (1 - parameter_alpha) * (
                    #             1 - l_ia / IA_len)
                    # }
                    if pred_case_id == 2:
                        debug = 1
                else:

                    parameter_alpha = parameter_rate / (2 + parameter_rate)
                    dl_score_ia = parameter_alpha * 2 / IA_len * d_ia + (1 - parameter_alpha) * (1 - l_ia / IA_len) \
                        if IA_len != 0 else 0
                    if pred_case_id == 2:
                        print("  parameter_alpha", parameter_alpha)
                        print("  d_ia", d_ia)
                        print("  d_score", 2 / IA_len * d_ia)
                        print("  d_score_parameter", parameter_alpha * 2 / IA_len * d_ia)
                        print("  (1 - parameter_alpha)", (1 - parameter_alpha))
                        print("  l_ia", l_ia)
                        print("  l_score", (1 - l_ia / IA_len))
                        print("  l_score_parameter", (1 - parameter_alpha) * (1 - l_ia / IA_len))
                        print("  IA_len", IA_len)
                        debug = 1
                dl_part_ia_score_dict
        else:
            if print_msg:
                print("basic_interval_list num", len(basic_interval_list))
            dl_score_ia_sum = 0
            for basic_interval in basic_interval_list:
                if i == 0:
                    base_position = IA_dict[area_id][1]
                elif i == label_interval_num:
                    base_position = IA_dict[area_id][0]
                else:
                    base_position = IA_mid

                l_ia = basic_interval[1] - basic_interval[0]
                d_ia = abs((basic_interval[1] + basic_interval[0]) / 2 - base_position)
                if pred_case_id == 2:
                    debug = 1
                dl_score_ia = 0
                #
                if l_ia != 0:
                    if i == 0:
                        parameter_alpha = parameter_rate / (1 + parameter_rate)
                        dl_score_ia1 = parameter_alpha / IA_len * (IA_len - d_ia) + (1 - parameter_alpha) * (
                                1 - l_ia / IA_len) \
                            if IA_len != 0 else 0
                        # 对抗边界是2
                        if parameter_rate <= 2:
                            min_value = parameter_alpha / IA_len * (IA_len - 1 / 2 * IA_len)
                            dl_score_ia = map_value(dl_score_ia1, min_value, 1, 1e-10, 1)
                        else:
                            min_value = (1 - parameter_alpha) * (1 - 0 / IA_len)
                            dl_score_ia = map_value(dl_score_ia1, min_value, 1, 1e-10, 1)
                        if print_msg:
                            print("0 dl_score_ia", dl_score_ia)
                        if pred_case_id == 2:
                            debug = 1
                    elif i == label_interval_num:
                        parameter_alpha = parameter_rate / (1 + parameter_rate)
                        debug11 = (IA_len - d_ia)
                        debug21 = 1 / IA_len * (IA_len - d_ia)
                        debug31 = parameter_alpha / IA_len * (IA_len - d_ia)

                        debug41 = (1 - l_ia / IA_len)
                        debug51 = (1 - parameter_alpha) * (1 - l_ia / IA_len)
                        dl_score_ia1 = parameter_alpha / IA_len * (IA_len - d_ia) + (1 - parameter_alpha) * (
                                1 - l_ia / IA_len) \
                            if IA_len != 0 else 0
                        # x, a, b, c, d
                        if parameter_rate <= 2:
                            min_value = parameter_alpha / IA_len * (IA_len - 1 / 2 * IA_len)
                            # min_value = min_value + 0.0000001
                            dl_score_ia = map_value(dl_score_ia1, min_value, 1, 1e-10, 1)
                        else:
                            min_value = (1 - parameter_alpha) * (1 - 0 / IA_len)
                            dl_score_ia = map_value(dl_score_ia1, min_value, 1, 1e-10, 1)
                            # dl_score_ia = map_value(dl_score_ia,parameter_alpha,1,0,1)
                        if pred_case_id == 2:
                            debug = 1
                    else:

                        parameter_alpha = parameter_rate / (2 + parameter_rate)
                        dl_score_ia = parameter_alpha * 2 / IA_len * d_ia + (1 - parameter_alpha) * (1 - l_ia / IA_len) \
                            if IA_len != 0 else 0
                        if pred_case_id == 2:
                            if print_msg:
                                print("  parameter_alpha", parameter_alpha)
                                print("  d_ia", d_ia)
                                print("  d_score", 2 / IA_len * d_ia)
                                print("  d_score_parameter", parameter_alpha * 2 / IA_len * d_ia)
                                print("  (1 - parameter_alpha)", (1 - parameter_alpha))
                                print("  l_ia", l_ia)
                                print("  l_score", (1 - l_ia / IA_len))
                                print("  l_score_parameter", (1 - parameter_alpha) * (1 - l_ia / IA_len))
                                print("  IA_len", IA_len)
                                debug = 1

                # sum
                dl_score_ia_sum += dl_score_ia
                l_ia_set_sum += basic_interval[1] - basic_interval[0]
                d_ia_set_sum += abs((basic_interval[1] + basic_interval[0]) / 2 - base_position)
                d_ia_sign_set_sum += (basic_interval[1] + basic_interval[0]) / 2 - base_position

            # sum
            # test_no_group_type = "sum"
            test_no_group_type = "average"
            if test_no_group_type == "sum":
                dl_score_ia = dl_score_ia_sum
                l_ia_set_dict[area_id] = l_ia_set_sum
                d_ia_set_dict[area_id] = d_ia_set_sum
                d_ia_sign_set_dict[area_id] = d_ia_sign_set_sum
            else:
                dl_score_ia = dl_score_ia_sum / len(basic_interval_list) if len(basic_interval_list) != 0 else 0
                l_ia_set_dict[area_id] = l_ia_set_sum / len(basic_interval_list) if len(basic_interval_list) != 0 else 0
                d_ia_set_dict[area_id] = d_ia_set_sum / len(basic_interval_list) if len(basic_interval_list) != 0 else 0
                d_ia_sign_set_dict[area_id] = d_ia_sign_set_sum
        dl_score_ia_dict[area_id] = dl_score_ia
    #     deal empty and no area
    if print_msg:
        print("d_ia_set_dict", d_ia_set_dict)
        print("d_ia_sign_set_dict", d_ia_sign_set_dict)
        print("l_ia_set_dict", l_ia_set_dict)
    # print("dl_score_ia_dict",dl_score_ia_dict)

    # cal label_function_gt
    # 6th circle
    for i, (area_id, area) in enumerate(VA_gt_dict.items()):

        VA_gt_len = area[1] - area[0]
        x1, x2 = cal_x1_x2(VA_gt_len, parameter_rho)
        x1_ori, x2_ori = x1 + area[0], x2 + area[0]
        x_mid = 2 * x1
        x_mid_ori = x_mid + area[0]
        # x1_ori,x2_ori = x1+area[0], x2+area[0]
        area_gt1 = [0, x1]
        area_gt2 = [x1, 2 * x1]
        area_gt3 = [2 * x1, x2]
        area_gt4 = [x2, VA_gt_len]
        area_gt1_ori = [area[0], x1 + area[0]]
        area_gt2_ori = [x1 + area[0], 2 * x1 + area[0]]
        area_gt3_ori = [2 * x1 + area[0], x2 + area[0]]
        area_gt4_ori = [x2 + area[0], area[1]]
        if print_msg:
            print("VA_gt_len", VA_gt_len)

        if parameter_rho < 0:
            parameter_b1, parameter_gama1 = 1 / 2, 0
            parameter_b2, parameter_gama2 = 1 / 2, 0
        else:
            parameter_b1, parameter_gama1 = cal_power_function_coefficient(x1, parameter_lamda)
            parameter_b2, parameter_gama2 = cal_power_function_coefficient(x2 - x_mid, parameter_lamda)

        func_dict.update({
            area_id + "_gt1": {"area": area_gt1_ori,
                               "area_start": area_gt1_ori[0],
                               "parameter_a": parameter_b1,
                               "parameter_beta": parameter_gama1,
                               "func": "va_gt1_func",
                               "x_tp": x1,
                               "gt_len": VA_gt_len,
                               "area_len": None,
                               "parameter_rho": parameter_rho
                               },
            area_id + "_gt2": {"area": area_gt2_ori,
                               "area_start": area_gt1_ori[0],
                               "parameter_a": parameter_b1,
                               "parameter_beta": parameter_gama1,
                               "func": "va_gt2_func",
                               "x_tp": x1,
                               "gt_len": VA_gt_len,
                               "area_len": None,
                               "parameter_rho": parameter_rho
                               },
            area_id + "_gt3": {"area": area_gt3_ori,
                               "area_start": area_gt1_ori[0],
                               "parameter_a": parameter_b2,
                               "parameter_beta": parameter_gama2,
                               "func": "va_gt3_func",
                               "x_tp": x2,
                               "gt_len": VA_gt_len,
                               "area_len": None,
                               "parameter_rho": parameter_rho
                               },
            area_id + "_gt4": {"area": area_gt4_ori,
                               "area_start": area_gt1_ori[0],
                               "parameter_a": parameter_b2,
                               "parameter_beta": parameter_gama2,
                               "func": "va_gt4_func",
                               "x_tp": x2,
                               "gt_len": VA_gt_len,
                               "area_len": None,
                               "parameter_rho": parameter_rho
                               },

        })

        s_a_gt1 = cal_integral_in_range_gt_power_func1(area_gt1_ori, parameter_b1, parameter_gama1, x_tp=x1,
                                                       gt_len=VA_gt_len, area_start=area_gt1_ori[0],
                                                       parameter_rho=parameter_rho)
        s_a_gt2 = cal_integral_in_range_gt_power_func2(area_gt2_ori, parameter_b1, parameter_gama1, x_tp=x1,
                                                       gt_len=VA_gt_len, area_start=area_gt1_ori[0],
                                                       parameter_rho=parameter_rho)
        s_a_gt3 = cal_integral_in_range_gt_power_func3(area_gt3_ori, parameter_b2, parameter_gama2, x_tp=x2,
                                                       gt_len=VA_gt_len, area_start=area_gt1_ori[0],
                                                       parameter_rho=parameter_rho)
        s_a_gt4 = cal_integral_in_range_gt_power_func4(area_gt4_ori, parameter_b2, parameter_gama2, x_tp=x2,
                                                       gt_len=VA_gt_len, area_start=area_gt1_ori[0],
                                                       parameter_rho=parameter_rho)

        s_a_gt = s_a_gt1 + s_a_gt2 + s_a_gt3 + s_a_gt4

        func_dict[area_id + "_gt1"]["area_s"] = s_a_gt1
        func_dict[area_id + "_gt2"]["area_s"] = s_a_gt2
        func_dict[area_id + "_gt3"]["area_s"] = s_a_gt3
        func_dict[area_id + "_gt4"]["area_s"] = s_a_gt4

        method = 2

        va_f_area = VA_f_dict[area_id]
        s_a_f = 0
        if va_f_area == []:
            pass
        else:
            VA_f_len = va_f_area[1] - va_f_area[0]
            parameter_beta1 = 1
            parameter_a1 = 1 / (2 * VA_f_len)

            greater_than_th = False

            func_dict[area_id + "_f"] = {"area": va_f_area,
                                         "area_start": va_f_area[0],
                                         "parameter_a": parameter_a1,
                                         "parameter_beta": parameter_beta1,
                                         "func": "va_f_func",
                                         "x_tp": None,
                                         "gt_len": None,
                                         "area_len": VA_f_len,
                                         "greater_than_th": greater_than_th
                                         }

            # f
            s_max_f = cal_integral_in_range_f_power_func(va_f_area, parameter_a1, parameter_beta1, VA_f_len, VA_f_len,
                                                         area_start=va_f_area[0], greater_than_th=greater_than_th)
            func_dict[area_id + "_f"]["area_s"] = s_max_f

            s_rate = s_max_f / s_a_gt
            func_dict[area_id + "_f_a"] = func_dict[area_id + "_f"]
            s_a_f = s_max_f
            if s_rate > parameter_theta:
                greater_than_th = True

                # method1
                if method == 1:
                    parameter_beta1_a = VA_f_len / (2 * parameter_theta * s_a_gt) - 1
                    parameter_a1_a = 1 / (2 * pow(VA_f_len, parameter_beta1_a))
                # method2
                else:
                    parameter_beta1_a = VA_f_len / (2 * parameter_theta * s_a_gt) - 1  # same with method1
                    parameter_a1_a = VA_f_len / pow(1 / 2, parameter_beta1_a)

                s_a_f = cal_integral_in_range_f_power_func(va_f_area, parameter_a1_a, parameter_beta1_a, VA_f_len,
                                                           VA_f_len, area_start=va_f_area[0], method=method,
                                                           greater_than_th=greater_than_th)
                func_dict[area_id + "_f_a"] = {
                    "area": va_f_area,
                    "area_start": va_f_area[0],
                    "area_s": s_a_f,
                    "parameter_a": parameter_a1_a,
                    "parameter_beta": parameter_beta1_a,
                    # "func": "va_f_func",
                    # "func": "va_f_reverse_func",
                    "x_tp": None,
                    "gt_len": None,
                    "area_len": VA_f_len,
                    "greater_than_th": greater_than_th
                }
                if method == 1:
                    func_dict[area_id + "_f_a"]["func"] = "va_f_func"
                else:
                    func_dict[area_id + "_f_a"]["func"] = "va_f_reverse_func"

        # d
        va_d_area = VA_d_dict[area_id]
        s_a_d = 0
        if va_d_area == []:
            pass
        else:

            if pred_case_id == 7 and area_id == "id_2":
                debug = 1

            VA_d_len = va_d_area[1] - va_d_area[0]
            parameter_beta2 = 1
            parameter_a2 = 1 / (2 * VA_d_len)

            greater_than_th = False
            s_max_d = cal_integral_in_range_d_power_func(va_d_area, parameter_a2, parameter_beta2, VA_d_len, VA_d_len,
                                                         area_start=va_d_area[0], greater_than_th=greater_than_th)
            func_dict[area_id + "_d"] = {
                "area": va_d_area,
                "area_start": va_d_area[0],
                "parameter_a": parameter_a2,
                "parameter_beta": parameter_beta2,
                "func": "va_d_func",
                "x_tp": None,
                "gt_len": None,
                "area_len": VA_d_len,
                "greater_than_th": greater_than_th
            }

            func_dict[area_id + "_d"]["area_s"] = s_max_d

            func_dict[area_id + "_d_a"] = func_dict[area_id + "_d"]
            s_rate = s_max_d / s_a_gt
            s_a_d = s_max_d

            if s_rate > parameter_theta:
                greater_than_th = True

                # method1
                if method == 1:
                    parameter_beta2_a = VA_d_len / (2 * parameter_theta * s_a_gt) - 1
                    parameter_a2_a = 1 / (2 * pow(VA_d_len, parameter_beta2_a))
                else:
                    # method2
                    parameter_beta2_a = VA_d_len / (2 * parameter_theta * s_a_gt) - 1  # same with method1
                    parameter_a2_a = VA_d_len / pow(1 / 2, parameter_beta2_a)

                s_a_d = cal_integral_in_range_d_power_func(va_d_area, parameter_a2_a, parameter_beta2_a, VA_d_len,
                                                           VA_d_len,
                                                           area_start=va_d_area[0], method=method,
                                                           greater_than_th=greater_than_th)
                func_dict[area_id + "_d_a"] = {
                    "area": va_d_area,
                    "area_start": va_d_area[0],
                    "area_s": s_a_d,
                    "parameter_a": parameter_a2_a,
                    "parameter_beta": parameter_beta2_a,
                    # "func": "va_d_func",
                    # "func": "va_d_reverse_func",
                    "x_tp": None,
                    "gt_len": None,
                    "area_len": VA_d_len,
                    "greater_than_th": greater_than_th

                }

                if method == 1:
                    func_dict[area_id + "_d_a"]["func"] = "va_d_func"
                else:
                    func_dict[area_id + "_d_a"]["func"] = "va_d_reverse_func"

        s_a_func_dict[area_id] = {
            "s_a_va_gt_len": VA_gt_len,
            "s_a_va_gt": s_a_gt,
            "s_a_va": s_a_f + s_a_gt + s_a_d,
            "x1": x1,
            "x2": x2,
            "x_mid": x_mid,
            "x1_ori": x1_ori,
            "x2_ori": x2_ori,
            "x_mid_ori": x_mid_ori,
        }
    # plot func
    if pred_case_id == 7 and area_id == "id_2":
        debug = 1
        # plot_func_multi(func_dict)
    # plot_func_multi(func_dict)
    # plot_func_multi_paper(func_dict,window_length)

    import pprint

    # print("s_a_func_dict")
    # pprint(s_a_func_dict, indent=4)

    # cal recall_va,detection_score
    recall_va_list = []
    recall_va_dict = {}
    detection_score_ia_i_dict = {}
    precision_va_dict = {}

    gt_num = label_interval_num
    pred_func_dict = {}
    basic_interval_integral_dict = {}

    # 7th circle
    for i in range(gt_num):
        area_id = "id_" + str(i)
        # print("area_id",area_id)
        area_id_back = "id_" + str(i + 1)
        VA_f_i_pred_group = VA_f_pred_group_dict[area_id]
        VA_gt1_i_pred_group = VA_gt1_pred_group_dict[area_id]
        VA_gt2_i_pred_group = VA_gt2_pred_group_dict[area_id]
        VA_gt3_i_pred_group = VA_gt3_pred_group_dict[area_id]
        VA_gt4_i_pred_group = VA_gt4_pred_group_dict[area_id]
        VA_d_i_pred_group = VA_d_pred_group_dict[area_id]

        IA_pred_i_group = IA_pred_group_dict[area_id]
        IA_pred_i_back_group = IA_pred_group_dict[area_id_back]
        VA_pred_i_group = VA_pred_group_dict[area_id]

        pred_integral_all = 0

        # # f
        va_f_area = VA_f_dict[area_id]
        if va_f_area == []:
            pass
        else:
            VA_f_len = va_f_area[1] - va_f_area[0]
            parameter_a1 = func_dict[area_id + "_f_a"]["parameter_a"]
            parameter_beta1 = func_dict[area_id + "_f_a"]["parameter_beta"]
            greater_than_th = func_dict[area_id + "_f_a"]["greater_than_th"]

            for j, basic_interval in enumerate(VA_f_i_pred_group):
                if basic_interval != []:

                    pred_func_dict[area_id + "_f_a" + "_" + str(j)] = {
                        "area": basic_interval,
                        "area_start": va_f_area[0],
                        "parameter_a": parameter_a1,
                        "parameter_beta": parameter_beta1,
                        # "func":"va_f_func",
                        "x_tp": None,
                        "gt_len": None,
                        "area_len": VA_f_len
                    }
                    if not greater_than_th:
                        pred_func_dict[area_id + "_f_a" + "_" + str(j)]["func"] = "va_d_func"
                    else:
                        if method == 1:
                            pred_func_dict[area_id + "_f_a" + "_" + str(j)]["func"] = "va_f_func"
                        else:
                            pred_func_dict[area_id + "_f_a" + "_" + str(j)]["func"] = "va_f_reverse_func"

                    cal_integral_basic_interval_f = cal_integral_in_range_f_power_func(basic_interval, parameter_a1,
                                                                                       parameter_beta1, VA_f_len,
                                                                                       VA_f_len,
                                                                                       area_start=va_f_area[0],
                                                                                       method=method,
                                                                                       greater_than_th=greater_than_th)
                    pred_integral_all += cal_integral_basic_interval_f
                    basic_interval_integral_dict[area_id + "_f" + "_" + str(j)] = cal_integral_basic_interval_f

        # gt1-gt4
        s_a_gt_len = s_a_func_dict[area_id]["s_a_va_gt_len"]
        s_a_va = s_a_func_dict[area_id]["s_a_va"]
        parameter_b1 = func_dict[area_id + "_gt1"]["parameter_a"]
        parameter_gama1 = func_dict[area_id + "_gt1"]["parameter_beta"]
        parameter_b2 = func_dict[area_id + "_gt3"]["parameter_a"]
        parameter_gama2 = func_dict[area_id + "_gt3"]["parameter_beta"]
        x1 = s_a_func_dict[area_id]["x1"]
        x2 = s_a_func_dict[area_id]["x2"]
        area_gt = VA_gt_dict[area_id]
        VA_gt_len = area_gt[1] - area_gt[0]

        area_gt1 = func_dict[area_id + "_gt1"]["area"]
        area_gt2 = func_dict[area_id + "_gt2"]["area"]
        area_gt3 = func_dict[area_id + "_gt3"]["area"]
        area_gt4 = func_dict[area_id + "_gt4"]["area"]
        area_start = func_dict[area_id + "_gt1"]["area_start"]

        for j, basic_interval in enumerate(VA_gt1_i_pred_group):
            if basic_interval != []:
                pred_func_dict[area_id + "_gt1" + "_" + str(j)] = {
                    "area": basic_interval,
                    "area_start": area_start,
                    "parameter_a": parameter_b1,
                    "parameter_beta": parameter_gama1,
                    "func": "va_gt1_func",
                    "x_tp": None,
                    "gt_len": VA_gt_len,
                    "area_len": None,
                    "parameter_rho": parameter_rho
                }
                cal_integral_basic_interval_gt1 = cal_integral_in_range_gt_power_func1(basic_interval, parameter_b1,
                                                                                       parameter_gama1, x_tp=x1,
                                                                                       gt_len=VA_gt_len,
                                                                                       area_start=area_start,
                                                                                       parameter_rho=parameter_rho)
                pred_integral_all += cal_integral_basic_interval_gt1
                basic_interval_integral_dict[area_id + "_gt1" + "_" + str(j)] = cal_integral_basic_interval_gt1

        for j, basic_interval in enumerate(VA_gt2_i_pred_group):
            if basic_interval != []:
                pred_func_dict[area_id + "_gt2" + "_" + str(j)] = {
                    "area": basic_interval,
                    "area_start": area_start,
                    "parameter_a": parameter_b1,
                    "parameter_beta": parameter_gama1,
                    "func": "va_gt2_func",
                    "x_tp": x1,
                    "gt_len": VA_gt_len,
                    "area_len": None,
                    "parameter_rho": parameter_rho
                }
                cal_integral_basic_interval_gt2 = cal_integral_in_range_gt_power_func2(basic_interval, parameter_b1,
                                                                                       parameter_gama1, x_tp=x1,
                                                                                       gt_len=VA_gt_len,
                                                                                       area_start=area_start,
                                                                                       parameter_rho=parameter_rho)
                pred_integral_all += cal_integral_basic_interval_gt2
                basic_interval_integral_dict[area_id + "_gt2" + "_" + str(j)] = cal_integral_basic_interval_gt2

        for j, basic_interval in enumerate(VA_gt3_i_pred_group):
            if basic_interval != []:
                pred_func_dict[area_id + "_gt3" + "_" + str(j)] = {
                    "area": basic_interval,
                    "area_start": area_start,
                    "parameter_a": parameter_b2,
                    "parameter_beta": parameter_gama2,
                    "func": "va_gt3_func",
                    "x_tp": x2,
                    "gt_len": VA_gt_len,
                    "area_len": None,
                    "parameter_rho": parameter_rho

                }
                cal_integral_basic_interval_gt3 = cal_integral_in_range_gt_power_func3(basic_interval, parameter_b2,
                                                                                       parameter_gama2, x_tp=x2,
                                                                                       gt_len=VA_gt_len,
                                                                                       area_start=area_start,
                                                                                       parameter_rho=parameter_rho)
                pred_integral_all += cal_integral_basic_interval_gt3
                basic_interval_integral_dict[area_id + "_gt3" + "_" + str(j)] = cal_integral_basic_interval_gt3

        for j, basic_interval in enumerate(VA_gt4_i_pred_group):
            if basic_interval != []:
                pred_func_dict[area_id + "_gt4" + "_" + str(j)] = {
                    "area": basic_interval,
                    "area_start": area_start,
                    "parameter_a": parameter_b2,
                    "parameter_beta": parameter_gama2,
                    "func": "va_gt4_func",
                    "x_tp": x2,
                    "gt_len": VA_gt_len,
                    "area_len": None,
                    "parameter_rho": parameter_rho
                }
                cal_integral_basic_interval_gt4 = cal_integral_in_range_gt_power_func4(basic_interval, parameter_b2,
                                                                                       parameter_gama2, x_tp=x2,
                                                                                       gt_len=VA_gt_len,
                                                                                       area_start=area_start,
                                                                                       parameter_rho=parameter_rho)
                pred_integral_all += cal_integral_basic_interval_gt4
                basic_interval_integral_dict[area_id + "_gt4" + "_" + str(j)] = cal_integral_basic_interval_gt4

        # # d
        va_d_area = VA_d_dict[area_id]
        if va_d_area == []:
            pass
        else:
            VA_d_len = va_d_area[1] - va_d_area[0]
            parameter_a2 = func_dict[area_id + "_d_a"]["parameter_a"]
            parameter_beta2 = func_dict[area_id + "_d_a"]["parameter_beta"]
            greater_than_th = func_dict[area_id + "_d_a"]["greater_than_th"]
            if pred_case_id == 7 and area_id == "id_2":
                debug = 1

            for j, basic_interval in enumerate(VA_d_i_pred_group):
                if basic_interval != []:
                    pred_func_dict[area_id + "_d_a" + "_" + str(j)] = {
                        "area": basic_interval,
                        "area_start": va_d_area[0],
                        "parameter_a": parameter_a2,
                        "parameter_beta": parameter_beta2,
                        # "func":"va_d_func",
                        "x_tp": None,
                        "gt_len": None,
                        "area_len": VA_d_len
                    }
                    if not greater_than_th:
                        pred_func_dict[area_id + "_d_a" + "_" + str(j)]["func"] = "va_d_func"
                    else:
                        if method == 1:
                            pred_func_dict[area_id + "_d_a" + "_" + str(j)]["func"] = "va_d_func"
                        else:
                            pred_func_dict[area_id + "_d_a" + "_" + str(j)]["func"] = "va_d_reverse_func"

                    cal_integral_basic_interval_d = cal_integral_in_range_d_power_func(basic_interval, parameter_a2,
                                                                                       parameter_beta2, VA_d_len,
                                                                                       VA_d_len,
                                                                                       area_start=va_d_area[0],
                                                                                       method=method,
                                                                                       greater_than_th=greater_than_th)
                    pred_integral_all += cal_integral_basic_interval_d
                    basic_interval_integral_dict[area_id + "_d" + "_" + str(j)] = cal_integral_basic_interval_d

        # print("pred_integral_all",pred_integral_all)
        va_area = VA_dict[area_id]
        recall_va_i = pred_integral_all / s_a_va
        recall_va_list.append(recall_va_i)
        if pred_case_id == 7:
            debug = 1
        recall_va_dict[area_id] = recall_va_i

        detection_score_i = 1 if VA_pred_group_dict[area_id] != [] else 0
        detection_score_ia_i_dict[area_id] = detection_score_i
        # invalid part punish in is not enough
        # detection
        if cal_mode == "detection":
            precision_va_dict[area_id] = detection_score_i

        precision_i_group = IA_pred_i_group + IA_pred_i_back_group + VA_pred_i_group

        precision_i_all = 0
        precision_i_valid = 0
        for precision_i_item in precision_i_group:
            precision_i_all += precision_i_item[1] - precision_i_item[0]
        for precision_i_item in VA_pred_i_group:
            precision_i_valid += precision_i_item[1] - precision_i_item[0]
        if pred_case_id == 6:
            debug = 1
        precision_va_i = precision_i_valid / precision_i_all if precision_i_all != 0 else 0

        # precision also consider invalid part
        # proportion
        if cal_mode == "proportion":
            precision_va_dict[area_id] = precision_va_i

    if print_msg:
        # print basic_interval_integral
        print("basic_interval_integral_dict", basic_interval_integral_dict)

    # pot pred func
    # if pred_case_id == 7 and area_id == "id_2":
    #     debug = 1
    #     plot_func_multi(pred_func_dict)
    # plot_func_multi(pred_func_dict)

    if print_msg:
        # print recall_va_dict,precision_va_dict,dl_score_ia_dict
        print("recall_va_dict", recall_va_dict)
        print("precision_va_dict", precision_va_dict)
        print("dl_score_ia_dict", dl_score_ia_dict)

    adjusted_dl_score_ia_dict = {}
    adjusted_dl_score_ia_dict_ratio_contribution = {}

    if pred_case_id == 2:
        debug = 1
    area_id_0 = "id_" + str(0)
    area_id_num = "id_" + str(gt_num)
    # two logics
    logic = 1
    # logic = 3
    if logic == 1:
        # logic 1
        #     one_exist_value = 1/2
        one_exist_value = 1
        # 8th circle
        for i in range(gt_num + 1):
            area_id = "id_" + str(i)
            area_id_front = "id_" + str(i - 1)
            area_id_back = "id_" + str(i + 1)

            ia_area = IA_dict[area_id]
            l_ia = l_ia_set_dict[area_id]
            d_ia = d_ia_set_dict[area_id]
            d_ia_sign = d_ia_sign_set_dict[area_id]
            ia_len = ia_area[1] - ia_area[0] if ia_area != [] else 0

            if pred_case_id == 22:
                debug = 1

            if i == 0:
                if IA_dict[area_id] == []:
                    # if IA_dict[area_id_back] == [] and area_id_back == area_id_num:
                    if area_id_back == area_id_num:
                        if IA_dict[area_id_back] == []:  # 自己为空，后面一个为最后一个，后面一个为空
                            ratio_before = 0
                            ratio_after = 1 / 2
                        else:  # 自己为空，后面一个为最后一个，后面一个不为空
                            ratio_before = 0
                            ratio_after = 0
                    else:
                        ratio_before = 0
                        ratio_after = 1 / 2

                    if VA_pred_group_dict[area_id] != []:
                        dl_score_ia_i_contribution = 1
                    else:
                        dl_score_ia_i_contribution = 0
                else:
                    # 数据里至少有一个异常，至少有两个ia，如果没有异常，那么IA_dict=[]
                    if IA_pred_group_dict[area_id] != []:
                        if area_id_back == area_id_num:
                            if IA_dict[area_id_back] == []:  # 自己不为空，后面一个为最后一个，后面一个为空
                                ratio_before = 0
                                ratio_after = 1
                            else:  # 自己不为空，后面一个为最后一个，后面一个不为空
                                ratio_before = 0
                                ratio_after = 1 / 2
                        else:
                            ratio_before = 0
                            ratio_after = 1 / 2
                        dl_score_ia_i_contribution = dl_score_ia_dict[area_id]
                    else:
                        if VA_pred_group_dict[area_id] != []:
                            if area_id_back == area_id_num:
                                if IA_dict[area_id_back] == []:  # 自己不为空，后面一个为最后一个，后面一个为空
                                    ratio_before = 0
                                    ratio_after = 1
                                else:  # 自己不为空，后面一个为最后一个，后面一个不为空
                                    ratio_before = 0
                                    ratio_after = 1 / 2
                            else:
                                ratio_before = 0
                                ratio_after = 1 / 2
                            dl_score_ia_i_contribution = 1
                        else:
                            if area_id_back == area_id_num:
                                if IA_dict[area_id_back] == []:  # 自己不为空，后面一个为最后一个，后面一个为空
                                    ratio_before = 0
                                    ratio_after = 1
                                else:  # 自己不为空，后面一个为最后一个，后面一个不为空
                                    ratio_before = 0
                                    ratio_after = 1 / 2
                            else:
                                ratio_before = 0
                                ratio_after = 1 / 2
                            dl_score_ia_i_contribution = 0
            elif i == gt_num:
                if IA_dict[area_id] == []:
                    if area_id_front == area_id_0:
                        if IA_dict[area_id_front] == []:  # 自己为空，前面一个为最后一个，前面一个为空
                            ratio_before = 1 / 2
                            ratio_after = 0
                        else:  # 自己为空，前面一个为第一个，前面一个不为空
                            ratio_before = 0
                            ratio_after = 0
                    else:
                        ratio_before = 1 / 2
                        ratio_after = 0

                    if VA_pred_group_dict[area_id_front] != []:
                        dl_score_ia_i_contribution = 1
                    else:
                        dl_score_ia_i_contribution = 0
                else:
                    if IA_pred_group_dict[area_id] != []:
                        if area_id_front == area_id_0:
                            if IA_dict[area_id_front] == []:
                                ratio_before = 1
                                ratio_after = 0
                            else:
                                ratio_before = 1 / 2
                                ratio_after = 0
                        else:
                            ratio_before = 1 / 2
                            ratio_after = 0
                        dl_score_ia_i_contribution = dl_score_ia_dict[area_id]
                    else:
                        if VA_pred_group_dict[area_id_front] != []:
                            if area_id_front == area_id_0:
                                if IA_dict[area_id_front] == []:
                                    ratio_before = 1
                                    ratio_after = 0
                                else:
                                    ratio_before = 1 / 2
                                    ratio_after = 0
                            else:
                                ratio_before = 1 / 2
                                ratio_after = 0
                            dl_score_ia_i_contribution = 1
                        else:
                            if area_id_front == area_id_0:
                                if IA_dict[area_id_front] == []:
                                    ratio_before = 1
                                    ratio_after = 0
                                else:
                                    ratio_before = 1 / 2
                                    ratio_after = 0
                            else:
                                ratio_before = 1 / 2
                                ratio_after = 0
                            dl_score_ia_i_contribution = 0
            else:
                if IA_pred_group_dict[area_id] != []:
                    ratio_before = 1 / 2
                    ratio_after = 1 / 2
                    IA_len = IA_dict[area_id][1] - IA_dict[area_id][0]
                    IA_mid = (IA_dict[area_id][1] + IA_dict[area_id][0])/2
                    IA_start = IA_dict[area_id][0]
                    ratio_before_proportion = (IA_mid + dl_score_ia_dict[area_id]-IA_start) /IA_len
                    ratio_after_proportion = 1-ratio_before_proportion
                    # if d_ia_sign > 0:
                    #     ratio_before = (ia_len / 2 - d_ia) / ia_len
                    #     ratio_after = (ia_len / 2 + d_ia) / ia_len
                    # elif d_ia_sign < 0:
                    #     ratio_before = (ia_len / 2 + d_ia) / ia_len
                    #     ratio_after = (ia_len / 2 - d_ia) / ia_len
                    # else:
                    #     ratio_before = 1/2
                    #     ratio_after = 1/2
                    dl_score_ia_i_contribution = dl_score_ia_dict[area_id]
                else:
                    ratio_before = 1 / 2
                    ratio_after = 1 / 2
                    if VA_pred_group_dict[area_id_front] != [] and VA_pred_group_dict[area_id] != []:
                        # ratio_before = 1/2
                        # ratio_after = 1/2
                        dl_score_ia_i_contribution = 1
                    elif VA_pred_group_dict[area_id_front] == [] and VA_pred_group_dict[area_id] == []:
                        # ratio_before = 1/2
                        # ratio_after = 1/2
                        dl_score_ia_i_contribution = 0
                    else:
                        # if VA_pred_group_dict[area_id_front] != []:
                        #     ratio_before = 1
                        #     ratio_after = 0
                        # else:
                        #     ratio_before = 0
                        #     ratio_after = 1
                        dl_score_ia_i_contribution = one_exist_value

            adjusted_dl_score_ia_dict[area_id] = dl_score_ia_i_contribution
            dl_score_ia_i_before_contribution = dl_score_ia_i_contribution * ratio_before
            dl_score_ia_i_after_contribution = dl_score_ia_i_contribution * ratio_after
            adjusted_dl_score_ia_dict_ratio_contribution[area_id] = [dl_score_ia_i_before_contribution,
                                                                     dl_score_ia_i_after_contribution]





    debug = 1
    # logic 2
    # one_exist_value = 1/2
    # for i in range(gt_num + 1):
    #     area_id = "id_" + str(i)
    #     area_id_front = "id_" + str(i - 1)
    #     area_id_back = "id_" + str(i + 1)
    #
    #     ia_area = IA_dict[area_id]
    #     l_ia = l_ia_set_dict[area_id]
    #     d_ia = d_ia_set_dict[area_id]
    #     d_ia_sign = d_ia_sign_set_dict[area_id]
    #     ia_len = ia_area[1] - ia_area[0] if ia_area != [] else 0
    #
    #     if i == 0:
    #         ratio_before = 0
    #         ratio_after = 1
    #         if IA_pred_group_dict[area_id] != []:
    #             dl_score_ia_i_contribution = dl_score_ia_dict[area_id]
    #         else:
    #             if VA_pred_group_dict[area_id] != []:
    #                 dl_score_ia_i_contribution = 1 / 2
    #             else:
    #                 dl_score_ia_i_contribution = 0
    #     elif i == gt_num:
    #         ratio_before = 1
    #         ratio_after = 0
    #         if IA_pred_group_dict[area_id] != []:
    #             dl_score_ia_i_contribution = dl_score_ia_dict[area_id]
    #         else:
    #             if VA_pred_group_dict[area_id_front] != []:
    #                 dl_score_ia_i_contribution = 1 / 2
    #             else:
    #                 dl_score_ia_i_contribution = 0
    #     else:
    #         if IA_pred_group_dict[area_id] != []:
    #             ratio_before = 1 / 2
    #             ratio_after = 1 / 2
    #             # if d_ia_sign > 0:
    #             #     ratio_before = (ia_len / 2 - d_ia) / ia_len
    #             #     ratio_after = (ia_len / 2 + d_ia) / ia_len
    #             # elif d_ia_sign < 0:
    #             #     ratio_before = (ia_len / 2 + d_ia) / ia_len
    #             #     ratio_after = (ia_len / 2 - d_ia) / ia_len
    #             # else:
    #             #     ratio_before = 1/2
    #             #     ratio_after = 1/2
    #             dl_score_ia_i_contribution = dl_score_ia_dict[area_id]
    #         else:
    #             # ratio_before = 1 / 2
    #             # ratio_after = 1 / 2
    #             if VA_pred_group_dict[area_id_front] != [] and VA_pred_group_dict[area_id] != []:
    #                 ratio_before = 1/2
    #                 ratio_after = 1/2
    #                 dl_score_ia_i_contribution = 1
    #             elif VA_pred_group_dict[area_id_front] == [] and VA_pred_group_dict[area_id] == []:
    #                 ratio_before = 1/2
    #                 ratio_after = 1/2
    #                 dl_score_ia_i_contribution = 0
    #             else:
    #                 if VA_pred_group_dict[area_id_front] != []:
    #                     ratio_before = 1
    #                     ratio_after = 0
    #                 else:
    #                     ratio_before = 0
    #                     ratio_after = 1
    #                 dl_score_ia_i_contribution = one_exist_value

    # logic 3
    if logic == 3:
        # one_exist_value = 1/2
        one_exist_value = 1
        for i in range(gt_num + 1):
            area_id = "id_" + str(i)
            area_id_front = "id_" + str(i - 1)
            area_id_back = "id_" + str(i + 1)

            ia_area = IA_dict[area_id]
            l_ia = l_ia_set_dict[area_id]
            d_ia = d_ia_set_dict[area_id]
            d_ia_sign = d_ia_sign_set_dict[area_id]
            ia_len = ia_area[1] - ia_area[0] if ia_area != [] else 0

            if i == 0:
                ratio_before = 0
                ratio_after = 1
                if IA_dict[area_id] == []:
                    dl_score_ia_i_contribution = 0
                else:
                    if IA_pred_group_dict[area_id] != []:
                        dl_score_ia_i_contribution = dl_score_ia_dict[area_id]
                    else:
                        if VA_pred_group_dict[area_id] != []:
                            dl_score_ia_i_contribution = 1
                        else:
                            dl_score_ia_i_contribution = 0
            elif i == gt_num:
                ratio_before = 1
                ratio_after = 0
                if IA_dict[area_id] == []:
                    dl_score_ia_i_contribution = 0
                else:
                    if IA_pred_group_dict[area_id] != []:
                        dl_score_ia_i_contribution = dl_score_ia_dict[area_id]
                    else:
                        if VA_pred_group_dict[area_id_front] != []:
                            dl_score_ia_i_contribution = 1
                        else:
                            dl_score_ia_i_contribution = 0
            else:
                if IA_pred_group_dict[area_id] != []:
                    # ratio_before = 1 / 2
                    # ratio_after = 1 / 2
                    if d_ia_sign > 0:
                        ratio_before = (ia_len / 2 - d_ia) / ia_len
                        ratio_after = (ia_len / 2 + d_ia) / ia_len
                    elif d_ia_sign < 0:
                        ratio_before = (ia_len / 2 + d_ia) / ia_len
                        ratio_after = (ia_len / 2 - d_ia) / ia_len
                    else:
                        ratio_before = 1 / 2
                        ratio_after = 1 / 2
                    dl_score_ia_i_contribution = dl_score_ia_dict[area_id]
                else:
                    # ratio_before = 1 / 2
                    # ratio_after = 1 / 2
                    if VA_pred_group_dict[area_id_front] != [] and VA_pred_group_dict[area_id] != []:
                        ratio_before = 1 / 2
                        ratio_after = 1 / 2
                        dl_score_ia_i_contribution = 2
                    elif VA_pred_group_dict[area_id_front] == [] and VA_pred_group_dict[area_id] == []:
                        ratio_before = 1 / 2
                        ratio_after = 1 / 2
                        dl_score_ia_i_contribution = 0
                    else:
                        if VA_pred_group_dict[area_id_front] != []:
                            ratio_before = 1
                            ratio_after = 0
                        else:
                            ratio_before = 0
                            ratio_after = 1
                        dl_score_ia_i_contribution = one_exist_value

            adjusted_dl_score_ia_dict[area_id] = dl_score_ia_i_contribution
            dl_score_ia_i_before_contribution = dl_score_ia_i_contribution * ratio_before
            dl_score_ia_i_after_contribution = dl_score_ia_i_contribution * ratio_after
            adjusted_dl_score_ia_dict_ratio_contribution[area_id] = [dl_score_ia_i_before_contribution,
                                                                     dl_score_ia_i_after_contribution]

    if print_msg:
        print("adjusted_dl_score_ia_dict", adjusted_dl_score_ia_dict)
        print("adjusted_dl_score_ia_dict_ratio_contribution", adjusted_dl_score_ia_dict_ratio_contribution)
    event_recall_i_list = []
    event_recall_i_dict = {}
    event_recall_i_dict_w0 = {}
    event_recall_i_dict_w1 = {}
    event_precision_i_dict = {}
    event_precision_i_dict_w0 = {}
    event_precision_i_dict_w1 = {}

    # 9th circle
    for i in range(gt_num):
        area_id = "id_" + str(i)
        if print_msg:
            print(area_id)
        area_id_back = "id_" + str(i + 1)
        recall_va_i = recall_va_dict[area_id]
        # if VA_pred_group_dict[area_id] != [] and IA_pred_group_dict[area_id] == [] and IA_pred_group_dict[area_id_back] == [] :
        #     dl_score_ia_i = 1
        #     dl_score_ia_i_back = 1
        # else:
        #     dl_score_ia_i = dl_score_ia_dict[area_id]
        #     dl_score_ia_i_back = dl_score_ia_dict[area_id_back]

        dl_score_ia_i_after_contribution = adjusted_dl_score_ia_dict_ratio_contribution[area_id][1]
        dl_score_ia_i_back_before_contribution = adjusted_dl_score_ia_dict_ratio_contribution[area_id_back][0]

        if print_msg:
            print(" dl_score_ia_i_after_contribution", dl_score_ia_i_after_contribution)
            print(" recall_va_i,", recall_va_i)
            print(" dl_score_ia_i_back_before_contribution", dl_score_ia_i_back_before_contribution)

        # event_recall_i = (1-2*parameter_eta)*recall_va_i + parameter_eta*(dl_score_ia_i_after_contribution+ dl_score_ia_i_back_before_contribution)
        if pred_case_id == 26:
            debug = 1

        # logic 1
        if logic == 1:
            event_recall_i = (1 - parameter_eta) * recall_va_i + parameter_eta * (
                        dl_score_ia_i_after_contribution + dl_score_ia_i_back_before_contribution)
            # w=0
            event_recall_i_w0 = (1 - 0) * recall_va_i + 0 * (
                        dl_score_ia_i_after_contribution + dl_score_ia_i_back_before_contribution)
            # w=1
            event_recall_i_w1 = (1 - 1) * recall_va_i + 1 * (
                        dl_score_ia_i_after_contribution + dl_score_ia_i_back_before_contribution)
            # w=1
            event_recall_i_w1 = (1 - 1) * recall_va_i + 1 * (
                        dl_score_ia_i_after_contribution + dl_score_ia_i_back_before_contribution)
        # logic 3
        if logic == 3:
            if i == 0 and IA_dict[area_id] == []:
                event_recall_i = (1 - parameter_eta) * recall_va_i + parameter_eta * 1 / 1 * (
                    dl_score_ia_i_back_before_contribution)
            elif i == gt_num - 1 and IA_dict[area_id_back] == []:
                event_recall_i = (1 - parameter_eta) * recall_va_i + parameter_eta * 1 / 1 * (
                    dl_score_ia_i_after_contribution)
            else:
                event_recall_i = (1 - parameter_eta) * recall_va_i + parameter_eta * 1 / 2 * (
                            dl_score_ia_i_after_contribution + dl_score_ia_i_back_before_contribution)
        # event_recall_i_list
        event_recall_i_dict[area_id] = event_recall_i
        event_recall_i_dict_w0[area_id] = event_recall_i_w0
        event_recall_i_dict_w1[area_id] = event_recall_i_w1

        precision_va_i = precision_va_dict[area_id]
        if print_msg:
            print(" dl_score_ia_i_after_contribution", dl_score_ia_i_after_contribution)
            print(" precision_va_i,", precision_va_i)
            print(" dl_score_ia_i_back_before_contribution", dl_score_ia_i_back_before_contribution)

        # event_precision_i = (1-2*parameter_eta)*precision_va_i + parameter_eta*(dl_score_ia_i_after_contribution+ dl_score_ia_i_back_before_contribution)

        # logic 1
        if logic == 1:
            event_precision_i = (1 - parameter_eta) * precision_va_i + parameter_eta * (
                        dl_score_ia_i_after_contribution + dl_score_ia_i_back_before_contribution)
            # w=0
            event_precision_i_w0 = (1 - 0) * precision_va_i + 0 * (
                        dl_score_ia_i_after_contribution + dl_score_ia_i_back_before_contribution)
            # w=1
            event_precision_i_w1 = (1 - 1) * precision_va_i + 1 * (
                        dl_score_ia_i_after_contribution + dl_score_ia_i_back_before_contribution)

        # logic 3
        if logic == 3:
            if i == 0 and IA_dict[area_id] == []:
                event_precision_i = (1 - parameter_eta) * precision_va_i + parameter_eta * 1 / 1 * (
                    dl_score_ia_i_back_before_contribution)
            elif i == gt_num - 1 and IA_dict[area_id_back] == []:
                event_precision_i = (1 - parameter_eta) * precision_va_i + parameter_eta * 1 / 1 * (
                    dl_score_ia_i_after_contribution)
            else:
                event_precision_i = (1 - parameter_eta) * precision_va_i + parameter_eta * 1 / 2 * (
                            dl_score_ia_i_after_contribution + dl_score_ia_i_back_before_contribution)
        event_precision_i_dict[area_id] = event_precision_i
        event_precision_i_dict_w0[area_id] = event_precision_i_w0
        event_precision_i_dict_w1[area_id] = event_precision_i_w1

    if print_msg:
        print("event_recall_i_dict", event_recall_i_dict)
        print("event_precision_i_dict", event_precision_i_dict)

    if pred_case_id == 2:
        debug = 1
    event_recall = np.array(list(event_recall_i_dict.values())).mean()
    # print(event_recall)
    event_precision = np.array(list(event_precision_i_dict.values())).mean()
    # print(event_precision)

    event_recall_w0 = np.array(list(event_recall_i_dict_w0.values())).mean()
    event_precision_w0 = np.array(list(event_precision_i_dict_w0.values())).mean()
    event_recall_w1 = np.array(list(event_recall_i_dict_w1.values())).mean()
    event_precision_w1 = np.array(list(event_precision_i_dict_w1.values())).mean()

    #
    # print("======== meata_recall")
    # print(" event_recall",event_recall)
    # print(" event_recall_w0",event_recall_w0)
    # print(" event_recall_w1",event_recall_w1)
    #
    # print(" event_recall_i_dict",event_recall_i_dict)
    # print(" event_recall_i_dict_w0",event_recall_i_dict_w0)
    # print(" event_recall_i_dict_w1",event_recall_i_dict_w1)
    # print(" adjusted_dl_score_ia_dict_ratio_contribution",adjusted_dl_score_ia_dict_ratio_contribution)

    recall_info = {
        "event_recall": event_recall,
        "event_recall_w0": event_recall_w0,
        "event_recall_w1": event_recall_w1,
        "event_recall_i_dict": event_recall_i_dict,
        "event_recall_i_dict_w0": event_recall_i_dict_w0,
        "event_recall_i_dict_w1": event_recall_i_dict_w1,
        "adjusted_dl_score_ia_dict_ratio_contribution": adjusted_dl_score_ia_dict_ratio_contribution,
    }

    meata_f1 = compute_f1_score(event_precision, event_recall)
    meata_f1_w0 = compute_f1_score(event_precision_w0, event_recall_w0)
    meata_f1_w1 = compute_f1_score(event_precision_w1, event_recall_w1)

    # gt_label_point_array = convert_interval_to_point_array(label_interval_ranges, time_series_length=window_length)
    # ont_count = np.count_nonzero(gt_label_point_array == 1)
    # anomaly_rate = ont_count / window_length
    # if anomaly_rate > 0.2:
    #     print("anomaly_rate exceeds threshold")
    #     no_random_coefficient = 1
    # else:
    #     no_random_coefficient = cal_no_random_measure_coefficient(pred_label_point_array, window_length)

    if thresh_id == 94:
        d=1
    coefficient_meata_f1 = no_random_coefficient * meata_f1
    coefficient_event_recall = no_random_coefficient * event_recall
    coefficient_event_precision = no_random_coefficient * event_precision
    print("meata_f1", meata_f1)
    print("coefficient_meata_f1", coefficient_meata_f1)

    coefficient_meata_f1_w0 = no_random_coefficient * meata_f1_w0
    coefficient_meata_f1_w1 = no_random_coefficient * meata_f1_w1

    # return event_recall,event_precision,meata_f1,no_random_meata_f1
    # return event_recall,event_precision,meata_f1,coefficient_meata_f1,VA_f_dict,VA_d_dict,coefficient_event_recall,coefficient_event_precision
    return event_recall, event_precision, meata_f1, coefficient_meata_f1, VA_f_dict, VA_d_dict, \
        coefficient_event_recall, coefficient_event_precision, \
        meata_f1_w0, coefficient_meata_f1_w0, meata_f1_w1, coefficient_meata_f1_w1, \
        recall_info

# gt计算的时候给出斜线
def meata_v4(label_interval_ranges, IA1_VA1_IA2_relative_list, window_length, parameter_dict, pred_label_point_array,
          output=None,pred_case_id=None, thresh_id=None,cal_mode="proportion"):
    # print_msg = False
    print_msg = True

    # parameter_alpha = parameter_dict["parameter_alpha"]
    parameter_rate = parameter_dict["parameter_rate"]

    parameter_theta = parameter_dict["parameter_theta"]
    # parameter_w = parameter_dict["parameter_w"]
    # parameter_switch_f =parameter_dict["parameter_switch_f"]
    # parameter_switch_d = parameter_dict["parameter_switch_d"]
    VA_f_window = parameter_dict["forecasting_window"]
    VA_d_window = parameter_dict["delay_window"]

    parameter_rho = parameter_dict["parameter_rho"]
    parameter_lamda = parameter_dict["parameter_lamda"]

    parameter_eta = parameter_dict["parameter_eta"]

    # no random
    no_random_coefficient = cal_no_random_measure_coefficient_method2(output,window_length)

    if print_msg:
        print("VA_f_window", VA_f_window)
        print("VA_d_window", VA_d_window)

    f_window_rate = 0
    d_window_rate = 0
    f_d_window_len = VA_f_window + VA_d_window
    if f_d_window_len != 0:
        f_window_rate = VA_f_window / f_d_window_len
        d_window_rate = 1 - f_window_rate

    IA_dict = {}
    VA_dict = {}
    VA_f_dict = {}
    VA_gt_dict = {}
    VA_d_dict = {}

    # pred generated is continuous range a,b(len) -> [a,a+b(len)),range of [a,b] -> [a,b) -> point of [a,b-1],point of [a,b] -> [a,b+1)

    # cal every f/d of gt,split IA and VA

    label_interval_num = len(label_interval_ranges)

    s_a_func_dict = {}

    # first circle
    for i, label_interval_range in enumerate(label_interval_ranges):
        # adjust now and before,add before
        VA_gt_now_start, VA_gt_now_end = label_interval_range
        # VA_f_now_start = max(VA_gt_now_start - VA_f_window,0) if parameter_switch_f == True else max(VA_gt_now_start - 3,0)
        VA_f_now_start = max(VA_gt_now_start - VA_f_window, 0)
        VA_f_now_end = VA_gt_now_start
        VA_d_now_start = VA_gt_now_end
        # VA_d_now_end = min(VA_gt_now_end + VA_f_window,window_length) if parameter_switch_d == True else min(VA_gt_now_end + 3,window_length)
        VA_d_now_end = min(VA_gt_now_end + VA_d_window, window_length)
        # # this way below is more fast
        # if i == 0:
        #     VA_f_now_start = max(VA_f_now_start,0)
        # if i == label_interval_num:
        #     VA_d_now_end = min(VA_d_now_end,window_length)
        if i > 0:
            # adjust now and before
            if VA_d_before_end > VA_f_now_start:
                intersection_range_start = max(VA_f_now_start, VA_gt_before_end)
                intersection_range_end = min(VA_d_before_end, VA_gt_now_start)
                intersection_range = [intersection_range_start, intersection_range_end]
                intersection_len = intersection_range_end - intersection_range_start
                #         if intersection_len == 1 and (parameter_switch_f == True and parameter_switch_d == True) and parameter_rho == 1/2:
                # #             f prior
                #             VA_d_before_end -=1
                #         else:
                #         VA_d_before_end = VA_f_now_start = intersection_range_start + round(parameter_w*intersection_len)
                #         VA_d_before_end = VA_f_now_start = intersection_range_start + round(parameter_w*intersection_len,2)
                #         VA_f_now_start = intersection_range_start + round(f_window_rate*intersection_len,2)
                #         VA_d_before_end = intersection_range_start + round(d_window_rate*intersection_len,2)
                VA_f_now_start = intersection_range_start + round(f_window_rate * intersection_len)
                VA_d_before_end = intersection_range_start + round(d_window_rate * intersection_len)
            #     add before
            IA_dict["id_" + str(i)] = [VA_d_before_end, VA_f_now_start]
            VA_dict["id_" + str(i - 1)] = [VA_f_before_start, VA_d_before_end]
            VA_f_dict["id_" + str(i - 1)] = [VA_f_before_start, VA_f_before_end]
            VA_gt_dict["id_" + str(i - 1)] = [VA_gt_before_start, VA_gt_before_end]
            VA_d_dict["id_" + str(i - 1)] = [VA_d_before_start, VA_d_before_end]

            # add by order

            split_line_set.add(VA_f_before_start)
            split_line_set.add(VA_f_before_end)

            split_line_set.add(VA_gt_before_start)
            split_line_set.add(VA_gt_before_end)

            split_line_set.add(VA_d_before_start)
            split_line_set.add(VA_d_before_end)

            split_line_set.add(VA_d_before_end)
            split_line_set.add(VA_f_now_start)

            VA_gt_before_len = VA_gt_before_end - VA_gt_before_start
            x1, x2 = cal_x1_x2(VA_gt_before_len, parameter_rho)

            # # print("area_id,area",area_id,area)
            # # # print("x1,x2",x1, x2)
            # # print("x1_ori,x2_ori",x1+VA_gt_before_start[0], x2+VA_gt_before_start[0])
            x1_ori, x2_ori = x1 + VA_gt_before_start, x2 + VA_gt_before_start
            x_mid_ori = 2 * x1 + VA_gt_before_start
            split_line_set.add(x1_ori)
            split_line_set.add(x_mid_ori)
            split_line_set.add(x2_ori)
        else:
            IA_dict["id_" + str(i)] = [0, VA_f_now_start]
            split_line_set.add(VA_f_now_start)

        VA_gt_before_start, VA_gt_before_end = VA_gt_now_start, VA_gt_now_end
        VA_f_before_start = VA_f_now_start
        VA_f_before_end = VA_f_now_end
        VA_d_before_start = VA_d_now_start
        VA_d_before_end = VA_d_now_end

    # last one VA and last two IA
    IA_dict["id_" + str(label_interval_num)] = [VA_d_before_end, window_length]
    VA_dict["id_" + str(label_interval_num - 1)] = [VA_f_before_start, VA_d_before_end]
    VA_f_dict["id_" + str(label_interval_num - 1)] = [VA_f_before_start, VA_f_before_end]
    VA_gt_dict["id_" + str(label_interval_num - 1)] = [VA_gt_before_start, VA_gt_before_end]
    VA_d_dict["id_" + str(label_interval_num - 1)] = [VA_d_before_start, VA_d_before_end]

    VA_gt_before_len = VA_gt_before_end - VA_gt_before_start
    x1, x2 = cal_x1_x2(VA_gt_before_len, parameter_rho)

    # # print("area_id,area",area_id,area)
    # # # print("x1,x2",x1, x2)
    # # print("x1_ori,x2_ori",x1+VA_gt_before_start[0], x2+VA_gt_before_start[0])
    x1_ori, x2_ori = x1 + VA_gt_before_start, x2 + VA_gt_before_start
    x_mid_ori = 2 * x1 + VA_gt_before_start
    split_line_set.add(x1_ori)
    split_line_set.add(x_mid_ori)
    split_line_set.add(x2_ori)

    # add by order

    split_line_set.add(VA_f_before_start)
    split_line_set.add(VA_f_before_end)

    split_line_set.add(VA_gt_before_start)
    split_line_set.add(VA_gt_before_end)

    split_line_set.add(VA_d_before_start)
    split_line_set.add(VA_d_before_end)

    if print_msg:
        # print area dict
        # # print("split area")
        print("IA_dict", IA_dict)
        # print("VA_dict",VA_dict)
        print("VA_f_dict", VA_f_dict)
        print("VA_gt_dict", VA_gt_dict)
        print("VA_d_dict", VA_d_dict)

    def dict_filer(dict_data: dict):
        for i, (area_id, area_range) in enumerate(dict_data.items()):
            if area_range != []:
                dict_data[area_id] = [] if area_range[0] == area_range[1] else area_range
        return dict_data

    # second circle
    IA_dict = dict_filer(IA_dict)
    VA_dict = dict_filer(VA_dict)
    VA_f_dict = dict_filer(VA_f_dict)
    VA_gt_dict = dict_filer(VA_gt_dict)
    VA_d_dict = dict_filer(VA_d_dict)

    if print_msg:
        print("dict adjust +++++++++++")
        print("IA_dict", IA_dict)
        # print("VA_dict",VA_dict)
        print("VA_f_dict", VA_f_dict)
        print("VA_gt_dict", VA_gt_dict)
        print("VA_d_dict", VA_d_dict)

        # split pred to basic interval
        print("split_line_set", split_line_set)

    # third circle
    split_IA1_VA1_IA2_relative_list = split_intervals(IA1_VA1_IA2_relative_list, split_line_set)
    # print("split_IA1_VA1_IA2_relative_list",split_IA1_VA1_IA2_relative_list)

    # forecasting_len=10
    # delay_len=10
    IA_pred_group_dict = {}
    VA_pred_group_dict = {}
    VA_f_pred_group_dict = {}
    VA_gt_pred_group_dict = {}
    VA_d_pred_group_dict = {}

    VA_gt1_pred_group_dict = {}
    VA_gt2_pred_group_dict = {}
    VA_gt3_pred_group_dict = {}
    VA_gt4_pred_group_dict = {}
    func_dict = {}

    # "id_"+str(label_interval_num)
    for i in range(label_interval_num):
        IA_pred_group_dict["id_" + str(i)] = []
        VA_pred_group_dict["id_" + str(i)] = []
        VA_f_pred_group_dict["id_" + str(i)] = []
        VA_gt_pred_group_dict["id_" + str(i)] = []
        VA_d_pred_group_dict["id_" + str(i)] = []
        VA_gt1_pred_group_dict["id_" + str(i)] = []
        VA_gt2_pred_group_dict["id_" + str(i)] = []
        VA_gt3_pred_group_dict["id_" + str(i)] = []
        VA_gt4_pred_group_dict["id_" + str(i)] = []
    IA_pred_group_dict["id_" + str(label_interval_num)] = []

    # 4th circle
    # for i, basic_interval in enumerate(IA1_VA1_IA2_relative_list):
    for i, basic_interval in enumerate(split_IA1_VA1_IA2_relative_list):
        # IA
        for j, (area_id, area) in enumerate(IA_dict.items()):
            if pred_in_area(basic_interval, area):
                if area_id not in IA_pred_group_dict.keys():
                    IA_pred_group_dict[area_id] = [basic_interval]
                else:
                    IA_pred_group_dict[area_id].append(basic_interval)

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

            # group by x1,x2
            VA_gt_len = area[1] - area[0]
            x1, x2 = cal_x1_x2(VA_gt_len, parameter_rho)

            # # print("area_id,area",area_id,area)
            # # # print("x1,x2",x1, x2)
            # # print("x1_ori,x2_ori",x1+area[0], x2+area[0])
            area_gt1 = [0, x1]
            area_gt2 = [x1, 2 * x1]
            area_gt3 = [2 * x1, x2]
            area_gt4 = [x2, area[1] - area[0]]
            area_gt1_ori = [area[0], x1 + area[0]]
            area_gt2_ori = [x1 + area[0], 2 * x1 + area[0]]
            area_gt3_ori = [2 * x1 + area[0], x2 + area[0]]
            area_gt4_ori = [x2 + area[0], area[1]]
            # # print("area_gt1",area_gt1)
            # # print("area_gt2",area_gt2)
            # # print("area_gt3",area_gt3)
            # # print("area_gt4",area_gt4)

            # # print("area_gt1_ori",area_gt1_ori)
            # # print("area_gt2_ori",area_gt2_ori)
            # # print("area_gt3_ori",area_gt3_ori)
            # # print("area_gt4_ori",area_gt4_ori)

            if pred_in_area(basic_interval, area_gt1_ori):
                # if area_id not in VA_gt1_pred_group_dict.keys():
                #     VA_gt1_pred_group_dict[area_id]=[basic_interval]
                # else:
                #     VA_gt1_pred_group_dict[area_id].append(basic_interval)
                VA_gt1_pred_group_dict[area_id].append(basic_interval)
            # if basic_interval == [135, 141] and area == [130, 150]:
            #     a =1
            if pred_in_area(basic_interval, area_gt2_ori):
                # if area_id not in VA_gt2_pred_group_dict.keys():
                #     VA_gt2_pred_group_dict[area_id]=[basic_interval]
                # else:
                #     VA_gt2_pred_group_dict[area_id].append(basic_interval)
                VA_gt2_pred_group_dict[area_id].append(basic_interval)
            if pred_in_area(basic_interval, area_gt3_ori):
                # if area_id not in VA_gt3_pred_group_dict.keys():
                #     VA_gt3_pred_group_dict[area_id]=[basic_interval]
                # else:
                VA_gt3_pred_group_dict[area_id].append(basic_interval)
            if pred_in_area(basic_interval, area_gt4_ori):
                # if area_id not in VA_gt4_pred_group_dict.keys():
                #     VA_gt4_pred_group_dict[area_id]=[basic_interval]
                # else:
                VA_gt4_pred_group_dict[area_id].append(basic_interval)

        # VA-d
        for j, (area_id, area) in enumerate(VA_d_dict.items()):
            if pred_in_area(basic_interval, area):
                if area_id not in VA_d_pred_group_dict.keys():
                    VA_d_pred_group_dict[area_id] = [basic_interval]
                else:
                    VA_d_pred_group_dict[area_id].append(basic_interval)
        # VA
        for j, (area_id, area) in enumerate(VA_dict.items()):
            if pred_in_area(basic_interval, area):
                if area_id not in VA_pred_group_dict.keys():
                    VA_pred_group_dict[area_id] = [basic_interval]
                else:
                    VA_pred_group_dict[area_id].append(basic_interval)
    if print_msg:
        # print pred dict
        print("IA_pred_group_dict", IA_pred_group_dict)
        print("VA_f_pred_group_dict", VA_f_pred_group_dict)
        # print("VA_gt_pred_group_dict",VA_gt_pred_group_dict)
        print("VA_gt1_pred_group_dict", VA_gt1_pred_group_dict)
        print("VA_gt2_pred_group_dict", VA_gt2_pred_group_dict)
        print("VA_gt3_pred_group_dict", VA_gt3_pred_group_dict)
        print("VA_gt4_pred_group_dict", VA_gt4_pred_group_dict)
        print("VA_d_pred_group_dict", VA_d_pred_group_dict)
        # print("VA_pred_group_dict",VA_pred_group_dict)

    d_ia_set_dict = {}
    d_ia_sign_set_dict = {}
    l_ia_set_dict = {}
    dl_score_ia_dict = {}

    dl_part_ia_score_dict = {}
    gt_num = label_interval_num


    # 5th circle
    for i, (area_id, basic_interval_list) in enumerate(IA_pred_group_dict.items()):
        if basic_interval_list == []:
            dl_score_ia_dict[area_id] = 0
            l_ia_set_dict[area_id] = 0
            d_ia_set_dict[
                area_id] = 0  # only when basic_interval_list ！= [],the value are used,so set value to 0 have boe problem
            d_ia_sign_set_dict[
                area_id] = 0  # only when basic_interval_list ！= [],the value are used,so set value to 0 have boe problem
            continue
        l_ia = 0
        l_ia_set_sum = 0
        d_ia_set_sum = 0
        d_ia_sign_set_sum = 0
        ia_set_num = len(basic_interval_list)
        IA_mid = (IA_dict[area_id][0] + IA_dict[area_id][1]) / 2
        IA_len = IA_dict[area_id][1] - IA_dict[area_id][0]


        test_group = "group"
        # test_group = "no group"
        if test_group == "group":
            for basic_interval in basic_interval_list:
                l_ia += basic_interval[1] - basic_interval[0]
                # the length of fitst and last IA could be set a fixed value,min(一个周期的长度,ori len),or min(mean len,ori len)
                if i == 0:
                    base_position = IA_dict[area_id][1]
                    # d_ia_set_sum+= abs((basic_interval[1] + basic_interval[0])/2 -base_position)
                # last id of IA is label_interval_num,not label_interval_num -1
                elif i == label_interval_num:
                    base_position = IA_dict[area_id][0]
                else:
                    base_position = IA_mid
                # d_ia_set_sum+= abs((basic_interval[1] + basic_interval[0])/2 -IA_mid)
                d_ia_set_sum += abs((basic_interval[1] + basic_interval[0]) / 2 - base_position)
                # i =0 or i = label_interval_num,only have positive distance

                d_ia_sign_set_sum += (basic_interval[1] + basic_interval[0]) / 2 - base_position

            l_ia_set_dict[area_id] = l_ia
            d_ia = d_ia_set_sum / ia_set_num if ia_set_num != 0 else 0
            d_ia_set_dict[area_id] = d_ia
            d_ia_sign = d_ia_sign_set_sum / ia_set_num if ia_set_num != 0 else 0
            d_ia_sign_set_dict[area_id] = d_ia_sign

            dl_score_ia = 0
            #
            if l_ia != 0:
                if i == 0:
                    # 一边情况下就没有1了
                    # parameter_alpha1 = parameter_rate/(2+parameter_rate)
                    #
                    # debug1 = (1/2*IA_len-d_ia)
                    # debug2 = 2 / IA_len * (1/2*IA_len-d_ia)
                    # debug3 = parameter_alpha1 * 2 / IA_len * (1/2*IA_len-d_ia)
                    #
                    # debug4 = (1 - l_ia / IA_len)
                    # debug5 = (1 - parameter_alpha1) * (1 - l_ia / IA_len)

                    # dl_score_ia = parameter_alpha1 * 2 / IA_len * (1/2*IA_len-d_ia) + (1 - parameter_alpha1) * (1 - l_ia / IA_len) \
                    #     if IA_len != 0 else 0

                    parameter_alpha = parameter_rate / (1 + parameter_rate)
                    debug11 = (IA_len - d_ia)
                    debug21 = 1 / IA_len * (IA_len - d_ia)
                    debug31 = parameter_alpha / IA_len * (IA_len - d_ia)

                    debug41 = (1 - l_ia / IA_len)
                    debug51 = (1 - parameter_alpha) * (1 - l_ia / IA_len)
                    # 得分跟系数也有关系

                    dl_score_ia1 = parameter_alpha / IA_len * (IA_len - d_ia) + (1 - parameter_alpha) * (
                            1 - l_ia / IA_len) \
                        if IA_len != 0 else 0
                    # 对抗边界是2
                    if parameter_rate <= 2:
                        min_value = parameter_alpha / IA_len * (IA_len - 1 / 2 * IA_len)
                        # min_value = min_value + 0.0000001
                        dl_score_ia = map_value(dl_score_ia1, min_value, 1, 1e-10, 1)
                    else:
                        min_value = (1 - parameter_alpha) * (1 - 0 / IA_len)
                        dl_score_ia = map_value(dl_score_ia1, min_value, 1, 1e-10, 1)

                        # dl_score_ia = dl_score_ia1
                    # print("0 dl_score_ia",dl_score_ia)
                    # dl_part_ia_score_dict[area_id] = {
                    #     "parameter_alpha":parameter_alpha,
                    #     "d_score":1/IA_len * (IA_len - d_ia),
                    #     "(1 - parameter_alpha)":(1 - parameter_alpha),
                    #     "d_score_parameter":(1 - parameter_alpha) * (
                    #                 1 - l_ia / IA_len)
                    # }
                    if pred_case_id == 2:
                        debug = 1
                # last id of IA is label_interval_num,not label_interval_num -1
                elif i == label_interval_num:

                    # debug1 = (1 / 2 * IA_len - d_ia)
                    # debug2 = 2 / IA_len * (1 / 2 * IA_len - d_ia)
                    # debug3 = parameter_alpha * 2 / IA_len * (1 / 2 * IA_len - d_ia)
                    #
                    # debug4 = (1 - l_ia / IA_len)
                    # debug5 = (1 - parameter_alpha) * (1 - l_ia / IA_len)

                    # dl_score_ia = parameter_alpha * 2 / IA_len * (1/2*IA_len-d_ia) + (1 - parameter_alpha) * (1 - l_ia / IA_len) \
                    #     if IA_len != 0 else 0

                    parameter_alpha = parameter_rate / (1 + parameter_rate)
                    debug11 = (IA_len - d_ia)
                    debug21 = 1 / IA_len * (IA_len - d_ia)
                    debug31 = parameter_alpha / IA_len * (IA_len - d_ia)

                    debug41 = (1 - l_ia / IA_len)
                    debug51 = (1 - parameter_alpha) * (1 - l_ia / IA_len)
                    dl_score_ia1 = parameter_alpha / IA_len * (IA_len - d_ia) + (1 - parameter_alpha) * (
                                1 - l_ia / IA_len) \
                        if IA_len != 0 else 0
                    # x, a, b, c, d
                    if parameter_rate <= 2:
                        min_value = parameter_alpha / IA_len * (IA_len - 1 / 2 * IA_len)
                        # min_value = min_value + 0.0000001
                        dl_score_ia = map_value(dl_score_ia1, min_value, 1, 1e-10, 1)
                    else:
                        min_value = (1 - parameter_alpha) * (1 - 0 / IA_len)
                        dl_score_ia = map_value(dl_score_ia1, min_value, 1, 1e-10, 1)
                        # dl_score_ia = map_value(dl_score_ia,parameter_alpha,1,0,1)
                    # dl_part_ia_score_dict[area_id] = {
                    #     "parameter_alpha": parameter_alpha,
                    #     "d_score": 1 / IA_len * (IA_len - d_ia),
                    #     "(1 - parameter_alpha)": (1 - parameter_alpha),
                    #     "d_score_parameter": (1 - parameter_alpha) * (
                    #             1 - l_ia / IA_len)
                    # }
                    if pred_case_id == 2:
                        debug = 1
                else:

                    parameter_alpha = parameter_rate / (2 + parameter_rate)
                    dl_score_ia = parameter_alpha * 2 / IA_len * d_ia + (1 - parameter_alpha) * (1 - l_ia / IA_len) \
                        if IA_len != 0 else 0
                    if pred_case_id == 2:
                        print("  parameter_alpha", parameter_alpha)
                        print("  d_ia", d_ia)
                        print("  d_score", 2 / IA_len * d_ia)
                        print("  d_score_parameter", parameter_alpha * 2 / IA_len * d_ia)
                        print("  (1 - parameter_alpha)", (1 - parameter_alpha))
                        print("  l_ia", l_ia)
                        print("  l_score", (1 - l_ia / IA_len))
                        print("  l_score_parameter", (1 - parameter_alpha) * (1 - l_ia / IA_len))
                        print("  IA_len", IA_len)
                        debug = 1
                dl_part_ia_score_dict
        else:
            if print_msg:
                print("basic_interval_list num", len(basic_interval_list))
            dl_score_ia_sum = 0
            for basic_interval in basic_interval_list:
                if i == 0:
                    base_position = IA_dict[area_id][1]
                elif i == label_interval_num:
                    base_position = IA_dict[area_id][0]
                else:
                    base_position = IA_mid

                l_ia = basic_interval[1] - basic_interval[0]
                d_ia = abs((basic_interval[1] + basic_interval[0]) / 2 - base_position)
                if pred_case_id == 2:
                    debug = 1
                dl_score_ia = 0
                #
                if l_ia != 0:
                    if i == 0:
                        parameter_alpha = parameter_rate / (1 + parameter_rate)
                        dl_score_ia1 = parameter_alpha / IA_len * (IA_len - d_ia) + (1 - parameter_alpha) * (
                                1 - l_ia / IA_len) \
                            if IA_len != 0 else 0
                        # 对抗边界是2
                        if parameter_rate <= 2:
                            min_value = parameter_alpha / IA_len * (IA_len - 1 / 2 * IA_len)
                            dl_score_ia = map_value(dl_score_ia1, min_value, 1, 1e-10, 1)
                        else:
                            min_value = (1 - parameter_alpha) * (1 - 0 / IA_len)
                            dl_score_ia = map_value(dl_score_ia1, min_value, 1, 1e-10, 1)
                        if print_msg:
                            print("0 dl_score_ia", dl_score_ia)
                        if pred_case_id == 2:
                            debug = 1
                    elif i == label_interval_num:
                        parameter_alpha = parameter_rate / (1 + parameter_rate)
                        debug11 = (IA_len - d_ia)
                        debug21 = 1 / IA_len * (IA_len - d_ia)
                        debug31 = parameter_alpha / IA_len * (IA_len - d_ia)

                        debug41 = (1 - l_ia / IA_len)
                        debug51 = (1 - parameter_alpha) * (1 - l_ia / IA_len)
                        dl_score_ia1 = parameter_alpha / IA_len * (IA_len - d_ia) + (1 - parameter_alpha) * (
                                1 - l_ia / IA_len) \
                            if IA_len != 0 else 0
                        # x, a, b, c, d
                        if parameter_rate <= 2:
                            min_value = parameter_alpha / IA_len * (IA_len - 1 / 2 * IA_len)
                            # min_value = min_value + 0.0000001
                            dl_score_ia = map_value(dl_score_ia1, min_value, 1, 1e-10, 1)
                        else:
                            min_value = (1 - parameter_alpha) * (1 - 0 / IA_len)
                            dl_score_ia = map_value(dl_score_ia1, min_value, 1, 1e-10, 1)
                            # dl_score_ia = map_value(dl_score_ia,parameter_alpha,1,0,1)
                        if pred_case_id == 2:
                            debug = 1
                    else:

                        parameter_alpha = parameter_rate / (2 + parameter_rate)
                        dl_score_ia = parameter_alpha * 2 / IA_len * d_ia + (1 - parameter_alpha) * (1 - l_ia / IA_len) \
                            if IA_len != 0 else 0
                        if pred_case_id == 2:
                            if print_msg:
                                print("  parameter_alpha", parameter_alpha)
                                print("  d_ia", d_ia)
                                print("  d_score", 2 / IA_len * d_ia)
                                print("  d_score_parameter", parameter_alpha * 2 / IA_len * d_ia)
                                print("  (1 - parameter_alpha)", (1 - parameter_alpha))
                                print("  l_ia", l_ia)
                                print("  l_score", (1 - l_ia / IA_len))
                                print("  l_score_parameter", (1 - parameter_alpha) * (1 - l_ia / IA_len))
                                print("  IA_len", IA_len)
                                debug = 1

                # sum
                dl_score_ia_sum += dl_score_ia
                l_ia_set_sum += basic_interval[1] - basic_interval[0]
                d_ia_set_sum += abs((basic_interval[1] + basic_interval[0]) / 2 - base_position)
                d_ia_sign_set_sum += (basic_interval[1] + basic_interval[0]) / 2 - base_position

            # sum
            # test_no_group_type = "sum"
            test_no_group_type = "average"
            if test_no_group_type == "sum":
                dl_score_ia = dl_score_ia_sum
                l_ia_set_dict[area_id] = l_ia_set_sum
                d_ia_set_dict[area_id] = d_ia_set_sum
                d_ia_sign_set_dict[area_id] = d_ia_sign_set_sum
            else:
                dl_score_ia = dl_score_ia_sum / len(basic_interval_list) if len(basic_interval_list) != 0 else 0
                l_ia_set_dict[area_id] = l_ia_set_sum / len(basic_interval_list) if len(basic_interval_list) != 0 else 0
                d_ia_set_dict[area_id] = d_ia_set_sum / len(basic_interval_list) if len(basic_interval_list) != 0 else 0
                d_ia_sign_set_dict[area_id] = d_ia_sign_set_sum
        dl_score_ia_dict[area_id] = dl_score_ia
    #     deal empty and no area
    if print_msg:
        print("d_ia_set_dict", d_ia_set_dict)
        print("d_ia_sign_set_dict", d_ia_sign_set_dict)
        print("l_ia_set_dict", l_ia_set_dict)
    # print("dl_score_ia_dict",dl_score_ia_dict)

    # cal label_function_gt
    # 6th circle
    for i, (area_id, area) in enumerate(VA_gt_dict.items()):

        VA_gt_len = area[1] - area[0]
        x1, x2 = cal_x1_x2(VA_gt_len, parameter_rho)
        x1_ori, x2_ori = x1 + area[0], x2 + area[0]
        x_mid = 2 * x1
        x_mid_ori = x_mid + area[0]
        # x1_ori,x2_ori = x1+area[0], x2+area[0]
        area_gt1 = [0, x1]
        area_gt2 = [x1, 2 * x1]
        area_gt3 = [2 * x1, x2]
        area_gt4 = [x2, VA_gt_len]
        area_gt1_ori = [area[0], x1 + area[0]]
        area_gt2_ori = [x1 + area[0], 2 * x1 + area[0]]
        area_gt3_ori = [2 * x1 + area[0], x2 + area[0]]
        area_gt4_ori = [x2 + area[0], area[1]]
        if print_msg:
            print("VA_gt_len", VA_gt_len)

        if parameter_rho < 0:
            parameter_b1, parameter_gama1 = 1 / 2, 0
            parameter_b2, parameter_gama2 = 1 / 2, 0
        else:
            parameter_b1, parameter_gama1 = cal_power_function_coefficient(x1, parameter_lamda)
            parameter_b2, parameter_gama2 = cal_power_function_coefficient(x2 - x_mid, parameter_lamda)

        func_dict.update({
            area_id + "_gt1": {"area": area_gt1_ori,
                               "area_start": area_gt1_ori[0],
                               "parameter_a": parameter_b1,
                               "parameter_beta": parameter_gama1,
                               "func": "va_gt1_func",
                               "x_tp": x1,
                               "gt_len": VA_gt_len,
                               "area_len": None,
                               "parameter_rho": parameter_rho
                               },
            area_id + "_gt2": {"area": area_gt2_ori,
                               "area_start": area_gt1_ori[0],
                               "parameter_a": parameter_b1,
                               "parameter_beta": parameter_gama1,
                               "func": "va_gt2_func",
                               "x_tp": x1,
                               "gt_len": VA_gt_len,
                               "area_len": None,
                               "parameter_rho": parameter_rho
                               },
            area_id + "_gt3": {"area": area_gt3_ori,
                               "area_start": area_gt1_ori[0],
                               "parameter_a": parameter_b2,
                               "parameter_beta": parameter_gama2,
                               "func": "va_gt3_func",
                               "x_tp": x2,
                               "gt_len": VA_gt_len,
                               "area_len": None,
                               "parameter_rho": parameter_rho
                               },
            area_id + "_gt4": {"area": area_gt4_ori,
                               "area_start": area_gt1_ori[0],
                               "parameter_a": parameter_b2,
                               "parameter_beta": parameter_gama2,
                               "func": "va_gt4_func",
                               "x_tp": x2,
                               "gt_len": VA_gt_len,
                               "area_len": None,
                               "parameter_rho": parameter_rho
                               },

        })

        s_a_gt1 = cal_integral_in_range_gt_power_func1(area_gt1_ori, parameter_b1, parameter_gama1, x_tp=x1,
                                                       gt_len=VA_gt_len, area_start=area_gt1_ori[0],
                                                       parameter_rho=parameter_rho)
        s_a_gt2 = cal_integral_in_range_gt_power_func2(area_gt2_ori, parameter_b1, parameter_gama1, x_tp=x1,
                                                       gt_len=VA_gt_len, area_start=area_gt1_ori[0],
                                                       parameter_rho=parameter_rho)
        s_a_gt3 = cal_integral_in_range_gt_power_func3(area_gt3_ori, parameter_b2, parameter_gama2, x_tp=x2,
                                                       gt_len=VA_gt_len, area_start=area_gt1_ori[0],
                                                       parameter_rho=parameter_rho)
        s_a_gt4 = cal_integral_in_range_gt_power_func4(area_gt4_ori, parameter_b2, parameter_gama2, x_tp=x2,
                                                       gt_len=VA_gt_len, area_start=area_gt1_ori[0],
                                                       parameter_rho=parameter_rho)

        s_a_gt = s_a_gt1 + s_a_gt2 + s_a_gt3 + s_a_gt4

        func_dict[area_id + "_gt1"]["area_s"] = s_a_gt1
        func_dict[area_id + "_gt2"]["area_s"] = s_a_gt2
        func_dict[area_id + "_gt3"]["area_s"] = s_a_gt3
        func_dict[area_id + "_gt4"]["area_s"] = s_a_gt4

        method = 2

        va_f_area = VA_f_dict[area_id]
        s_a_f = 0
        if va_f_area == []:
            pass
        else:
            VA_f_len = va_f_area[1] - va_f_area[0]
            parameter_beta1 = 1
            parameter_a1 = 1 / (2 * VA_f_len)

            greater_than_th = False

            func_dict[area_id + "_f"] = {"area": va_f_area,
                                         "area_start": va_f_area[0],
                                         "parameter_a": parameter_a1,
                                         "parameter_beta": parameter_beta1,
                                         "func": "va_f_func",
                                         "x_tp": None,
                                         "gt_len": None,
                                         "area_len": VA_f_len,
                                         "greater_than_th": greater_than_th
                                         }

            # f
            s_max_f = cal_integral_in_range_f_power_func(va_f_area, parameter_a1, parameter_beta1, VA_f_len, VA_f_len,
                                                         area_start=va_f_area[0], greater_than_th=greater_than_th)
            func_dict[area_id + "_f"]["area_s"] = s_max_f

            s_rate = s_max_f / s_a_gt
            func_dict[area_id + "_f_a"] = func_dict[area_id + "_f"]
            s_a_f = s_max_f
            if s_rate > parameter_theta:
                greater_than_th = True

                # method1
                if method == 1:
                    parameter_beta1_a = VA_f_len / (2 * parameter_theta * s_a_gt) - 1
                    parameter_a1_a = 1 / (2 * pow(VA_f_len, parameter_beta1_a))
                # method2
                else:
                    parameter_beta1_a = VA_f_len / (2 * parameter_theta * s_a_gt) - 1  # same with method1
                    parameter_a1_a = VA_f_len / pow(1 / 2, parameter_beta1_a)

                s_a_f = cal_integral_in_range_f_power_func(va_f_area, parameter_a1_a, parameter_beta1_a, VA_f_len,
                                                           VA_f_len, area_start=va_f_area[0], method=method,
                                                           greater_than_th=greater_than_th)
                func_dict[area_id + "_f_a"] = {
                    "area": va_f_area,
                    "area_start": va_f_area[0],
                    "area_s": s_a_f,
                    "parameter_a": parameter_a1_a,
                    "parameter_beta": parameter_beta1_a,
                    # "func": "va_f_func",
                    # "func": "va_f_reverse_func",
                    "x_tp": None,
                    "gt_len": None,
                    "area_len": VA_f_len,
                    "greater_than_th": greater_than_th
                }
                if method == 1:
                    func_dict[area_id + "_f_a"]["func"] = "va_f_func"
                else:
                    func_dict[area_id + "_f_a"]["func"] = "va_f_reverse_func"

        # d
        va_d_area = VA_d_dict[area_id]
        s_a_d = 0
        if va_d_area == []:
            pass
        else:

            if pred_case_id == 7 and area_id == "id_2":
                debug = 1

            VA_d_len = va_d_area[1] - va_d_area[0]
            parameter_beta2 = 1
            parameter_a2 = 1 / (2 * VA_d_len)

            greater_than_th = False
            s_max_d = cal_integral_in_range_d_power_func(va_d_area, parameter_a2, parameter_beta2, VA_d_len, VA_d_len,
                                                         area_start=va_d_area[0], greater_than_th=greater_than_th)
            func_dict[area_id + "_d"] = {
                "area": va_d_area,
                "area_start": va_d_area[0],
                "parameter_a": parameter_a2,
                "parameter_beta": parameter_beta2,
                "func": "va_d_func",
                "x_tp": None,
                "gt_len": None,
                "area_len": VA_d_len,
                "greater_than_th": greater_than_th
            }

            func_dict[area_id + "_d"]["area_s"] = s_max_d

            func_dict[area_id + "_d_a"] = func_dict[area_id + "_d"]
            s_rate = s_max_d / s_a_gt
            s_a_d = s_max_d

            if s_rate > parameter_theta:
                greater_than_th = True

                # method1
                if method == 1:
                    parameter_beta2_a = VA_d_len / (2 * parameter_theta * s_a_gt) - 1
                    parameter_a2_a = 1 / (2 * pow(VA_d_len, parameter_beta2_a))
                else:
                    # method2
                    parameter_beta2_a = VA_d_len / (2 * parameter_theta * s_a_gt) - 1  # same with method1
                    parameter_a2_a = VA_d_len / pow(1 / 2, parameter_beta2_a)

                s_a_d = cal_integral_in_range_d_power_func(va_d_area, parameter_a2_a, parameter_beta2_a, VA_d_len,
                                                           VA_d_len,
                                                           area_start=va_d_area[0], method=method,
                                                           greater_than_th=greater_than_th)
                func_dict[area_id + "_d_a"] = {
                    "area": va_d_area,
                    "area_start": va_d_area[0],
                    "area_s": s_a_d,
                    "parameter_a": parameter_a2_a,
                    "parameter_beta": parameter_beta2_a,
                    # "func": "va_d_func",
                    # "func": "va_d_reverse_func",
                    "x_tp": None,
                    "gt_len": None,
                    "area_len": VA_d_len,
                    "greater_than_th": greater_than_th

                }

                if method == 1:
                    func_dict[area_id + "_d_a"]["func"] = "va_d_func"
                else:
                    func_dict[area_id + "_d_a"]["func"] = "va_d_reverse_func"

        s_a_func_dict[area_id] = {
            "s_a_va_gt_len": VA_gt_len,
            "s_a_va_gt": s_a_gt,
            "s_a_va": s_a_f + s_a_gt + s_a_d,
            "x1": x1,
            "x2": x2,
            "x_mid": x_mid,
            "x1_ori": x1_ori,
            "x2_ori": x2_ori,
            "x_mid_ori": x_mid_ori,
        }
    # plot func
    if pred_case_id == 7 and area_id == "id_2":
        debug = 1
        # plot_func_multi(func_dict)
    # plot_func_multi(func_dict)
    # plot_func_multi_paper(func_dict,window_length)

    import pprint

    # print("s_a_func_dict")
    # pprint(s_a_func_dict, indent=4)

    # cal recall_va,detection_score
    recall_va_list = []
    recall_va_dict = {}
    detection_score_ia_i_dict = {}
    precision_va_dict = {}

    gt_num = label_interval_num
    pred_func_dict = {}
    basic_interval_integral_dict = {}

    # 7th circle
    for i in range(gt_num):
        area_id = "id_" + str(i)
        # print("area_id",area_id)
        area_id_back = "id_" + str(i + 1)

        VA_start = VA_dict[area_id][0]
        VA_end = VA_dict[area_id][1]
        IA_len = IA_dict[area_id][1] - IA_dict[area_id][0]
        IA_start = IA_dict[area_id][0]
        IA_end = IA_dict[area_id][1]
        IA_back_start = IA_dict[area_id_back][0]
        IA_back_end = IA_dict[area_id_back][1]
        IA_back_len = IA_dict[area_id_back][1] - IA_dict[area_id_back][0]

        VA_f_i_pred_group = VA_f_pred_group_dict[area_id]
        VA_gt1_i_pred_group = VA_gt1_pred_group_dict[area_id]
        VA_gt2_i_pred_group = VA_gt2_pred_group_dict[area_id]
        VA_gt3_i_pred_group = VA_gt3_pred_group_dict[area_id]
        VA_gt4_i_pred_group = VA_gt4_pred_group_dict[area_id]
        VA_d_i_pred_group = VA_d_pred_group_dict[area_id]

        IA_pred_i_group = IA_pred_group_dict[area_id]
        IA_pred_i_back_group = IA_pred_group_dict[area_id_back]
        VA_pred_i_group = VA_pred_group_dict[area_id]

        pred_integral_all = 0
        pred_integral_all_before = 0
        pred_integral_all_after = 0

        # # f
        va_f_area = VA_f_dict[area_id]
        if va_f_area == []:
            pass
        else:
            VA_f_len = va_f_area[1] - va_f_area[0]
            parameter_a1 = func_dict[area_id + "_f_a"]["parameter_a"]
            parameter_beta1 = func_dict[area_id + "_f_a"]["parameter_beta"]
            greater_than_th = func_dict[area_id + "_f_a"]["greater_than_th"]

            for j, basic_interval in enumerate(VA_f_i_pred_group):
                if basic_interval != []:

                    pred_func_dict[area_id + "_f_a" + "_" + str(j)] = {
                        "area": basic_interval,
                        "area_start": va_f_area[0],
                        "parameter_a": parameter_a1,
                        "parameter_beta": parameter_beta1,
                        # "func":"va_f_func",
                        "x_tp": None,
                        "gt_len": None,
                        "area_len": VA_f_len
                    }
                    if not greater_than_th:
                        pred_func_dict[area_id + "_f_a" + "_" + str(j)]["func"] = "va_d_func"
                    else:
                        if method == 1:
                            pred_func_dict[area_id + "_f_a" + "_" + str(j)]["func"] = "va_f_func"
                        else:
                            pred_func_dict[area_id + "_f_a" + "_" + str(j)]["func"] = "va_f_reverse_func"

                    cal_integral_basic_interval_f = cal_integral_in_range_f_power_func(basic_interval, parameter_a1,
                                                                                       parameter_beta1, VA_f_len,
                                                                                       VA_f_len,
                                                                                       area_start=va_f_area[0],
                                                                                       method=method,
                                                                                       greater_than_th=greater_than_th)
                    pred_integral_all += cal_integral_basic_interval_f
                    basic_interval_integral_dict[area_id + "_f" + "_" + str(j)] = cal_integral_basic_interval_f

        # gt1-gt4
        s_a_gt_len = s_a_func_dict[area_id]["s_a_va_gt_len"]
        s_a_va = s_a_func_dict[area_id]["s_a_va"]
        parameter_b1 = func_dict[area_id + "_gt1"]["parameter_a"]
        parameter_gama1 = func_dict[area_id + "_gt1"]["parameter_beta"]
        parameter_b2 = func_dict[area_id + "_gt3"]["parameter_a"]
        parameter_gama2 = func_dict[area_id + "_gt3"]["parameter_beta"]
        x1 = s_a_func_dict[area_id]["x1"]
        x2 = s_a_func_dict[area_id]["x2"]
        area_gt = VA_gt_dict[area_id]
        VA_gt_len = area_gt[1] - area_gt[0]

        area_gt1 = func_dict[area_id + "_gt1"]["area"]
        area_gt2 = func_dict[area_id + "_gt2"]["area"]
        area_gt3 = func_dict[area_id + "_gt3"]["area"]
        area_gt4 = func_dict[area_id + "_gt4"]["area"]
        area_start = func_dict[area_id + "_gt1"]["area_start"]

        for j, basic_interval in enumerate(VA_gt1_i_pred_group):
            if basic_interval != []:
                pred_func_dict[area_id + "_gt1" + "_" + str(j)] = {
                    "area": basic_interval,
                    "area_start": area_start,
                    "parameter_a": parameter_b1,
                    "parameter_beta": parameter_gama1,
                    "func": "va_gt1_func",
                    "x_tp": None,
                    "gt_len": VA_gt_len,
                    "area_len": None,
                    "parameter_rho": parameter_rho
                }
                cal_integral_basic_interval_gt1 = cal_integral_in_range_gt_power_func1(basic_interval, parameter_b1,
                                                                                       parameter_gama1, x_tp=x1,
                                                                                       gt_len=VA_gt_len,
                                                                                       area_start=area_start,
                                                                                       parameter_rho=parameter_rho)
                pred_integral_all += cal_integral_basic_interval_gt1
                basic_interval_integral_dict[area_id + "_gt1" + "_" + str(j)] = cal_integral_basic_interval_gt1

        for j, basic_interval in enumerate(VA_gt2_i_pred_group):
            if basic_interval != []:
                pred_func_dict[area_id + "_gt2" + "_" + str(j)] = {
                    "area": basic_interval,
                    "area_start": area_start,
                    "parameter_a": parameter_b1,
                    "parameter_beta": parameter_gama1,
                    "func": "va_gt2_func",
                    "x_tp": x1,
                    "gt_len": VA_gt_len,
                    "area_len": None,
                    "parameter_rho": parameter_rho
                }
                cal_integral_basic_interval_gt2 = cal_integral_in_range_gt_power_func2(basic_interval, parameter_b1,
                                                                                       parameter_gama1, x_tp=x1,
                                                                                       gt_len=VA_gt_len,
                                                                                       area_start=area_start,
                                                                                       parameter_rho=parameter_rho)
                pred_integral_all += cal_integral_basic_interval_gt2
                basic_interval_integral_dict[area_id + "_gt2" + "_" + str(j)] = cal_integral_basic_interval_gt2

        for j, basic_interval in enumerate(VA_gt3_i_pred_group):
            if basic_interval != []:
                pred_func_dict[area_id + "_gt3" + "_" + str(j)] = {
                    "area": basic_interval,
                    "area_start": area_start,
                    "parameter_a": parameter_b2,
                    "parameter_beta": parameter_gama2,
                    "func": "va_gt3_func",
                    "x_tp": x2,
                    "gt_len": VA_gt_len,
                    "area_len": None,
                    "parameter_rho": parameter_rho

                }
                cal_integral_basic_interval_gt3 = cal_integral_in_range_gt_power_func3(basic_interval, parameter_b2,
                                                                                       parameter_gama2, x_tp=x2,
                                                                                       gt_len=VA_gt_len,
                                                                                       area_start=area_start,
                                                                                       parameter_rho=parameter_rho)
                pred_integral_all += cal_integral_basic_interval_gt3
                basic_interval_integral_dict[area_id + "_gt3" + "_" + str(j)] = cal_integral_basic_interval_gt3

        for j, basic_interval in enumerate(VA_gt4_i_pred_group):
            if basic_interval != []:
                pred_func_dict[area_id + "_gt4" + "_" + str(j)] = {
                    "area": basic_interval,
                    "area_start": area_start,
                    "parameter_a": parameter_b2,
                    "parameter_beta": parameter_gama2,
                    "func": "va_gt4_func",
                    "x_tp": x2,
                    "gt_len": VA_gt_len,
                    "area_len": None,
                    "parameter_rho": parameter_rho
                }
                cal_integral_basic_interval_gt4 = cal_integral_in_range_gt_power_func4(basic_interval, parameter_b2,
                                                                                       parameter_gama2, x_tp=x2,
                                                                                       gt_len=VA_gt_len,
                                                                                       area_start=area_start,
                                                                                       parameter_rho=parameter_rho)
                pred_integral_all += cal_integral_basic_interval_gt4
                basic_interval_integral_dict[area_id + "_gt4" + "_" + str(j)] = cal_integral_basic_interval_gt4

        # # d
        va_d_area = VA_d_dict[area_id]
        if va_d_area == []:
            pass
        else:
            VA_d_len = va_d_area[1] - va_d_area[0]
            parameter_a2 = func_dict[area_id + "_d_a"]["parameter_a"]
            parameter_beta2 = func_dict[area_id + "_d_a"]["parameter_beta"]
            greater_than_th = func_dict[area_id + "_d_a"]["greater_than_th"]
            if pred_case_id == 7 and area_id == "id_2":
                debug = 1

            for j, basic_interval in enumerate(VA_d_i_pred_group):
                if basic_interval != []:
                    pred_func_dict[area_id + "_d_a" + "_" + str(j)] = {
                        "area": basic_interval,
                        "area_start": va_d_area[0],
                        "parameter_a": parameter_a2,
                        "parameter_beta": parameter_beta2,
                        # "func":"va_d_func",
                        "x_tp": None,
                        "gt_len": None,
                        "area_len": VA_d_len
                    }
                    if not greater_than_th:
                        pred_func_dict[area_id + "_d_a" + "_" + str(j)]["func"] = "va_d_func"
                    else:
                        if method == 1:
                            pred_func_dict[area_id + "_d_a" + "_" + str(j)]["func"] = "va_d_func"
                        else:
                            pred_func_dict[area_id + "_d_a" + "_" + str(j)]["func"] = "va_d_reverse_func"

                    cal_integral_basic_interval_d = cal_integral_in_range_d_power_func(basic_interval, parameter_a2,
                                                                                       parameter_beta2, VA_d_len,
                                                                                       VA_d_len,
                                                                                       area_start=va_d_area[0],
                                                                                       method=method,
                                                                                       greater_than_th=greater_than_th)
                    pred_integral_all += cal_integral_basic_interval_d
                    basic_interval_integral_dict[area_id + "_d" + "_" + str(j)] = cal_integral_basic_interval_d

        # print("pred_integral_all",pred_integral_all)
        va_area = VA_dict[area_id]
        recall_va_i = pred_integral_all / s_a_va
        recall_va_list.append(recall_va_i)
        if pred_case_id == 7:
            debug = 1
        recall_va_dict[area_id] = recall_va_i

        detection_score_i = 1 if VA_pred_group_dict[area_id] != [] else 0
        detection_score_ia_i_dict[area_id] = detection_score_i
        # invalid part punish in is not enough
        # detection
        if cal_mode == "detection":
            precision_va_dict[area_id] = detection_score_i

        cal_precision_method = 1

        if cal_precision_method == 1:
            # method1

            precision_i_group = IA_pred_i_group + IA_pred_i_back_group + VA_pred_i_group

            precision_i_all = 0
            precision_i_valid = 0
            for precision_i_item in precision_i_group:
                precision_i_all += precision_i_item[1] - precision_i_item[0]
            for precision_i_item in VA_pred_i_group:
                precision_i_valid += precision_i_item[1] - precision_i_item[0]

            # precision_i_all = {float} 415.0
            # precision_i_group = {list: 6} [[100, 201], [300, 401], [600, 701], [800, 901], [495, 500.5], [500.5, 506]]
            # precision_i_item = {list: 2} [500.5, 506]
            # precision_i_valid = {float} 11.0
            # precision_va_dict = {dict: 1} {'id_0': 0.02650602409638554}
            # precision_va_i = {float} 0.02650602409638554

        else:
            precision_i_all = pred_integral_all
            precision_i_valid = pred_integral_all
            precision_i_group = IA_pred_i_group + IA_pred_i_back_group

            if cal_precision_method == 2:
                # method2
                for precision_i_item in precision_i_group:
                    precision_i_all += parameter_eta*(precision_i_item[1] - precision_i_item[0])
                # precision_i_all = {float} 175.59227919873044
                # precision_i_group = {list: 4} [[100, 201], [300, 401], [600, 701], [800, 901]]
                # precision_i_item = {list: 2} [800, 901]
                # precision_i_valid = {float} 8.25
                # precision_va_dict = {dict: 1} {'id_0': 0.046983842556442244}
                # precision_va_i = {float} 0.046983842556442244

            elif cal_precision_method == 3:
                # method3
                parameter_a_left = 1/(2*IA_len)
                parameter_beta_left = 1
                parameter_a_right = 1/(2*IA_back_len)
                parameter_beta_right = 1

                func_dict[area_id + "_ia_before"] = {
                    "area": IA_dict[area_id],
                    "area_start": IA_start,
                    "parameter_a": parameter_a_left,
                    "parameter_beta": parameter_beta_left,
                    "func": "ia_before_func",
                    "x_tp": None,
                    "gt_len": None,
                    "ia_len": IA_len,
                    "area_len": None,
                    "parameter_rho": parameter_rho
                }
                for interval_id, IA_basic_interval in enumerate(IA_pred_i_group):
                    # IA_left_area_relative = [VA_start -  IA_basic_interval[1],VA_start -  IA_basic_interval[0]]

                    # func_dict[area_id + "_ia_before" + "_" + str(interval_id)] = {
                    #     "area": IA_basic_interval,
                    #     "area_start": IA_start,
                    #     "parameter_a": parameter_a_left,
                    #     "parameter_beta": parameter_beta_left,
                    #     "func": "ia_before_func",
                    #     "x_tp": None,
                    #     "gt_len": None,
                    #     "ia_len": IA_len,
                    #     "area_len": None,
                    #     "parameter_rho": parameter_rho
                    # }
                    # plot_func_multi(pred_func_dict)

                    cal_integral_gt_before = cal_integral_gt_before_ia_power_func(IA_basic_interval, parameter_a_left, parameter_beta_left,area_start=IA_start)
                    pred_integral_all_before += cal_integral_gt_before

                func_dict[area_id + "_ia_after"] = {
                    "area": IA_dict[area_id_back],
                    "area_start": IA_back_start,
                    "parameter_a": parameter_a_right,
                    "parameter_beta": parameter_beta_right,
                    "func": "ia_after_func",
                    "x_tp": None,
                    "gt_len": None,
                    "ia_len": IA_back_len,
                    "area_len": None,
                    "parameter_rho": parameter_rho
                }
                for interval_id, IA_basic_interval in enumerate(IA_pred_i_back_group):
                    IA_right_area_relative = [IA_basic_interval[0]- VA_end,IA_basic_interval[1]- VA_end]
                    # func_dict[area_id + "_ia_after" + "_" + str(interval_id)] = {
                    #     "area": IA_basic_interval,
                    #     "area_start": IA_back_start,
                    #     "parameter_a": parameter_a_right,
                    #     "parameter_beta": parameter_beta_right,
                    #     "func": "ia_after_func",
                    #     "x_tp": None,
                    #     "gt_len": None,
                    #     "ia_len": IA_back_len,
                    #     "area_len": None,
                    #     "parameter_rho": parameter_rho
                    # }
                    # plot_func_multi(pred_func_dict)
                    cal_integral_gt_after = cal_integral_gt_after_ia_power_func(IA_basic_interval, parameter_a_right, parameter_beta_right,area_start=IA_back_start)
                    pred_integral_all_after += cal_integral_gt_after


                precision_i_all += pred_integral_all_before
                precision_i_all += pred_integral_all_after

                #         precision_i_all = {float} 111.41639330578857
                # precision_i_group = {list: 4} [[100, 201], [300, 401], [600, 701], [800, 901]]
                # precision_i_valid = {float} 8.25
                # precision_va_dict = {dict: 1} {'id_0': 0.07404655414896989}
                # precision_va_i = {float} 0.07404655414896989
                # pred_case_id = {int} 1
                # pred_func_dict = {dict: 2} {'id_0_gt3_0': {'area': [495, 500.5], 'area_len': None, 'area_start': 490, 'func': 'va_gt3_func', 'gt_len': 21, 'parameter_a': 0.00021595939963286903, 'parameter_beta': 3.0, 'parameter_rho': 0, 'x_tp': 10.5}, 'id_0_gt4_0': {'area': [500.5, 506], 'area_len'
                # pred_integral_all = {float} 8.25
                # pred_integral_all_after = {float} 51.532719836400815
                # pred_integral_all_before = {float} 51.63367346938775



        precision_va_i = precision_i_valid / precision_i_all if precision_i_all != 0 else 0
        if pred_case_id == 11 or pred_case_id == 12:
             d = 1
        # precision also consider invalid part
        # proportion
        if cal_mode == "proportion":
            precision_va_dict[area_id] = precision_va_i

    if pred_case_id == 1:
        debug = 1
    if print_msg:
        # print basic_interval_integral
        print("basic_interval_integral_dict", basic_interval_integral_dict)

    # pot pred func
    # if pred_case_id == 7 and area_id == "id_2":
    #     debug = 1
    #     plot_func_multi(pred_func_dict)
    # plot_func_multi_paper(func_dict)

    if print_msg:
        # print recall_va_dict,precision_va_dict,dl_score_ia_dict
        print("recall_va_dict", recall_va_dict)
        print("precision_va_dict", precision_va_dict)
        print("dl_score_ia_dict", dl_score_ia_dict)

    adjusted_dl_score_ia_dict = {}
    adjusted_dl_score_ia_dict_ratio_contribution = {}

    if pred_case_id == 2:
        debug = 1
    area_id_0 = "id_" + str(0)
    area_id_num = "id_" + str(gt_num)
    # two logics
    logic = 1
    # logic = 3
    if logic == 1:
        # logic 1
        #     one_exist_value = 1/2
        one_exist_value = 1
        # 8th circle
        for i in range(gt_num + 1):
            area_id = "id_" + str(i)
            area_id_front = "id_" + str(i - 1)
            area_id_back = "id_" + str(i + 1)

            ia_area = IA_dict[area_id]
            l_ia = l_ia_set_dict[area_id]
            d_ia = d_ia_set_dict[area_id]
            d_ia_sign = d_ia_sign_set_dict[area_id]
            ia_len = ia_area[1] - ia_area[0] if ia_area != [] else 0

            if pred_case_id == 22:
                debug = 1

            if i == 0:
                if IA_dict[area_id] == []:
                    # if IA_dict[area_id_back] == [] and area_id_back == area_id_num:
                    if area_id_back == area_id_num:
                        if IA_dict[area_id_back] == []:  # 自己为空，后面一个为最后一个，后面一个为空
                            ratio_before = 0
                            ratio_after = 1 / 2
                        else:  # 自己为空，后面一个为最后一个，后面一个不为空
                            ratio_before = 0
                            ratio_after = 0
                    else:
                        ratio_before = 0
                        ratio_after = 1 / 2

                    if VA_pred_group_dict[area_id] != []:
                        dl_score_ia_i_contribution = 1
                    else:
                        dl_score_ia_i_contribution = 0
                else:
                    # 数据里至少有一个异常，至少有两个ia，如果没有异常，那么IA_dict=[]
                    if IA_pred_group_dict[area_id] != []:
                        if area_id_back == area_id_num:
                            if IA_dict[area_id_back] == []:  # 自己不为空，后面一个为最后一个，后面一个为空
                                ratio_before = 0
                                ratio_after = 1
                            else:  # 自己不为空，后面一个为最后一个，后面一个不为空
                                ratio_before = 0
                                ratio_after = 1 / 2
                        else:
                            ratio_before = 0
                            ratio_after = 1 / 2
                        dl_score_ia_i_contribution = dl_score_ia_dict[area_id]
                    else:
                        if VA_pred_group_dict[area_id] != []:
                            if area_id_back == area_id_num:
                                if IA_dict[area_id_back] == []:  # 自己不为空，后面一个为最后一个，后面一个为空
                                    ratio_before = 0
                                    ratio_after = 1
                                else:  # 自己不为空，后面一个为最后一个，后面一个不为空
                                    ratio_before = 0
                                    ratio_after = 1 / 2
                            else:
                                ratio_before = 0
                                ratio_after = 1 / 2
                            dl_score_ia_i_contribution = 1
                        else:
                            if area_id_back == area_id_num:
                                if IA_dict[area_id_back] == []:  # 自己不为空，后面一个为最后一个，后面一个为空
                                    ratio_before = 0
                                    ratio_after = 1
                                else:  # 自己不为空，后面一个为最后一个，后面一个不为空
                                    ratio_before = 0
                                    ratio_after = 1 / 2
                            else:
                                ratio_before = 0
                                ratio_after = 1 / 2
                            dl_score_ia_i_contribution = 0
            elif i == gt_num:
                if IA_dict[area_id] == []:
                    if area_id_front == area_id_0:
                        if IA_dict[area_id_front] == []:  # 自己为空，前面一个为最后一个，前面一个为空
                            ratio_before = 1 / 2
                            ratio_after = 0
                        else:  # 自己为空，前面一个为第一个，前面一个不为空
                            ratio_before = 0
                            ratio_after = 0
                    else:
                        ratio_before = 1 / 2
                        ratio_after = 0

                    if VA_pred_group_dict[area_id_front] != []:
                        dl_score_ia_i_contribution = 1
                    else:
                        dl_score_ia_i_contribution = 0
                else:
                    if IA_pred_group_dict[area_id] != []:
                        if area_id_front == area_id_0:
                            if IA_dict[area_id_front] == []:
                                ratio_before = 1
                                ratio_after = 0
                            else:
                                ratio_before = 1 / 2
                                ratio_after = 0
                        else:
                            ratio_before = 1 / 2
                            ratio_after = 0
                        dl_score_ia_i_contribution = dl_score_ia_dict[area_id]
                    else:
                        if VA_pred_group_dict[area_id_front] != []:
                            if area_id_front == area_id_0:
                                if IA_dict[area_id_front] == []:
                                    ratio_before = 1
                                    ratio_after = 0
                                else:
                                    ratio_before = 1 / 2
                                    ratio_after = 0
                            else:
                                ratio_before = 1 / 2
                                ratio_after = 0
                            dl_score_ia_i_contribution = 1
                        else:
                            if area_id_front == area_id_0:
                                if IA_dict[area_id_front] == []:
                                    ratio_before = 1
                                    ratio_after = 0
                                else:
                                    ratio_before = 1 / 2
                                    ratio_after = 0
                            else:
                                ratio_before = 1 / 2
                                ratio_after = 0
                            dl_score_ia_i_contribution = 0
            else:
                if IA_pred_group_dict[area_id] != []:
                    ratio_before = 1 / 2
                    ratio_after = 1 / 2
                    IA_len = IA_dict[area_id][1] - IA_dict[area_id][0]
                    IA_mid = (IA_dict[area_id][1] + IA_dict[area_id][0])/2
                    IA_start = IA_dict[area_id][0]
                    ratio_before_proportion = (IA_mid + dl_score_ia_dict[area_id]-IA_start) /IA_len
                    ratio_after_proportion = 1-ratio_before_proportion
                    # if d_ia_sign > 0:
                    #     ratio_before = (ia_len / 2 - d_ia) / ia_len
                    #     ratio_after = (ia_len / 2 + d_ia) / ia_len
                    # elif d_ia_sign < 0:
                    #     ratio_before = (ia_len / 2 + d_ia) / ia_len
                    #     ratio_after = (ia_len / 2 - d_ia) / ia_len
                    # else:
                    #     ratio_before = 1/2
                    #     ratio_after = 1/2
                    dl_score_ia_i_contribution = dl_score_ia_dict[area_id]
                else:
                    ratio_before = 1 / 2
                    ratio_after = 1 / 2
                    if VA_pred_group_dict[area_id_front] != [] and VA_pred_group_dict[area_id] != []:
                        # ratio_before = 1/2
                        # ratio_after = 1/2
                        dl_score_ia_i_contribution = 1
                    elif VA_pred_group_dict[area_id_front] == [] and VA_pred_group_dict[area_id] == []:
                        # ratio_before = 1/2
                        # ratio_after = 1/2
                        dl_score_ia_i_contribution = 0
                    else:
                        # if VA_pred_group_dict[area_id_front] != []:
                        #     ratio_before = 1
                        #     ratio_after = 0
                        # else:
                        #     ratio_before = 0
                        #     ratio_after = 1
                        dl_score_ia_i_contribution = one_exist_value

            adjusted_dl_score_ia_dict[area_id] = dl_score_ia_i_contribution
            dl_score_ia_i_before_contribution = dl_score_ia_i_contribution * ratio_before
            dl_score_ia_i_after_contribution = dl_score_ia_i_contribution * ratio_after
            adjusted_dl_score_ia_dict_ratio_contribution[area_id] = [dl_score_ia_i_before_contribution,
                                                                     dl_score_ia_i_after_contribution]





    debug = 1
    # logic 2
    # one_exist_value = 1/2
    # for i in range(gt_num + 1):
    #     area_id = "id_" + str(i)
    #     area_id_front = "id_" + str(i - 1)
    #     area_id_back = "id_" + str(i + 1)
    #
    #     ia_area = IA_dict[area_id]
    #     l_ia = l_ia_set_dict[area_id]
    #     d_ia = d_ia_set_dict[area_id]
    #     d_ia_sign = d_ia_sign_set_dict[area_id]
    #     ia_len = ia_area[1] - ia_area[0] if ia_area != [] else 0
    #
    #     if i == 0:
    #         ratio_before = 0
    #         ratio_after = 1
    #         if IA_pred_group_dict[area_id] != []:
    #             dl_score_ia_i_contribution = dl_score_ia_dict[area_id]
    #         else:
    #             if VA_pred_group_dict[area_id] != []:
    #                 dl_score_ia_i_contribution = 1 / 2
    #             else:
    #                 dl_score_ia_i_contribution = 0
    #     elif i == gt_num:
    #         ratio_before = 1
    #         ratio_after = 0
    #         if IA_pred_group_dict[area_id] != []:
    #             dl_score_ia_i_contribution = dl_score_ia_dict[area_id]
    #         else:
    #             if VA_pred_group_dict[area_id_front] != []:
    #                 dl_score_ia_i_contribution = 1 / 2
    #             else:
    #                 dl_score_ia_i_contribution = 0
    #     else:
    #         if IA_pred_group_dict[area_id] != []:
    #             ratio_before = 1 / 2
    #             ratio_after = 1 / 2
    #             # if d_ia_sign > 0:
    #             #     ratio_before = (ia_len / 2 - d_ia) / ia_len
    #             #     ratio_after = (ia_len / 2 + d_ia) / ia_len
    #             # elif d_ia_sign < 0:
    #             #     ratio_before = (ia_len / 2 + d_ia) / ia_len
    #             #     ratio_after = (ia_len / 2 - d_ia) / ia_len
    #             # else:
    #             #     ratio_before = 1/2
    #             #     ratio_after = 1/2
    #             dl_score_ia_i_contribution = dl_score_ia_dict[area_id]
    #         else:
    #             # ratio_before = 1 / 2
    #             # ratio_after = 1 / 2
    #             if VA_pred_group_dict[area_id_front] != [] and VA_pred_group_dict[area_id] != []:
    #                 ratio_before = 1/2
    #                 ratio_after = 1/2
    #                 dl_score_ia_i_contribution = 1
    #             elif VA_pred_group_dict[area_id_front] == [] and VA_pred_group_dict[area_id] == []:
    #                 ratio_before = 1/2
    #                 ratio_after = 1/2
    #                 dl_score_ia_i_contribution = 0
    #             else:
    #                 if VA_pred_group_dict[area_id_front] != []:
    #                     ratio_before = 1
    #                     ratio_after = 0
    #                 else:
    #                     ratio_before = 0
    #                     ratio_after = 1
    #                 dl_score_ia_i_contribution = one_exist_value

    # logic 3
    if logic == 3:
        # one_exist_value = 1/2
        one_exist_value = 1
        for i in range(gt_num + 1):
            area_id = "id_" + str(i)
            area_id_front = "id_" + str(i - 1)
            area_id_back = "id_" + str(i + 1)

            ia_area = IA_dict[area_id]
            l_ia = l_ia_set_dict[area_id]
            d_ia = d_ia_set_dict[area_id]
            d_ia_sign = d_ia_sign_set_dict[area_id]
            ia_len = ia_area[1] - ia_area[0] if ia_area != [] else 0

            if i == 0:
                ratio_before = 0
                ratio_after = 1
                if IA_dict[area_id] == []:
                    dl_score_ia_i_contribution = 0
                else:
                    if IA_pred_group_dict[area_id] != []:
                        dl_score_ia_i_contribution = dl_score_ia_dict[area_id]
                    else:
                        if VA_pred_group_dict[area_id] != []:
                            dl_score_ia_i_contribution = 1
                        else:
                            dl_score_ia_i_contribution = 0
            elif i == gt_num:
                ratio_before = 1
                ratio_after = 0
                if IA_dict[area_id] == []:
                    dl_score_ia_i_contribution = 0
                else:
                    if IA_pred_group_dict[area_id] != []:
                        dl_score_ia_i_contribution = dl_score_ia_dict[area_id]
                    else:
                        if VA_pred_group_dict[area_id_front] != []:
                            dl_score_ia_i_contribution = 1
                        else:
                            dl_score_ia_i_contribution = 0
            else:
                if IA_pred_group_dict[area_id] != []:
                    # ratio_before = 1 / 2
                    # ratio_after = 1 / 2
                    if d_ia_sign > 0:
                        ratio_before = (ia_len / 2 - d_ia) / ia_len
                        ratio_after = (ia_len / 2 + d_ia) / ia_len
                    elif d_ia_sign < 0:
                        ratio_before = (ia_len / 2 + d_ia) / ia_len
                        ratio_after = (ia_len / 2 - d_ia) / ia_len
                    else:
                        ratio_before = 1 / 2
                        ratio_after = 1 / 2
                    dl_score_ia_i_contribution = dl_score_ia_dict[area_id]
                else:
                    # ratio_before = 1 / 2
                    # ratio_after = 1 / 2
                    if VA_pred_group_dict[area_id_front] != [] and VA_pred_group_dict[area_id] != []:
                        ratio_before = 1 / 2
                        ratio_after = 1 / 2
                        dl_score_ia_i_contribution = 2
                    elif VA_pred_group_dict[area_id_front] == [] and VA_pred_group_dict[area_id] == []:
                        ratio_before = 1 / 2
                        ratio_after = 1 / 2
                        dl_score_ia_i_contribution = 0
                    else:
                        if VA_pred_group_dict[area_id_front] != []:
                            ratio_before = 1
                            ratio_after = 0
                        else:
                            ratio_before = 0
                            ratio_after = 1
                        dl_score_ia_i_contribution = one_exist_value

            adjusted_dl_score_ia_dict[area_id] = dl_score_ia_i_contribution
            dl_score_ia_i_before_contribution = dl_score_ia_i_contribution * ratio_before
            dl_score_ia_i_after_contribution = dl_score_ia_i_contribution * ratio_after
            adjusted_dl_score_ia_dict_ratio_contribution[area_id] = [dl_score_ia_i_before_contribution,
                                                                     dl_score_ia_i_after_contribution]

    if print_msg:
        print("adjusted_dl_score_ia_dict", adjusted_dl_score_ia_dict)
        print("adjusted_dl_score_ia_dict_ratio_contribution", adjusted_dl_score_ia_dict_ratio_contribution)
    event_recall_i_list = []
    event_recall_i_dict = {}
    event_recall_i_dict_w0 = {}
    event_recall_i_dict_w1 = {}
    event_precision_i_dict = {}
    event_precision_i_dict_w0 = {}
    event_precision_i_dict_w1 = {}

    # 9th circle
    for i in range(gt_num):
        area_id = "id_" + str(i)
        if print_msg:
            print(area_id)
        area_id_back = "id_" + str(i + 1)
        recall_va_i = recall_va_dict[area_id]
        # if VA_pred_group_dict[area_id] != [] and IA_pred_group_dict[area_id] == [] and IA_pred_group_dict[area_id_back] == [] :
        #     dl_score_ia_i = 1
        #     dl_score_ia_i_back = 1
        # else:
        #     dl_score_ia_i = dl_score_ia_dict[area_id]
        #     dl_score_ia_i_back = dl_score_ia_dict[area_id_back]

        dl_score_ia_i_after_contribution = adjusted_dl_score_ia_dict_ratio_contribution[area_id][1]
        dl_score_ia_i_back_before_contribution = adjusted_dl_score_ia_dict_ratio_contribution[area_id_back][0]

        if print_msg:
            print(" dl_score_ia_i_after_contribution", dl_score_ia_i_after_contribution)
            print(" recall_va_i,", recall_va_i)
            print(" dl_score_ia_i_back_before_contribution", dl_score_ia_i_back_before_contribution)

        # event_recall_i = (1-2*parameter_eta)*recall_va_i + parameter_eta*(dl_score_ia_i_after_contribution+ dl_score_ia_i_back_before_contribution)
        if pred_case_id == 26:
            debug = 1

        # logic 1
        if logic == 1:
            event_recall_i = (1 - parameter_eta) * recall_va_i + parameter_eta * (
                        dl_score_ia_i_after_contribution + dl_score_ia_i_back_before_contribution)
            # w=0
            event_recall_i_w0 = (1 - 0) * recall_va_i + 0 * (
                        dl_score_ia_i_after_contribution + dl_score_ia_i_back_before_contribution)
            # w=1
            event_recall_i_w1 = (1 - 1) * recall_va_i + 1 * (
                        dl_score_ia_i_after_contribution + dl_score_ia_i_back_before_contribution)
            # w=1
            event_recall_i_w1 = (1 - 1) * recall_va_i + 1 * (
                        dl_score_ia_i_after_contribution + dl_score_ia_i_back_before_contribution)
        # logic 3
        if logic == 3:
            if i == 0 and IA_dict[area_id] == []:
                event_recall_i = (1 - parameter_eta) * recall_va_i + parameter_eta * 1 / 1 * (
                    dl_score_ia_i_back_before_contribution)
            elif i == gt_num - 1 and IA_dict[area_id_back] == []:
                event_recall_i = (1 - parameter_eta) * recall_va_i + parameter_eta * 1 / 1 * (
                    dl_score_ia_i_after_contribution)
            else:
                event_recall_i = (1 - parameter_eta) * recall_va_i + parameter_eta * 1 / 2 * (
                            dl_score_ia_i_after_contribution + dl_score_ia_i_back_before_contribution)
        # event_recall_i_list
        event_recall_i_dict[area_id] = event_recall_i
        event_recall_i_dict_w0[area_id] = event_recall_i_w0
        event_recall_i_dict_w1[area_id] = event_recall_i_w1

        precision_va_i = precision_va_dict[area_id]
        if print_msg:
            print(" dl_score_ia_i_after_contribution", dl_score_ia_i_after_contribution)
            print(" precision_va_i,", precision_va_i)
            print(" dl_score_ia_i_back_before_contribution", dl_score_ia_i_back_before_contribution)

        # event_precision_i = (1-2*parameter_eta)*precision_va_i + parameter_eta*(dl_score_ia_i_after_contribution+ dl_score_ia_i_back_before_contribution)

        # logic 1
        if logic == 1:
            event_precision_i = (1 - parameter_eta) * precision_va_i + parameter_eta * (
                        dl_score_ia_i_after_contribution + dl_score_ia_i_back_before_contribution)
            # w=0
            event_precision_i_w0 = (1 - 0) * precision_va_i + 0 * (
                        dl_score_ia_i_after_contribution + dl_score_ia_i_back_before_contribution)
            # w=1
            event_precision_i_w1 = (1 - 1) * precision_va_i + 1 * (
                        dl_score_ia_i_after_contribution + dl_score_ia_i_back_before_contribution)

        # logic 3
        if logic == 3:
            if i == 0 and IA_dict[area_id] == []:
                event_precision_i = (1 - parameter_eta) * precision_va_i + parameter_eta * 1 / 1 * (
                    dl_score_ia_i_back_before_contribution)
            elif i == gt_num - 1 and IA_dict[area_id_back] == []:
                event_precision_i = (1 - parameter_eta) * precision_va_i + parameter_eta * 1 / 1 * (
                    dl_score_ia_i_after_contribution)
            else:
                event_precision_i = (1 - parameter_eta) * precision_va_i + parameter_eta * 1 / 2 * (
                            dl_score_ia_i_after_contribution + dl_score_ia_i_back_before_contribution)
        event_precision_i_dict[area_id] = event_precision_i
        event_precision_i_dict_w0[area_id] = event_precision_i_w0
        event_precision_i_dict_w1[area_id] = event_precision_i_w1

    if print_msg:
        print("event_recall_i_dict", event_recall_i_dict)
        print("event_precision_i_dict", event_precision_i_dict)

    if pred_case_id == 2:
        debug = 1
    event_recall = np.array(list(event_recall_i_dict.values())).mean()
    # print(event_recall)
    event_precision = np.array(list(event_precision_i_dict.values())).mean()
    # print(event_precision)

    event_recall_w0 = np.array(list(event_recall_i_dict_w0.values())).mean()
    event_precision_w0 = np.array(list(event_precision_i_dict_w0.values())).mean()
    event_recall_w1 = np.array(list(event_recall_i_dict_w1.values())).mean()
    event_precision_w1 = np.array(list(event_precision_i_dict_w1.values())).mean()

    #
    # print("======== meata_recall")
    # print(" event_recall",event_recall)
    # print(" event_recall_w0",event_recall_w0)
    # print(" event_recall_w1",event_recall_w1)
    #
    # print(" event_recall_i_dict",event_recall_i_dict)
    # print(" event_recall_i_dict_w0",event_recall_i_dict_w0)
    # print(" event_recall_i_dict_w1",event_recall_i_dict_w1)
    # print(" adjusted_dl_score_ia_dict_ratio_contribution",adjusted_dl_score_ia_dict_ratio_contribution)

    recall_info = {
        "event_recall": event_recall,
        "event_recall_w0": event_recall_w0,
        "event_recall_w1": event_recall_w1,
        "event_recall_i_dict": event_recall_i_dict,
        "event_recall_i_dict_w0": event_recall_i_dict_w0,
        "event_recall_i_dict_w1": event_recall_i_dict_w1,
        "adjusted_dl_score_ia_dict_ratio_contribution": adjusted_dl_score_ia_dict_ratio_contribution,
    }

    meata_f1 = compute_f1_score(event_precision, event_recall)
    meata_f1_w0 = compute_f1_score(event_precision_w0, event_recall_w0)
    meata_f1_w1 = compute_f1_score(event_precision_w1, event_recall_w1)

    # gt_label_point_array = convert_interval_to_point_array(label_interval_ranges, time_series_length=window_length)
    # ont_count = np.count_nonzero(gt_label_point_array == 1)
    # anomaly_rate = ont_count / window_length
    # if anomaly_rate > 0.2:
    #     print("anomaly_rate exceeds threshold")
    #     no_random_coefficient = 1
    # else:
    #     no_random_coefficient = cal_no_random_measure_coefficient(pred_label_point_array, window_length)

    if thresh_id == 94:
        d=1
    coefficient_meata_f1 = no_random_coefficient * meata_f1
    coefficient_event_recall = no_random_coefficient * event_recall
    coefficient_event_precision = no_random_coefficient * event_precision
    print("meata_f1", meata_f1)
    print("coefficient_meata_f1", coefficient_meata_f1)

    coefficient_meata_f1_w0 = no_random_coefficient * meata_f1_w0
    coefficient_meata_f1_w1 = no_random_coefficient * meata_f1_w1

    # return event_recall,event_precision,meata_f1,no_random_meata_f1
    # return event_recall,event_precision,meata_f1,coefficient_meata_f1,VA_f_dict,VA_d_dict,coefficient_event_recall,coefficient_event_precision
    return event_recall, event_precision, meata_f1, coefficient_meata_f1, VA_f_dict, VA_d_dict, \
        coefficient_event_recall, coefficient_event_precision, \
        meata_f1_w0, coefficient_meata_f1_w0, meata_f1_w1, coefficient_meata_f1_w1, \
        recall_info

def ddl_func(x,max_area_len):
    parameter_alpha = 2
    parameter_a = 1/max_area_len**parameter_alpha
    score = parameter_a*(max_area_len-x)**parameter_alpha
    return score


def ddl_func1(x,max_area_len):
    score = -1 / max_area_len * x + 1
    score = -1 / max_area_len * x + 1
    return score

# metric in three ranges,distant range considers to both sides
def meata_v5(label_interval_ranges, IA1_VA1_IA2_relative_list, window_length, parameter_dict, pred_label_point_array,
          output=None,pred_case_id=None, max_ia_distant_length=-1,thresh_id=None,cal_mode="proportion",find_type="ts_section"):
    print_msg = False
    # print_msg = True

    # parameter_alpha = parameter_dict["parameter_alpha"]
    parameter_rate = parameter_dict["parameter_rate"]

    parameter_theta = parameter_dict["parameter_theta"]
    # parameter_w = parameter_dict["parameter_w"]
    # parameter_switch_f =parameter_dict["parameter_switch_f"]
    # parameter_switch_d = parameter_dict["parameter_switch_d"]
    VA_f_window = parameter_dict["forecasting_window"]
    VA_d_window = parameter_dict["delay_window"]

    parameter_rho = parameter_dict["parameter_rho"]
    parameter_lamda = parameter_dict["parameter_lamda"]

    parameter_w_gt = parameter_dict["parameter_w_gt"]
    parameter_w_near_ngt = parameter_dict["parameter_w_near_ngt"]
    parameter_w_near_2_ngt_gt_left = parameter_dict["parameter_w_near_2_ngt_gt_left"]
    parameter_w_near_0_ngt_gt_right = parameter_dict["parameter_w_near_0_ngt_gt_right"]

    parameter_near_r_onset = parameter_dict["parameter_near_r_onset"]
    parameter_near_r_mean = parameter_dict["parameter_near_r_mean"]
    parameter_near_l = parameter_dict["parameter_near_l"]
    parameter_distant_r_onset = parameter_dict["parameter_distant_r_onset"]
    parameter_distant_r_mean = parameter_dict["parameter_distant_r_mean"]
    parameter_distant_l = parameter_dict["parameter_distant_l"]
    parameter_near_single_side_range = parameter_dict["parameter_near_single_side_range"]
    parameter_gama = parameter_dict["parameter_gama"]
    parameter_distant_method = parameter_dict["parameter_distant_method"]
    parameter_distant_direction = parameter_dict["parameter_distant_direction"]
    parameter_ddl_parameter_strategy = parameter_dict["parameter_ddl_parameter_strategy"]

    distant_method = parameter_distant_method


    # 区域要合并，gt_before=0,需要和ngt_middle合并
    IA_dict = {}
    IA_left_dict = {}
    IA_right_dict = {}
    VA_dict = {}
    VA_f_dict = {}
    VA_gt_dict = {}
    VA_d_dict = {}

    # pred generated is continuous range a,b(len) -> [a,a+b(len)),range of [a,b] -> [a,b) -> point of [a,b-1],point of [a,b] -> [a,b+1)

    # cal every f/d of gt,split IA and VA

    # split
    gt_num = len(label_interval_ranges)
    for i, label_interval_range in enumerate(label_interval_ranges):
        # adjust now and before,add before
        # gt index:0-(gt_num-1)
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
                # IA_now_mid = (IA_dict["id_" + str(i)][0] + IA_dict["id_" + str(i)][1])/2
                IA_left_dict["id_" + str(i)] = [0, IA_dict["id_" + str(i)][1]]
                # IA_right_dict["id_" + str(i)] = [IA_now_mid, VA_f_now_start]


                # VA_dict["id_" + str(i)] = [VA_f_before_start, VA_d_before_end]
                VA_f_dict["id_" + str(i)] = [VA_f_now_start, VA_f_now_end]
                VA_gt_dict["id_" + str(i)] = [VA_gt_now_start, VA_gt_now_end]
                VA_d_dict["id_" + str(i + 1)] = [VA_d_now_start, VA_d_now_end]

                # 下一个区
                IA_dict["id_" + str(i+1)] = [VA_d_now_end, window_length]
                # IA_after_mid = (IA_dict["id_" + str(i+1)][0] + IA_dict["id_" + str(i+1)][1])/2
                IA_right_dict["id_" + str(i+1)] = [VA_d_now_end, IA_dict["id_" + str(i+1)][1]]
                # IA_right_dict["id_" + str(i+1)] = [IA_after_mid, window_length]

                if parameter_w_near_2_ngt_gt_left <= 0:
                    IA_dict["id_" + str(i)] = [0, VA_gt_now_start]
                    IA_left_dict["id_" + str(i)] = [0, IA_dict["id_" + str(i)][1]]

                    VA_f_dict["id_" + str(i)] = [VA_gt_now_start, VA_gt_now_start]

                if parameter_w_near_0_ngt_gt_right <= 0:
                    IA_dict["id_" + str(i)] = [0, VA_f_now_start]
                    IA_left_dict["id_" + str(i)] = [0, IA_dict["id_" + str(i)][1]]

                    VA_d_dict["id_" + str(i + 1)] = [VA_gt_now_end, VA_gt_now_end]

                    # 下一个区
                    IA_dict["id_" + str(i + 1)] = [VA_gt_now_end, window_length]
                    # IA_after_mid = (IA_dict["id_" + str(i+1)][0] + IA_dict["id_" + str(i+1)][1])/2
                    IA_right_dict["id_" + str(i + 1)] = [VA_gt_now_end, IA_dict["id_" + str(i + 1)][1]]
                    # IA_right_dict["id_" + str(i+1)] = [IA_after_mid, window_length]



            else:
                VA_f_now_start = max(VA_gt_now_start - parameter_near_single_side_range, 0)
                VA_f_now_end = VA_gt_now_start
                VA_gt_after_start, VA_gt_after_end = label_interval_ranges[i+1]
                VA_d_now_start = VA_gt_now_end
                VA_d_now_end = min(VA_gt_now_end + parameter_near_single_side_range, VA_gt_after_start)

                IA_dict["id_" + str(i)] = [0, VA_f_now_start]
                # IA_now_mid = (IA_dict["id_" + str(i)][0] + IA_dict["id_" + str(i)][1])/2
                IA_left_dict["id_" + str(i)] = [0, IA_dict["id_" + str(i)][1]]
                # IA_right_dict["id_" + str(i)] = [IA_now_mid, VA_f_now_start]


                # VA_dict["id_" + str(i)] = [VA_f_before_start, VA_d_before_end]
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


            # VA_dict["id_" + str(i)] = [VA_f_before_start, VA_d_before_end]
            VA_f_dict["id_" + str(i)] = [VA_f_now_start, VA_f_now_end]
            VA_gt_dict["id_" + str(i)] = [VA_gt_now_start, VA_gt_now_end]
            VA_d_dict["id_" + str(i + 1)] = [VA_d_now_start, VA_d_now_end]

            # 下一个区
            IA_dict["id_" + str(i + 1)] = [VA_d_now_end, window_length]
            # IA_after_mid = (IA_dict["id_" + str(i + 1)][0] + IA_dict["id_" + str(i + 1)][1]) / 2
            # IA_left_dict["id_" + str(i + 1)] = [VA_d_now_end, IA_after_mid]
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

                # 下一个区
                IA_dict["id_" + str(i + 1)] = [VA_gt_now_end, window_length]
                # IA_after_mid = (IA_dict["id_" + str(i + 1)][0] + IA_dict["id_" + str(i + 1)][1]) / 2
                # IA_left_dict["id_" + str(i + 1)] = [VA_d_now_end, IA_after_mid]
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

            # VA_dict["id_" + str(i)] = [VA_f_before_start, VA_d_before_end]
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
        # print area dict
        # # print("split area")
        print("IA_dict", IA_dict)
        print("IA_left_dict", IA_left_dict)
        print("IA_right_dict", IA_right_dict)
        # print("VA_dict",VA_dict)
        print("VA_f_dict", VA_f_dict)
        print("VA_gt_dict", VA_gt_dict)
        print("VA_d_dict", VA_d_dict)

    # def dict_filer(dict_data: dict):
    #     for i, (area_id, area_range) in enumerate(dict_data.items()):
    #         if area_range != []:
    #             dict_data[area_id] = [] if area_range[0] == area_range[1] else area_range
    #     return dict_data
    #
    # # second circle
    # IA_dict = dict_filer(IA_dict)
    # # VA_dict = dict_filer(VA_dict)
    # VA_f_dict = dict_filer(VA_f_dict)
    # VA_gt_dict = dict_filer(VA_gt_dict)
    # VA_d_dict = dict_filer(VA_d_dict)

    if print_msg:
        print()
        # print("dict adjust +++++++++++")
        # print("IA_dict", IA_dict)
        # # print("VA_dict",VA_dict)
        # print("VA_f_dict", VA_f_dict)
        # print("VA_gt_dict", VA_gt_dict)
        # print("VA_d_dict", VA_d_dict)

        # split pred to basic interval
        print("split_line_set", split_line_set)

    # third circle
    split_IA1_VA1_IA2_relative_list = split_intervals(IA1_VA1_IA2_relative_list, split_line_set)
    if print_msg:
        print()
        print("split_IA1_VA1_IA2_relative_list",split_IA1_VA1_IA2_relative_list)

    # forecasting_len=10
    # delay_len=10
    IA_pred_group_dict = {}
    IA_left_pred_group_dict = {}
    IA_right_pred_group_dict = {}
    VA_pred_group_dict = {}
    VA_f_pred_group_dict = {}
    VA_gt_pred_group_dict = {}
    VA_d_pred_group_dict = {}


    # VA_gt3_pred_group_dict = {}
    func_dict = {}
    func_dict_copy = {}

    # "id_"+str(label_interval_num)
    for i in range(gt_num):
        IA_pred_group_dict["id_" + str(i)] = []
        VA_pred_group_dict["id_" + str(i)] = []
        VA_f_pred_group_dict["id_" + str(i)] = []
        VA_gt_pred_group_dict["id_" + str(i)] = []
        VA_d_pred_group_dict["id_" + str(i+1)] = [] # VA_d要加1
        if distant_method == 2:
            IA_left_pred_group_dict["id_" + str(i)] = []
            IA_right_pred_group_dict["id_" + str(i+1)] = []

    IA_pred_group_dict["id_" + str(gt_num)] = []

    # 4th circle
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
        # VA
        # for j, (area_id, area) in enumerate(VA_dict.items()):
        #     if pred_in_area(basic_interval, area):
        #         if area_id not in VA_pred_group_dict.keys():
        #             VA_pred_group_dict[area_id] = [basic_interval]
        #         else:
        #             VA_pred_group_dict[area_id].append(basic_interval)
    if print_msg:
        # print pred dict
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

    ddl_method = 2

    # cal label_function_gt
    for i, (area_id, area) in enumerate(VA_gt_dict.items()):
        area_id_back = "id_" + (i+1).__str__()
        VA_gt_len = area[1] - area[0]
        # if print_msg:
        #     print("VA_gt_len", VA_gt_len)

        # parameter_b= cal_power_function_parameter_b(parameter_gama,parameter_w_gt,parameter_w_near_ngt,VA_gt_len)
        parameter_w_gt_recall = 1
        # parameter_w_gt_recall = 1/2
        parameter_b_recall= cal_power_function_parameter_b(parameter_gama,parameter_w_gt_recall,0,VA_gt_len)
        parameter_w_gt_precision = 1/2

        parameter_b_precision= cal_power_function_parameter_b(parameter_gama,parameter_w_gt_precision,0,VA_gt_len)

        # area, parameter_b, parameter_gama, parameter_w_gt = 0, area_end = None
        s_a_gt_precision = cal_integral_in_range_gt_power_func(area, parameter_b_precision, parameter_gama, \
                                                     parameter_w_gt=parameter_w_gt_precision, \
                                                     area_end=area[1])

        s_a_gt_recall = cal_integral_in_range_gt_power_func(area, parameter_b_recall, parameter_gama, \
                                                     parameter_w_gt=parameter_w_gt_recall, \
                                                     area_end=area[1])
        # VA_gt_dict {'id_0': [3914, 3925], 'id_1': [5259, 5269], 'id_2': [9513, 9527]}
        # VA_gt_pred_group_dict {'id_0': [[3918, 3920]], 'id_1': [[5264, 5266]], 'id_2': [[9520, 9521]]}
        # s_a_gt_recall_gt = cal_integral_in_range_gt_power_func([9513, 9527], parameter_b_recall, parameter_gama, \
        #                                              parameter_w_gt=parameter_w_gt_recall, \
        #                                              area_end=9527)
        # s_a_gt_recall1 = cal_integral_in_range_gt_power_func([9513, 9527], parameter_b_recall, parameter_gama, \
        #                                              parameter_w_gt=parameter_w_gt_recall, \
        #                                              area_end=9527)
        # s_a_gt_recall2 = cal_integral_in_range_gt_power_func([9520, 9521], parameter_b_recall, parameter_gama, \
        #                                              parameter_w_gt=parameter_w_gt_recall, \
        #                                              area_end=9527)
        # s_a_gt_recall3 = cal_integral_in_range_gt_power_func([9513, 9520], parameter_b_recall, parameter_gama, \
        #                                              parameter_w_gt=parameter_w_gt_recall, \
        #                                              area_end=9527)
        # recall_rate1 = s_a_gt_recall1/s_a_gt_recall_gt
        # recall_rate2 = s_a_gt_recall2/s_a_gt_recall_gt
        # recall_rate3 = s_a_gt_recall3/s_a_gt_recall_gt
        # print("")
        # print("debug recall_rate")
        # print("s_a_gt_recall_gt",s_a_gt_recall_gt)
        # print("s_a_gt_recall1",s_a_gt_recall1)
        # print("recall_rate1",recall_rate1)
        # print("s_a_gt_recall2",s_a_gt_recall2)zz
        # print("recall_rate2",recall_rate2)
        # print("s_a_gt_recall3",s_a_gt_recall3)
        # print("recall_rate3",recall_rate3)

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





        # plot_func_multi_paper(func_dict,window_length)

        # s_a_gt1 = cal_integral_in_range_gt_power_func([9523, 9527], parameter_b, parameter_gama, \
        #                                              parameter_w_gt, \
        #                                              area_end=area[1])
        #
        # s_a_gt2 = cal_integral_in_range_gt_power_func([9520, 9521], parameter_b, parameter_gama, \
        #                                               parameter_w_gt, \
        #                                               area_end=area[1])
        # [3914, 3925]
        # 8.555555555555555
        # 3918, 3920
        # 1.5335169880624426
        # 0.17924224535794783
        # [3923, 3925]
        # 0.1567028013309005
        # b=0.002754820936639119
        # parameter_gama=2

        # [9513, 9527]
        #
        # [9520, 9521]
        #
        # s_a_gt_rate1 = s_a_gt1/s_a_gt
        # s_a_gt_rate2 = s_a_gt2/s_a_gt

        # parameter_theta = 0.5
        # f
        VA_f_area = VA_f_dict[area_id]
        VA_f_len = VA_f_area[1] - VA_f_area[0]
        VA_f_start = VA_f_area[0]
        VA_f_end = VA_f_area[1]
        # parameter_beta_left = 1
        # parameter_a_left = (1 - parameter_w_gt_precision) / parameter_near_single_side_range
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
        # s_gt_after = cal_integral_gt_after_ia_power_func(VA_d_area,
        #                                                             parameter_a_right,
        #                                                             parameter_beta_right,
        #                                                             parameter_w_gt,
        #                                                             area_start=VA_d_start)


        s_rate = s_a_gt_precision/(s_gt_before*2+s_a_gt_precision)
        # if s_rate < parameter_theta:
        if False:
            s_gt_before_target = s_gt_after_target = 1/2*s_a_gt_precision*(1/parameter_theta - 1)
            parameter_beta1 = (1-parameter_w_gt_precision)*VA_f_len / s_gt_before_target - 1
            parameter_a1 = (1-parameter_w_gt_precision)/pow(VA_f_len, parameter_beta1)

            # func_dict[area_id + "_f_a"] = {
            func_dict_copy[area_id + "_f_a"] = {
                "area": VA_f_area,
                "area_start": VA_f_area[0],
                "area_end": VA_f_area[1],
                "area_s": s_gt_before_target,
                "parameter_a": parameter_a1,
                "parameter_beta": parameter_beta1,
                "func": "ia_before_func",
                # "func": "va_f_reverse_func",
                "x_tp": None,
                "gt_len": None,
                "area_len": VA_f_len,
                # "greater_than_th": greater_than_th
                "parameter_w_gt": parameter_w_gt_precision,
                "parameter_w_near_ngt": parameter_w_near_ngt
            }


            # parameter_beta2 = (1 - parameter_w_gt) * VA_d_len / ( (1 - parameter_w_gt) * VA_d_len-s_gt_after_target) - 1
            # parameter_a2 = -(1 - parameter_w_gt) / pow(VA_d_len, parameter_beta2)

            # func_dict[area_id + "_d_a"] = {
            func_dict_copy[area_id + "_d_a"] = {
                "area": VA_d_area,
                "area_start": VA_d_area[0],
                "area_end": VA_d_area[1],
                "area_s": s_gt_after_target,
                "parameter_a": parameter_a1,
                "parameter_beta": parameter_beta1,
                "func": "ia_after_func",
                # "func": "va_f_reverse_func",
                "x_tp": None,
                "gt_len": None,
                "area_len": VA_d_len,
                # "greater_than_th": greater_than_th
                "parameter_w_gt": parameter_w_gt_precision,
                "parameter_w_near_ngt": parameter_w_near_ngt
            }
        else:
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

        # func_dict[area_id + "_gt_precision"]["area_s"] = s_a_gt_precision

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
            # the length of fitst and last IA could be set a fixed value,min(一个周期的长度,ori len),or min(mean len,ori len)
            d_ia_before_mean_set_sum+= abs(VA_f_end - (basic_interval[1] + basic_interval[0])/2)
            # onset只看第一个
            # d_ia_before_onset_set_sum+= abs(VA_f_end - basic_interval[1])
            if interval_idx == VA_f_i_pred_group_len-1:
                d_ia_before_onset_set_sum += abs(VA_f_end - basic_interval[1])


        l_ia_before_set_dict[area_id] = l_ia_before_set_sum
        d_ia_before_mean =  d_ia_before_mean_set_sum/ia_set_num if ia_set_num != 0 else 0
        d_ia_before_mean_set_dict[area_id] = d_ia_before_mean
        d_ia_before_onset =  d_ia_before_onset_set_sum/ia_set_num if ia_set_num != 0 else 0
        d_ia_before_onset_set_dict[area_id] = d_ia_before_onset

        # dl_score_ia_before = parameter_near_r_mean*(-1/parameter_near_single_side_range * d_ia_before_mean +1) + \
        #                    parameter_near_r_onset*(-1/parameter_near_single_side_range * d_ia_before_onset +1) + \
        #                      (1-parameter_near_r_mean-parameter_near_r_onset)*(-1/parameter_near_single_side_range * l_ia_before_set_sum +1) \
        #                 if parameter_near_single_side_range != 0 else 0
        if parameter_ddl_parameter_strategy == 1:
            score_ia_near_before_mean_d = ddl_func1(d_ia_before_mean,parameter_near_single_side_range) if parameter_near_single_side_range !=0 else 1
            score_ia_near_before_onset_d = ddl_func1(d_ia_before_onset,parameter_near_single_side_range) if parameter_near_single_side_range !=0 else 1
        elif parameter_ddl_parameter_strategy == 2:
            score_ia_near_before_mean_d = ddl_func(d_ia_before_mean,parameter_near_single_side_range) if parameter_near_single_side_range !=0 else 1
            score_ia_near_before_onset_d = ddl_func(d_ia_before_onset,parameter_near_single_side_range) if parameter_near_single_side_range !=0 else 1
        else:
            score_ia_near_before_mean_d = ddl_func1(d_ia_before_mean,parameter_near_single_side_range) if parameter_near_single_side_range !=0 else 1
            score_ia_near_before_onset_d = ddl_func1(d_ia_before_onset,parameter_near_single_side_range) if parameter_near_single_side_range !=0 else 1
        # 惩罚越大，曲线越陡
        score_ia_near_before_l = ddl_func1(l_ia_before_set_sum, parameter_near_single_side_range) if parameter_near_single_side_range !=0 else 1
        if ddl_method == 1:
            dl_score_ia_before = (parameter_near_r_mean*score_ia_near_before_mean_d + \
                               parameter_near_r_onset*score_ia_near_before_onset_d)*score_ia_near_before_l
        else:
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
            # the length of fitst and last IA could be set a fixed value,min(一个周期的长度,ori len),or min(mean len,ori len)
            d_ia_after_mean_set_sum += abs((basic_interval[1] + basic_interval[0]) / 2 - VA_d_start)
            # d_ia_after_onset_set_sum = abs(VA_d_start - basic_interval[0])
            # onset应该只加第一个
            if interval_idx == 0:
                d_ia_after_onset_set_sum = abs(VA_d_start - basic_interval[0])

        l_ia_after_set_dict[area_id_back] = l_ia_after_set_sum
        d_ia_after_mean = d_ia_after_mean_set_sum / ia_set_num if ia_set_num != 0 else 0
        d_ia_after_mean_set_dict[area_id_back] = d_ia_after_mean
        d_ia_after_onset = d_ia_after_onset_set_sum / ia_set_num if ia_set_num != 0 else 0
        d_ia_after_onset_set_dict[area_id_back] = d_ia_after_onset

        # dl_score_ia_after = parameter_near_r_mean * (-1 / parameter_near_single_side_range * d_ia_after_mean + 1) + \
        #                      parameter_near_r_onset * (-1 / parameter_near_single_side_range * d_ia_after_onset + 1) + \
        #                      (1 - parameter_near_r_mean - parameter_near_r_onset) * (
        #                                  -1 / parameter_near_single_side_range * l_ia_after_set_sum + 1) \
        #     if parameter_near_single_side_range != 0 else 0

        if parameter_ddl_parameter_strategy == 1:
            score_ia_near_after_mean_d = ddl_func1(d_ia_after_mean,parameter_near_single_side_range) if parameter_near_single_side_range !=0 else 1
            score_ia_near_after_onset_d = ddl_func1(d_ia_after_onset,parameter_near_single_side_range) if parameter_near_single_side_range !=0 else 1
        elif parameter_ddl_parameter_strategy == 2:
            score_ia_near_after_mean_d = ddl_func(d_ia_after_mean,parameter_near_single_side_range) if parameter_near_single_side_range !=0 else 1
            score_ia_near_after_onset_d = ddl_func(d_ia_after_onset,parameter_near_single_side_range) if parameter_near_single_side_range !=0 else 1
        else:
            score_ia_near_after_mean_d = ddl_func1(d_ia_after_mean,parameter_near_single_side_range) if parameter_near_single_side_range !=0 else 1
            score_ia_near_after_onset_d = ddl_func1(d_ia_after_onset,parameter_near_single_side_range) if parameter_near_single_side_range !=0 else 1
        score_ia_near_after_l = ddl_func1(l_ia_after_set_sum, parameter_near_single_side_range) if parameter_near_single_side_range !=0 else 1


        if thresh_id == 50:
            d = 1
        if ddl_method == 1:
            dl_score_ia_after = (parameter_near_r_mean*score_ia_near_after_mean_d + \
                               parameter_near_r_onset*score_ia_near_after_onset_d)*score_ia_near_after_l
        else:
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

    # distant ia method1
    # max_ia_distant_length = 0
    ddl_score_ia_distant_dict = {}
    if distant_method ==1:

        if print_msg:
            print()
            print("max_ia_distant_length", max_ia_distant_length)

        l_ia_distant_set_dict = {}
        d_ia_distant_mean_set_dict = {}
        d_ia_distant_onset_set_dict = {}

        # 5th circle
        for i, (area_id, basic_interval_list) in enumerate(IA_pred_group_dict.items()):
            # if basic_interval_list == []:
            #     ddl_score_ia_distant_dict[area_id] = 0
            #     l_ia_distant_set_dict[area_id] = 0
            #     d_ia_distant_onset_set_dict[area_id] = 0  # only when basic_interval_list ！= [],the value are used,so set value to 0 have boe problem
            #     d_ia_distant_mean_set_dict[area_id] = 0  # only when basic_interval_list ！= [],the value are used,so set value to 0 have boe problem
            #     ddl_score_ia_distant_dict[area_id] = 1
            #     continue
            l_ia_distant_set_sum = 0
            d_ia_distant_mean_set_sum = 0
            d_ia_distant_onset_set_sum = 0
            ia_set_num = len(basic_interval_list)

            IA_mid = (IA_dict[area_id][0] + IA_dict[area_id][1]) / 2
            IA_len = IA_dict[area_id][1] - IA_dict[area_id][0]
            IA_start = IA_dict[area_id][0]
            IA_end = IA_dict[area_id][1]
            
            # for basic_interval in basic_interval_list:
            for interval_idx, basic_interval in enumerate(basic_interval_list):
                # the length of fitst and last IA could be set a fixed value,min(一个周期的长度,ori len),or min(mean len,ori len)
                basic_interval_mid = (basic_interval[1] + basic_interval[0]) / 2
                if parameter_distant_direction == "both":
                    l_ia_distant_set_sum += basic_interval[1] - basic_interval[0]
                    if i == 0:
                        base_position = IA_dict[area_id][1]
                        d_ia_distant_mean_set_sum += abs(basic_interval_mid - base_position)
                        # d_ia_set_sum+= abs((basic_interval[1] + basic_interval[0])/2 -base_position)
                    # last id of IA is label_interval_num,not label_interval_num -1
                    elif i == gt_num:
                        base_position = IA_dict[area_id][0]
                        d_ia_distant_mean_set_sum += abs(basic_interval_mid - base_position)
                    else:
                        # base_position = IA_mid
                        d_ia_distant_mean_set_sum += min(abs(basic_interval_mid - IA_dict[area_id][0]),\
                                                         abs(basic_interval_mid - IA_dict[area_id][1]))
                    # d_ia_set_sum+= abs((basic_interval[1] + basic_interval[0])/2 -IA_mid)
                    # d_ia_distant_mean_set_sum += abs((basic_interval[1] + basic_interval[0]) / 2 - base_position)

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
            # d_ia_distant_onset = d_ia_distant_onset_set_sum / ia_set_num if ia_set_num != 0 else 0
            d_ia_distant_onset = d_ia_distant_onset_set_sum
            d_ia_distant_onset_set_dict[area_id] = d_ia_distant_onset

            # dl_score_ia_distant = 1
            #
            # if l_ia_distant_set_sum != 0:
            #     dl_score_ia_distant = parameter_distant_r_mean * (-1 / max_ia_distant_length * d_ia_distant_mean + 1) + \
            #                 parameter_distant_r_onset * (-1 / max_ia_distant_length * d_ia_distant_onset + 1) + \
            #                 (1 - parameter_distant_r_mean - parameter_distant_r_onset) * (-1 / max_ia_distant_length * l_ia_distant_set_sum + 1) \
            #         if max_ia_distant_length != 0 else 0
            # if l_ia_distant_set_sum != 0:
            
            score_ia_distant_mean_d = ddl_func1(d_ia_distant_mean,max_ia_distant_length) if max_ia_distant_length !=0 else 1
            score_ia_distant_onset_d = ddl_func1(d_ia_distant_onset,max_ia_distant_length) if max_ia_distant_length !=0 else 1
            if parameter_ddl_parameter_strategy == 1:
                score_ia_distant_l = ddl_func1(l_ia_distant_set_sum, max_ia_distant_length) if max_ia_distant_length !=0 else 1
            elif parameter_ddl_parameter_strategy == 2:
                score_ia_distant_l = ddl_func(l_ia_distant_set_sum, max_ia_distant_length) if max_ia_distant_length !=0 else 1
            else:
                score_ia_distant_l = ddl_func(l_ia_distant_set_sum, max_ia_distant_length) if max_ia_distant_length !=0 else 1

            if ddl_method == 1:
                dl_score_ia_distant = (parameter_distant_r_mean * score_ia_distant_mean_d + \
                                              parameter_distant_r_onset * score_ia_distant_onset_d) * score_ia_distant_l
                # if max_ia_distant_length_half != 0 else 0
            else:
                dl_score_ia_distant = parameter_distant_r_mean * score_ia_distant_mean_d * \
                                             parameter_distant_r_onset * score_ia_distant_onset_d * \
                                             parameter_distant_l * score_ia_distant_l

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
            # if basic_interval_list == []:
            #     ddl_score_ia_distant_dict[area_id] = 0
            #     l_ia_distant_set_dict[area_id] = 0
            #     d_ia_distant_onset_set_dict[area_id] = 0  # only when basic_interval_list ！= [],the value are used,so set value to 0 have boe problem
            #     d_ia_distant_mean_set_dict[area_id] = 0  # only when basic_interval_list ！= [],the value are used,so set value to 0 have boe problem
            #     ddl_score_ia_distant_dict[area_id] = 1
            #     continue
            l_ia_distant_set_sum = 0
            d_ia_distant_mean_set_sum = 0
            d_ia_distant_onset_set_sum = 0
            ia_set_num = len(basic_interval_list)

            IA_mid = (IA_dict[area_id][0] + IA_dict[area_id][1]) / 2
            IA_len = IA_dict[area_id][1] - IA_dict[area_id][0]
            IA_start = IA_dict[area_id][0]
            IA_end = IA_dict[area_id][1]

            # for basic_interval in basic_interval_list:
            for interval_idx, basic_interval in enumerate(basic_interval_list):
                l_ia_distant_set_sum += basic_interval[1] - basic_interval[0]
                # the length of fitst and last IA could be set a fixed value,min(一个周期的长度,ori len),or min(mean len,ori len)
                if i == 0:
                    base_position = IA_dict[area_id][1]
                    d_ia_distant_mean_set_sum += abs((basic_interval[1] + basic_interval[0]) / 2 - base_position)
                    # d_ia_set_sum+= abs((basic_interval[1] + basic_interval[0])/2 -base_position)
                # last id of IA is label_interval_num,not label_interval_num -1
                elif i == gt_num:
                    base_position = IA_dict[area_id][0]
                    d_ia_distant_mean_set_sum += abs((basic_interval[1] + basic_interval[0]) / 2 - base_position)
                else:
                    # base_position = IA_mid
                    basic_interval_mid = (basic_interval[1] + basic_interval[0]) / 2
                    d_ia_distant_mean_set_sum += min(abs(basic_interval_mid - IA_dict[area_id][0]), \
                                                     abs(basic_interval_mid - IA_dict[area_id][1]))
                # d_ia_set_sum+= abs((basic_interval[1] + basic_interval[0])/2 -IA_mid)
                # d_ia_distant_mean_set_sum += abs((basic_interval[1] + basic_interval[0]) / 2 - base_position)

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

                # if i == 0:
                #     distant_onset_distance = abs(IA_end-basic_interval[1])
                # elif i == gt_num:
                #     distant_onset_distance = abs(basic_interval[0]-IA_start)
                # else:
                #     distant_onset_distance = min(abs(basic_interval[0]-IA_start),abs(IA_end-basic_interval[1]))
                # d_ia_distant_onset_set_sum += distant_onset_distance
                # i =0 or i = label_interval_num,only have positive distance

            l_ia_distant_set_dict[area_id] = l_ia_distant_set_sum
            d_ia_distant_mean = d_ia_distant_mean_set_sum / ia_set_num if ia_set_num != 0 else 0
            d_ia_distant_mean_set_dict[area_id] = d_ia_distant_mean
            # d_ia_distant_onset = d_ia_distant_onset_set_sum / ia_set_num if ia_set_num != 0 else 0
            d_ia_distant_onset = d_ia_distant_onset_set_sum
            d_ia_distant_onset_set_dict[area_id] = d_ia_distant_onset

            # dl_score_ia_distant = 1
            #
            # if l_ia_distant_set_sum != 0:
            #     dl_score_ia_distant = parameter_distant_r_mean * (-1 / max_ia_distant_length * d_ia_distant_mean + 1) + \
            #                 parameter_distant_r_onset * (-1 / max_ia_distant_length * d_ia_distant_onset + 1) + \
            #                 (1 - parameter_distant_r_mean - parameter_distant_r_onset) * (-1 / max_ia_distant_length * l_ia_distant_set_sum + 1) \
            #         if max_ia_distant_length != 0 else 0
            # if l_ia_distant_set_sum != 0:

            score_ia_distant_mean_d = ddl_func1(d_ia_distant_mean,
                                                max_ia_distant_length) if max_ia_distant_length != 0 else 1
            score_ia_distant_onset_d = ddl_func1(d_ia_distant_onset,
                                                 max_ia_distant_length) if max_ia_distant_length != 0 else 1
            if parameter_ddl_parameter_strategy == 1:
                score_ia_distant_l = ddl_func1(l_ia_distant_set_sum,
                                               max_ia_distant_length) if max_ia_distant_length != 0 else 1
            elif parameter_ddl_parameter_strategy == 2:
                score_ia_distant_l = ddl_func(l_ia_distant_set_sum,
                                              max_ia_distant_length) if max_ia_distant_length != 0 else 1
            else:
                score_ia_distant_l = ddl_func(l_ia_distant_set_sum,
                                              max_ia_distant_length) if max_ia_distant_length != 0 else 1

            if ddl_method == 1:
                dl_score_ia_distant = (parameter_distant_r_mean * score_ia_distant_mean_d + \
                                       parameter_distant_r_onset * score_ia_distant_onset_d) * score_ia_distant_l
                # if max_ia_distant_length_half != 0 else 0
            else:
                dl_score_ia_distant = parameter_distant_r_mean * score_ia_distant_mean_d * \
                                      parameter_distant_r_onset * score_ia_distant_onset_d * \
                                      parameter_distant_l * score_ia_distant_l

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
            VA_gt_len = area[1] - area[0]
            # if print_msg:
            #     print("VA_gt_len", VA_gt_len)


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

            # IA_f_i_pred_group_len = len(IA_f_i_pred_group)
            for interval_idx, basic_interval in enumerate(IA_f_i_pred_group):
                l_ia_distant_before_set_sum += basic_interval[1] - basic_interval[0]
                # the length of fitst and last IA could be set a fixed value,min(一个周期的长度,ori len),or min(mean len,ori len)

                d_ia_distant_before_mean_set_sum += abs(IA_f_end - (basic_interval[1] + basic_interval[0]) / 2)
                # onset只看第一个
                # d_ia_before_onset_set_sum+= abs(VA_f_end - basic_interval[1])
                if interval_idx == ia_distant_before_set_num - 1:
                    d_ia_distant_before_onset_set_sum += abs(IA_f_end - basic_interval[1])

            l_ia_distant_before_set_dict[area_id] = l_ia_distant_before_set_sum
            d_ia_distant_before_mean = d_ia_distant_before_mean_set_sum / ia_distant_before_set_num if ia_distant_before_set_num != 0 else 0
            d_ia_distant_before_mean_set_dict[area_id] = d_ia_distant_before_mean
            # d_ia_distant_before_onset = d_ia_distant_before_onset_set_sum / ia_distant_before_set_num if ia_distant_before_set_num != 0 else 0
            d_ia_distant_before_onset = d_ia_distant_before_onset_set_sum
            d_ia_distant_before_onset_set_dict[area_id] = d_ia_distant_before_onset

            # score_ia_distant_before_mean_d = parameter_distant_r_mean * (-1 / max_ia_distant_length * d_ia_distant_before_mean + 1)
            # score_ia_distant_before_onset_d = parameter_distant_r_onset * (-1 / max_ia_distant_length * d_ia_distant_before_onset + 1)
            # score_ia_distant_before_l = (1 - parameter_distant_r_mean - parameter_distant_r_onset) * (-1 / max_ia_distant_length * l_ia_distant_before_set_sum + 1)


            # score_ia_distant_before_mean_d = parameter_distant_r_mean * ddl_func(d_ia_distant_before_mean,max_ia_distant_length)
            # score_ia_distant_before_onset_d = parameter_distant_r_onset * ddl_func(d_ia_distant_before_onset,max_ia_distant_length)
            # score_ia_distant_before_l = (1 - parameter_distant_r_mean - parameter_distant_r_onset) * ddl_func(l_ia_distant_before_set_sum,max_ia_distant_length)

            score_ia_distant_before_mean_d = ddl_func1(d_ia_distant_before_mean,
                                                                                 max_ia_distant_length) if max_ia_distant_length !=0 else 1
            score_ia_distant_before_onset_d = ddl_func1(d_ia_distant_before_onset,
                                                                                   max_ia_distant_length) if max_ia_distant_length !=0 else 1
            score_ia_distant_before_l = ddl_func1(l_ia_distant_before_set_sum, IA_f_len) if IA_f_len !=0 else 1
            # score_ia_distant_before_l = ddl_func(l_ia_distant_before_set_sum, IA_f_len) if IA_f_len !=0 else 1

            # score_ia_distant_before_l = (IA_f_len-l_ia_distant_before_set_sum)/IA_f_len




            # dl_score_ia_distant_before = parameter_distant_r_mean * score_ia_distant_before_mean_d + \
            #              parameter_distant_r_onset * score_ia_distant_before_onset_d + \
            #              (1 - parameter_distant_r_mean - parameter_distant_r_onset) * score_ia_distant_before_l
            if ddl_method == 1:
                dl_score_ia_distant_before = (parameter_distant_r_mean * score_ia_distant_before_mean_d + \
                             parameter_distant_r_onset * score_ia_distant_before_onset_d) * score_ia_distant_before_l
                # if max_ia_distant_length_half != 0 else 0
            else:
                dl_score_ia_distant_before = parameter_distant_r_mean*score_ia_distant_before_mean_d * \
                                             parameter_distant_r_onset*score_ia_distant_before_onset_d * \
                                             parameter_distant_l*score_ia_distant_before_l

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
                # the length of fitst and last IA could be set a fixed value,min(一个周期的长度,ori len),or min(mean len,ori len)
                d_ia_distant_after_mean_set_sum += abs((basic_interval[1] + basic_interval[0]) / 2 - IA_d_start)
                # d_ia_after_onset_set_sum = abs(VA_d_start - basic_interval[0])
                # onset应该只加第一个
                if interval_idx == 0:
                    d_ia_distant_after_onset_set_sum = abs(IA_d_start - basic_interval[0])

            l_ia_distant_after_set_dict[area_id_back] = l_ia_distant_after_set_sum
            d_ia_distant_after_mean = d_ia_distant_after_mean_set_sum / ia_distant_after_set_num if ia_distant_after_set_num != 0 else 0
            d_ia_distant_after_mean_set_dict[area_id_back] = d_ia_distant_after_mean
            # d_ia_distant_after_onset = d_ia_distant_after_onset_set_sum / ia_distant_after_set_num if ia_distant_after_set_num != 0 else 0
            d_ia_distant_after_onset = d_ia_distant_after_onset_set_sum
            d_ia_distant_after_onset_set_dict[area_id_back] = d_ia_distant_after_onset

            # score_ia_distant_after_mean_d = parameter_distant_r_mean * (-1 / max_ia_distant_length * d_ia_distant_after_mean + 1)
            # score_ia_distant_after_onset_d = parameter_distant_r_onset * (-1 / max_ia_distant_length * d_ia_distant_after_onset + 1)
            # score_ia_distant_after_l = (1 - parameter_distant_r_mean - parameter_distant_r_onset) * (-1 / max_ia_distant_length * l_ia_distant_after_set_sum + 1)

            # ddl_func(d_ia_distant_before_mean, max_ia_distant_length)
            # score_ia_distant_after_mean_d = parameter_distant_r_mean * ddl_func(d_ia_distant_after_mean, max_ia_distant_length)
            # score_ia_distant_after_onset_d = parameter_distant_r_onset * ddl_func(d_ia_distant_after_onset, max_ia_distant_length)
            # score_ia_distant_after_l = (1 - parameter_distant_r_mean - parameter_distant_r_onset) * ddl_func(l_ia_distant_after_set_sum, max_ia_distant_length)

            score_ia_distant_after_mean_d = ddl_func1(d_ia_distant_after_mean, max_ia_distant_length) if max_ia_distant_length !=0 else 1
            score_ia_distant_after_onset_d = ddl_func1(d_ia_distant_after_onset, max_ia_distant_length) if max_ia_distant_length !=0 else 1
            # try:
            score_ia_distant_after_l = ddl_func1(l_ia_distant_after_set_sum, IA_d_len) if IA_d_len !=0 else 1
            # score_ia_distant_after_l = l_ia_distant_after_set_sum/IA_d_len
            # except:
            #     d=1



            # dl_score_ia_distant_after = parameter_distant_r_mean * score_ia_distant_after_mean_d + \
            #                     parameter_distant_r_onset * score_ia_distant_after_onset_d + \
            #                     (1 - parameter_distant_r_mean - parameter_distant_r_onset) * score_ia_distant_after_l \
            if ddl_method == 1:
                dl_score_ia_distant_after = (parameter_distant_r_mean * score_ia_distant_after_mean_d + \
                                    parameter_distant_r_onset * score_ia_distant_after_onset_d) * score_ia_distant_after_l
                # if max_ia_distant_length_half != 0 else 1
            else:
                dl_score_ia_distant_after = parameter_distant_r_mean*score_ia_distant_after_mean_d * \
                                            parameter_distant_r_onset*score_ia_distant_after_onset_d * \
                                            parameter_distant_l*score_ia_distant_after_l

            if print_msg:
                print()
                print("score_ia_distant_after")
                print(" score_ia_distant_after_mean_d",score_ia_distant_after_mean_d)
                print(" score_ia_distant_after_onset_d",score_ia_distant_after_onset_d)
                print(" score_ia_distant_after_l",score_ia_distant_after_l)
                print(" dl_score_ia_distant_after",dl_score_ia_distant_after)

            ddl_score_ia_distant_after_dict[area_id_back] = dl_score_ia_distant_after
    if False:
        # distant ia method2
        max_ia_distant_length_half = max_ia_distant_length/2

        # calc left
        l_ia_distant_set_dict = {}
        d_ia_distant_mean_set_dict = {}
        d_ia_distant_onset_set_dict = {}

        # 5th circle
        # ia left,IA_start
        for i, (area_id, basic_interval_list) in enumerate(IA_left_pred_group_dict.items()):
            if basic_interval_list == []:
                ddl_score_ia_distant_dict[area_id] = [None,None]
                ddl_score_ia_distant_dict[area_id][0] = 0
                l_ia_distant_set_dict[area_id] = 0
                d_ia_distant_mean_set_dict[area_id] = 0
                d_ia_distant_onset_set_dict[area_id] = 1
                continue
            l_ia_distant_set_sum = 0
            d_ia_distant_mean_set_sum = 0
            d_ia_distant_onset_set_sum = 0
            ia_set_num = len(basic_interval_list)
            IA_mid = (IA_dict[area_id][0] + IA_dict[area_id][1]) / 2
            IA_len = IA_dict[area_id][1] - IA_dict[area_id][0]
            IA_start = IA_dict[area_id][0]
            IA_end = IA_dict[area_id][1]

            for basic_interval in basic_interval_list:
                l_ia_distant_set_sum += basic_interval[1] - basic_interval[0]

                # d_ia_set_sum+= abs((basic_interval[1] + basic_interval[0])/2 -IA_mid)
                d_ia_distant_mean_set_sum += abs((basic_interval[1] + basic_interval[0]) / 2 - IA_start)
                d_ia_distant_onset_set_sum += abs(basic_interval[0] - IA_start)
                # i =0 or i = label_interval_num,only have positive distance

            l_ia_distant_set_dict[area_id] = l_ia_distant_set_sum
            d_ia_distant_mean = d_ia_distant_mean_set_sum / ia_set_num if ia_set_num != 0 else 0
            d_ia_distant_mean_set_dict[area_id] = d_ia_distant_mean
            d_ia_distant_onset = d_ia_distant_onset_set_sum / ia_set_num if ia_set_num != 0 else 0
            d_ia_distant_onset_set_dict[area_id] = d_ia_distant_onset

            dl_score_ia_distant = 0
            #
            if l_ia_distant_set_sum != 0:
                dl_score_ia_distant = parameter_distant_r_mean * (-1 / max_ia_distant_length_half * d_ia_distant_mean + 1) + \
                                      parameter_distant_r_onset * (
                                                  -1 / max_ia_distant_length_half * d_ia_distant_onset + 1) + \
                                      (1 - parameter_distant_r_mean - parameter_distant_r_onset) * (
                                                  -1 / max_ia_distant_length_half * l_ia_distant_set_sum + 1) \
                    if max_ia_distant_length_half != 0 else 0

            ddl_score_ia_distant_dict[area_id] = [None,None]
            ddl_score_ia_distant_dict[area_id][0] = dl_score_ia_distant

        # ia right,IA_end
        for i, (area_id, basic_interval_list) in enumerate(IA_right_pred_group_dict.items()):
            if basic_interval_list == []:
                ddl_score_ia_distant_dict[area_id][1] = 0
                l_ia_distant_set_dict[area_id] = 0
                d_ia_distant_mean_set_dict[area_id] = 0
                d_ia_distant_onset_set_dict[area_id] = 0
                continue
            l_ia_distant_set_sum = 0
            d_ia_distant_mean_set_sum = 0
            d_ia_distant_onset_set_sum = 0
            ia_set_num = len(basic_interval_list)
            IA_mid = (IA_dict[area_id][0] + IA_dict[area_id][1]) / 2
            IA_len = IA_dict[area_id][1] - IA_dict[area_id][0]
            IA_start = IA_dict[area_id][0]
            IA_end = IA_dict[area_id][1]

            for basic_interval in basic_interval_list:
                l_ia_distant_set_sum += basic_interval[1] - basic_interval[0]

                # d_ia_set_sum+= abs((basic_interval[1] + basic_interval[0])/2 -IA_mid)
                d_ia_distant_mean_set_sum += abs((IA_end -basic_interval[1] + basic_interval[0]) / 2)
                d_ia_distant_onset_set_sum += abs(IA_end - basic_interval[1])
                # i =0 or i = label_interval_num,only have positive distance

            l_ia_distant_set_dict[area_id] = l_ia_distant_set_sum
            d_ia_distant_mean = d_ia_distant_mean_set_sum / ia_set_num if ia_set_num != 0 else 0
            d_ia_distant_mean_set_dict[area_id] = d_ia_distant_mean
            d_ia_distant_onset = d_ia_distant_onset_set_sum / ia_set_num if ia_set_num != 0 else 0
            d_ia_distant_onset_set_dict[area_id] = d_ia_distant_onset

            dl_score_ia_distant = 0
            #
            if l_ia_distant_set_sum != 0:
                dl_score_ia_distant = parameter_distant_r_mean * (-1 / max_ia_distant_length_half * d_ia_distant_mean + 1) + \
                                      parameter_distant_r_onset * (
                                                  -1 / max_ia_distant_length_half * d_ia_distant_onset + 1) + \
                                      (1 - parameter_distant_r_mean - parameter_distant_r_onset) * (
                                                  -1 / max_ia_distant_length_half * l_ia_distant_set_sum + 1) \
                    if max_ia_distant_length_half != 0 else 0

            ddl_score_ia_distant_dict[area_id][1] = dl_score_ia_distant



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

    area_id_0 = "id_" + str(0)
    area_id_num = "id_" + str(gt_num)

    # 调整left,right边界情况的贡献,把权重值也传回去

    adjusted_ddl_score_ia_near_left_dict = deepcopy(ddl_score_ia_before_dict)
    adjusted_ddl_score_ia_near_right_dict = deepcopy(ddl_score_ia_after_dict)
    adjusted_ddl_score_ia_near_dict_ratio = {}
    # adjusted_ddl_score_ia_near_left_dict_ratio_contribution = {}
    # adjusted_ddl_score_ia_near_right_dict_ratio_contribution = {}
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


        # ia_near_ratio_before = 1 / 2
        # ia_near_ratio_after = 1 / 2

        ia_near_ratio_before = parameter_w_near_0_ngt_gt_right
        ia_near_ratio_after = parameter_w_near_2_ngt_gt_left

        # 调整ia near weight
        # ia near before:0-gt_num-1
        # ia near after:1-gt_num
        if parameter_w_near_2_ngt_gt_left > 0 and parameter_w_near_0_ngt_gt_right > 0:
            if i == 0:
                # 默认
                ia_near_ratio_before = 0
                # if IA_dict[area_id] == []:
                if VA_f_dict[area_id][0] >= VA_f_dict[area_id][1] and VA_d_dict[area_id_back][0] < VA_d_dict[area_id_back][1]: # 下一个是最后一个,后一个不空
                        ia_near_ratio_after = 0 # 比例调整
                # else:  # 自己不空
                #     # if IA_pred_group_dict[area_id] == [] and VA_pred_group_dict[area_id] == []:  # 自己组空,同id的VA组不空
                #     #     adjusted_ddl_score_ia_dict[area_id] = 0
                #     # if area_id_back == area_id_num:
                #     if VA_d_dict[area_id_back][0] >= VA_d_dict[area_id_back][1]:  # 下一个是最后一个,后一个空
                #         ia_near_ratio_after = 1
            elif i == gt_num:
                # 默认
                ia_near_ratio_after = 0
                if VA_d_dict[area_id][0] >= VA_d_dict[area_id][1] and VA_f_dict[area_id_front][0] < VA_f_dict[area_id_front][1]:# 前面一个不空
                        ia_near_ratio_before = 0
                # else:# 自己不空
                #     # if IA_pred_group_dict[area_id] == [] and VA_pred_group_dict[area_id_front] == []:
                #     #     adjusted_ddl_score_ia_dict[area_id] = 0
                #     # if area_id_front == area_id_0:
                #     if VA_f_dict[area_id_front][0] >= VA_f_dict[area_id_front][1]:
                #         ia_near_ratio_before = 1
            else:
                # if IA_pred_group_dict[area_id] == []:
                #     if VA_pred_group_dict[area_id_front] == [] and VA_pred_group_dict[area_id] == []:
                #         adjusted_ddl_score_ia_dict[area_id] = 0
                if i == 1:
                    # if area_id_front == area_id_0:
                    if VA_d_dict[area_id][0] < VA_d_dict[area_id][1] and VA_f_dict[area_id_front][0] >= VA_f_dict[area_id_front][1]:
                            ia_near_ratio_before = 1
                if i == gt_num-1:
                    # if area_id_back == area_id_num:
                    if VA_f_dict[area_id][0] < VA_f_dict[area_id][1] and VA_d_dict[area_id_back][0] >= VA_d_dict[area_id_back][1]:  # 下一个是最后一个,后一个空
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
        # parameter_w_distant_ngt = 1 - parameter_w_gt - parameter_w_near_ngt
        # adjust ia distant
        for i in range(gt_num + 1):
            area_id = "id_" + str(i)
            area_id_front = "id_" + str(i - 1)
            area_id_back = "id_" + str(i + 1)


            # ia_near_ratio_before = 1 / 2
            # ia_near_ratio_after = 1 / 2
            #
            # # 调整ia near weight
            # # ia near before:0-gt_num-1
            # # ia near after:1-gt_num
            # if i == 0:
            #     # 默认
            #     ia_near_ratio_before = 0
            #     # if IA_dict[area_id] == []:
            #     if VA_f_dict[area_id][0] >= VA_f_dict[area_id][1] and VA_d_dict[area_id_back][0] < VA_d_dict[area_id_back][1]: # 下一个是最后一个,后一个不空
            #             ia_near_ratio_after = 0 # 比例调整
            #     # else:  # 自己不空
            #     #     # if IA_pred_group_dict[area_id] == [] and VA_pred_group_dict[area_id] == []:  # 自己组空,同id的VA组不空
            #     #     #     adjusted_ddl_score_ia_dict[area_id] = 0
            #     #     # if area_id_back == area_id_num:
            #     #     if VA_d_dict[area_id_back][0] >= VA_d_dict[area_id_back][1]:  # 下一个是最后一个,后一个空
            #     #         ia_near_ratio_after = 1
            # elif i == gt_num:
            #     # 默认
            #     ia_near_ratio_after = 0
            #     if VA_d_dict[area_id][0] >= VA_d_dict[area_id][1] and VA_f_dict[area_id_front][0] < VA_f_dict[area_id_front][1]:# 前面一个不空
            #             ia_near_ratio_before = 0
            #     # else:# 自己不空
            #     #     # if IA_pred_group_dict[area_id] == [] and VA_pred_group_dict[area_id_front] == []:
            #     #     #     adjusted_ddl_score_ia_dict[area_id] = 0
            #     #     # if area_id_front == area_id_0:
            #     #     if VA_f_dict[area_id_front][0] >= VA_f_dict[area_id_front][1]:
            #     #         ia_near_ratio_before = 1
            # else:
            #     # if IA_pred_group_dict[area_id] == []:
            #     #     if VA_pred_group_dict[area_id_front] == [] and VA_pred_group_dict[area_id] == []:
            #     #         adjusted_ddl_score_ia_dict[area_id] = 0
            #     if i == 1:
            #         # if area_id_front == area_id_0:
            #         if VA_d_dict[area_id][0] < VA_d_dict[area_id][1] and VA_f_dict[area_id_front][0] >= VA_f_dict[area_id_front][1]:
            #                 ia_near_ratio_before = 1
            #     if i == gt_num-1:
            #         # if area_id_back == area_id_num:
            #         if VA_f_dict[area_id][0] < VA_f_dict[area_id][1] and VA_d_dict[area_id_back][0] >= VA_d_dict[area_id_back][1]:  # 下一个是最后一个,后一个空
            #                 ia_near_ratio_after = 1
            #
            # adjusted_ddl_score_ia_near_dict_ratio[area_id] = [ia_near_ratio_before,ia_near_ratio_after]

            ia_distant_ratio_before = 1 / 2
            ia_distant_ratio_after = 1 / 2

            # 调整ia distant
            if i == 0:
                # 默认
                ia_distant_ratio_before = 0
                # 调分数
                if VA_pred_group_dict[area_id] == [] and IA_pred_group_dict[area_id] == []:
                    adjusted_ddl_score_ia_dict[area_id] = 0 # 分数调整
                    # if area_id_back == area_id_num:
                # 调比例
                if IA_dict[area_id][0] >= IA_dict[area_id][1] and IA_dict[area_id_back][0] < IA_dict[area_id_back][1]:  # 自己空,下一个不空
                    ia_distant_ratio_after = 0 # 比例调整
                # else:  # 自己不空
                #     if IA_pred_group_dict[area_id] == [] and VA_pred_group_dict[area_id] == []:  # 自己组空,同id的VA组不空
                #         adjusted_ddl_score_ia_dict[area_id] = 0
                #     # if area_id_back == area_id_num:
                #     # if IA_dict[area_id][0] < IA_dict[area_id][1]: # 下一个是最后一个,后一个不空
                #     if IA_dict[area_id_back][0] >= IA_dict[area_id_back][1]:  # 下一个是最后一个,后一个空
                #         ia_distant_ratio_after = 1
            elif i == gt_num:
                # 默认
                ia_distant_ratio_after = 0
                if VA_pred_group_dict[area_id_front] == [] and IA_pred_group_dict[area_id] == []:
                    adjusted_ddl_score_ia_dict[area_id] = 0 # 分数调整
                if IA_dict[area_id][0] >= IA_dict[area_id][1] and IA_dict[area_id_front][0] < IA_dict[area_id_front][1]:  # 自己空,下一个不空
                    ia_distant_ratio_before = 0 # 比例调整
                # if IA_dict[area_id][0] >= IA_dict[area_id][1]:# 自己空
                #     if VA_pred_group_dict[area_id_front] == []:
                #         adjusted_ddl_score_ia_dict[area_id] = 0
                #     # if area_id_front == area_id_0:
                #     if IA_dict[area_id_front][0] < IA_dict[area_id_front][1]:# 前面一个不空
                #             ia_distant_ratio_before = 0
                # else:
                #     if IA_pred_group_dict[area_id] == [] and VA_pred_group_dict[area_id_front] == []:
                #         adjusted_ddl_score_ia_dict[area_id] = 0
                #     # if area_id_front == area_id_0:
                #     if IA_dict[area_id_front][0] >= IA_dict[area_id_front][1]:
                #         ia_distant_ratio_before = 1
            else:
                if IA_pred_group_dict[area_id] == [] and \
                        VA_pred_group_dict[area_id_front] == [] and VA_pred_group_dict[area_id] == []:
                    adjusted_ddl_score_ia_dict[area_id] = 0

                if i == 1:
                    if IA_dict[area_id][0] < IA_dict[area_id][1] and IA_dict[area_id_front][0] >= IA_dict[area_id_front][1]:
                            ia_distant_ratio_before = 1
                if i == gt_num-1:
                    if IA_dict[area_id][0] < IA_dict[area_id][1] and IA_dict[area_id_back][0] >= IA_dict[area_id_back][1]:  # 下一个是最后一个,后一个空
                            ia_distant_ratio_after = 1

            adjusted_ddl_score_ia_distant_dict_ratio[area_id] = [ia_distant_ratio_before,ia_distant_ratio_after]
            ddl_score_ia_i_before_contribution = adjusted_ddl_score_ia_dict[area_id] * ia_distant_ratio_before
            ddl_score_ia_i_after_contribution = adjusted_ddl_score_ia_dict[area_id] * ia_distant_ratio_after
            adjusted_ddl_score_ia_dict_ratio_contribution[area_id] = [ddl_score_ia_i_before_contribution,
                                                                     ddl_score_ia_i_after_contribution]
        # # adjust ia near
        # for i in range(gt_num):
        #     area_id = "id_" + str(i)
        #     area_id_front = "id_" + str(i - 1)
        #     area_id_back = "id_" + str(i + 1)
        #     # adjust left
        #     if VA_f_pred_group_dict[area_id] == [] \
        #             and VA_gt_pred_group_dict[area_id] == [] and VA_d_pred_group_dict[area_id_back] == []:
        #         adjusted_ddl_score_ia_near_left_dict[area_id] = 0
        #
        #     # adjust right
        #     if VA_d_pred_group_dict[area_id_back] == [] \
        #             and VA_gt_pred_group_dict[area_id] == [] and VA_f_pred_group_dict[area_id] == []:
        #         adjusted_ddl_score_ia_near_right_dict[area_id_back] = 0
    else:
        for i in range(gt_num + 1):
            area_id = "id_" + str(i)
            area_id_front = "id_" + str(i - 1)
            area_id_back = "id_" + str(i + 1)


            ia_distant_ratio_before = 1 / 2
            ia_distant_ratio_after = 1 / 2

            # 调整ia distant weight
            # ia distant before:0-gt_num-1 
            # ia distant after:1-gt_num 
            if i == 0:
                # 默认
                ia_distant_ratio_before = 0
                # if IA_dict[area_id] == []:
                if IA_left_dict[area_id][0] >= IA_left_dict[area_id][1] and IA_right_dict[area_id_back][0] < IA_right_dict[area_id_back][1]: # 下一个是最后一个,后一个不空
                        ia_distant_ratio_after = 0 # 比例调整
                # else:  # 自己不空
                #     # if IA_pred_group_dict[area_id] == [] and VA_pred_group_dict[area_id] == []:  # 自己组空,同id的VA组不空
                #     #     adjusted_ddl_score_ia_dict[area_id] = 0
                #     # if area_id_back == area_id_num:
                #     if VA_d_dict[area_id_back][0] >= VA_d_dict[area_id_back][1]:  # 下一个是最后一个,后一个空
                #         ia_near_ratio_after = 1
            elif i == gt_num:
                # 默认
                ia_distant_ratio_after = 0
                if IA_right_dict[area_id][0] >= IA_right_dict[area_id][1] and IA_left_dict[area_id_front][0] < IA_left_dict[area_id_front][1]:# 前面一个不空
                        ia_distant_ratio_before = 0
                # else:# 自己不空
                #     # if IA_pred_group_dict[area_id] == [] and VA_pred_group_dict[area_id_front] == []:
                #     #     adjusted_ddl_score_ia_dict[area_id] = 0
                #     # if area_id_front == area_id_0:
                #     if IA_left_dict[area_id_front][0] >= IA_left_dict[area_id_front][1]:
                #         ia_near_ratio_before = 1
            else:
                # if IA_pred_group_dict[area_id] == []:
                #     if VA_pred_group_dict[area_id_front] == [] and VA_pred_group_dict[area_id] == []:
                #         adjusted_ddl_score_ia_dict[area_id] = 0
                if i == 1:
                    # if area_id_front == area_id_0:
                    if IA_right_dict[area_id][0] < IA_right_dict[area_id][1] and IA_left_dict[area_id_front][0] >= IA_left_dict[area_id_front][1]:
                            ia_distant_ratio_before = 1
                if i == gt_num-1:
                    # if area_id_back == area_id_num:
                    if IA_left_dict[area_id][0] < IA_left_dict[area_id][1] and IA_right_dict[area_id_back][0] >= IA_right_dict[area_id_back][1]:  # 下一个是最后一个,后一个空
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



    #  weight add


    # plot_func_multi(func_dict)
    # plot_func_multi_paper(func_dict,window_length)


    import pprint

    # print("s_a_func_dict")
    # pprint(s_a_func_dict, indent=4)

    # calc f1,3 methods
    # method1,横，加，for auc，1:横，auc(纵)，加,2:auc(纵)，横,加
    # method2,加，横,for auc，1:auc(纵)，加,横
    # f1_type = "row_add"
    # f1_type = "add_row"

    # auc调f1算local p,r and local fq matrix
    # auc_type = "row_auc_add"
    # auc_type = "auc_row_add"
    # auc_type = "auc_add_row"

    # cal recall,precision

    local_recall_matrix = []
    local_precision_matrix = []
    local_f1_matrix = []
    local_f1_add_fp_matrix = []
    local_fq_near_matrix = []
    local_fq_distant_matrix = []



    recall_va_dict = {}
    detection_score_ia_i_dict = {}
    precision_va_dict = {}

    pred_func_dict = {}
    basic_interval_precision_integral_dict = {}
    basic_interval_recall_integral_dict = {}

    gt_detected_num = 0
    # 7th circle
    for i in range(gt_num):
        area_id = "id_" + str(i)
        area_id_back = "id_" + str(i + 1)

        VA_gt_area = VA_gt_dict[area_id]
        VA_gt_len = VA_gt_dict[area_id][1] - VA_gt_dict[area_id][0]
        VA_gt_start = VA_gt_dict[area_id][0]
        VA_gt_end = VA_gt_dict[area_id][1]

        # 前后判空
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

        # IA_pred_i_group = IA_pred_group_dict[area_id]
        # IA_pred_i_back_group = VA_f_pred_group_dict[area_id]
        # IA_pred_i_back_group = VA_d_pred_group_dict[area_id_back]
        # VA_pred_i_group = VA_pred_group_dict[area_id]

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
                   "parameter_rho":parameter_rho,
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
                   "parameter_rho":parameter_rho,
                    "parameter_w_gt": parameter_w_gt_r,
                    "parameter_w_near_ngt": 0
                }

                # [3914, 3925]
                # 8.555555555555555
                # 3918, 3920
                # 1.5335169880624426
                # 0.17924224535794783
                # [3923, 3925]
                # 0.1567028013309005
                # b=0.002754820936639119
                # parameter_gama=2

                # cal_integral_basic_interval_gt = cal_integral_in_range_gt_power_func(basic_interval, parameter_a, parameter_beta)
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



        # [3914, 3925]
        # 8.555555555555555
        # 3918, 3920
        # 1.5335169880624426
        # 0.17924224535794783

        # gt_i_precision_integration = gt_precision_integration_dict[area_id]
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

                # precision_i_all = {float} 415.0
                # precision_i_group = {list: 6} [[100, 201], [300, 401], [600, 701], [800, 901], [495, 500.5], [500.5, 506]]
                # precision_i_item = {list: 2} [500.5, 506]
                # precision_i_valid = {float} 11.0
                # precision_va_dict = {dict: 1} {'id_0': 0.02650602409638554}
                # precision_va_i = {float} 0.02650602409638554

            else:
                precision_i_all = pred_precision_integral_all
                precision_i_valid = pred_precision_integral_all
                precision_i_group = VA_f_i_pred_group + VA_d_i_after_pred_group

                if cal_precision_method == 2:
                    # method2
                    for precision_i_item in precision_i_group:
                        precision_i_all += (1-parameter_w_gt_p) * (precision_i_item[1] - precision_i_item[0])
                    # precision_i_all = {float} 175.59227919873044
                    # precision_i_group = {list: 4} [[100, 201], [300, 401], [600, 701], [800, 901]]
                    # precision_i_item = {list: 2} [800, 901]
                    # precision_i_valid = {float} 8.25
                    # precision_va_dict = {dict: 1} {'id_0': 0.046983842556442244}
                    # precision_va_i = {float} 0.046983842556442244

                elif cal_precision_method == 3:
                    # method3
                    # parameter_a_left = (1-parameter_w_gt) / parameter_near_single_side_range
                    # parameter_beta_left = 1
                    # # parameter_a_right = -(1-parameter_w_gt) / parameter_near_single_side_range
                    # parameter_a_right = parameter_beta_left
                    # parameter_beta_right = parameter_beta_left

                    parameter_a_left = func_dict_copy[area_id + "_f_a"]["parameter_a"]
                    parameter_beta_left = func_dict_copy[area_id + "_f_a"]["parameter_beta"]
                    parameter_a_right = parameter_a_left
                    parameter_beta_right = parameter_beta_left

                    # func_dict[area_id + "_ia_before"] = {
                    #     "area": VA_f_dict[area_id],
                    #     "area_start": VA_f_dict[area_id][0],
                    #     "area_end": VA_f_dict[area_id][1],
                    #     "parameter_a": parameter_a_left,
                    #     "parameter_beta": parameter_beta_left,
                    #     "func": "ia_before_func",
                    #     "x_tp": None,
                    #     "gt_len": None,
                    #     "ia_len": VA_f_dict[area_id][1]-VA_f_dict[area_id][0],
                    #     "area_len": None,
                    #     "parameter_rho": parameter_rho,
                    #     "parameter_w_gt": parameter_w_near_ngt,
                    #     "parameter_w_near_ngt": parameter_w_near_ngt
                    # }
                    for interval_id, IA_basic_interval in enumerate(VA_f_i_pred_group):
                        # IA_left_area_relative = [VA_start -  IA_basic_interval[1],VA_start -  IA_basic_interval[0]]

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
                            "parameter_rho": parameter_rho,
                            "parameter_w_gt": 0,
                            "parameter_w_near_ngt": 0
                        }
                        # plot_func_multi(pred_func_dict) 2.7254361799816342,  2.270101256772354      4.198347107438016，2.270123215340426
                        # 4.277737593386269

                        cal_integral_gt_before = cal_integral_gt_before_ia_power_func(IA_basic_interval,
                                                                                      parameter_a_left,
                                                                                      parameter_beta_left,
                                                                                      parameter_near_single_side_range=parameter_near_single_side_range,
                                                                                      area_end=VA_f_end)
                        pred_precision_integral_all_before += cal_integral_gt_before

                    #     "area": VA_d_dict[area_id_back],
                    #     "area_start": VA_d_dict[area_id_back][0],
                    #     "area_end": VA_d_dict[area_id_back][1],
                    #     "parameter_a": parameter_a_right,
                    #     "parameter_beta": parameter_beta_right,
                    #     "func": "ia_after_func",
                    #     "x_tp": None,
                    #     "gt_len": None,
                    #     "ia_len": VA_d_dict[area_id_back][1]-VA_d_dict[area_id_back][0],
                    #     "area_len": None,
                    #     "parameter_rho": parameter_rho,
                    #     "parameter_w_gt": parameter_w_near_ngt,
                    #     "parameter_w_near_ngt": parameter_w_near_ngt
                    # }
                    a = func_dict_copy
                    for interval_id, IA_basic_interval in enumerate(VA_d_i_after_pred_group):
                        # IA_right_area_relative = [IA_basic_interval[0] - VA_end, IA_basic_interval[1] - VA_end]
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
                            "parameter_rho": parameter_rho,
                            "parameter_w_gt": 0,
                            "parameter_w_near_ngt": 0
                        }
                        # plot_func_multi(pred_func_dict)
                        cal_integral_gt_after = cal_integral_gt_after_ia_power_func(IA_basic_interval,
                                                                                    parameter_a_right,
                                                                                    parameter_beta_right,
                                                                                    parameter_near_single_side_range=parameter_near_single_side_range,
                                                                                    area_start=VA_d_start)
                        pred_precision_integral_all_after += cal_integral_gt_after

                    precision_i_all += pred_precision_integral_all_before
                    precision_i_all += pred_precision_integral_all_after
            # VA_f_dict {'id_0': [3789, 3914], 'id_1': [5134, 5259], 'id_2': [9388, 9513]}
            # VA_gt_dict {'id_0': [3914, 3925], 'id_1': [5259, 5269], 'id_2': [9513, 9527]}
            # VA_d_dict {'id_1': [3925, 4050], 'id_2': [5269, 5394], 'id_3': [9527, 9652]}

            # VA_f_pred_group_dict {'id_0': [[3861, 3910]], 'id_1': [], 'id_2': []}
            # VA_gt_pred_group_dict {'id_0': [[3921, 3925]], 'id_1': [], 'id_2': []}
            # VA_d_pred_group_dict {'id_1': [[3925, 3965], [3970, 3981]], 'id_2': [], 'id_3': []}

            # precision_i_all = {float} 9.271656369945495
            # precision_i_group = {list: 3} [[3861, 3910], [3925, 3965], [3970, 3981]]
            # precision_i_valid = {float} 2.7254361799816342
            # precision_va_dict = {dict: 1} {'id_0': 0.2939535365888087}
            # precision_va_i = {float} 0.2939535365888087

            # VA_f_pred_group_dict {'id_0': [[3860, 3910]], 'id_1': [[5241, 5246], [5247, 5249]], 'id_2': []}
            # VA_gt_pred_group_dict {'id_0': [[3919, 3925]], 'id_1': [], 'id_2': []}
            # VA_d_pred_group_dict {'id_1': [[3925, 3981]], 'id_2': [], 'id_3': []}

            # precision_i_all = {float} 10.74620791616471
            # precision_i_group = {list: 2} [[3860, 3910], [3925, 3981]]
            # precision_i_valid = {float} 4.198347107438016
            # precision_va_dict = {dict: 1} {'id_0': 0.3906817307268696}
            # precision_va_i = {float} 0.3906817307268696

            precision_va_i = precision_i_valid / precision_i_all if precision_i_all != 0 else 0
            precision_va_dict[area_id] = precision_va_i
            local_precision_matrix.append(precision_va_i)

            meata_f1_i = compute_f1_score(precision_va_i, recall_va_i)
            local_f1_matrix.append(meata_f1_i)

            if pred_case_id == 10 or pred_case_id == 11:
                d = 1

        # plot_func_multi_paper(func_dict)

        # near fq
        # adjusted_ddl_score_ia_near_left_contribution = adjusted_ddl_score_ia_near_left_dict[area_id]
        # adjusted_ddl_score_ia_near_right_contribution = adjusted_ddl_score_ia_near_right_dict[area_id_back]

        # distant fq
        # adjusted_ddl_score_ia_i_after_contribution = adjusted_ddl_score_ia_dict_ratio_contribution[area_id][1]
        # adjusted_ddl_score_ia_i_back_before_contribution = adjusted_ddl_score_ia_dict_ratio_contribution[area_id_back][0]

        # try:
        adjusted_ddl_score_ia_near = adjusted_ddl_score_ia_near_left_dict[area_id]*adjusted_ddl_score_ia_near_dict_ratio[area_id][1] + \
                                     adjusted_ddl_score_ia_near_right_dict[area_id_back]*adjusted_ddl_score_ia_near_dict_ratio[area_id_back][0]
        if print_msg:
            # print recall_va_dict,precision_va_dict,dl_score_ia_dict
            print()
            print("adjusted_ddl_score_ia_near_left_dict[area_id]", adjusted_ddl_score_ia_near_left_dict[area_id],"*")
            print("adjusted_ddl_score_ia_near_dict_ratio[area_id][1]", adjusted_ddl_score_ia_near_dict_ratio[area_id][1],"+")
            print("adjusted_ddl_score_ia_near_right_dict[area_id_back]", adjusted_ddl_score_ia_near_right_dict[area_id_back],"*")
            print("adjusted_ddl_score_ia_near_dict_ratio[area_id_back][0]", adjusted_ddl_score_ia_near_dict_ratio[area_id_back][0],"=")
            print("adjusted_ddl_score_ia_near", adjusted_ddl_score_ia_near)
        # except:
        #     d= 1
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

    # plot_func_multi_paper(func_dict,window_length=window_length)

    if print_msg:
        # print recall_va_dict,precision_va_dict,dl_score_ia_dict
        print()
        print("parameter_w_near_0_ngt_gt_right", parameter_w_near_0_ngt_gt_right)
        print("parameter_w_near_2_ngt_gt_left", parameter_w_near_2_ngt_gt_left)
        print("ddl_score_ia_before_dict", ddl_score_ia_before_dict)
        print("ddl_score_ia_after_dict", ddl_score_ia_after_dict)
        print("adjusted_ddl_score_ia_near_dict_ratio", adjusted_ddl_score_ia_near_dict_ratio)
        print("adjusted_ddl_score_ia_near_left_dict", adjusted_ddl_score_ia_near_left_dict)
        print("adjusted_ddl_score_ia_near_right_dict", adjusted_ddl_score_ia_near_right_dict)
        # adjusted_ddl_score_ia_near = adjusted_ddl_score_ia_near_left_dict[area_id]*adjusted_ddl_score_ia_near_dict_ratio[area_id][1] + \
        #                                      adjusted_ddl_score_ia_near_right_dict[area_id_back]*adjusted_ddl_score_ia_near_dict_ratio[area_id_back][0]
        #

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

# 分组双边绝对距离最小值
def meata_v3(label_interval_ranges, IA1_VA1_IA2_relative_list, window_length, parameter_dict, pred_label_point_array,
          output=None,pred_case_id=None, thresh_id=None,cal_mode="proportion"):
    # print_msg = False
    print_msg = True

    # parameter_alpha = parameter_dict["parameter_alpha"]
    parameter_rate = parameter_dict["parameter_rate"]

    parameter_theta = parameter_dict["parameter_theta"]
    # parameter_w = parameter_dict["parameter_w"]
    # parameter_switch_f =parameter_dict["parameter_switch_f"]
    # parameter_switch_d = parameter_dict["parameter_switch_d"]
    VA_f_window = parameter_dict["forecasting_window"]
    VA_d_window = parameter_dict["delay_window"]

    parameter_rho = parameter_dict["parameter_rho"]
    parameter_lamda = parameter_dict["parameter_lamda"]

    parameter_eta = parameter_dict["parameter_eta"]

    # no random
    no_random_coefficient = cal_no_random_measure_coefficient_method2(output,window_length)

    if print_msg:
        print("VA_f_window", VA_f_window)
        print("VA_d_window", VA_d_window)

    f_window_rate = 0
    d_window_rate = 0
    f_d_window_len = VA_f_window + VA_d_window
    if f_d_window_len != 0:
        f_window_rate = VA_f_window / f_d_window_len
        d_window_rate = 1 - f_window_rate

    IA_dict = {}
    VA_dict = {}
    VA_f_dict = {}
    VA_gt_dict = {}
    VA_d_dict = {}

    # pred generated is continuous range a,b(len) -> [a,a+b(len)),range of [a,b] -> [a,b) -> point of [a,b-1],point of [a,b] -> [a,b+1)

    # cal every f/d of gt,split IA and VA

    label_interval_num = len(label_interval_ranges)
    gt_num = label_interval_num
    s_a_func_dict = {}

    # first circle
    for i, label_interval_range in enumerate(label_interval_ranges):
        # adjust now and before,add before
        VA_gt_now_start, VA_gt_now_end = label_interval_range
        # VA_f_now_start = max(VA_gt_now_start - VA_f_window,0) if parameter_switch_f == True else max(VA_gt_now_start - 3,0)
        VA_f_now_start = max(VA_gt_now_start - VA_f_window, 0)
        VA_f_now_end = VA_gt_now_start
        VA_d_now_start = VA_gt_now_end
        # VA_d_now_end = min(VA_gt_now_end + VA_f_window,window_length) if parameter_switch_d == True else min(VA_gt_now_end + 3,window_length)
        VA_d_now_end = min(VA_gt_now_end + VA_d_window, window_length)
        # # this way below is more fast
        # if i == 0:
        #     VA_f_now_start = max(VA_f_now_start,0)
        # if i == label_interval_num:
        #     VA_d_now_end = min(VA_d_now_end,window_length)
        if i > 0:
            # adjust now and before
            if VA_d_before_end > VA_f_now_start:
                intersection_range_start = max(VA_f_now_start, VA_gt_before_end)
                intersection_range_end = min(VA_d_before_end, VA_gt_now_start)
                intersection_range = [intersection_range_start, intersection_range_end]
                intersection_len = intersection_range_end - intersection_range_start
                #         if intersection_len == 1 and (parameter_switch_f == True and parameter_switch_d == True) and parameter_rho == 1/2:
                # #             f prior
                #             VA_d_before_end -=1
                #         else:
                #         VA_d_before_end = VA_f_now_start = intersection_range_start + round(parameter_w*intersection_len)
                #         VA_d_before_end = VA_f_now_start = intersection_range_start + round(parameter_w*intersection_len,2)
                #         VA_f_now_start = intersection_range_start + round(f_window_rate*intersection_len,2)
                #         VA_d_before_end = intersection_range_start + round(d_window_rate*intersection_len,2)
                VA_f_now_start = intersection_range_start + round(f_window_rate * intersection_len)
                VA_d_before_end = intersection_range_start + round(d_window_rate * intersection_len)
            #     add before
            IA_dict["id_" + str(i)] = [VA_d_before_end, VA_f_now_start]
            VA_dict["id_" + str(i - 1)] = [VA_f_before_start, VA_d_before_end]
            VA_f_dict["id_" + str(i - 1)] = [VA_f_before_start, VA_f_before_end]
            VA_gt_dict["id_" + str(i - 1)] = [VA_gt_before_start, VA_gt_before_end]
            VA_d_dict["id_" + str(i - 1)] = [VA_d_before_start, VA_d_before_end]

            # add by order

            split_line_set.add(VA_f_before_start)
            split_line_set.add(VA_f_before_end)

            split_line_set.add(VA_gt_before_start)
            split_line_set.add(VA_gt_before_end)

            split_line_set.add(VA_d_before_start)
            split_line_set.add(VA_d_before_end)

            split_line_set.add(VA_d_before_end)
            split_line_set.add(VA_f_now_start)

            VA_gt_before_len = VA_gt_before_end - VA_gt_before_start
            x1, x2 = cal_x1_x2(VA_gt_before_len, parameter_rho)

            # # print("area_id,area",area_id,area)
            # # # print("x1,x2",x1, x2)
            # # print("x1_ori,x2_ori",x1+VA_gt_before_start[0], x2+VA_gt_before_start[0])
            x1_ori, x2_ori = x1 + VA_gt_before_start, x2 + VA_gt_before_start
            x_mid_ori = 2 * x1 + VA_gt_before_start
            split_line_set.add(x1_ori)
            split_line_set.add(x_mid_ori)
            split_line_set.add(x2_ori)
        else:
            IA_dict["id_" + str(i)] = [0, VA_f_now_start]
            split_line_set.add(VA_f_now_start)

        VA_gt_before_start, VA_gt_before_end = VA_gt_now_start, VA_gt_now_end
        VA_f_before_start = VA_f_now_start
        VA_f_before_end = VA_f_now_end
        VA_d_before_start = VA_d_now_start
        VA_d_before_end = VA_d_now_end

    # last one VA and last two IA
    IA_dict["id_" + str(label_interval_num)] = [VA_d_before_end, window_length]
    VA_dict["id_" + str(label_interval_num - 1)] = [VA_f_before_start, VA_d_before_end]
    VA_f_dict["id_" + str(label_interval_num - 1)] = [VA_f_before_start, VA_f_before_end]
    VA_gt_dict["id_" + str(label_interval_num - 1)] = [VA_gt_before_start, VA_gt_before_end]
    VA_d_dict["id_" + str(label_interval_num - 1)] = [VA_d_before_start, VA_d_before_end]

    VA_gt_before_len = VA_gt_before_end - VA_gt_before_start
    x1, x2 = cal_x1_x2(VA_gt_before_len, parameter_rho)

    # # print("area_id,area",area_id,area)
    # # # print("x1,x2",x1, x2)
    # # print("x1_ori,x2_ori",x1+VA_gt_before_start[0], x2+VA_gt_before_start[0])
    x1_ori, x2_ori = x1 + VA_gt_before_start, x2 + VA_gt_before_start
    x_mid_ori = 2 * x1 + VA_gt_before_start
    split_line_set.add(x1_ori)
    split_line_set.add(x_mid_ori)
    split_line_set.add(x2_ori)

    # add by order

    split_line_set.add(VA_f_before_start)
    split_line_set.add(VA_f_before_end)

    split_line_set.add(VA_gt_before_start)
    split_line_set.add(VA_gt_before_end)

    split_line_set.add(VA_d_before_start)
    split_line_set.add(VA_d_before_end)

    if print_msg:
        # print area dict
        # # print("split area")
        print("IA_dict", IA_dict)
        # print("VA_dict",VA_dict)
        print("VA_f_dict", VA_f_dict)
        print("VA_gt_dict", VA_gt_dict)
        print("VA_d_dict", VA_d_dict)

    def dict_filer(dict_data: dict):
        for i, (area_id, area_range) in enumerate(dict_data.items()):
            if area_range != []:
                dict_data[area_id] = [] if area_range[0] == area_range[1] else area_range
        return dict_data

    # second circle
    IA_dict = dict_filer(IA_dict)
    VA_dict = dict_filer(VA_dict)
    VA_f_dict = dict_filer(VA_f_dict)
    VA_gt_dict = dict_filer(VA_gt_dict)
    VA_d_dict = dict_filer(VA_d_dict)

    if print_msg:
        print("dict adjust +++++++++++")
        print("IA_dict", IA_dict)
        # print("VA_dict",VA_dict)
        print("VA_f_dict", VA_f_dict)
        print("VA_gt_dict", VA_gt_dict)
        print("VA_d_dict", VA_d_dict)

        # split pred to basic interval
        print("split_line_set", split_line_set)

    # third circle
    split_IA1_VA1_IA2_relative_list = split_intervals(IA1_VA1_IA2_relative_list, split_line_set)
    # print("split_IA1_VA1_IA2_relative_list",split_IA1_VA1_IA2_relative_list)

    # forecasting_len=10
    # delay_len=10
    IA_pred_group_dict = {}
    VA_pred_group_dict = {}
    VA_f_pred_group_dict = {}
    VA_gt_pred_group_dict = {}
    VA_d_pred_group_dict = {}

    VA_gt1_pred_group_dict = {}
    VA_gt2_pred_group_dict = {}
    VA_gt3_pred_group_dict = {}
    VA_gt4_pred_group_dict = {}
    func_dict = {}

    # "id_"+str(label_interval_num)
    for i in range(label_interval_num):
        IA_pred_group_dict["id_" + str(i)] = []
        VA_pred_group_dict["id_" + str(i)] = []
        VA_f_pred_group_dict["id_" + str(i)] = []
        VA_gt_pred_group_dict["id_" + str(i)] = []
        VA_d_pred_group_dict["id_" + str(i)] = []
        VA_gt1_pred_group_dict["id_" + str(i)] = []
        VA_gt2_pred_group_dict["id_" + str(i)] = []
        VA_gt3_pred_group_dict["id_" + str(i)] = []
        VA_gt4_pred_group_dict["id_" + str(i)] = []
    IA_pred_group_dict["id_" + str(label_interval_num)] = []

    # 4th circle
    # for i, basic_interval in enumerate(IA1_VA1_IA2_relative_list):
    for i, basic_interval in enumerate(split_IA1_VA1_IA2_relative_list):
        # IA
        for j, (area_id, area) in enumerate(IA_dict.items()):
            if pred_in_area(basic_interval, area):
                if area_id not in IA_pred_group_dict.keys():
                    IA_pred_group_dict[area_id] = [basic_interval]
                else:
                    IA_pred_group_dict[area_id].append(basic_interval)

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

            # group by x1,x2
            VA_gt_len = area[1] - area[0]
            x1, x2 = cal_x1_x2(VA_gt_len, parameter_rho)

            # # print("area_id,area",area_id,area)
            # # # print("x1,x2",x1, x2)
            # # print("x1_ori,x2_ori",x1+area[0], x2+area[0])
            area_gt1 = [0, x1]
            area_gt2 = [x1, 2 * x1]
            area_gt3 = [2 * x1, x2]
            area_gt4 = [x2, area[1] - area[0]]
            area_gt1_ori = [area[0], x1 + area[0]]
            area_gt2_ori = [x1 + area[0], 2 * x1 + area[0]]
            area_gt3_ori = [2 * x1 + area[0], x2 + area[0]]
            area_gt4_ori = [x2 + area[0], area[1]]
            # # print("area_gt1",area_gt1)
            # # print("area_gt2",area_gt2)
            # # print("area_gt3",area_gt3)
            # # print("area_gt4",area_gt4)

            # # print("area_gt1_ori",area_gt1_ori)
            # # print("area_gt2_ori",area_gt2_ori)
            # # print("area_gt3_ori",area_gt3_ori)
            # # print("area_gt4_ori",area_gt4_ori)

            if pred_in_area(basic_interval, area_gt1_ori):
                # if area_id not in VA_gt1_pred_group_dict.keys():
                #     VA_gt1_pred_group_dict[area_id]=[basic_interval]
                # else:
                #     VA_gt1_pred_group_dict[area_id].append(basic_interval)
                VA_gt1_pred_group_dict[area_id].append(basic_interval)
            # if basic_interval == [135, 141] and area == [130, 150]:
            #     a =1
            if pred_in_area(basic_interval, area_gt2_ori):
                # if area_id not in VA_gt2_pred_group_dict.keys():
                #     VA_gt2_pred_group_dict[area_id]=[basic_interval]
                # else:
                #     VA_gt2_pred_group_dict[area_id].append(basic_interval)
                VA_gt2_pred_group_dict[area_id].append(basic_interval)
            if pred_in_area(basic_interval, area_gt3_ori):
                # if area_id not in VA_gt3_pred_group_dict.keys():
                #     VA_gt3_pred_group_dict[area_id]=[basic_interval]
                # else:
                VA_gt3_pred_group_dict[area_id].append(basic_interval)
            if pred_in_area(basic_interval, area_gt4_ori):
                # if area_id not in VA_gt4_pred_group_dict.keys():
                #     VA_gt4_pred_group_dict[area_id]=[basic_interval]
                # else:
                VA_gt4_pred_group_dict[area_id].append(basic_interval)

        # VA-d
        for j, (area_id, area) in enumerate(VA_d_dict.items()):
            if pred_in_area(basic_interval, area):
                if area_id not in VA_d_pred_group_dict.keys():
                    VA_d_pred_group_dict[area_id] = [basic_interval]
                else:
                    VA_d_pred_group_dict[area_id].append(basic_interval)
        # VA
        for j, (area_id, area) in enumerate(VA_dict.items()):
            if pred_in_area(basic_interval, area):
                if area_id not in VA_pred_group_dict.keys():
                    VA_pred_group_dict[area_id] = [basic_interval]
                else:
                    VA_pred_group_dict[area_id].append(basic_interval)
    if print_msg:
        # print pred dict
        print("IA_pred_group_dict", IA_pred_group_dict)
        print("VA_f_pred_group_dict", VA_f_pred_group_dict)
        # print("VA_gt_pred_group_dict",VA_gt_pred_group_dict)
        print("VA_gt1_pred_group_dict", VA_gt1_pred_group_dict)
        print("VA_gt2_pred_group_dict", VA_gt2_pred_group_dict)
        print("VA_gt3_pred_group_dict", VA_gt3_pred_group_dict)
        print("VA_gt4_pred_group_dict", VA_gt4_pred_group_dict)
        print("VA_d_pred_group_dict", VA_d_pred_group_dict)
        # print("VA_pred_group_dict",VA_pred_group_dict)

    d_ia_set_dict = {}
    d_ia_sign_set_dict = {}
    l_ia_set_dict = {}
    dl_score_ia_dict = {}

    dl_part_ia_score_dict = {}

    d_ia_both_sides_dict = {}
    dl_score = {}
    adjust_dl_score = {}


    # 5th circle
    for i, (area_id, basic_interval_list) in enumerate(IA_pred_group_dict.items()):
        if basic_interval_list == []:
            dl_score_ia_dict[area_id] = 0
            l_ia_set_dict[area_id] = 0
            d_ia_set_dict[
                area_id] = 0  # only when basic_interval_list ！= [],the value are used,so set value to 0 have boe problem
            d_ia_sign_set_dict[
                area_id] = 0  # only when basic_interval_list ！= [],the value are used,so set value to 0 have boe problem
            d_ia_both_sides_dict[area_id] = [0,0]
            dl_score[area_id] = [1,1]
            continue
        l_ia = 0
        l_ia_set_sum = 0
        d_ia_set_sum = 0
        d_ia_set_before_side_sum = 0
        d_ia_sign_set_sum = 0
        ia_set_num = len(basic_interval_list)
        IA_mid = (IA_dict[area_id][0] + IA_dict[area_id][1]) / 2
        IA_len = IA_dict[area_id][1] - IA_dict[area_id][0]
        IA_start = IA_dict[area_id][0]
        IA_end = IA_dict[area_id][1]

        # 对两边的距离
        # 直接绝对距离的得分，调整函数
        test_group = "group"
        # test_group = "no group"
        if test_group == "group":
            for basic_interval in basic_interval_list:
                l_ia += basic_interval[1] - basic_interval[0]
                # the length of fitst and last IA could be set a fixed value,min(一个周期的长度,ori len),or min(mean len,ori len)
                if i == 0:
                    base_position = IA_dict[area_id][1]
                    # d_ia_set_sum+= abs((basic_interval[1] + basic_interval[0])/2 -base_position)
                # last id of IA is label_interval_num,not label_interval_num -1
                elif i == label_interval_num:
                    base_position = IA_dict[area_id][0]
                else:
                    base_position = IA_mid
                # d_ia_set_sum+= abs((basic_interval[1] + basic_interval[0])/2 -IA_mid)
                d_ia_set_sum += abs((basic_interval[1] + basic_interval[0]) / 2 - base_position)
                # i =0 or i = label_interval_num,only have positive distance

                d_ia_sign_set_sum += (basic_interval[1] + basic_interval[0]) / 2 - base_position
                # d_ia_set_before_side_sum += (basic_interval[1] + basic_interval[0]) / 2 - IA_start
                d_ia_before = (basic_interval[1] + basic_interval[0]) / 2 - IA_start
                d_ia_end = IA_end - (basic_interval[1] + basic_interval[0]) / 2
                if i == 0:
                    d_ia_set_before_side_sum += d_ia_end
                elif i == gt_num:
                    d_ia_set_before_side_sum += d_ia_before
                else:
                    d_ia_set_before_side_sum += min(d_ia_before,d_ia_end)

            l_ia_set_dict[area_id] = l_ia
            d_ia = d_ia_set_sum / ia_set_num if ia_set_num != 0 else 0
            d_ia_set_dict[area_id] = d_ia
            d_ia_sign = d_ia_sign_set_sum / ia_set_num if ia_set_num != 0 else 0
            d_ia_sign_set_dict[area_id] = d_ia_sign
            d_ia_before_side = d_ia_set_before_side_sum / ia_set_num if ia_set_num != 0 else 0
            # d_ia_min = min(d_ia_before_side,IA_len-d_ia_before_side)
            d_ia_both_sides_dict[area_id] = [d_ia_before_side,d_ia_before_side]

            # Sub_LOF
            # d_ia_both_sides_dict+++++++++ {'id_0': [3885.0, 29.0], 'id_1': [667.0, 667.0], 'id_2': [25.0, 4219.0], 'id_3': [34.0, 3676.0]}
            # l_ia_set_dict+++++++++ {'id_0': 58, 'id_1': 112, 'id_2': 50, 'id_3': 2}
            # FFT
            # d_ia_both_sides_dict+++++++++ {'id_0': [3901.0, 13.0], 'id_1': [28.5, 1305.5], 'id_2': [1.5, 4242.5], 'id_3': [2708.5, 1001.5]}
            # l_ia_set_dict+++++++++ {'id_0': 26, 'id_1': 57, 'id_2': 3, 'id_3': 11}

            # n = 0.01
            # n = 0.005
            # n = 0.001
            n = 0.0003
            dl_rate = 1/2
            # dl_rate = 2/3
            dl_score_before_d = dl_rate * math.e ** (-n * d_ia_both_sides_dict[area_id][0])
            dl_score_after_d = dl_rate * math.e ** (-n * d_ia_both_sides_dict[area_id][1])
            dl_score_after_l = (1-dl_rate) * math.e ** (-n * l_ia_set_dict[area_id])


            if i == 0:
                dl_score_before = 0
                dl_score_after = dl_score_after_d + dl_score_after_l
            elif i == gt_num:
                dl_score_before = dl_score_before_d + dl_score_after_l
                dl_score_after = 0
            else:
                dl_score_before = dl_score_before_d + dl_score_after_l
                dl_score_after = dl_score_after_d + dl_score_after_l
            dl_score[area_id] = [dl_score_before,dl_score_after]


            dl_score_ia = 0
            #
            if l_ia != 0:
                if i == 0:
                    # 一边情况下就没有1了
                    # parameter_alpha1 = parameter_rate/(2+parameter_rate)
                    #
                    # debug1 = (1/2*IA_len-d_ia)
                    # debug2 = 2 / IA_len * (1/2*IA_len-d_ia)
                    # debug3 = parameter_alpha1 * 2 / IA_len * (1/2*IA_len-d_ia)
                    #
                    # debug4 = (1 - l_ia / IA_len)
                    # debug5 = (1 - parameter_alpha1) * (1 - l_ia / IA_len)

                    # dl_score_ia = parameter_alpha1 * 2 / IA_len * (1/2*IA_len-d_ia) + (1 - parameter_alpha1) * (1 - l_ia / IA_len) \
                    #     if IA_len != 0 else 0

                    parameter_alpha = parameter_rate / (1 + parameter_rate)
                    debug11 = (IA_len - d_ia)
                    debug21 = 1 / IA_len * (IA_len - d_ia)
                    debug31 = parameter_alpha / IA_len * (IA_len - d_ia)

                    debug41 = (1 - l_ia / IA_len)
                    debug51 = (1 - parameter_alpha) * (1 - l_ia / IA_len)
                    # 得分跟系数也有关系

                    dl_score_ia1 = parameter_alpha / IA_len * (IA_len - d_ia) + (1 - parameter_alpha) * (
                            1 - l_ia / IA_len) \
                        if IA_len != 0 else 0
                    # 对抗边界是2
                    if parameter_rate <= 2:
                        min_value = parameter_alpha / IA_len * (IA_len - 1 / 2 * IA_len)
                        # min_value = min_value + 0.0000001
                        dl_score_ia = map_value(dl_score_ia1, min_value, 1, 1e-10, 1)
                    else:
                        min_value = (1 - parameter_alpha) * (1 - 0 / IA_len)
                        dl_score_ia = map_value(dl_score_ia1, min_value, 1, 1e-10, 1)

                        # dl_score_ia = dl_score_ia1
                    # print("0 dl_score_ia",dl_score_ia)
                    # dl_part_ia_score_dict[area_id] = {
                    #     "parameter_alpha":parameter_alpha,
                    #     "d_score":1/IA_len * (IA_len - d_ia),
                    #     "(1 - parameter_alpha)":(1 - parameter_alpha),
                    #     "d_score_parameter":(1 - parameter_alpha) * (
                    #                 1 - l_ia / IA_len)
                    # }
                    if pred_case_id == 2:
                        debug = 1
                # last id of IA is label_interval_num,not label_interval_num -1
                elif i == label_interval_num:

                    # debug1 = (1 / 2 * IA_len - d_ia)
                    # debug2 = 2 / IA_len * (1 / 2 * IA_len - d_ia)
                    # debug3 = parameter_alpha * 2 / IA_len * (1 / 2 * IA_len - d_ia)
                    #
                    # debug4 = (1 - l_ia / IA_len)
                    # debug5 = (1 - parameter_alpha) * (1 - l_ia / IA_len)

                    # dl_score_ia = parameter_alpha * 2 / IA_len * (1/2*IA_len-d_ia) + (1 - parameter_alpha) * (1 - l_ia / IA_len) \
                    #     if IA_len != 0 else 0

                    parameter_alpha = parameter_rate / (1 + parameter_rate)
                    debug11 = (IA_len - d_ia)
                    debug21 = 1 / IA_len * (IA_len - d_ia)
                    debug31 = parameter_alpha / IA_len * (IA_len - d_ia)

                    debug41 = (1 - l_ia / IA_len)
                    debug51 = (1 - parameter_alpha) * (1 - l_ia / IA_len)
                    dl_score_ia1 = parameter_alpha / IA_len * (IA_len - d_ia) + (1 - parameter_alpha) * (
                                1 - l_ia / IA_len) \
                        if IA_len != 0 else 0
                    # x, a, b, c, d
                    if parameter_rate <= 2:
                        min_value = parameter_alpha / IA_len * (IA_len - 1 / 2 * IA_len)
                        # min_value = min_value + 0.0000001
                        dl_score_ia = map_value(dl_score_ia1, min_value, 1, 1e-10, 1)
                    else:
                        min_value = (1 - parameter_alpha) * (1 - 0 / IA_len)
                        dl_score_ia = map_value(dl_score_ia1, min_value, 1, 1e-10, 1)
                        # dl_score_ia = map_value(dl_score_ia,parameter_alpha,1,0,1)
                    # dl_part_ia_score_dict[area_id] = {
                    #     "parameter_alpha": parameter_alpha,
                    #     "d_score": 1 / IA_len * (IA_len - d_ia),
                    #     "(1 - parameter_alpha)": (1 - parameter_alpha),
                    #     "d_score_parameter": (1 - parameter_alpha) * (
                    #             1 - l_ia / IA_len)
                    # }
                    if pred_case_id == 2:
                        debug = 1
                else:

                    parameter_alpha = parameter_rate / (2 + parameter_rate)
                    dl_score_ia = parameter_alpha * 2 / IA_len * d_ia + (1 - parameter_alpha) * (1 - l_ia / IA_len) \
                        if IA_len != 0 else 0
                    if pred_case_id == 2:
                        print("  parameter_alpha", parameter_alpha)
                        print("  d_ia", d_ia)
                        print("  d_score", 2 / IA_len * d_ia)
                        print("  d_score_parameter", parameter_alpha * 2 / IA_len * d_ia)
                        print("  (1 - parameter_alpha)", (1 - parameter_alpha))
                        print("  l_ia", l_ia)
                        print("  l_score", (1 - l_ia / IA_len))
                        print("  l_score_parameter", (1 - parameter_alpha) * (1 - l_ia / IA_len))
                        print("  IA_len", IA_len)
                        debug = 1
                dl_part_ia_score_dict
        else:
            if print_msg:
                print("basic_interval_list num", len(basic_interval_list))
            dl_score_ia_sum = 0
            for basic_interval in basic_interval_list:
                if i == 0:
                    base_position = IA_dict[area_id][1]
                elif i == label_interval_num:
                    base_position = IA_dict[area_id][0]
                else:
                    base_position = IA_mid

                l_ia = basic_interval[1] - basic_interval[0]
                d_ia = abs((basic_interval[1] + basic_interval[0]) / 2 - base_position)
                if pred_case_id == 2:
                    debug = 1
                dl_score_ia = 0
                #
                if l_ia != 0:
                    if i == 0:
                        parameter_alpha = parameter_rate / (1 + parameter_rate)
                        dl_score_ia1 = parameter_alpha / IA_len * (IA_len - d_ia) + (1 - parameter_alpha) * (
                                1 - l_ia / IA_len) \
                            if IA_len != 0 else 0
                        # 对抗边界是2
                        if parameter_rate <= 2:
                            min_value = parameter_alpha / IA_len * (IA_len - 1 / 2 * IA_len)
                            dl_score_ia = map_value(dl_score_ia1, min_value, 1, 1e-10, 1)
                        else:
                            min_value = (1 - parameter_alpha) * (1 - 0 / IA_len)
                            dl_score_ia = map_value(dl_score_ia1, min_value, 1, 1e-10, 1)
                        if print_msg:
                            print("0 dl_score_ia", dl_score_ia)
                        if pred_case_id == 2:
                            debug = 1
                    elif i == label_interval_num:
                        parameter_alpha = parameter_rate / (1 + parameter_rate)
                        debug11 = (IA_len - d_ia)
                        debug21 = 1 / IA_len * (IA_len - d_ia)
                        debug31 = parameter_alpha / IA_len * (IA_len - d_ia)

                        debug41 = (1 - l_ia / IA_len)
                        debug51 = (1 - parameter_alpha) * (1 - l_ia / IA_len)
                        dl_score_ia1 = parameter_alpha / IA_len * (IA_len - d_ia) + (1 - parameter_alpha) * (
                                1 - l_ia / IA_len) \
                            if IA_len != 0 else 0
                        # x, a, b, c, d
                        if parameter_rate <= 2:
                            min_value = parameter_alpha / IA_len * (IA_len - 1 / 2 * IA_len)
                            # min_value = min_value + 0.0000001
                            dl_score_ia = map_value(dl_score_ia1, min_value, 1, 1e-10, 1)
                        else:
                            min_value = (1 - parameter_alpha) * (1 - 0 / IA_len)
                            dl_score_ia = map_value(dl_score_ia1, min_value, 1, 1e-10, 1)
                            # dl_score_ia = map_value(dl_score_ia,parameter_alpha,1,0,1)
                        if pred_case_id == 2:
                            debug = 1
                    else:

                        parameter_alpha = parameter_rate / (2 + parameter_rate)
                        dl_score_ia = parameter_alpha * 2 / IA_len * d_ia + (1 - parameter_alpha) * (1 - l_ia / IA_len) \
                            if IA_len != 0 else 0
                        if pred_case_id == 2:
                            if print_msg:
                                print("  parameter_alpha", parameter_alpha)
                                print("  d_ia", d_ia)
                                print("  d_score", 2 / IA_len * d_ia)
                                print("  d_score_parameter", parameter_alpha * 2 / IA_len * d_ia)
                                print("  (1 - parameter_alpha)", (1 - parameter_alpha))
                                print("  l_ia", l_ia)
                                print("  l_score", (1 - l_ia / IA_len))
                                print("  l_score_parameter", (1 - parameter_alpha) * (1 - l_ia / IA_len))
                                print("  IA_len", IA_len)
                                debug = 1

                # sum
                dl_score_ia_sum += dl_score_ia
                l_ia_set_sum += basic_interval[1] - basic_interval[0]
                d_ia_set_sum += abs((basic_interval[1] + basic_interval[0]) / 2 - base_position)
                d_ia_sign_set_sum += (basic_interval[1] + basic_interval[0]) / 2 - base_position

            # sum
            # test_no_group_type = "sum"
            test_no_group_type = "average"
            if test_no_group_type == "sum":
                dl_score_ia = dl_score_ia_sum
                l_ia_set_dict[area_id] = l_ia_set_sum
                d_ia_set_dict[area_id] = d_ia_set_sum
                d_ia_sign_set_dict[area_id] = d_ia_sign_set_sum
            else:
                dl_score_ia = dl_score_ia_sum / len(basic_interval_list) if len(basic_interval_list) != 0 else 0
                l_ia_set_dict[area_id] = l_ia_set_sum / len(basic_interval_list) if len(basic_interval_list) != 0 else 0
                d_ia_set_dict[area_id] = d_ia_set_sum / len(basic_interval_list) if len(basic_interval_list) != 0 else 0
                d_ia_sign_set_dict[area_id] = d_ia_sign_set_sum
        dl_score_ia_dict[area_id] = dl_score_ia
    #     deal empty and no area
    if print_msg:
        print("d_ia_set_dict", d_ia_set_dict)
        print("d_ia_sign_set_dict", d_ia_sign_set_dict)
        print("l_ia_set_dict", l_ia_set_dict)
    # print("dl_score_ia_dict",dl_score_ia_dict)

    adjust_dl_score = dl_score

    # cal label_function_gt
    # 6th circle
    for i, (area_id, area) in enumerate(VA_gt_dict.items()):

        VA_gt_len = area[1] - area[0]
        x1, x2 = cal_x1_x2(VA_gt_len, parameter_rho)
        x1_ori, x2_ori = x1 + area[0], x2 + area[0]
        x_mid = 2 * x1
        x_mid_ori = x_mid + area[0]
        # x1_ori,x2_ori = x1+area[0], x2+area[0]
        area_gt1 = [0, x1]
        area_gt2 = [x1, 2 * x1]
        area_gt3 = [2 * x1, x2]
        area_gt4 = [x2, VA_gt_len]
        area_gt1_ori = [area[0], x1 + area[0]]
        area_gt2_ori = [x1 + area[0], 2 * x1 + area[0]]
        area_gt3_ori = [2 * x1 + area[0], x2 + area[0]]
        area_gt4_ori = [x2 + area[0], area[1]]
        if print_msg:
            print("VA_gt_len", VA_gt_len)

        if parameter_rho < 0:
            parameter_b1, parameter_gama1 = 1 / 2, 0
            parameter_b2, parameter_gama2 = 1 / 2, 0
        else:
            parameter_b1, parameter_gama1 = cal_power_function_coefficient(x1, parameter_lamda)
            parameter_b2, parameter_gama2 = cal_power_function_coefficient(x2 - x_mid, parameter_lamda)

        func_dict.update({
            area_id + "_gt1": {"area": area_gt1_ori,
                               "area_start": area_gt1_ori[0],
                               "parameter_a": parameter_b1,
                               "parameter_beta": parameter_gama1,
                               "func": "va_gt1_func",
                               "x_tp": x1,
                               "gt_len": VA_gt_len,
                               "area_len": None,
                               "parameter_rho": parameter_rho
                               },
            area_id + "_gt2": {"area": area_gt2_ori,
                               "area_start": area_gt1_ori[0],
                               "parameter_a": parameter_b1,
                               "parameter_beta": parameter_gama1,
                               "func": "va_gt2_func",
                               "x_tp": x1,
                               "gt_len": VA_gt_len,
                               "area_len": None,
                               "parameter_rho": parameter_rho
                               },
            area_id + "_gt3": {"area": area_gt3_ori,
                               "area_start": area_gt1_ori[0],
                               "parameter_a": parameter_b2,
                               "parameter_beta": parameter_gama2,
                               "func": "va_gt3_func",
                               "x_tp": x2,
                               "gt_len": VA_gt_len,
                               "area_len": None,
                               "parameter_rho": parameter_rho
                               },
            area_id + "_gt4": {"area": area_gt4_ori,
                               "area_start": area_gt1_ori[0],
                               "parameter_a": parameter_b2,
                               "parameter_beta": parameter_gama2,
                               "func": "va_gt4_func",
                               "x_tp": x2,
                               "gt_len": VA_gt_len,
                               "area_len": None,
                               "parameter_rho": parameter_rho
                               },

        })

        s_a_gt1 = cal_integral_in_range_gt_power_func1(area_gt1_ori, parameter_b1, parameter_gama1, x_tp=x1,
                                                       gt_len=VA_gt_len, area_start=area_gt1_ori[0],
                                                       parameter_rho=parameter_rho)
        s_a_gt2 = cal_integral_in_range_gt_power_func2(area_gt2_ori, parameter_b1, parameter_gama1, x_tp=x1,
                                                       gt_len=VA_gt_len, area_start=area_gt1_ori[0],
                                                       parameter_rho=parameter_rho)
        s_a_gt3 = cal_integral_in_range_gt_power_func3(area_gt3_ori, parameter_b2, parameter_gama2, x_tp=x2,
                                                       gt_len=VA_gt_len, area_start=area_gt1_ori[0],
                                                       parameter_rho=parameter_rho)
        s_a_gt4 = cal_integral_in_range_gt_power_func4(area_gt4_ori, parameter_b2, parameter_gama2, x_tp=x2,
                                                       gt_len=VA_gt_len, area_start=area_gt1_ori[0],
                                                       parameter_rho=parameter_rho)

        s_a_gt = s_a_gt1 + s_a_gt2 + s_a_gt3 + s_a_gt4

        func_dict[area_id + "_gt1"]["area_s"] = s_a_gt1
        func_dict[area_id + "_gt2"]["area_s"] = s_a_gt2
        func_dict[area_id + "_gt3"]["area_s"] = s_a_gt3
        func_dict[area_id + "_gt4"]["area_s"] = s_a_gt4

        method = 2

        va_f_area = VA_f_dict[area_id]
        s_a_f = 0
        if va_f_area == []:
            pass
        else:
            VA_f_len = va_f_area[1] - va_f_area[0]
            parameter_beta1 = 1
            parameter_a1 = 1 / (2 * VA_f_len)

            greater_than_th = False

            func_dict[area_id + "_f"] = {"area": va_f_area,
                                         "area_start": va_f_area[0],
                                         "parameter_a": parameter_a1,
                                         "parameter_beta": parameter_beta1,
                                         "func": "va_f_func",
                                         "x_tp": None,
                                         "gt_len": None,
                                         "area_len": VA_f_len,
                                         "greater_than_th": greater_than_th
                                         }

            # f
            s_max_f = cal_integral_in_range_f_power_func(va_f_area, parameter_a1, parameter_beta1, VA_f_len, VA_f_len,
                                                         area_start=va_f_area[0], greater_than_th=greater_than_th)
            func_dict[area_id + "_f"]["area_s"] = s_max_f

            s_rate = s_max_f / s_a_gt
            func_dict[area_id + "_f_a"] = func_dict[area_id + "_f"]
            s_a_f = s_max_f
            if s_rate > parameter_theta:
                greater_than_th = True

                # method1
                if method == 1:
                    parameter_beta1_a = VA_f_len / (2 * parameter_theta * s_a_gt) - 1
                    parameter_a1_a = 1 / (2 * pow(VA_f_len, parameter_beta1_a))
                # method2
                else:
                    parameter_beta1_a = VA_f_len / (2 * parameter_theta * s_a_gt) - 1  # same with method1
                    parameter_a1_a = VA_f_len / pow(1 / 2, parameter_beta1_a)

                s_a_f = cal_integral_in_range_f_power_func(va_f_area, parameter_a1_a, parameter_beta1_a, VA_f_len,
                                                           VA_f_len, area_start=va_f_area[0], method=method,
                                                           greater_than_th=greater_than_th)
                func_dict[area_id + "_f_a"] = {
                    "area": va_f_area,
                    "area_start": va_f_area[0],
                    "area_s": s_a_f,
                    "parameter_a": parameter_a1_a,
                    "parameter_beta": parameter_beta1_a,
                    # "func": "va_f_func",
                    # "func": "va_f_reverse_func",
                    "x_tp": None,
                    "gt_len": None,
                    "area_len": VA_f_len,
                    "greater_than_th": greater_than_th
                }
                if method == 1:
                    func_dict[area_id + "_f_a"]["func"] = "va_f_func"
                else:
                    func_dict[area_id + "_f_a"]["func"] = "va_f_reverse_func"

        # d
        va_d_area = VA_d_dict[area_id]
        s_a_d = 0
        if va_d_area == []:
            pass
        else:

            if pred_case_id == 7 and area_id == "id_2":
                debug = 1

            VA_d_len = va_d_area[1] - va_d_area[0]
            parameter_beta2 = 1
            parameter_a2 = 1 / (2 * VA_d_len)

            greater_than_th = False
            s_max_d = cal_integral_in_range_d_power_func(va_d_area, parameter_a2, parameter_beta2, VA_d_len, VA_d_len,
                                                         area_start=va_d_area[0], greater_than_th=greater_than_th)
            func_dict[area_id + "_d"] = {
                "area": va_d_area,
                "area_start": va_d_area[0],
                "parameter_a": parameter_a2,
                "parameter_beta": parameter_beta2,
                "func": "va_d_func",
                "x_tp": None,
                "gt_len": None,
                "area_len": VA_d_len,
                "greater_than_th": greater_than_th
            }

            func_dict[area_id + "_d"]["area_s"] = s_max_d

            func_dict[area_id + "_d_a"] = func_dict[area_id + "_d"]
            s_rate = s_max_d / s_a_gt
            s_a_d = s_max_d

            if s_rate > parameter_theta:
                greater_than_th = True

                # method1
                if method == 1:
                    parameter_beta2_a = VA_d_len / (2 * parameter_theta * s_a_gt) - 1
                    parameter_a2_a = 1 / (2 * pow(VA_d_len, parameter_beta2_a))
                else:
                    # method2
                    parameter_beta2_a = VA_d_len / (2 * parameter_theta * s_a_gt) - 1  # same with method1
                    parameter_a2_a = VA_d_len / pow(1 / 2, parameter_beta2_a)

                s_a_d = cal_integral_in_range_d_power_func(va_d_area, parameter_a2_a, parameter_beta2_a, VA_d_len,
                                                           VA_d_len,
                                                           area_start=va_d_area[0], method=method,
                                                           greater_than_th=greater_than_th)
                func_dict[area_id + "_d_a"] = {
                    "area": va_d_area,
                    "area_start": va_d_area[0],
                    "area_s": s_a_d,
                    "parameter_a": parameter_a2_a,
                    "parameter_beta": parameter_beta2_a,
                    # "func": "va_d_func",
                    # "func": "va_d_reverse_func",
                    "x_tp": None,
                    "gt_len": None,
                    "area_len": VA_d_len,
                    "greater_than_th": greater_than_th

                }

                if method == 1:
                    func_dict[area_id + "_d_a"]["func"] = "va_d_func"
                else:
                    func_dict[area_id + "_d_a"]["func"] = "va_d_reverse_func"

        s_a_func_dict[area_id] = {
            "s_a_va_gt_len": VA_gt_len,
            "s_a_va_gt": s_a_gt,
            "s_a_va": s_a_f + s_a_gt + s_a_d,
            "x1": x1,
            "x2": x2,
            "x_mid": x_mid,
            "x1_ori": x1_ori,
            "x2_ori": x2_ori,
            "x_mid_ori": x_mid_ori,
        }
    # plot func
    if pred_case_id == 7 and area_id == "id_2":
        debug = 1
        # plot_func_multi(func_dict)
    # plot_func_multi(func_dict)
    # plot_func_multi_paper(func_dict,window_length)

    import pprint

    # print("s_a_func_dict")
    # pprint(s_a_func_dict, indent=4)

    # cal recall_va,detection_score
    recall_va_list = []
    recall_va_dict = {}
    detection_score_ia_i_dict = {}
    precision_va_dict = {}

    gt_num = label_interval_num
    pred_func_dict = {}
    basic_interval_integral_dict = {}

    # 7th circle
    for i in range(gt_num):
        area_id = "id_" + str(i)
        # print("area_id",area_id)
        area_id_back = "id_" + str(i + 1)
        VA_f_i_pred_group = VA_f_pred_group_dict[area_id]
        VA_gt1_i_pred_group = VA_gt1_pred_group_dict[area_id]
        VA_gt2_i_pred_group = VA_gt2_pred_group_dict[area_id]
        VA_gt3_i_pred_group = VA_gt3_pred_group_dict[area_id]
        VA_gt4_i_pred_group = VA_gt4_pred_group_dict[area_id]
        VA_d_i_pred_group = VA_d_pred_group_dict[area_id]

        IA_pred_i_group = IA_pred_group_dict[area_id]
        IA_pred_i_back_group = IA_pred_group_dict[area_id_back]
        VA_pred_i_group = VA_pred_group_dict[area_id]

        pred_integral_all = 0

        # # f
        va_f_area = VA_f_dict[area_id]
        if va_f_area == []:
            pass
        else:
            VA_f_len = va_f_area[1] - va_f_area[0]
            parameter_a1 = func_dict[area_id + "_f_a"]["parameter_a"]
            parameter_beta1 = func_dict[area_id + "_f_a"]["parameter_beta"]
            greater_than_th = func_dict[area_id + "_f_a"]["greater_than_th"]

            for j, basic_interval in enumerate(VA_f_i_pred_group):
                if basic_interval != []:

                    pred_func_dict[area_id + "_f_a" + "_" + str(j)] = {
                        "area": basic_interval,
                        "area_start": va_f_area[0],
                        "parameter_a": parameter_a1,
                        "parameter_beta": parameter_beta1,
                        # "func":"va_f_func",
                        "x_tp": None,
                        "gt_len": None,
                        "area_len": VA_f_len
                    }
                    if not greater_than_th:
                        pred_func_dict[area_id + "_f_a" + "_" + str(j)]["func"] = "va_d_func"
                    else:
                        if method == 1:
                            pred_func_dict[area_id + "_f_a" + "_" + str(j)]["func"] = "va_f_func"
                        else:
                            pred_func_dict[area_id + "_f_a" + "_" + str(j)]["func"] = "va_f_reverse_func"

                    cal_integral_basic_interval_f = cal_integral_in_range_f_power_func(basic_interval, parameter_a1,
                                                                                       parameter_beta1, VA_f_len,
                                                                                       VA_f_len,
                                                                                       area_start=va_f_area[0],
                                                                                       method=method,
                                                                                       greater_than_th=greater_than_th)
                    pred_integral_all += cal_integral_basic_interval_f
                    basic_interval_integral_dict[area_id + "_f" + "_" + str(j)] = cal_integral_basic_interval_f

        # gt1-gt4
        s_a_gt_len = s_a_func_dict[area_id]["s_a_va_gt_len"]
        s_a_va = s_a_func_dict[area_id]["s_a_va"]
        parameter_b1 = func_dict[area_id + "_gt1"]["parameter_a"]
        parameter_gama1 = func_dict[area_id + "_gt1"]["parameter_beta"]
        parameter_b2 = func_dict[area_id + "_gt3"]["parameter_a"]
        parameter_gama2 = func_dict[area_id + "_gt3"]["parameter_beta"]
        x1 = s_a_func_dict[area_id]["x1"]
        x2 = s_a_func_dict[area_id]["x2"]
        area_gt = VA_gt_dict[area_id]
        VA_gt_len = area_gt[1] - area_gt[0]

        area_gt1 = func_dict[area_id + "_gt1"]["area"]
        area_gt2 = func_dict[area_id + "_gt2"]["area"]
        area_gt3 = func_dict[area_id + "_gt3"]["area"]
        area_gt4 = func_dict[area_id + "_gt4"]["area"]
        area_start = func_dict[area_id + "_gt1"]["area_start"]

        for j, basic_interval in enumerate(VA_gt1_i_pred_group):
            if basic_interval != []:
                pred_func_dict[area_id + "_gt1" + "_" + str(j)] = {
                    "area": basic_interval,
                    "area_start": area_start,
                    "parameter_a": parameter_b1,
                    "parameter_beta": parameter_gama1,
                    "func": "va_gt1_func",
                    "x_tp": None,
                    "gt_len": VA_gt_len,
                    "area_len": None,
                    "parameter_rho": parameter_rho
                }
                cal_integral_basic_interval_gt1 = cal_integral_in_range_gt_power_func1(basic_interval, parameter_b1,
                                                                                       parameter_gama1, x_tp=x1,
                                                                                       gt_len=VA_gt_len,
                                                                                       area_start=area_start,
                                                                                       parameter_rho=parameter_rho)
                pred_integral_all += cal_integral_basic_interval_gt1
                basic_interval_integral_dict[area_id + "_gt1" + "_" + str(j)] = cal_integral_basic_interval_gt1

        for j, basic_interval in enumerate(VA_gt2_i_pred_group):
            if basic_interval != []:
                pred_func_dict[area_id + "_gt2" + "_" + str(j)] = {
                    "area": basic_interval,
                    "area_start": area_start,
                    "parameter_a": parameter_b1,
                    "parameter_beta": parameter_gama1,
                    "func": "va_gt2_func",
                    "x_tp": x1,
                    "gt_len": VA_gt_len,
                    "area_len": None,
                    "parameter_rho": parameter_rho
                }
                cal_integral_basic_interval_gt2 = cal_integral_in_range_gt_power_func2(basic_interval, parameter_b1,
                                                                                       parameter_gama1, x_tp=x1,
                                                                                       gt_len=VA_gt_len,
                                                                                       area_start=area_start,
                                                                                       parameter_rho=parameter_rho)
                pred_integral_all += cal_integral_basic_interval_gt2
                basic_interval_integral_dict[area_id + "_gt2" + "_" + str(j)] = cal_integral_basic_interval_gt2

        for j, basic_interval in enumerate(VA_gt3_i_pred_group):
            if basic_interval != []:
                pred_func_dict[area_id + "_gt3" + "_" + str(j)] = {
                    "area": basic_interval,
                    "area_start": area_start,
                    "parameter_a": parameter_b2,
                    "parameter_beta": parameter_gama2,
                    "func": "va_gt3_func",
                    "x_tp": x2,
                    "gt_len": VA_gt_len,
                    "area_len": None,
                    "parameter_rho": parameter_rho

                }
                cal_integral_basic_interval_gt3 = cal_integral_in_range_gt_power_func3(basic_interval, parameter_b2,
                                                                                       parameter_gama2, x_tp=x2,
                                                                                       gt_len=VA_gt_len,
                                                                                       area_start=area_start,
                                                                                       parameter_rho=parameter_rho)
                pred_integral_all += cal_integral_basic_interval_gt3
                basic_interval_integral_dict[area_id + "_gt3" + "_" + str(j)] = cal_integral_basic_interval_gt3

        for j, basic_interval in enumerate(VA_gt4_i_pred_group):
            if basic_interval != []:
                pred_func_dict[area_id + "_gt4" + "_" + str(j)] = {
                    "area": basic_interval,
                    "area_start": area_start,
                    "parameter_a": parameter_b2,
                    "parameter_beta": parameter_gama2,
                    "func": "va_gt4_func",
                    "x_tp": x2,
                    "gt_len": VA_gt_len,
                    "area_len": None,
                    "parameter_rho": parameter_rho
                }
                cal_integral_basic_interval_gt4 = cal_integral_in_range_gt_power_func4(basic_interval, parameter_b2,
                                                                                       parameter_gama2, x_tp=x2,
                                                                                       gt_len=VA_gt_len,
                                                                                       area_start=area_start,
                                                                                       parameter_rho=parameter_rho)
                pred_integral_all += cal_integral_basic_interval_gt4
                basic_interval_integral_dict[area_id + "_gt4" + "_" + str(j)] = cal_integral_basic_interval_gt4

        # # d
        va_d_area = VA_d_dict[area_id]
        if va_d_area == []:
            pass
        else:
            VA_d_len = va_d_area[1] - va_d_area[0]
            parameter_a2 = func_dict[area_id + "_d_a"]["parameter_a"]
            parameter_beta2 = func_dict[area_id + "_d_a"]["parameter_beta"]
            greater_than_th = func_dict[area_id + "_d_a"]["greater_than_th"]
            if pred_case_id == 7 and area_id == "id_2":
                debug = 1

            for j, basic_interval in enumerate(VA_d_i_pred_group):
                if basic_interval != []:
                    pred_func_dict[area_id + "_d_a" + "_" + str(j)] = {
                        "area": basic_interval,
                        "area_start": va_d_area[0],
                        "parameter_a": parameter_a2,
                        "parameter_beta": parameter_beta2,
                        # "func":"va_d_func",
                        "x_tp": None,
                        "gt_len": None,
                        "area_len": VA_d_len
                    }
                    if not greater_than_th:
                        pred_func_dict[area_id + "_d_a" + "_" + str(j)]["func"] = "va_d_func"
                    else:
                        if method == 1:
                            pred_func_dict[area_id + "_d_a" + "_" + str(j)]["func"] = "va_d_func"
                        else:
                            pred_func_dict[area_id + "_d_a" + "_" + str(j)]["func"] = "va_d_reverse_func"

                    cal_integral_basic_interval_d = cal_integral_in_range_d_power_func(basic_interval, parameter_a2,
                                                                                       parameter_beta2, VA_d_len,
                                                                                       VA_d_len,
                                                                                       area_start=va_d_area[0],
                                                                                       method=method,
                                                                                       greater_than_th=greater_than_th)
                    pred_integral_all += cal_integral_basic_interval_d
                    basic_interval_integral_dict[area_id + "_d" + "_" + str(j)] = cal_integral_basic_interval_d

        # print("pred_integral_all",pred_integral_all)
        va_area = VA_dict[area_id]
        recall_va_i = pred_integral_all / s_a_va
        recall_va_list.append(recall_va_i)
        if pred_case_id == 7:
            debug = 1
        recall_va_dict[area_id] = recall_va_i

        detection_score_i = 1 if VA_pred_group_dict[area_id] != [] else 0
        detection_score_ia_i_dict[area_id] = detection_score_i
        # invalid part punish in is not enough
        # detection
        if cal_mode == "detection":
            precision_va_dict[area_id] = detection_score_i

        # precision_i_all = 0
        # precision_i_valid = 0
        # precision_i_group = IA_pred_i_group + IA_pred_i_back_group + VA_pred_i_group
        # for precision_i_item in precision_i_group:
        #     precision_i_all += precision_i_item[1] - precision_i_item[0]
        # for precision_i_item in VA_pred_i_group:
        #     precision_i_valid += precision_i_item[1] - precision_i_item[0]

        precision_i_all = pred_integral_all
        precision_i_valid = pred_integral_all
        precision_i_group = IA_pred_i_group + IA_pred_i_back_group

        for precision_i_item in precision_i_group:
            precision_i_all += parameter_eta*(precision_i_item[1] - precision_i_item[0])

        if pred_case_id == 6:
            debug = 1
        precision_va_i = precision_i_valid / precision_i_all if precision_i_all != 0 else 0

        # precision also consider invalid part
        # proportion
        if cal_mode == "proportion":
            precision_va_dict[area_id] = precision_va_i

    if print_msg:
        # print basic_interval_integral
        print("basic_interval_integral_dict", basic_interval_integral_dict)

    # pot pred func
    # if pred_case_id == 7 and area_id == "id_2":
    #     debug = 1
    #     plot_func_multi(pred_func_dict)
    # plot_func_multi(pred_func_dict)

    if print_msg:
        # print recall_va_dict,precision_va_dict,dl_score_ia_dict
        print("recall_va_dict", recall_va_dict)
        print("precision_va_dict", precision_va_dict)
        print("dl_score_ia_dict", dl_score_ia_dict)
        print("d_ia_both_sides_dict+++++++++", d_ia_both_sides_dict)
        print("l_ia_set_dict+++++++++", l_ia_set_dict)
        print("dl_score+++++++++", dl_score)

    adjusted_dl_score_ia_dict = {}
    adjusted_dl_score_ia_dict_ratio_contribution = {}
    adjusted_dl_score_ia_dict_ratio = {}
    adjusted_dl_score_absolute = {}

    if pred_case_id == 2:
        debug = 1
    area_id_0 = "id_" + str(0)
    area_id_num = "id_" + str(gt_num)
    # two logics
    logic = 1
    # logic = 3
    if logic == 1:
        # logic 1
        #     one_exist_value = 1/2
        one_exist_value = 1
        # 8th circle
        for i in range(gt_num + 1):
            area_id = "id_" + str(i)
            area_id_front = "id_" + str(i - 1)
            area_id_back = "id_" + str(i + 1)

            ia_area = IA_dict[area_id]
            l_ia = l_ia_set_dict[area_id]
            d_ia = d_ia_set_dict[area_id]
            d_ia_sign = d_ia_sign_set_dict[area_id]
            ia_len = ia_area[1] - ia_area[0] if ia_area != [] else 0

            if pred_case_id == 22:
                debug = 1

            if i == 0:
                if IA_dict[area_id] == []:
                    # if IA_dict[area_id_back] == [] and area_id_back == area_id_num:
                    if area_id_back == area_id_num:
                        if IA_dict[area_id_back] == []:  # 自己为空，后面一个为最后一个，后面一个为空
                            ratio_before = 0
                            ratio_after = 1 / 2
                        else:  # 自己为空，后面一个为最后一个，后面一个不为空
                            ratio_before = 0
                            ratio_after = 0
                    else:
                        ratio_before = 0
                        ratio_after = 1 / 2

                    if VA_pred_group_dict[area_id] != []:
                        dl_score_ia_i_contribution = 1
                        # dl_score_ia_i_contribution = [1,1]
                    else:
                        dl_score_ia_i_contribution = 0
                        # dl_score_ia_i_contribution = [0,0]
                else:
                    # 数据里至少有一个异常，至少有两个ia，如果没有异常，那么IA_dict=[]
                    if IA_pred_group_dict[area_id] != []:
                        if area_id_back == area_id_num:
                            if IA_dict[area_id_back] == []:  # 自己不为空，后面一个为最后一个，后面一个为空
                                ratio_before = 0
                                ratio_after = 1
                            else:  # 自己不为空，后面一个为最后一个，后面一个不为空
                                ratio_before = 0
                                ratio_after = 1 / 2
                        else:
                            ratio_before = 0
                            ratio_after = 1 / 2
                        dl_score_ia_i_contribution = dl_score_ia_dict[area_id]
                        # dl_score_ia_i_contribution = dl_score[area_id]
                    else:
                        if VA_pred_group_dict[area_id] != []:
                            if area_id_back == area_id_num:
                                if IA_dict[area_id_back] == []:  # 自己不为空，后面一个为最后一个，后面一个为空
                                    ratio_before = 0
                                    ratio_after = 1
                                else:  # 自己不为空，后面一个为最后一个，后面一个不为空
                                    ratio_before = 0
                                    ratio_after = 1 / 2
                            else:
                                ratio_before = 0
                                ratio_after = 1 / 2
                            dl_score_ia_i_contribution = 1
                        else:
                            if area_id_back == area_id_num:
                                if IA_dict[area_id_back] == []:  # 自己不为空，后面一个为最后一个，后面一个为空
                                    ratio_before = 0
                                    ratio_after = 1
                                else:  # 自己不为空，后面一个为最后一个，后面一个不为空
                                    ratio_before = 0
                                    ratio_after = 1 / 2
                            else:
                                ratio_before = 0
                                ratio_after = 1 / 2
                            dl_score_ia_i_contribution = 0
            elif i == gt_num:
                if IA_dict[area_id] == []:
                    if area_id_front == area_id_0:
                        if IA_dict[area_id_front] == []:  # 自己为空，前面一个为最后一个，前面一个为空
                            ratio_before = 1 / 2
                            ratio_after = 0
                        else:  # 自己为空，前面一个为第一个，前面一个不为空
                            ratio_before = 0
                            ratio_after = 0
                    else:
                        ratio_before = 1 / 2
                        ratio_after = 0

                    if VA_pred_group_dict[area_id_front] != []:
                        dl_score_ia_i_contribution = 1
                    else:
                        dl_score_ia_i_contribution = 0
                else:
                    if IA_pred_group_dict[area_id] != []:
                        if area_id_front == area_id_0:
                            if IA_dict[area_id_front] == []:
                                ratio_before = 1
                                ratio_after = 0
                            else:
                                ratio_before = 1 / 2
                                ratio_after = 0
                        else:
                            ratio_before = 1 / 2
                            ratio_after = 0
                        dl_score_ia_i_contribution = dl_score_ia_dict[area_id]
                        # dl_score_ia_i_contribution = dl_score[area_id]
                    else:
                        if VA_pred_group_dict[area_id_front] != []:
                            if area_id_front == area_id_0:
                                if IA_dict[area_id_front] == []:
                                    ratio_before = 1
                                    ratio_after = 0
                                else:
                                    ratio_before = 1 / 2
                                    ratio_after = 0
                            else:
                                ratio_before = 1 / 2
                                ratio_after = 0
                            dl_score_ia_i_contribution = 1
                        else:
                            if area_id_front == area_id_0:
                                if IA_dict[area_id_front] == []:
                                    ratio_before = 1
                                    ratio_after = 0
                                else:
                                    ratio_before = 1 / 2
                                    ratio_after = 0
                            else:
                                ratio_before = 1 / 2
                                ratio_after = 0
                            dl_score_ia_i_contribution = 0
            else:
                if IA_pred_group_dict[area_id] != []:
                    ratio_before = 1 / 2
                    ratio_after = 1 / 2
                    # IA_len = IA_dict[area_id][1] - IA_dict[area_id][0]
                    # IA_mid = (IA_dict[area_id][1] + IA_dict[area_id][0])/2
                    # IA_start = IA_dict[area_id][0]
                    # ratio_before_proportion = (IA_mid + dl_score_ia_dict[area_id]-IA_start) /IA_len
                    # ratio_after_proportion = 1-ratio_before_proportion
                    # if d_ia_sign > 0:
                    #     ratio_before = (ia_len / 2 - d_ia) / ia_len
                    #     ratio_after = (ia_len / 2 + d_ia) / ia_len
                    # elif d_ia_sign < 0:
                    #     ratio_before = (ia_len / 2 + d_ia) / ia_len
                    #     ratio_after = (ia_len / 2 - d_ia) / ia_len
                    # else:
                    #     ratio_before = 1/2
                    #     ratio_after = 1/2
                    dl_score_ia_i_contribution = dl_score_ia_dict[area_id]
                    # dl_score_ia_i_contribution = dl_score[area_id]
                else:
                    ratio_before = 1 / 2
                    ratio_after = 1 / 2
                    if VA_pred_group_dict[area_id_front] != [] and VA_pred_group_dict[area_id] != []:
                        # ratio_before = 1/2
                        # ratio_after = 1/2
                        dl_score_ia_i_contribution = 1
                    elif VA_pred_group_dict[area_id_front] == [] and VA_pred_group_dict[area_id] == []:
                        # ratio_before = 1/2
                        # ratio_after = 1/2
                        dl_score_ia_i_contribution = 0
                    else:
                        # if VA_pred_group_dict[area_id_front] != []:
                        #     ratio_before = 1
                        #     ratio_after = 0
                        # else:
                        #     ratio_before = 0
                        #     ratio_after = 1
                        dl_score_ia_i_contribution = one_exist_value

            if i == 0:
                adjust_dl_score[area_id][0] = 0
                if IA_pred_group_dict[area_id] == []:
                    if VA_pred_group_dict[area_id] == []:
                        adjust_dl_score[area_id][1] = 0
            elif i == gt_num:
                adjust_dl_score[area_id][1] = 0
                if IA_pred_group_dict[area_id] == []:
                    if VA_pred_group_dict[area_id_front] == []:
                        adjust_dl_score[area_id][0] = 0
            else:
                if IA_pred_group_dict[area_id] == []:
                    if VA_pred_group_dict[area_id_front] == []:
                        adjust_dl_score[area_id][0] = 0
                    if VA_pred_group_dict[area_id] == []:
                        adjust_dl_score[area_id][1] = 0

            adjusted_dl_score_ia_dict[area_id] = dl_score_ia_i_contribution
            dl_score_ia_i_before_contribution = dl_score_ia_i_contribution * ratio_before
            dl_score_ia_i_after_contribution = dl_score_ia_i_contribution * ratio_after
            adjusted_dl_score_ia_dict_ratio_contribution[area_id] = [dl_score_ia_i_before_contribution,
                                                                     dl_score_ia_i_after_contribution]
            adjusted_dl_score_ia_dict_ratio[area_id] = [ratio_before,
                                                                     ratio_after]
            # adjusted_dl_score_absolute[area_id] = [dl_score[area_id][0]*ratio_before,
            #                                                          dl_score[area_id][1]*ratio_after]





    debug = 1
    # logic 2
    # one_exist_value = 1/2
    # for i in range(gt_num + 1):
    #     area_id = "id_" + str(i)
    #     area_id_front = "id_" + str(i - 1)
    #     area_id_back = "id_" + str(i + 1)
    #
    #     ia_area = IA_dict[area_id]
    #     l_ia = l_ia_set_dict[area_id]
    #     d_ia = d_ia_set_dict[area_id]
    #     d_ia_sign = d_ia_sign_set_dict[area_id]
    #     ia_len = ia_area[1] - ia_area[0] if ia_area != [] else 0
    #
    #     if i == 0:
    #         ratio_before = 0
    #         ratio_after = 1
    #         if IA_pred_group_dict[area_id] != []:
    #             dl_score_ia_i_contribution = dl_score_ia_dict[area_id]
    #         else:
    #             if VA_pred_group_dict[area_id] != []:
    #                 dl_score_ia_i_contribution = 1 / 2
    #             else:
    #                 dl_score_ia_i_contribution = 0
    #     elif i == gt_num:
    #         ratio_before = 1
    #         ratio_after = 0
    #         if IA_pred_group_dict[area_id] != []:
    #             dl_score_ia_i_contribution = dl_score_ia_dict[area_id]
    #         else:
    #             if VA_pred_group_dict[area_id_front] != []:
    #                 dl_score_ia_i_contribution = 1 / 2
    #             else:
    #                 dl_score_ia_i_contribution = 0
    #     else:
    #         if IA_pred_group_dict[area_id] != []:
    #             ratio_before = 1 / 2
    #             ratio_after = 1 / 2
    #             # if d_ia_sign > 0:
    #             #     ratio_before = (ia_len / 2 - d_ia) / ia_len
    #             #     ratio_after = (ia_len / 2 + d_ia) / ia_len
    #             # elif d_ia_sign < 0:
    #             #     ratio_before = (ia_len / 2 + d_ia) / ia_len
    #             #     ratio_after = (ia_len / 2 - d_ia) / ia_len
    #             # else:
    #             #     ratio_before = 1/2
    #             #     ratio_after = 1/2
    #             dl_score_ia_i_contribution = dl_score_ia_dict[area_id]
    #         else:
    #             # ratio_before = 1 / 2
    #             # ratio_after = 1 / 2
    #             if VA_pred_group_dict[area_id_front] != [] and VA_pred_group_dict[area_id] != []:
    #                 ratio_before = 1/2
    #                 ratio_after = 1/2
    #                 dl_score_ia_i_contribution = 1
    #             elif VA_pred_group_dict[area_id_front] == [] and VA_pred_group_dict[area_id] == []:
    #                 ratio_before = 1/2
    #                 ratio_after = 1/2
    #                 dl_score_ia_i_contribution = 0
    #             else:
    #                 if VA_pred_group_dict[area_id_front] != []:
    #                     ratio_before = 1
    #                     ratio_after = 0
    #                 else:
    #                     ratio_before = 0
    #                     ratio_after = 1
    #                 dl_score_ia_i_contribution = one_exist_value

    # logic 3
    if logic == 3:
        # one_exist_value = 1/2
        one_exist_value = 1
        for i in range(gt_num + 1):
            area_id = "id_" + str(i)
            area_id_front = "id_" + str(i - 1)
            area_id_back = "id_" + str(i + 1)

            ia_area = IA_dict[area_id]
            l_ia = l_ia_set_dict[area_id]
            d_ia = d_ia_set_dict[area_id]
            d_ia_sign = d_ia_sign_set_dict[area_id]
            ia_len = ia_area[1] - ia_area[0] if ia_area != [] else 0

            if i == 0:
                ratio_before = 0
                ratio_after = 1
                if IA_dict[area_id] == []:
                    dl_score_ia_i_contribution = 0
                else:
                    if IA_pred_group_dict[area_id] != []:
                        dl_score_ia_i_contribution = dl_score_ia_dict[area_id]
                    else:
                        if VA_pred_group_dict[area_id] != []:
                            dl_score_ia_i_contribution = 1
                        else:
                            dl_score_ia_i_contribution = 0
            elif i == gt_num:
                ratio_before = 1
                ratio_after = 0
                if IA_dict[area_id] == []:
                    dl_score_ia_i_contribution = 0
                else:
                    if IA_pred_group_dict[area_id] != []:
                        dl_score_ia_i_contribution = dl_score_ia_dict[area_id]
                    else:
                        if VA_pred_group_dict[area_id_front] != []:
                            dl_score_ia_i_contribution = 1
                        else:
                            dl_score_ia_i_contribution = 0
            else:
                if IA_pred_group_dict[area_id] != []:
                    # ratio_before = 1 / 2
                    # ratio_after = 1 / 2
                    if d_ia_sign > 0:
                        ratio_before = (ia_len / 2 - d_ia) / ia_len
                        ratio_after = (ia_len / 2 + d_ia) / ia_len
                    elif d_ia_sign < 0:
                        ratio_before = (ia_len / 2 + d_ia) / ia_len
                        ratio_after = (ia_len / 2 - d_ia) / ia_len
                    else:
                        ratio_before = 1 / 2
                        ratio_after = 1 / 2
                    dl_score_ia_i_contribution = dl_score_ia_dict[area_id]
                else:
                    # ratio_before = 1 / 2
                    # ratio_after = 1 / 2
                    if VA_pred_group_dict[area_id_front] != [] and VA_pred_group_dict[area_id] != []:
                        ratio_before = 1 / 2
                        ratio_after = 1 / 2
                        dl_score_ia_i_contribution = 2
                    elif VA_pred_group_dict[area_id_front] == [] and VA_pred_group_dict[area_id] == []:
                        ratio_before = 1 / 2
                        ratio_after = 1 / 2
                        dl_score_ia_i_contribution = 0
                    else:
                        if VA_pred_group_dict[area_id_front] != []:
                            ratio_before = 1
                            ratio_after = 0
                        else:
                            ratio_before = 0
                            ratio_after = 1
                        dl_score_ia_i_contribution = one_exist_value

            adjusted_dl_score_ia_dict[area_id] = dl_score_ia_i_contribution
            dl_score_ia_i_before_contribution = dl_score_ia_i_contribution * ratio_before
            dl_score_ia_i_after_contribution = dl_score_ia_i_contribution * ratio_after
            adjusted_dl_score_ia_dict_ratio_contribution[area_id] = [dl_score_ia_i_before_contribution,
                                                                     dl_score_ia_i_after_contribution]

    if print_msg:
        # print("adjusted_dl_score_ia_dict", adjusted_dl_score_ia_dict)
        # print("adjusted_dl_score_ia_dict_ratio_contribution", adjusted_dl_score_ia_dict_ratio_contribution)
        print("adjusted_dl_score_ia_dict_ratio", adjusted_dl_score_ia_dict_ratio)
        # print("adjusted_dl_score_absolute", adjusted_dl_score_absolute)
        print("adjust_dl_score", adjust_dl_score)
    event_recall_i_list = []
    event_recall_i_dict = {}
    event_recall_i_dict_w0 = {}
    event_recall_i_dict_w1 = {}
    event_precision_i_dict = {}
    event_precision_i_dict_w0 = {}
    event_precision_i_dict_w1 = {}

    # 9th circle
    for i in range(gt_num):
        area_id = "id_" + str(i)
        if print_msg:
            print(area_id)
        area_id_back = "id_" + str(i + 1)
        recall_va_i = recall_va_dict[area_id]
        # if VA_pred_group_dict[area_id] != [] and IA_pred_group_dict[area_id] == [] and IA_pred_group_dict[area_id_back] == [] :
        #     dl_score_ia_i = 1
        #     dl_score_ia_i_back = 1
        # else:
        #     dl_score_ia_i = dl_score_ia_dict[area_id]
        #     dl_score_ia_i_back = dl_score_ia_dict[area_id_back]

        dl_score_ia_i_after_contribution = adjusted_dl_score_ia_dict_ratio_contribution[area_id][1]
        dl_score_ia_i_back_before_contribution = adjusted_dl_score_ia_dict_ratio_contribution[area_id_back][0]
        adjusted_dl_score_absolute_i_after = adjust_dl_score[area_id][1]
        adjusted_dl_score_absolute_i_back_before = adjust_dl_score[area_id_back][0]

        if print_msg:
            print(" adjusted_dl_score_absolute_i_after", adjusted_dl_score_absolute_i_after)
            # print(" dl_score_ia_i_after_contribution", dl_score_ia_i_after_contribution)
            print(" recall_va_i,", recall_va_i)
            # print(" dl_score_ia_i_back_before_contribution", dl_score_ia_i_back_before_contribution)
            print(" adjusted_dl_score_absolute_i_back_before", adjusted_dl_score_absolute_i_back_before)

        # event_recall_i = (1-2*parameter_eta)*recall_va_i + parameter_eta*(dl_score_ia_i_after_contribution+ dl_score_ia_i_back_before_contribution)
        if pred_case_id == 26:
            debug = 1

        # logic 1
        if logic == 1:
            event_recall_i = (1 - parameter_eta) * recall_va_i + parameter_eta * (
                        adjusted_dl_score_absolute_i_after + adjusted_dl_score_absolute_i_back_before)/2
            # w=0
            event_recall_i_w0 = (1 - 0) * recall_va_i + 0 * (
                        adjusted_dl_score_absolute_i_after + adjusted_dl_score_absolute_i_back_before)/2
            # w=1
            event_recall_i_w1 = (1 - 1) * recall_va_i + 1 * (
                        adjusted_dl_score_absolute_i_after + adjusted_dl_score_absolute_i_back_before)/2

            # event_recall_i = (1 - parameter_eta) * recall_va_i + parameter_eta * (
            #             adjusted_dl_score_absolute_i_after + adjusted_dl_score_absolute_i_back_before)
            # # w=0
            # event_recall_i_w0 = (1 - 0) * recall_va_i + 0 * (
            #             adjusted_dl_score_absolute_i_after + adjusted_dl_score_absolute_i_back_before)
            # # w=1
            # event_recall_i_w1 = (1 - 1) * recall_va_i + 1 * (
            #             adjusted_dl_score_absolute_i_after + adjusted_dl_score_absolute_i_back_before)
        # logic 3
        if logic == 3:
            if i == 0 and IA_dict[area_id] == []:
                event_recall_i = (1 - parameter_eta) * recall_va_i + parameter_eta * 1 / 1 * (
                    dl_score_ia_i_back_before_contribution)
            elif i == gt_num - 1 and IA_dict[area_id_back] == []:
                event_recall_i = (1 - parameter_eta) * recall_va_i + parameter_eta * 1 / 1 * (
                    dl_score_ia_i_after_contribution)
            else:
                event_recall_i = (1 - parameter_eta) * recall_va_i + parameter_eta * 1 / 2 * (
                            dl_score_ia_i_after_contribution + dl_score_ia_i_back_before_contribution)
        # event_recall_i_list
        event_recall_i_dict[area_id] = event_recall_i
        event_recall_i_dict_w0[area_id] = event_recall_i_w0
        event_recall_i_dict_w1[area_id] = event_recall_i_w1

        precision_va_i = precision_va_dict[area_id]
        if print_msg:
            print(" adjusted_dl_score_absolute_i_after", adjusted_dl_score_absolute_i_after)
            # print(" dl_score_ia_i_after_contribution", dl_score_ia_i_after_contribution)
            print(" precision_va_i,", precision_va_i)
            # print(" dl_score_ia_i_back_before_contribution", dl_score_ia_i_back_before_contribution)
            print(" adjusted_dl_score_absolute_i_back_before", adjusted_dl_score_absolute_i_back_before)


        # event_precision_i = (1-2*parameter_eta)*precision_va_i + parameter_eta*(dl_score_ia_i_after_contribution+ dl_score_ia_i_back_before_contribution)

        # logic 1
        if logic == 1:
            event_precision_i = (1 - parameter_eta) * precision_va_i + parameter_eta * (
                        adjusted_dl_score_absolute_i_after + adjusted_dl_score_absolute_i_back_before)/2
            # w=0
            event_precision_i_w0 = (1 - 0) * precision_va_i + 0 * (
                        adjusted_dl_score_absolute_i_after + adjusted_dl_score_absolute_i_back_before)/2
            # w=1
            event_precision_i_w1 = (1 - 1) * precision_va_i + 1 * (
                        adjusted_dl_score_absolute_i_after + adjusted_dl_score_absolute_i_back_before)/2

            # event_precision_i = (1 - parameter_eta) * precision_va_i + parameter_eta * (
            #             adjusted_dl_score_absolute_i_after + adjusted_dl_score_absolute_i_back_before)
            # # w=0
            # event_precision_i_w0 = (1 - 0) * precision_va_i + 0 * (
            #             adjusted_dl_score_absolute_i_after + adjusted_dl_score_absolute_i_back_before)
            # # w=1
            # event_precision_i_w1 = (1 - 1) * precision_va_i + 1 * (
            #             adjusted_dl_score_absolute_i_after + adjusted_dl_score_absolute_i_back_before)

        # logic 3
        if logic == 3:
            if i == 0 and IA_dict[area_id] == []:
                event_precision_i = (1 - parameter_eta) * precision_va_i + parameter_eta * 1 / 1 * (
                    dl_score_ia_i_back_before_contribution)
            elif i == gt_num - 1 and IA_dict[area_id_back] == []:
                event_precision_i = (1 - parameter_eta) * precision_va_i + parameter_eta * 1 / 1 * (
                    dl_score_ia_i_after_contribution)
            else:
                event_precision_i = (1 - parameter_eta) * precision_va_i + parameter_eta * 1 / 2 * (
                            dl_score_ia_i_after_contribution + dl_score_ia_i_back_before_contribution)
        event_precision_i_dict[area_id] = event_precision_i
        event_precision_i_dict_w0[area_id] = event_precision_i_w0
        event_precision_i_dict_w1[area_id] = event_precision_i_w1

    if print_msg:
        print("event_recall_i_dict", event_recall_i_dict)
        print("event_precision_i_dict", event_precision_i_dict)

    if pred_case_id == 2:
        debug = 1
    event_recall = np.array(list(event_recall_i_dict.values())).mean()
    # print(event_recall)
    event_precision = np.array(list(event_precision_i_dict.values())).mean()
    # print(event_precision)

    event_recall_w0 = np.array(list(event_recall_i_dict_w0.values())).mean()
    event_precision_w0 = np.array(list(event_precision_i_dict_w0.values())).mean()
    event_recall_w1 = np.array(list(event_recall_i_dict_w1.values())).mean()
    event_precision_w1 = np.array(list(event_precision_i_dict_w1.values())).mean()

    #
    # print("======== meata_recall")
    # print(" event_recall",event_recall)
    # print(" event_recall_w0",event_recall_w0)
    # print(" event_recall_w1",event_recall_w1)
    #
    # print(" event_recall_i_dict",event_recall_i_dict)
    # print(" event_recall_i_dict_w0",event_recall_i_dict_w0)
    # print(" event_recall_i_dict_w1",event_recall_i_dict_w1)
    # print(" adjusted_dl_score_ia_dict_ratio_contribution",adjusted_dl_score_ia_dict_ratio_contribution)

    recall_info = {
        "event_recall": event_recall,
        "event_recall_w0": event_recall_w0,
        "event_recall_w1": event_recall_w1,
        "event_recall_i_dict": event_recall_i_dict,
        "event_recall_i_dict_w0": event_recall_i_dict_w0,
        "event_recall_i_dict_w1": event_recall_i_dict_w1,
        "adjusted_dl_score_ia_dict_ratio_contribution": adjusted_dl_score_ia_dict_ratio_contribution,
        "adjusted_dl_score_absolute": adjusted_dl_score_absolute,
    }

    meata_f1 = compute_f1_score(event_precision, event_recall)
    meata_f1_w0 = compute_f1_score(event_precision_w0, event_recall_w0)
    meata_f1_w1 = compute_f1_score(event_precision_w1, event_recall_w1)

    # gt_label_point_array = convert_interval_to_point_array(label_interval_ranges, time_series_length=window_length)
    # ont_count = np.count_nonzero(gt_label_point_array == 1)
    # anomaly_rate = ont_count / window_length
    # if anomaly_rate > 0.2:
    #     print("anomaly_rate exceeds threshold")
    #     no_random_coefficient = 1
    # else:
    #     no_random_coefficient = cal_no_random_measure_coefficient(pred_label_point_array, window_length)

    if thresh_id == 94:
        d=1
    coefficient_meata_f1 = no_random_coefficient * meata_f1
    coefficient_event_recall = no_random_coefficient * event_recall
    coefficient_event_precision = no_random_coefficient * event_precision
    print("meata_f1", meata_f1)
    print("coefficient_meata_f1", coefficient_meata_f1)

    coefficient_meata_f1_w0 = no_random_coefficient * meata_f1_w0
    coefficient_meata_f1_w1 = no_random_coefficient * meata_f1_w1

    # return event_recall,event_precision,meata_f1,no_random_meata_f1
    # return event_recall,event_precision,meata_f1,coefficient_meata_f1,VA_f_dict,VA_d_dict,coefficient_event_recall,coefficient_event_precision
    return event_recall, event_precision, meata_f1, coefficient_meata_f1, VA_f_dict, VA_d_dict, \
        coefficient_event_recall, coefficient_event_precision, \
        meata_f1_w0, coefficient_meata_f1_w0, meata_f1_w1, coefficient_meata_f1_w1, \
        recall_info

# 分组双边绝对距离单独计算
def meata_v2(label_interval_ranges, IA1_VA1_IA2_relative_list, window_length, parameter_dict, pred_label_point_array,
             output=None, pred_case_id=None, thresh_id=None, cal_mode="proportion"):
    # print_msg = False
    print_msg = True

    # parameter_alpha = parameter_dict["parameter_alpha"]
    parameter_rate = parameter_dict["parameter_rate"]

    parameter_theta = parameter_dict["parameter_theta"]
    # parameter_w = parameter_dict["parameter_w"]
    # parameter_switch_f =parameter_dict["parameter_switch_f"]
    # parameter_switch_d = parameter_dict["parameter_switch_d"]
    VA_f_window = parameter_dict["forecasting_window"]
    VA_d_window = parameter_dict["delay_window"]

    parameter_rho = parameter_dict["parameter_rho"]
    parameter_lamda = parameter_dict["parameter_lamda"]

    parameter_eta = parameter_dict["parameter_eta"]

    # no random
    no_random_coefficient = cal_no_random_measure_coefficient_method2(output, window_length)

    if print_msg:
        print("VA_f_window", VA_f_window)
        print("VA_d_window", VA_d_window)

    f_window_rate = 0
    d_window_rate = 0
    f_d_window_len = VA_f_window + VA_d_window
    if f_d_window_len != 0:
        f_window_rate = VA_f_window / f_d_window_len
        d_window_rate = 1 - f_window_rate

    IA_dict = {}
    VA_dict = {}
    VA_f_dict = {}
    VA_gt_dict = {}
    VA_d_dict = {}

    # pred generated is continuous range a,b(len) -> [a,a+b(len)),range of [a,b] -> [a,b) -> point of [a,b-1],point of [a,b] -> [a,b+1)

    # cal every f/d of gt,split IA and VA

    label_interval_num = len(label_interval_ranges)

    s_a_func_dict = {}

    # first circle
    for i, label_interval_range in enumerate(label_interval_ranges):
        # adjust now and before,add before
        VA_gt_now_start, VA_gt_now_end = label_interval_range
        # VA_f_now_start = max(VA_gt_now_start - VA_f_window,0) if parameter_switch_f == True else max(VA_gt_now_start - 3,0)
        VA_f_now_start = max(VA_gt_now_start - VA_f_window, 0)
        VA_f_now_end = VA_gt_now_start
        VA_d_now_start = VA_gt_now_end
        # VA_d_now_end = min(VA_gt_now_end + VA_f_window,window_length) if parameter_switch_d == True else min(VA_gt_now_end + 3,window_length)
        VA_d_now_end = min(VA_gt_now_end + VA_d_window, window_length)
        # # this way below is more fast
        # if i == 0:
        #     VA_f_now_start = max(VA_f_now_start,0)
        # if i == label_interval_num:
        #     VA_d_now_end = min(VA_d_now_end,window_length)
        if i > 0:
            # adjust now and before
            if VA_d_before_end > VA_f_now_start:
                intersection_range_start = max(VA_f_now_start, VA_gt_before_end)
                intersection_range_end = min(VA_d_before_end, VA_gt_now_start)
                intersection_range = [intersection_range_start, intersection_range_end]
                intersection_len = intersection_range_end - intersection_range_start
                #         if intersection_len == 1 and (parameter_switch_f == True and parameter_switch_d == True) and parameter_rho == 1/2:
                # #             f prior
                #             VA_d_before_end -=1
                #         else:
                #         VA_d_before_end = VA_f_now_start = intersection_range_start + round(parameter_w*intersection_len)
                #         VA_d_before_end = VA_f_now_start = intersection_range_start + round(parameter_w*intersection_len,2)
                #         VA_f_now_start = intersection_range_start + round(f_window_rate*intersection_len,2)
                #         VA_d_before_end = intersection_range_start + round(d_window_rate*intersection_len,2)
                VA_f_now_start = intersection_range_start + round(f_window_rate * intersection_len)
                VA_d_before_end = intersection_range_start + round(d_window_rate * intersection_len)
            #     add before
            IA_dict["id_" + str(i)] = [VA_d_before_end, VA_f_now_start]
            VA_dict["id_" + str(i - 1)] = [VA_f_before_start, VA_d_before_end]
            VA_f_dict["id_" + str(i - 1)] = [VA_f_before_start, VA_f_before_end]
            VA_gt_dict["id_" + str(i - 1)] = [VA_gt_before_start, VA_gt_before_end]
            VA_d_dict["id_" + str(i - 1)] = [VA_d_before_start, VA_d_before_end]

            # add by order

            split_line_set.add(VA_f_before_start)
            split_line_set.add(VA_f_before_end)

            split_line_set.add(VA_gt_before_start)
            split_line_set.add(VA_gt_before_end)

            split_line_set.add(VA_d_before_start)
            split_line_set.add(VA_d_before_end)

            split_line_set.add(VA_d_before_end)
            split_line_set.add(VA_f_now_start)

            VA_gt_before_len = VA_gt_before_end - VA_gt_before_start
            x1, x2 = cal_x1_x2(VA_gt_before_len, parameter_rho)

            # # print("area_id,area",area_id,area)
            # # # print("x1,x2",x1, x2)
            # # print("x1_ori,x2_ori",x1+VA_gt_before_start[0], x2+VA_gt_before_start[0])
            x1_ori, x2_ori = x1 + VA_gt_before_start, x2 + VA_gt_before_start
            x_mid_ori = 2 * x1 + VA_gt_before_start
            split_line_set.add(x1_ori)
            split_line_set.add(x_mid_ori)
            split_line_set.add(x2_ori)
        else:
            IA_dict["id_" + str(i)] = [0, VA_f_now_start]
            split_line_set.add(VA_f_now_start)

        VA_gt_before_start, VA_gt_before_end = VA_gt_now_start, VA_gt_now_end
        VA_f_before_start = VA_f_now_start
        VA_f_before_end = VA_f_now_end
        VA_d_before_start = VA_d_now_start
        VA_d_before_end = VA_d_now_end

    # last one VA and last two IA
    IA_dict["id_" + str(label_interval_num)] = [VA_d_before_end, window_length]
    VA_dict["id_" + str(label_interval_num - 1)] = [VA_f_before_start, VA_d_before_end]
    VA_f_dict["id_" + str(label_interval_num - 1)] = [VA_f_before_start, VA_f_before_end]
    VA_gt_dict["id_" + str(label_interval_num - 1)] = [VA_gt_before_start, VA_gt_before_end]
    VA_d_dict["id_" + str(label_interval_num - 1)] = [VA_d_before_start, VA_d_before_end]

    VA_gt_before_len = VA_gt_before_end - VA_gt_before_start
    x1, x2 = cal_x1_x2(VA_gt_before_len, parameter_rho)

    # # print("area_id,area",area_id,area)
    # # # print("x1,x2",x1, x2)
    # # print("x1_ori,x2_ori",x1+VA_gt_before_start[0], x2+VA_gt_before_start[0])
    x1_ori, x2_ori = x1 + VA_gt_before_start, x2 + VA_gt_before_start
    x_mid_ori = 2 * x1 + VA_gt_before_start
    split_line_set.add(x1_ori)
    split_line_set.add(x_mid_ori)
    split_line_set.add(x2_ori)

    # add by order

    split_line_set.add(VA_f_before_start)
    split_line_set.add(VA_f_before_end)

    split_line_set.add(VA_gt_before_start)
    split_line_set.add(VA_gt_before_end)

    split_line_set.add(VA_d_before_start)
    split_line_set.add(VA_d_before_end)

    if print_msg:
        # print area dict
        # # print("split area")
        print("IA_dict", IA_dict)
        # print("VA_dict",VA_dict)
        print("VA_f_dict", VA_f_dict)
        print("VA_gt_dict", VA_gt_dict)
        print("VA_d_dict", VA_d_dict)

    def dict_filer(dict_data: dict):
        for i, (area_id, area_range) in enumerate(dict_data.items()):
            if area_range != []:
                dict_data[area_id] = [] if area_range[0] == area_range[1] else area_range
        return dict_data

    # second circle
    IA_dict = dict_filer(IA_dict)
    VA_dict = dict_filer(VA_dict)
    VA_f_dict = dict_filer(VA_f_dict)
    VA_gt_dict = dict_filer(VA_gt_dict)
    VA_d_dict = dict_filer(VA_d_dict)

    if print_msg:
        print("dict adjust +++++++++++")
        print("IA_dict", IA_dict)
        # print("VA_dict",VA_dict)
        print("VA_f_dict", VA_f_dict)
        print("VA_gt_dict", VA_gt_dict)
        print("VA_d_dict", VA_d_dict)

        # split pred to basic interval
        print("split_line_set", split_line_set)

    # third circle
    split_IA1_VA1_IA2_relative_list = split_intervals(IA1_VA1_IA2_relative_list, split_line_set)
    # print("split_IA1_VA1_IA2_relative_list",split_IA1_VA1_IA2_relative_list)

    # forecasting_len=10
    # delay_len=10
    IA_pred_group_dict = {}
    VA_pred_group_dict = {}
    VA_f_pred_group_dict = {}
    VA_gt_pred_group_dict = {}
    VA_d_pred_group_dict = {}

    VA_gt1_pred_group_dict = {}
    VA_gt2_pred_group_dict = {}
    VA_gt3_pred_group_dict = {}
    VA_gt4_pred_group_dict = {}
    func_dict = {}

    # "id_"+str(label_interval_num)
    for i in range(label_interval_num):
        IA_pred_group_dict["id_" + str(i)] = []
        VA_pred_group_dict["id_" + str(i)] = []
        VA_f_pred_group_dict["id_" + str(i)] = []
        VA_gt_pred_group_dict["id_" + str(i)] = []
        VA_d_pred_group_dict["id_" + str(i)] = []
        VA_gt1_pred_group_dict["id_" + str(i)] = []
        VA_gt2_pred_group_dict["id_" + str(i)] = []
        VA_gt3_pred_group_dict["id_" + str(i)] = []
        VA_gt4_pred_group_dict["id_" + str(i)] = []
    IA_pred_group_dict["id_" + str(label_interval_num)] = []

    # 4th circle
    # for i, basic_interval in enumerate(IA1_VA1_IA2_relative_list):
    for i, basic_interval in enumerate(split_IA1_VA1_IA2_relative_list):
        # IA
        for j, (area_id, area) in enumerate(IA_dict.items()):
            if pred_in_area(basic_interval, area):
                if area_id not in IA_pred_group_dict.keys():
                    IA_pred_group_dict[area_id] = [basic_interval]
                else:
                    IA_pred_group_dict[area_id].append(basic_interval)

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

            # group by x1,x2
            VA_gt_len = area[1] - area[0]
            x1, x2 = cal_x1_x2(VA_gt_len, parameter_rho)

            # # print("area_id,area",area_id,area)
            # # # print("x1,x2",x1, x2)
            # # print("x1_ori,x2_ori",x1+area[0], x2+area[0])
            area_gt1 = [0, x1]
            area_gt2 = [x1, 2 * x1]
            area_gt3 = [2 * x1, x2]
            area_gt4 = [x2, area[1] - area[0]]
            area_gt1_ori = [area[0], x1 + area[0]]
            area_gt2_ori = [x1 + area[0], 2 * x1 + area[0]]
            area_gt3_ori = [2 * x1 + area[0], x2 + area[0]]
            area_gt4_ori = [x2 + area[0], area[1]]
            # # print("area_gt1",area_gt1)
            # # print("area_gt2",area_gt2)
            # # print("area_gt3",area_gt3)
            # # print("area_gt4",area_gt4)

            # # print("area_gt1_ori",area_gt1_ori)
            # # print("area_gt2_ori",area_gt2_ori)
            # # print("area_gt3_ori",area_gt3_ori)
            # # print("area_gt4_ori",area_gt4_ori)

            if pred_in_area(basic_interval, area_gt1_ori):
                # if area_id not in VA_gt1_pred_group_dict.keys():
                #     VA_gt1_pred_group_dict[area_id]=[basic_interval]
                # else:
                #     VA_gt1_pred_group_dict[area_id].append(basic_interval)
                VA_gt1_pred_group_dict[area_id].append(basic_interval)
            # if basic_interval == [135, 141] and area == [130, 150]:
            #     a =1
            if pred_in_area(basic_interval, area_gt2_ori):
                # if area_id not in VA_gt2_pred_group_dict.keys():
                #     VA_gt2_pred_group_dict[area_id]=[basic_interval]
                # else:
                #     VA_gt2_pred_group_dict[area_id].append(basic_interval)
                VA_gt2_pred_group_dict[area_id].append(basic_interval)
            if pred_in_area(basic_interval, area_gt3_ori):
                # if area_id not in VA_gt3_pred_group_dict.keys():
                #     VA_gt3_pred_group_dict[area_id]=[basic_interval]
                # else:
                VA_gt3_pred_group_dict[area_id].append(basic_interval)
            if pred_in_area(basic_interval, area_gt4_ori):
                # if area_id not in VA_gt4_pred_group_dict.keys():
                #     VA_gt4_pred_group_dict[area_id]=[basic_interval]
                # else:
                VA_gt4_pred_group_dict[area_id].append(basic_interval)

        # VA-d
        for j, (area_id, area) in enumerate(VA_d_dict.items()):
            if pred_in_area(basic_interval, area):
                if area_id not in VA_d_pred_group_dict.keys():
                    VA_d_pred_group_dict[area_id] = [basic_interval]
                else:
                    VA_d_pred_group_dict[area_id].append(basic_interval)
        # VA
        for j, (area_id, area) in enumerate(VA_dict.items()):
            if pred_in_area(basic_interval, area):
                if area_id not in VA_pred_group_dict.keys():
                    VA_pred_group_dict[area_id] = [basic_interval]
                else:
                    VA_pred_group_dict[area_id].append(basic_interval)
    if print_msg:
        # print pred dict
        print("IA_pred_group_dict", IA_pred_group_dict)
        print("VA_f_pred_group_dict", VA_f_pred_group_dict)
        # print("VA_gt_pred_group_dict",VA_gt_pred_group_dict)
        print("VA_gt1_pred_group_dict", VA_gt1_pred_group_dict)
        print("VA_gt2_pred_group_dict", VA_gt2_pred_group_dict)
        print("VA_gt3_pred_group_dict", VA_gt3_pred_group_dict)
        print("VA_gt4_pred_group_dict", VA_gt4_pred_group_dict)
        print("VA_d_pred_group_dict", VA_d_pred_group_dict)
        # print("VA_pred_group_dict",VA_pred_group_dict)

    d_ia_set_dict = {}
    d_ia_sign_set_dict = {}
    l_ia_set_dict = {}
    dl_score_ia_dict = {}

    dl_part_ia_score_dict = {}

    # 5th circle
    for i, (area_id, basic_interval_list) in enumerate(IA_pred_group_dict.items()):
        if basic_interval_list == []:
            dl_score_ia_dict[area_id] = 0
            l_ia_set_dict[area_id] = 0
            d_ia_set_dict[
                area_id] = 0  # only when basic_interval_list ！= [],the value are used,so set value to 0 have boe problem
            d_ia_sign_set_dict[
                area_id] = 0  # only when basic_interval_list ！= [],the value are used,so set value to 0 have boe problem
            continue
        l_ia = 0
        l_ia_set_sum = 0
        d_ia_set_sum = 0
        d_ia_sign_set_sum = 0
        ia_set_num = len(basic_interval_list)
        IA_mid = (IA_dict[area_id][0] + IA_dict[area_id][1]) / 2
        IA_len = IA_dict[area_id][1] - IA_dict[area_id][0]

        test_group = "group"
        # test_group = "no group"
        if test_group == "group":
            for basic_interval in basic_interval_list:
                l_ia += basic_interval[1] - basic_interval[0]
                # the length of fitst and last IA could be set a fixed value,min(一个周期的长度,ori len),or min(mean len,ori len)
                if i == 0:
                    base_position = IA_dict[area_id][1]
                    # d_ia_set_sum+= abs((basic_interval[1] + basic_interval[0])/2 -base_position)
                # last id of IA is label_interval_num,not label_interval_num -1
                elif i == label_interval_num:
                    base_position = IA_dict[area_id][0]
                else:
                    base_position = IA_mid
                # d_ia_set_sum+= abs((basic_interval[1] + basic_interval[0])/2 -IA_mid)
                d_ia_set_sum += abs((basic_interval[1] + basic_interval[0]) / 2 - base_position)
                # i =0 or i = label_interval_num,only have positive distance

                d_ia_sign_set_sum += (basic_interval[1] + basic_interval[0]) / 2 - base_position

            l_ia_set_dict[area_id] = l_ia
            d_ia = d_ia_set_sum / ia_set_num if ia_set_num != 0 else 0
            d_ia_set_dict[area_id] = d_ia
            d_ia_sign = d_ia_sign_set_sum / ia_set_num if ia_set_num != 0 else 0
            d_ia_sign_set_dict[area_id] = d_ia_sign

            dl_score_ia = 0
            #
            if l_ia != 0:
                if i == 0:
                    # 一边情况下就没有1了
                    # parameter_alpha1 = parameter_rate/(2+parameter_rate)
                    #
                    # debug1 = (1/2*IA_len-d_ia)
                    # debug2 = 2 / IA_len * (1/2*IA_len-d_ia)
                    # debug3 = parameter_alpha1 * 2 / IA_len * (1/2*IA_len-d_ia)
                    #
                    # debug4 = (1 - l_ia / IA_len)
                    # debug5 = (1 - parameter_alpha1) * (1 - l_ia / IA_len)

                    # dl_score_ia = parameter_alpha1 * 2 / IA_len * (1/2*IA_len-d_ia) + (1 - parameter_alpha1) * (1 - l_ia / IA_len) \
                    #     if IA_len != 0 else 0

                    parameter_alpha = parameter_rate / (1 + parameter_rate)
                    debug11 = (IA_len - d_ia)
                    debug21 = 1 / IA_len * (IA_len - d_ia)
                    debug31 = parameter_alpha / IA_len * (IA_len - d_ia)

                    debug41 = (1 - l_ia / IA_len)
                    debug51 = (1 - parameter_alpha) * (1 - l_ia / IA_len)
                    # 得分跟系数也有关系

                    dl_score_ia1 = parameter_alpha / IA_len * (IA_len - d_ia) + (1 - parameter_alpha) * (
                            1 - l_ia / IA_len) \
                        if IA_len != 0 else 0
                    # 对抗边界是2
                    if parameter_rate <= 2:
                        min_value = parameter_alpha / IA_len * (IA_len - 1 / 2 * IA_len)
                        # min_value = min_value + 0.0000001
                        dl_score_ia = map_value(dl_score_ia1, min_value, 1, 1e-10, 1)
                    else:
                        min_value = (1 - parameter_alpha) * (1 - 0 / IA_len)
                        dl_score_ia = map_value(dl_score_ia1, min_value, 1, 1e-10, 1)

                        # dl_score_ia = dl_score_ia1
                    # print("0 dl_score_ia",dl_score_ia)
                    # dl_part_ia_score_dict[area_id] = {
                    #     "parameter_alpha":parameter_alpha,
                    #     "d_score":1/IA_len * (IA_len - d_ia),
                    #     "(1 - parameter_alpha)":(1 - parameter_alpha),
                    #     "d_score_parameter":(1 - parameter_alpha) * (
                    #                 1 - l_ia / IA_len)
                    # }
                    if pred_case_id == 2:
                        debug = 1
                # last id of IA is label_interval_num,not label_interval_num -1
                elif i == label_interval_num:

                    # debug1 = (1 / 2 * IA_len - d_ia)
                    # debug2 = 2 / IA_len * (1 / 2 * IA_len - d_ia)
                    # debug3 = parameter_alpha * 2 / IA_len * (1 / 2 * IA_len - d_ia)
                    #
                    # debug4 = (1 - l_ia / IA_len)
                    # debug5 = (1 - parameter_alpha) * (1 - l_ia / IA_len)

                    # dl_score_ia = parameter_alpha * 2 / IA_len * (1/2*IA_len-d_ia) + (1 - parameter_alpha) * (1 - l_ia / IA_len) \
                    #     if IA_len != 0 else 0

                    parameter_alpha = parameter_rate / (1 + parameter_rate)
                    debug11 = (IA_len - d_ia)
                    debug21 = 1 / IA_len * (IA_len - d_ia)
                    debug31 = parameter_alpha / IA_len * (IA_len - d_ia)

                    debug41 = (1 - l_ia / IA_len)
                    debug51 = (1 - parameter_alpha) * (1 - l_ia / IA_len)
                    dl_score_ia1 = parameter_alpha / IA_len * (IA_len - d_ia) + (1 - parameter_alpha) * (
                            1 - l_ia / IA_len) \
                        if IA_len != 0 else 0
                    # x, a, b, c, d
                    if parameter_rate <= 2:
                        min_value = parameter_alpha / IA_len * (IA_len - 1 / 2 * IA_len)
                        # min_value = min_value + 0.0000001
                        dl_score_ia = map_value(dl_score_ia1, min_value, 1, 1e-10, 1)
                    else:
                        min_value = (1 - parameter_alpha) * (1 - 0 / IA_len)
                        dl_score_ia = map_value(dl_score_ia1, min_value, 1, 1e-10, 1)
                        # dl_score_ia = map_value(dl_score_ia,parameter_alpha,1,0,1)
                    # dl_part_ia_score_dict[area_id] = {
                    #     "parameter_alpha": parameter_alpha,
                    #     "d_score": 1 / IA_len * (IA_len - d_ia),
                    #     "(1 - parameter_alpha)": (1 - parameter_alpha),
                    #     "d_score_parameter": (1 - parameter_alpha) * (
                    #             1 - l_ia / IA_len)
                    # }
                    if pred_case_id == 2:
                        debug = 1
                else:

                    parameter_alpha = parameter_rate / (2 + parameter_rate)
                    dl_score_ia = parameter_alpha * 2 / IA_len * d_ia + (1 - parameter_alpha) * (1 - l_ia / IA_len) \
                        if IA_len != 0 else 0
                    if pred_case_id == 2:
                        print("  parameter_alpha", parameter_alpha)
                        print("  d_ia", d_ia)
                        print("  d_score", 2 / IA_len * d_ia)
                        print("  d_score_parameter", parameter_alpha * 2 / IA_len * d_ia)
                        print("  (1 - parameter_alpha)", (1 - parameter_alpha))
                        print("  l_ia", l_ia)
                        print("  l_score", (1 - l_ia / IA_len))
                        print("  l_score_parameter", (1 - parameter_alpha) * (1 - l_ia / IA_len))
                        print("  IA_len", IA_len)
                        debug = 1
                dl_part_ia_score_dict
        else:
            if print_msg:
                print("basic_interval_list num", len(basic_interval_list))
            dl_score_ia_sum = 0
            for basic_interval in basic_interval_list:
                if i == 0:
                    base_position = IA_dict[area_id][1]
                elif i == label_interval_num:
                    base_position = IA_dict[area_id][0]
                else:
                    base_position = IA_mid

                l_ia = basic_interval[1] - basic_interval[0]
                d_ia = abs((basic_interval[1] + basic_interval[0]) / 2 - base_position)
                if pred_case_id == 2:
                    debug = 1
                dl_score_ia = 0
                #
                if l_ia != 0:
                    if i == 0:
                        parameter_alpha = parameter_rate / (1 + parameter_rate)
                        dl_score_ia1 = parameter_alpha / IA_len * (IA_len - d_ia) + (1 - parameter_alpha) * (
                                1 - l_ia / IA_len) \
                            if IA_len != 0 else 0
                        # 对抗边界是2
                        if parameter_rate <= 2:
                            min_value = parameter_alpha / IA_len * (IA_len - 1 / 2 * IA_len)
                            dl_score_ia = map_value(dl_score_ia1, min_value, 1, 1e-10, 1)
                        else:
                            min_value = (1 - parameter_alpha) * (1 - 0 / IA_len)
                            dl_score_ia = map_value(dl_score_ia1, min_value, 1, 1e-10, 1)
                        if print_msg:
                            print("0 dl_score_ia", dl_score_ia)
                        if pred_case_id == 2:
                            debug = 1
                    elif i == label_interval_num:
                        parameter_alpha = parameter_rate / (1 + parameter_rate)
                        debug11 = (IA_len - d_ia)
                        debug21 = 1 / IA_len * (IA_len - d_ia)
                        debug31 = parameter_alpha / IA_len * (IA_len - d_ia)

                        debug41 = (1 - l_ia / IA_len)
                        debug51 = (1 - parameter_alpha) * (1 - l_ia / IA_len)
                        dl_score_ia1 = parameter_alpha / IA_len * (IA_len - d_ia) + (1 - parameter_alpha) * (
                                1 - l_ia / IA_len) \
                            if IA_len != 0 else 0
                        # x, a, b, c, d
                        if parameter_rate <= 2:
                            min_value = parameter_alpha / IA_len * (IA_len - 1 / 2 * IA_len)
                            # min_value = min_value + 0.0000001
                            dl_score_ia = map_value(dl_score_ia1, min_value, 1, 1e-10, 1)
                        else:
                            min_value = (1 - parameter_alpha) * (1 - 0 / IA_len)
                            dl_score_ia = map_value(dl_score_ia1, min_value, 1, 1e-10, 1)
                            # dl_score_ia = map_value(dl_score_ia,parameter_alpha,1,0,1)
                        if pred_case_id == 2:
                            debug = 1
                    else:

                        parameter_alpha = parameter_rate / (2 + parameter_rate)
                        dl_score_ia = parameter_alpha * 2 / IA_len * d_ia + (1 - parameter_alpha) * (1 - l_ia / IA_len) \
                            if IA_len != 0 else 0
                        if pred_case_id == 2:
                            if print_msg:
                                print("  parameter_alpha", parameter_alpha)
                                print("  d_ia", d_ia)
                                print("  d_score", 2 / IA_len * d_ia)
                                print("  d_score_parameter", parameter_alpha * 2 / IA_len * d_ia)
                                print("  (1 - parameter_alpha)", (1 - parameter_alpha))
                                print("  l_ia", l_ia)
                                print("  l_score", (1 - l_ia / IA_len))
                                print("  l_score_parameter", (1 - parameter_alpha) * (1 - l_ia / IA_len))
                                print("  IA_len", IA_len)
                                debug = 1

                # sum
                dl_score_ia_sum += dl_score_ia
                l_ia_set_sum += basic_interval[1] - basic_interval[0]
                d_ia_set_sum += abs((basic_interval[1] + basic_interval[0]) / 2 - base_position)
                d_ia_sign_set_sum += (basic_interval[1] + basic_interval[0]) / 2 - base_position

            # sum
            # test_no_group_type = "sum"
            test_no_group_type = "average"
            if test_no_group_type == "sum":
                dl_score_ia = dl_score_ia_sum
                l_ia_set_dict[area_id] = l_ia_set_sum
                d_ia_set_dict[area_id] = d_ia_set_sum
                d_ia_sign_set_dict[area_id] = d_ia_sign_set_sum
            else:
                dl_score_ia = dl_score_ia_sum / len(basic_interval_list) if len(basic_interval_list) != 0 else 0
                l_ia_set_dict[area_id] = l_ia_set_sum / len(basic_interval_list) if len(basic_interval_list) != 0 else 0
                d_ia_set_dict[area_id] = d_ia_set_sum / len(basic_interval_list) if len(basic_interval_list) != 0 else 0
                d_ia_sign_set_dict[area_id] = d_ia_sign_set_sum
        dl_score_ia_dict[area_id] = dl_score_ia
    #     deal empty and no area
    if print_msg:
        print("d_ia_set_dict", d_ia_set_dict)
        print("d_ia_sign_set_dict", d_ia_sign_set_dict)
        print("l_ia_set_dict", l_ia_set_dict)
    # print("dl_score_ia_dict",dl_score_ia_dict)

    # cal label_function_gt
    # 6th circle
    for i, (area_id, area) in enumerate(VA_gt_dict.items()):

        VA_gt_len = area[1] - area[0]
        x1, x2 = cal_x1_x2(VA_gt_len, parameter_rho)
        x1_ori, x2_ori = x1 + area[0], x2 + area[0]
        x_mid = 2 * x1
        x_mid_ori = x_mid + area[0]
        # x1_ori,x2_ori = x1+area[0], x2+area[0]
        area_gt1 = [0, x1]
        area_gt2 = [x1, 2 * x1]
        area_gt3 = [2 * x1, x2]
        area_gt4 = [x2, VA_gt_len]
        area_gt1_ori = [area[0], x1 + area[0]]
        area_gt2_ori = [x1 + area[0], 2 * x1 + area[0]]
        area_gt3_ori = [2 * x1 + area[0], x2 + area[0]]
        area_gt4_ori = [x2 + area[0], area[1]]
        if print_msg:
            print("VA_gt_len", VA_gt_len)

        if parameter_rho < 0:
            parameter_b1, parameter_gama1 = 1 / 2, 0
            parameter_b2, parameter_gama2 = 1 / 2, 0
        else:
            parameter_b1, parameter_gama1 = cal_power_function_coefficient(x1, parameter_lamda)
            parameter_b2, parameter_gama2 = cal_power_function_coefficient(x2 - x_mid, parameter_lamda)

        func_dict.update({
            area_id + "_gt1": {"area": area_gt1_ori,
                               "area_start": area_gt1_ori[0],
                               "parameter_a": parameter_b1,
                               "parameter_beta": parameter_gama1,
                               "func": "va_gt1_func",
                               "x_tp": x1,
                               "gt_len": VA_gt_len,
                               "area_len": None,
                               "parameter_rho": parameter_rho
                               },
            area_id + "_gt2": {"area": area_gt2_ori,
                               "area_start": area_gt1_ori[0],
                               "parameter_a": parameter_b1,
                               "parameter_beta": parameter_gama1,
                               "func": "va_gt2_func",
                               "x_tp": x1,
                               "gt_len": VA_gt_len,
                               "area_len": None,
                               "parameter_rho": parameter_rho
                               },
            area_id + "_gt3": {"area": area_gt3_ori,
                               "area_start": area_gt1_ori[0],
                               "parameter_a": parameter_b2,
                               "parameter_beta": parameter_gama2,
                               "func": "va_gt3_func",
                               "x_tp": x2,
                               "gt_len": VA_gt_len,
                               "area_len": None,
                               "parameter_rho": parameter_rho
                               },
            area_id + "_gt4": {"area": area_gt4_ori,
                               "area_start": area_gt1_ori[0],
                               "parameter_a": parameter_b2,
                               "parameter_beta": parameter_gama2,
                               "func": "va_gt4_func",
                               "x_tp": x2,
                               "gt_len": VA_gt_len,
                               "area_len": None,
                               "parameter_rho": parameter_rho
                               },

        })

        s_a_gt1 = cal_integral_in_range_gt_power_func1(area_gt1_ori, parameter_b1, parameter_gama1, x_tp=x1,
                                                       gt_len=VA_gt_len, area_start=area_gt1_ori[0],
                                                       parameter_rho=parameter_rho)
        s_a_gt2 = cal_integral_in_range_gt_power_func2(area_gt2_ori, parameter_b1, parameter_gama1, x_tp=x1,
                                                       gt_len=VA_gt_len, area_start=area_gt1_ori[0],
                                                       parameter_rho=parameter_rho)
        s_a_gt3 = cal_integral_in_range_gt_power_func3(area_gt3_ori, parameter_b2, parameter_gama2, x_tp=x2,
                                                       gt_len=VA_gt_len, area_start=area_gt1_ori[0],
                                                       parameter_rho=parameter_rho)
        s_a_gt4 = cal_integral_in_range_gt_power_func4(area_gt4_ori, parameter_b2, parameter_gama2, x_tp=x2,
                                                       gt_len=VA_gt_len, area_start=area_gt1_ori[0],
                                                       parameter_rho=parameter_rho)

        s_a_gt = s_a_gt1 + s_a_gt2 + s_a_gt3 + s_a_gt4

        func_dict[area_id + "_gt1"]["area_s"] = s_a_gt1
        func_dict[area_id + "_gt2"]["area_s"] = s_a_gt2
        func_dict[area_id + "_gt3"]["area_s"] = s_a_gt3
        func_dict[area_id + "_gt4"]["area_s"] = s_a_gt4

        method = 2

        va_f_area = VA_f_dict[area_id]
        s_a_f = 0
        if va_f_area == []:
            pass
        else:
            VA_f_len = va_f_area[1] - va_f_area[0]
            parameter_beta1 = 1
            parameter_a1 = 1 / (2 * VA_f_len)

            greater_than_th = False

            func_dict[area_id + "_f"] = {"area": va_f_area,
                                         "area_start": va_f_area[0],
                                         "parameter_a": parameter_a1,
                                         "parameter_beta": parameter_beta1,
                                         "func": "va_f_func",
                                         "x_tp": None,
                                         "gt_len": None,
                                         "area_len": VA_f_len,
                                         "greater_than_th": greater_than_th
                                         }

            # f
            s_max_f = cal_integral_in_range_f_power_func(va_f_area, parameter_a1, parameter_beta1, VA_f_len, VA_f_len,
                                                         area_start=va_f_area[0], greater_than_th=greater_than_th)
            func_dict[area_id + "_f"]["area_s"] = s_max_f

            s_rate = s_max_f / s_a_gt
            func_dict[area_id + "_f_a"] = func_dict[area_id + "_f"]
            s_a_f = s_max_f
            if s_rate > parameter_theta:
                greater_than_th = True

                # method1
                if method == 1:
                    parameter_beta1_a = VA_f_len / (2 * parameter_theta * s_a_gt) - 1
                    parameter_a1_a = 1 / (2 * pow(VA_f_len, parameter_beta1_a))
                # method2
                else:
                    parameter_beta1_a = VA_f_len / (2 * parameter_theta * s_a_gt) - 1  # same with method1
                    parameter_a1_a = VA_f_len / pow(1 / 2, parameter_beta1_a)

                s_a_f = cal_integral_in_range_f_power_func(va_f_area, parameter_a1_a, parameter_beta1_a, VA_f_len,
                                                           VA_f_len, area_start=va_f_area[0], method=method,
                                                           greater_than_th=greater_than_th)
                func_dict[area_id + "_f_a"] = {
                    "area": va_f_area,
                    "area_start": va_f_area[0],
                    "area_s": s_a_f,
                    "parameter_a": parameter_a1_a,
                    "parameter_beta": parameter_beta1_a,
                    # "func": "va_f_func",
                    # "func": "va_f_reverse_func",
                    "x_tp": None,
                    "gt_len": None,
                    "area_len": VA_f_len,
                    "greater_than_th": greater_than_th
                }
                if method == 1:
                    func_dict[area_id + "_f_a"]["func"] = "va_f_func"
                else:
                    func_dict[area_id + "_f_a"]["func"] = "va_f_reverse_func"

        # d
        va_d_area = VA_d_dict[area_id]
        s_a_d = 0
        if va_d_area == []:
            pass
        else:

            if pred_case_id == 7 and area_id == "id_2":
                debug = 1

            VA_d_len = va_d_area[1] - va_d_area[0]
            parameter_beta2 = 1
            parameter_a2 = 1 / (2 * VA_d_len)

            greater_than_th = False
            s_max_d = cal_integral_in_range_d_power_func(va_d_area, parameter_a2, parameter_beta2, VA_d_len, VA_d_len,
                                                         area_start=va_d_area[0], greater_than_th=greater_than_th)
            func_dict[area_id + "_d"] = {
                "area": va_d_area,
                "area_start": va_d_area[0],
                "parameter_a": parameter_a2,
                "parameter_beta": parameter_beta2,
                "func": "va_d_func",
                "x_tp": None,
                "gt_len": None,
                "area_len": VA_d_len,
                "greater_than_th": greater_than_th
            }

            func_dict[area_id + "_d"]["area_s"] = s_max_d

            func_dict[area_id + "_d_a"] = func_dict[area_id + "_d"]
            s_rate = s_max_d / s_a_gt
            s_a_d = s_max_d

            if s_rate > parameter_theta:
                greater_than_th = True

                # method1
                if method == 1:
                    parameter_beta2_a = VA_d_len / (2 * parameter_theta * s_a_gt) - 1
                    parameter_a2_a = 1 / (2 * pow(VA_d_len, parameter_beta2_a))
                else:
                    # method2
                    parameter_beta2_a = VA_d_len / (2 * parameter_theta * s_a_gt) - 1  # same with method1
                    parameter_a2_a = VA_d_len / pow(1 / 2, parameter_beta2_a)

                s_a_d = cal_integral_in_range_d_power_func(va_d_area, parameter_a2_a, parameter_beta2_a, VA_d_len,
                                                           VA_d_len,
                                                           area_start=va_d_area[0], method=method,
                                                           greater_than_th=greater_than_th)
                func_dict[area_id + "_d_a"] = {
                    "area": va_d_area,
                    "area_start": va_d_area[0],
                    "area_s": s_a_d,
                    "parameter_a": parameter_a2_a,
                    "parameter_beta": parameter_beta2_a,
                    # "func": "va_d_func",
                    # "func": "va_d_reverse_func",
                    "x_tp": None,
                    "gt_len": None,
                    "area_len": VA_d_len,
                    "greater_than_th": greater_than_th

                }

                if method == 1:
                    func_dict[area_id + "_d_a"]["func"] = "va_d_func"
                else:
                    func_dict[area_id + "_d_a"]["func"] = "va_d_reverse_func"

        s_a_func_dict[area_id] = {
            "s_a_va_gt_len": VA_gt_len,
            "s_a_va_gt": s_a_gt,
            "s_a_va": s_a_f + s_a_gt + s_a_d,
            "x1": x1,
            "x2": x2,
            "x_mid": x_mid,
            "x1_ori": x1_ori,
            "x2_ori": x2_ori,
            "x_mid_ori": x_mid_ori,
        }
    # plot func
    if pred_case_id == 7 and area_id == "id_2":
        debug = 1
        # plot_func_multi(func_dict)
    # plot_func_multi(func_dict)
    # plot_func_multi_paper(func_dict,window_length)

    import pprint

    # print("s_a_func_dict")
    # pprint(s_a_func_dict, indent=4)

    # cal recall_va,detection_score
    recall_va_list = []
    recall_va_dict = {}
    detection_score_ia_i_dict = {}
    precision_va_dict = {}

    gt_num = label_interval_num
    pred_func_dict = {}
    basic_interval_integral_dict = {}

    # 7th circle
    for i in range(gt_num):
        area_id = "id_" + str(i)
        # print("area_id",area_id)
        area_id_back = "id_" + str(i + 1)
        VA_f_i_pred_group = VA_f_pred_group_dict[area_id]
        VA_gt1_i_pred_group = VA_gt1_pred_group_dict[area_id]
        VA_gt2_i_pred_group = VA_gt2_pred_group_dict[area_id]
        VA_gt3_i_pred_group = VA_gt3_pred_group_dict[area_id]
        VA_gt4_i_pred_group = VA_gt4_pred_group_dict[area_id]
        VA_d_i_pred_group = VA_d_pred_group_dict[area_id]

        IA_pred_i_group = IA_pred_group_dict[area_id]
        IA_pred_i_back_group = IA_pred_group_dict[area_id_back]
        VA_pred_i_group = VA_pred_group_dict[area_id]

        pred_integral_all = 0

        # # f
        va_f_area = VA_f_dict[area_id]
        if va_f_area == []:
            pass
        else:
            VA_f_len = va_f_area[1] - va_f_area[0]
            parameter_a1 = func_dict[area_id + "_f_a"]["parameter_a"]
            parameter_beta1 = func_dict[area_id + "_f_a"]["parameter_beta"]
            greater_than_th = func_dict[area_id + "_f_a"]["greater_than_th"]

            for j, basic_interval in enumerate(VA_f_i_pred_group):
                if basic_interval != []:

                    pred_func_dict[area_id + "_f_a" + "_" + str(j)] = {
                        "area": basic_interval,
                        "area_start": va_f_area[0],
                        "parameter_a": parameter_a1,
                        "parameter_beta": parameter_beta1,
                        # "func":"va_f_func",
                        "x_tp": None,
                        "gt_len": None,
                        "area_len": VA_f_len
                    }
                    if not greater_than_th:
                        pred_func_dict[area_id + "_f_a" + "_" + str(j)]["func"] = "va_d_func"
                    else:
                        if method == 1:
                            pred_func_dict[area_id + "_f_a" + "_" + str(j)]["func"] = "va_f_func"
                        else:
                            pred_func_dict[area_id + "_f_a" + "_" + str(j)]["func"] = "va_f_reverse_func"

                    cal_integral_basic_interval_f = cal_integral_in_range_f_power_func(basic_interval, parameter_a1,
                                                                                       parameter_beta1, VA_f_len,
                                                                                       VA_f_len,
                                                                                       area_start=va_f_area[0],
                                                                                       method=method,
                                                                                       greater_than_th=greater_than_th)
                    pred_integral_all += cal_integral_basic_interval_f
                    basic_interval_integral_dict[area_id + "_f" + "_" + str(j)] = cal_integral_basic_interval_f

        # gt1-gt4
        s_a_gt_len = s_a_func_dict[area_id]["s_a_va_gt_len"]
        s_a_va = s_a_func_dict[area_id]["s_a_va"]
        parameter_b1 = func_dict[area_id + "_gt1"]["parameter_a"]
        parameter_gama1 = func_dict[area_id + "_gt1"]["parameter_beta"]
        parameter_b2 = func_dict[area_id + "_gt3"]["parameter_a"]
        parameter_gama2 = func_dict[area_id + "_gt3"]["parameter_beta"]
        x1 = s_a_func_dict[area_id]["x1"]
        x2 = s_a_func_dict[area_id]["x2"]
        area_gt = VA_gt_dict[area_id]
        VA_gt_len = area_gt[1] - area_gt[0]

        area_gt1 = func_dict[area_id + "_gt1"]["area"]
        area_gt2 = func_dict[area_id + "_gt2"]["area"]
        area_gt3 = func_dict[area_id + "_gt3"]["area"]
        area_gt4 = func_dict[area_id + "_gt4"]["area"]
        area_start = func_dict[area_id + "_gt1"]["area_start"]

        for j, basic_interval in enumerate(VA_gt1_i_pred_group):
            if basic_interval != []:
                pred_func_dict[area_id + "_gt1" + "_" + str(j)] = {
                    "area": basic_interval,
                    "area_start": area_start,
                    "parameter_a": parameter_b1,
                    "parameter_beta": parameter_gama1,
                    "func": "va_gt1_func",
                    "x_tp": None,
                    "gt_len": VA_gt_len,
                    "area_len": None,
                    "parameter_rho": parameter_rho
                }
                cal_integral_basic_interval_gt1 = cal_integral_in_range_gt_power_func1(basic_interval, parameter_b1,
                                                                                       parameter_gama1, x_tp=x1,
                                                                                       gt_len=VA_gt_len,
                                                                                       area_start=area_start,
                                                                                       parameter_rho=parameter_rho)
                pred_integral_all += cal_integral_basic_interval_gt1
                basic_interval_integral_dict[area_id + "_gt1" + "_" + str(j)] = cal_integral_basic_interval_gt1

        for j, basic_interval in enumerate(VA_gt2_i_pred_group):
            if basic_interval != []:
                pred_func_dict[area_id + "_gt2" + "_" + str(j)] = {
                    "area": basic_interval,
                    "area_start": area_start,
                    "parameter_a": parameter_b1,
                    "parameter_beta": parameter_gama1,
                    "func": "va_gt2_func",
                    "x_tp": x1,
                    "gt_len": VA_gt_len,
                    "area_len": None,
                    "parameter_rho": parameter_rho
                }
                cal_integral_basic_interval_gt2 = cal_integral_in_range_gt_power_func2(basic_interval, parameter_b1,
                                                                                       parameter_gama1, x_tp=x1,
                                                                                       gt_len=VA_gt_len,
                                                                                       area_start=area_start,
                                                                                       parameter_rho=parameter_rho)
                pred_integral_all += cal_integral_basic_interval_gt2
                basic_interval_integral_dict[area_id + "_gt2" + "_" + str(j)] = cal_integral_basic_interval_gt2

        for j, basic_interval in enumerate(VA_gt3_i_pred_group):
            if basic_interval != []:
                pred_func_dict[area_id + "_gt3" + "_" + str(j)] = {
                    "area": basic_interval,
                    "area_start": area_start,
                    "parameter_a": parameter_b2,
                    "parameter_beta": parameter_gama2,
                    "func": "va_gt3_func",
                    "x_tp": x2,
                    "gt_len": VA_gt_len,
                    "area_len": None,
                    "parameter_rho": parameter_rho

                }
                cal_integral_basic_interval_gt3 = cal_integral_in_range_gt_power_func3(basic_interval, parameter_b2,
                                                                                       parameter_gama2, x_tp=x2,
                                                                                       gt_len=VA_gt_len,
                                                                                       area_start=area_start,
                                                                                       parameter_rho=parameter_rho)
                pred_integral_all += cal_integral_basic_interval_gt3
                basic_interval_integral_dict[area_id + "_gt3" + "_" + str(j)] = cal_integral_basic_interval_gt3

        for j, basic_interval in enumerate(VA_gt4_i_pred_group):
            if basic_interval != []:
                pred_func_dict[area_id + "_gt4" + "_" + str(j)] = {
                    "area": basic_interval,
                    "area_start": area_start,
                    "parameter_a": parameter_b2,
                    "parameter_beta": parameter_gama2,
                    "func": "va_gt4_func",
                    "x_tp": x2,
                    "gt_len": VA_gt_len,
                    "area_len": None,
                    "parameter_rho": parameter_rho
                }
                cal_integral_basic_interval_gt4 = cal_integral_in_range_gt_power_func4(basic_interval, parameter_b2,
                                                                                       parameter_gama2, x_tp=x2,
                                                                                       gt_len=VA_gt_len,
                                                                                       area_start=area_start,
                                                                                       parameter_rho=parameter_rho)
                pred_integral_all += cal_integral_basic_interval_gt4
                basic_interval_integral_dict[area_id + "_gt4" + "_" + str(j)] = cal_integral_basic_interval_gt4

        # # d
        va_d_area = VA_d_dict[area_id]
        if va_d_area == []:
            pass
        else:
            VA_d_len = va_d_area[1] - va_d_area[0]
            parameter_a2 = func_dict[area_id + "_d_a"]["parameter_a"]
            parameter_beta2 = func_dict[area_id + "_d_a"]["parameter_beta"]
            greater_than_th = func_dict[area_id + "_d_a"]["greater_than_th"]
            if pred_case_id == 7 and area_id == "id_2":
                debug = 1

            for j, basic_interval in enumerate(VA_d_i_pred_group):
                if basic_interval != []:
                    pred_func_dict[area_id + "_d_a" + "_" + str(j)] = {
                        "area": basic_interval,
                        "area_start": va_d_area[0],
                        "parameter_a": parameter_a2,
                        "parameter_beta": parameter_beta2,
                        # "func":"va_d_func",
                        "x_tp": None,
                        "gt_len": None,
                        "area_len": VA_d_len
                    }
                    if not greater_than_th:
                        pred_func_dict[area_id + "_d_a" + "_" + str(j)]["func"] = "va_d_func"
                    else:
                        if method == 1:
                            pred_func_dict[area_id + "_d_a" + "_" + str(j)]["func"] = "va_d_func"
                        else:
                            pred_func_dict[area_id + "_d_a" + "_" + str(j)]["func"] = "va_d_reverse_func"

                    cal_integral_basic_interval_d = cal_integral_in_range_d_power_func(basic_interval, parameter_a2,
                                                                                       parameter_beta2, VA_d_len,
                                                                                       VA_d_len,
                                                                                       area_start=va_d_area[0],
                                                                                       method=method,
                                                                                       greater_than_th=greater_than_th)
                    pred_integral_all += cal_integral_basic_interval_d
                    basic_interval_integral_dict[area_id + "_d" + "_" + str(j)] = cal_integral_basic_interval_d

        # print("pred_integral_all",pred_integral_all)
        va_area = VA_dict[area_id]
        recall_va_i = pred_integral_all / s_a_va
        recall_va_list.append(recall_va_i)
        if pred_case_id == 7:
            debug = 1
        recall_va_dict[area_id] = recall_va_i

        detection_score_i = 1 if VA_pred_group_dict[area_id] != [] else 0
        detection_score_ia_i_dict[area_id] = detection_score_i
        # invalid part punish in is not enough
        # detection
        if cal_mode == "detection":
            precision_va_dict[area_id] = detection_score_i

        precision_i_group = IA_pred_i_group + IA_pred_i_back_group + VA_pred_i_group

        precision_i_all = 0
        precision_i_valid = 0
        for precision_i_item in precision_i_group:
            precision_i_all += precision_i_item[1] - precision_i_item[0]
        for precision_i_item in VA_pred_i_group:
            precision_i_valid += precision_i_item[1] - precision_i_item[0]
        if pred_case_id == 6:
            debug = 1
        precision_va_i = precision_i_valid / precision_i_all if precision_i_all != 0 else 0

        # precision also consider invalid part
        # proportion
        if cal_mode == "proportion":
            precision_va_dict[area_id] = precision_va_i

    if print_msg:
        # print basic_interval_integral
        print("basic_interval_integral_dict", basic_interval_integral_dict)

    # pot pred func
    # if pred_case_id == 7 and area_id == "id_2":
    #     debug = 1
    #     plot_func_multi(pred_func_dict)
    # plot_func_multi(pred_func_dict)

    if print_msg:
        # print recall_va_dict,precision_va_dict,dl_score_ia_dict
        print("recall_va_dict", recall_va_dict)
        print("precision_va_dict", precision_va_dict)
        print("dl_score_ia_dict", dl_score_ia_dict)

    adjusted_dl_score_ia_dict = {}
    adjusted_dl_score_ia_rate_dict = {}
    adjusted_dl_score_ia_dict_ratio_contribution = {}

    if pred_case_id == 2:
        debug = 1
    area_id_0 = "id_" + str(0)
    area_id_num = "id_" + str(gt_num)
    # two logics
    logic = 1
    # logic = 3
    if logic == 1:
        # logic 1
        #     one_exist_value = 1/2
        one_exist_value = 1
        # 8th circle
        for i in range(gt_num + 1):
            area_id = "id_" + str(i)
            area_id_front = "id_" + str(i - 1)
            area_id_back = "id_" + str(i + 1)

            ia_area = IA_dict[area_id]
            l_ia = l_ia_set_dict[area_id]
            d_ia = d_ia_set_dict[area_id]
            d_ia_sign = d_ia_sign_set_dict[area_id]
            ia_len = ia_area[1] - ia_area[0] if ia_area != [] else 0

            if pred_case_id == 22:
                debug = 1

            if i == 0:
                if IA_dict[area_id] == []:
                    # if IA_dict[area_id_back] == [] and area_id_back == area_id_num:
                    if area_id_back == area_id_num:
                        if IA_dict[area_id_back] == []:  # 自己为空，后面一个为最后一个，后面一个为空
                            ratio_before = 0
                            ratio_after = 1 / 2
                        else:  # 自己为空，后面一个为最后一个，后面一个不为空
                            ratio_before = 0
                            ratio_after = 0
                    else:
                        ratio_before = 0
                        ratio_after = 1 / 2

                    if VA_pred_group_dict[area_id] != []:
                        dl_score_ia_i_contribution = 1
                    else:
                        dl_score_ia_i_contribution = 0
                else:
                    # 数据里至少有一个异常，至少有两个ia，如果没有异常，那么IA_dict=[]
                    if IA_pred_group_dict[area_id] != []:
                        if area_id_back == area_id_num:
                            if IA_dict[area_id_back] == []:  # 自己不为空，后面一个为最后一个，后面一个为空
                                ratio_before = 0
                                ratio_after = 1
                            else:  # 自己不为空，后面一个为最后一个，后面一个不为空
                                ratio_before = 0
                                ratio_after = 1 / 2
                        else:
                            ratio_before = 0
                            ratio_after = 1 / 2
                        dl_score_ia_i_contribution = dl_score_ia_dict[area_id]
                    else:
                        if VA_pred_group_dict[area_id] != []:
                            if area_id_back == area_id_num:
                                if IA_dict[area_id_back] == []:  # 自己不为空，后面一个为最后一个，后面一个为空
                                    ratio_before = 0
                                    ratio_after = 1
                                else:  # 自己不为空，后面一个为最后一个，后面一个不为空
                                    ratio_before = 0
                                    ratio_after = 1 / 2
                            else:
                                ratio_before = 0
                                ratio_after = 1 / 2
                            dl_score_ia_i_contribution = 1
                        else:
                            if area_id_back == area_id_num:
                                if IA_dict[area_id_back] == []:  # 自己不为空，后面一个为最后一个，后面一个为空
                                    ratio_before = 0
                                    ratio_after = 1
                                else:  # 自己不为空，后面一个为最后一个，后面一个不为空
                                    ratio_before = 0
                                    ratio_after = 1 / 2
                            else:
                                ratio_before = 0
                                ratio_after = 1 / 2
                            dl_score_ia_i_contribution = 0
            elif i == gt_num:
                if IA_dict[area_id] == []:
                    if area_id_front == area_id_0:
                        if IA_dict[area_id_front] == []:  # 自己为空，前面一个为最后一个，前面一个为空
                            ratio_before = 1 / 2
                            ratio_after = 0
                        else:  # 自己为空，前面一个为第一个，前面一个不为空
                            ratio_before = 0
                            ratio_after = 0
                    else:
                        ratio_before = 1 / 2
                        ratio_after = 0

                    if VA_pred_group_dict[area_id_front] != []:
                        dl_score_ia_i_contribution = 1
                    else:
                        dl_score_ia_i_contribution = 0
                else:
                    if IA_pred_group_dict[area_id] != []:
                        if area_id_front == area_id_0:
                            if IA_dict[area_id_front] == []:
                                ratio_before = 1
                                ratio_after = 0
                            else:
                                ratio_before = 1 / 2
                                ratio_after = 0
                        else:
                            ratio_before = 1 / 2
                            ratio_after = 0
                        dl_score_ia_i_contribution = dl_score_ia_dict[area_id]
                    else:
                        if VA_pred_group_dict[area_id_front] != []:
                            if area_id_front == area_id_0:
                                if IA_dict[area_id_front] == []:
                                    ratio_before = 1
                                    ratio_after = 0
                                else:
                                    ratio_before = 1 / 2
                                    ratio_after = 0
                            else:
                                ratio_before = 1 / 2
                                ratio_after = 0
                            dl_score_ia_i_contribution = 1
                        else:
                            if area_id_front == area_id_0:
                                if IA_dict[area_id_front] == []:
                                    ratio_before = 1
                                    ratio_after = 0
                                else:
                                    ratio_before = 1 / 2
                                    ratio_after = 0
                            else:
                                ratio_before = 1 / 2
                                ratio_after = 0
                            dl_score_ia_i_contribution = 0
            else:
                if IA_pred_group_dict[area_id] != []:
                    IA_len = IA_dict[area_id][1] - IA_dict[area_id][0]
                    IA_mid = (IA_dict[area_id][1] + IA_dict[area_id][0]) / 2
                    # IA_start = IA_dict[area_id][0]
                    IA_end = IA_dict[area_id][1]
                    ratio_before = (IA_end - (IA_mid + d_ia_sign_set_dict[area_id])) / IA_len
                    ratio_after = 1 - ratio_before

                    # ratio_before = 1 / 2
                    # ratio_after = 1 / 2

                    # if d_ia_sign > 0:
                    #     ratio_before = (ia_len / 2 - d_ia) / ia_len
                    #     ratio_after = (ia_len / 2 + d_ia) / ia_len
                    # elif d_ia_sign < 0:
                    #     ratio_before = (ia_len / 2 + d_ia) / ia_len
                    #     ratio_after = (ia_len / 2 - d_ia) / ia_len
                    # else:
                    #     ratio_before = 1/2
                    #     ratio_after = 1/2
                    dl_score_ia_i_contribution = dl_score_ia_dict[area_id]
                else:
                    ratio_before = 1 / 2
                    ratio_after = 1 / 2
                    if VA_pred_group_dict[area_id_front] != [] and VA_pred_group_dict[area_id] != []:
                        # ratio_before = 1/2
                        # ratio_after = 1/2
                        dl_score_ia_i_contribution = 1
                    elif VA_pred_group_dict[area_id_front] == [] and VA_pred_group_dict[area_id] == []:
                        # ratio_before = 1/2
                        # ratio_after = 1/2
                        dl_score_ia_i_contribution = 0
                    else:
                        # if VA_pred_group_dict[area_id_front] != []:
                        #     ratio_before = 1
                        #     ratio_after = 0
                        # else:
                        #     ratio_before = 0
                        #     ratio_after = 1
                        dl_score_ia_i_contribution = one_exist_value

            adjusted_dl_score_ia_dict[area_id] = dl_score_ia_i_contribution
            dl_score_ia_i_before_contribution = dl_score_ia_i_contribution * ratio_before
            dl_score_ia_i_after_contribution = dl_score_ia_i_contribution * ratio_after
            adjusted_dl_score_ia_rate_dict[area_id] = [ratio_before,ratio_after]
            adjusted_dl_score_ia_dict_ratio_contribution[area_id] = [dl_score_ia_i_before_contribution,
                                                                     dl_score_ia_i_after_contribution]

    debug = 1
    # logic 2
    # one_exist_value = 1/2
    # for i in range(gt_num + 1):
    #     area_id = "id_" + str(i)
    #     area_id_front = "id_" + str(i - 1)
    #     area_id_back = "id_" + str(i + 1)
    #
    #     ia_area = IA_dict[area_id]
    #     l_ia = l_ia_set_dict[area_id]
    #     d_ia = d_ia_set_dict[area_id]
    #     d_ia_sign = d_ia_sign_set_dict[area_id]
    #     ia_len = ia_area[1] - ia_area[0] if ia_area != [] else 0
    #
    #     if i == 0:
    #         ratio_before = 0
    #         ratio_after = 1
    #         if IA_pred_group_dict[area_id] != []:
    #             dl_score_ia_i_contribution = dl_score_ia_dict[area_id]
    #         else:
    #             if VA_pred_group_dict[area_id] != []:
    #                 dl_score_ia_i_contribution = 1 / 2
    #             else:
    #                 dl_score_ia_i_contribution = 0
    #     elif i == gt_num:
    #         ratio_before = 1
    #         ratio_after = 0
    #         if IA_pred_group_dict[area_id] != []:
    #             dl_score_ia_i_contribution = dl_score_ia_dict[area_id]
    #         else:
    #             if VA_pred_group_dict[area_id_front] != []:
    #                 dl_score_ia_i_contribution = 1 / 2
    #             else:
    #                 dl_score_ia_i_contribution = 0
    #     else:
    #         if IA_pred_group_dict[area_id] != []:
    #             ratio_before = 1 / 2
    #             ratio_after = 1 / 2
    #             # if d_ia_sign > 0:
    #             #     ratio_before = (ia_len / 2 - d_ia) / ia_len
    #             #     ratio_after = (ia_len / 2 + d_ia) / ia_len
    #             # elif d_ia_sign < 0:
    #             #     ratio_before = (ia_len / 2 + d_ia) / ia_len
    #             #     ratio_after = (ia_len / 2 - d_ia) / ia_len
    #             # else:
    #             #     ratio_before = 1/2
    #             #     ratio_after = 1/2
    #             dl_score_ia_i_contribution = dl_score_ia_dict[area_id]
    #         else:
    #             # ratio_before = 1 / 2
    #             # ratio_after = 1 / 2
    #             if VA_pred_group_dict[area_id_front] != [] and VA_pred_group_dict[area_id] != []:
    #                 ratio_before = 1/2
    #                 ratio_after = 1/2
    #                 dl_score_ia_i_contribution = 1
    #             elif VA_pred_group_dict[area_id_front] == [] and VA_pred_group_dict[area_id] == []:
    #                 ratio_before = 1/2
    #                 ratio_after = 1/2
    #                 dl_score_ia_i_contribution = 0
    #             else:
    #                 if VA_pred_group_dict[area_id_front] != []:
    #                     ratio_before = 1
    #                     ratio_after = 0
    #                 else:
    #                     ratio_before = 0
    #                     ratio_after = 1
    #                 dl_score_ia_i_contribution = one_exist_value

    # logic 3
    if logic == 3:
        # one_exist_value = 1/2
        one_exist_value = 1
        for i in range(gt_num + 1):
            area_id = "id_" + str(i)
            area_id_front = "id_" + str(i - 1)
            area_id_back = "id_" + str(i + 1)

            ia_area = IA_dict[area_id]
            l_ia = l_ia_set_dict[area_id]
            d_ia = d_ia_set_dict[area_id]
            d_ia_sign = d_ia_sign_set_dict[area_id]
            ia_len = ia_area[1] - ia_area[0] if ia_area != [] else 0

            if i == 0:
                ratio_before = 0
                ratio_after = 1
                if IA_dict[area_id] == []:
                    dl_score_ia_i_contribution = 0
                else:
                    if IA_pred_group_dict[area_id] != []:
                        dl_score_ia_i_contribution = dl_score_ia_dict[area_id]
                    else:
                        if VA_pred_group_dict[area_id] != []:
                            dl_score_ia_i_contribution = 1
                        else:
                            dl_score_ia_i_contribution = 0
            elif i == gt_num:
                ratio_before = 1
                ratio_after = 0
                if IA_dict[area_id] == []:
                    dl_score_ia_i_contribution = 0
                else:
                    if IA_pred_group_dict[area_id] != []:
                        dl_score_ia_i_contribution = dl_score_ia_dict[area_id]
                    else:
                        if VA_pred_group_dict[area_id_front] != []:
                            dl_score_ia_i_contribution = 1
                        else:
                            dl_score_ia_i_contribution = 0
            else:
                if IA_pred_group_dict[area_id] != []:
                    # ratio_before = 1 / 2
                    # ratio_after = 1 / 2
                    if d_ia_sign > 0:
                        ratio_before = (ia_len / 2 - d_ia) / ia_len
                        ratio_after = (ia_len / 2 + d_ia) / ia_len
                    elif d_ia_sign < 0:
                        ratio_before = (ia_len / 2 + d_ia) / ia_len
                        ratio_after = (ia_len / 2 - d_ia) / ia_len
                    else:
                        ratio_before = 1 / 2
                        ratio_after = 1 / 2
                    dl_score_ia_i_contribution = dl_score_ia_dict[area_id]
                else:
                    # ratio_before = 1 / 2
                    # ratio_after = 1 / 2
                    if VA_pred_group_dict[area_id_front] != [] and VA_pred_group_dict[area_id] != []:
                        ratio_before = 1 / 2
                        ratio_after = 1 / 2
                        dl_score_ia_i_contribution = 2
                    elif VA_pred_group_dict[area_id_front] == [] and VA_pred_group_dict[area_id] == []:
                        ratio_before = 1 / 2
                        ratio_after = 1 / 2
                        dl_score_ia_i_contribution = 0
                    else:
                        if VA_pred_group_dict[area_id_front] != []:
                            ratio_before = 1
                            ratio_after = 0
                        else:
                            ratio_before = 0
                            ratio_after = 1
                        dl_score_ia_i_contribution = one_exist_value

            adjusted_dl_score_ia_dict[area_id] = dl_score_ia_i_contribution
            dl_score_ia_i_before_contribution = dl_score_ia_i_contribution * ratio_before
            dl_score_ia_i_after_contribution = dl_score_ia_i_contribution * ratio_after
            adjusted_dl_score_ia_dict_ratio_contribution[area_id] = [dl_score_ia_i_before_contribution,
                                                                     dl_score_ia_i_after_contribution]

    if print_msg:
        print("adjusted_dl_score_ia_dict", adjusted_dl_score_ia_dict)
        print("adjusted_dl_score_ia_rate_dict", adjusted_dl_score_ia_rate_dict)
        print("adjusted_dl_score_ia_dict_ratio_contribution", adjusted_dl_score_ia_dict_ratio_contribution)
    event_recall_i_list = []
    event_recall_i_dict = {}
    event_recall_i_dict_w0 = {}
    event_recall_i_dict_w1 = {}
    event_precision_i_dict = {}
    event_precision_i_dict_w0 = {}
    event_precision_i_dict_w1 = {}

    # 9th circle
    for i in range(gt_num):
        area_id = "id_" + str(i)
        if print_msg:
            print(area_id)
        area_id_back = "id_" + str(i + 1)
        recall_va_i = recall_va_dict[area_id]
        # if VA_pred_group_dict[area_id] != [] and IA_pred_group_dict[area_id] == [] and IA_pred_group_dict[area_id_back] == [] :
        #     dl_score_ia_i = 1
        #     dl_score_ia_i_back = 1
        # else:
        #     dl_score_ia_i = dl_score_ia_dict[area_id]
        #     dl_score_ia_i_back = dl_score_ia_dict[area_id_back]

        dl_score_ia_i_after_contribution = adjusted_dl_score_ia_dict_ratio_contribution[area_id][1]
        dl_score_ia_i_back_before_contribution = adjusted_dl_score_ia_dict_ratio_contribution[area_id_back][0]

        if print_msg:
            print(" dl_score_ia_i_after_contribution", dl_score_ia_i_after_contribution)
            print(" recall_va_i,", recall_va_i)
            print(" dl_score_ia_i_back_before_contribution", dl_score_ia_i_back_before_contribution)

        # event_recall_i = (1-2*parameter_eta)*recall_va_i + parameter_eta*(dl_score_ia_i_after_contribution+ dl_score_ia_i_back_before_contribution)
        if pred_case_id == 26:
            debug = 1

        # logic 1
        if logic == 1:
            event_recall_i = (1 - parameter_eta/2) * recall_va_i + parameter_eta/2 * (
                    dl_score_ia_i_after_contribution + dl_score_ia_i_back_before_contribution)
            # w=0
            event_recall_i_w0 = (1 - 0/2) * recall_va_i + 0/2 * (
                    dl_score_ia_i_after_contribution + dl_score_ia_i_back_before_contribution)
            # w=1
            event_recall_i_w1 = (1 - 1/2) * recall_va_i + 1/2 * (
                    dl_score_ia_i_after_contribution + dl_score_ia_i_back_before_contribution)
            # w=1
            event_recall_i_w1 = (1 - 1/2) * recall_va_i + 1/2 * (
                    dl_score_ia_i_after_contribution + dl_score_ia_i_back_before_contribution)
        # logic 3
        if logic == 3:
            if i == 0 and IA_dict[area_id] == []:
                event_recall_i = (1 - parameter_eta) * recall_va_i + parameter_eta * 1 / 1 * (
                    dl_score_ia_i_back_before_contribution)
            elif i == gt_num - 1 and IA_dict[area_id_back] == []:
                event_recall_i = (1 - parameter_eta) * recall_va_i + parameter_eta * 1 / 1 * (
                    dl_score_ia_i_after_contribution)
            else:
                event_recall_i = (1 - parameter_eta) * recall_va_i + parameter_eta * 1 / 2 * (
                        dl_score_ia_i_after_contribution + dl_score_ia_i_back_before_contribution)
        # event_recall_i_list
        event_recall_i_dict[area_id] = event_recall_i
        event_recall_i_dict_w0[area_id] = event_recall_i_w0
        event_recall_i_dict_w1[area_id] = event_recall_i_w1

        precision_va_i = precision_va_dict[area_id]
        if print_msg:
            print(" dl_score_ia_i_after_contribution", dl_score_ia_i_after_contribution)
            print(" precision_va_i,", precision_va_i)
            print(" dl_score_ia_i_back_before_contribution", dl_score_ia_i_back_before_contribution)

        # event_precision_i = (1-2*parameter_eta)*precision_va_i + parameter_eta*(dl_score_ia_i_after_contribution+ dl_score_ia_i_back_before_contribution)

        # logic 1
        if logic == 1:
            event_precision_i = (1 - parameter_eta/2) * precision_va_i + parameter_eta/2 * (
                    dl_score_ia_i_after_contribution + dl_score_ia_i_back_before_contribution)
            # w=0
            event_precision_i_w0 = (1 - 0/2) * precision_va_i + 0/2 * (
                    dl_score_ia_i_after_contribution + dl_score_ia_i_back_before_contribution)
            # w=1
            event_precision_i_w1 = (1 - 1/2) * precision_va_i + 1/2 * (
                    dl_score_ia_i_after_contribution + dl_score_ia_i_back_before_contribution)

        # logic 3
        if logic == 3:
            if i == 0 and IA_dict[area_id] == []:
                event_precision_i = (1 - parameter_eta) * precision_va_i + parameter_eta * 1 / 1 * (
                    dl_score_ia_i_back_before_contribution)
            elif i == gt_num - 1 and IA_dict[area_id_back] == []:
                event_precision_i = (1 - parameter_eta) * precision_va_i + parameter_eta * 1 / 1 * (
                    dl_score_ia_i_after_contribution)
            else:
                event_precision_i = (1 - parameter_eta) * precision_va_i + parameter_eta * 1 / 2 * (
                        dl_score_ia_i_after_contribution + dl_score_ia_i_back_before_contribution)
        event_precision_i_dict[area_id] = event_precision_i
        event_precision_i_dict_w0[area_id] = event_precision_i_w0
        event_precision_i_dict_w1[area_id] = event_precision_i_w1

    if print_msg:
        print("event_recall_i_dict", event_recall_i_dict)
        print("event_precision_i_dict", event_precision_i_dict)

    if pred_case_id == 2:
        debug = 1
    event_recall = np.array(list(event_recall_i_dict.values())).mean()
    # print(event_recall)
    event_precision = np.array(list(event_precision_i_dict.values())).mean()
    # print(event_precision)

    event_recall_w0 = np.array(list(event_recall_i_dict_w0.values())).mean()
    event_precision_w0 = np.array(list(event_precision_i_dict_w0.values())).mean()
    event_recall_w1 = np.array(list(event_recall_i_dict_w1.values())).mean()
    event_precision_w1 = np.array(list(event_precision_i_dict_w1.values())).mean()

    #
    # print("======== meata_recall")
    # print(" event_recall",event_recall)
    # print(" event_recall_w0",event_recall_w0)
    # print(" event_recall_w1",event_recall_w1)
    #
    # print(" event_recall_i_dict",event_recall_i_dict)
    # print(" event_recall_i_dict_w0",event_recall_i_dict_w0)
    # print(" event_recall_i_dict_w1",event_recall_i_dict_w1)
    # print(" adjusted_dl_score_ia_dict_ratio_contribution",adjusted_dl_score_ia_dict_ratio_contribution)

    recall_info = {
        "event_recall": event_recall,
        "event_recall_w0": event_recall_w0,
        "event_recall_w1": event_recall_w1,
        "event_recall_i_dict": event_recall_i_dict,
        "event_recall_i_dict_w0": event_recall_i_dict_w0,
        "event_recall_i_dict_w1": event_recall_i_dict_w1,
        "adjusted_dl_score_ia_dict_ratio_contribution": adjusted_dl_score_ia_dict_ratio_contribution,
    }

    meata_f1 = compute_f1_score(event_precision, event_recall)
    meata_f1_w0 = compute_f1_score(event_precision_w0, event_recall_w0)
    meata_f1_w1 = compute_f1_score(event_precision_w1, event_recall_w1)

    # gt_label_point_array = convert_interval_to_point_array(label_interval_ranges, time_series_length=window_length)
    # ont_count = np.count_nonzero(gt_label_point_array == 1)
    # anomaly_rate = ont_count / window_length
    # if anomaly_rate > 0.2:
    #     print("anomaly_rate exceeds threshold")
    #     no_random_coefficient = 1
    # else:
    #     no_random_coefficient = cal_no_random_measure_coefficient(pred_label_point_array, window_length)

    if thresh_id == 94:
        d = 1
    coefficient_meata_f1 = no_random_coefficient * meata_f1
    coefficient_event_recall = no_random_coefficient * event_recall
    coefficient_event_precision = no_random_coefficient * event_precision
    print("meata_f1", meata_f1)
    print("coefficient_meata_f1", coefficient_meata_f1)

    coefficient_meata_f1_w0 = no_random_coefficient * meata_f1_w0
    coefficient_meata_f1_w1 = no_random_coefficient * meata_f1_w1

    # return event_recall,event_precision,meata_f1,no_random_meata_f1
    # return event_recall,event_precision,meata_f1,coefficient_meata_f1,VA_f_dict,VA_d_dict,coefficient_event_recall,coefficient_event_precision
    return event_recall, event_precision, meata_f1, coefficient_meata_f1, VA_f_dict, VA_d_dict, \
        coefficient_event_recall, coefficient_event_precision, \
        meata_f1_w0, coefficient_meata_f1_w0, meata_f1_w1, coefficient_meata_f1_w1, \
        recall_info

def meata_tp_merge_first(label_interval_ranges, IA1_VA1_IA2_relative_list, window_length, parameter_dict, pred_label_point_array,
          pred_case_id=None, cal_mode="proportion"):
    print_msg = False
    # print_msg = True

    # parameter_alpha = parameter_dict["parameter_alpha"]
    parameter_rate = parameter_dict["parameter_rate"]

    parameter_theta = parameter_dict["parameter_theta"]
    # parameter_w = parameter_dict["parameter_w"]
    # parameter_switch_f =parameter_dict["parameter_switch_f"]
    # parameter_switch_d = parameter_dict["parameter_switch_d"]
    VA_f_window = parameter_dict["forecasting_window"]
    VA_d_window = parameter_dict["delay_window"]

    parameter_rho = parameter_dict["parameter_rho"]
    parameter_lamda = parameter_dict["parameter_lamda"]

    parameter_eta = parameter_dict["parameter_eta"]

    if print_msg:
        print("VA_f_window", VA_f_window)
        print("VA_d_window", VA_d_window)

    f_window_rate = 0
    d_window_rate = 0
    f_d_window_len = VA_f_window + VA_d_window
    if f_d_window_len != 0:
        f_window_rate = VA_f_window / f_d_window_len
        d_window_rate = 1 - f_window_rate

    IA_dict = {}
    VA_dict = {}
    VA_f_dict = {}
    VA_gt_dict = {}
    VA_d_dict = {}

    # pred generated is continuous range a,b(len) -> [a,a+b(len)),range of [a,b] -> [a,b) -> point of [a,b-1],point of [a,b] -> [a,b+1)

    # cal every f/d of gt,split IA and VA

    label_interval_num = len(label_interval_ranges)

    s_a_func_dict = {}

    for i, label_interval_range in enumerate(label_interval_ranges):
        # adjust now and before,add before
        VA_gt_now_start, VA_gt_now_end = label_interval_range
        # VA_f_now_start = max(VA_gt_now_start - VA_f_window,0) if parameter_switch_f == True else max(VA_gt_now_start - 3,0)
        VA_f_now_start = max(VA_gt_now_start - VA_f_window, 0)
        VA_f_now_end = VA_gt_now_start
        VA_d_now_start = VA_gt_now_end
        # VA_d_now_end = min(VA_gt_now_end + VA_f_window,window_length) if parameter_switch_d == True else min(VA_gt_now_end + 3,window_length)
        VA_d_now_end = min(VA_gt_now_end + VA_d_window, window_length)
        # # this way below is more fast
        # if i == 0:
        #     VA_f_now_start = max(VA_f_now_start,0)
        # if i == label_interval_num:
        #     VA_d_now_end = min(VA_d_now_end,window_length)
        if i > 0:
            # adjust now and before
            if VA_d_before_end > VA_f_now_start:
                intersection_range_start = max(VA_f_now_start, VA_gt_before_end)
                intersection_range_end = min(VA_d_before_end, VA_gt_now_start)
                intersection_range = [intersection_range_start, intersection_range_end]
                intersection_len = intersection_range_end - intersection_range_start
                #         if intersection_len == 1 and (parameter_switch_f == True and parameter_switch_d == True) and parameter_rho == 1/2:
                # #             f prior
                #             VA_d_before_end -=1
                #         else:
                #         VA_d_before_end = VA_f_now_start = intersection_range_start + round(parameter_w*intersection_len)
                #         VA_d_before_end = VA_f_now_start = intersection_range_start + round(parameter_w*intersection_len,2)
                #         VA_f_now_start = intersection_range_start + round(f_window_rate*intersection_len,2)
                #         VA_d_before_end = intersection_range_start + round(d_window_rate*intersection_len,2)
                VA_f_now_start = intersection_range_start + round(f_window_rate * intersection_len)
                VA_d_before_end = intersection_range_start + round(d_window_rate * intersection_len)
            #     add before
            IA_dict["id_" + str(i)] = [VA_d_before_end, VA_f_now_start]
            VA_dict["id_" + str(i - 1)] = [VA_f_before_start, VA_d_before_end]
            VA_f_dict["id_" + str(i - 1)] = [VA_f_before_start, VA_f_before_end]
            VA_gt_dict["id_" + str(i - 1)] = [VA_gt_before_start, VA_gt_before_end]
            VA_d_dict["id_" + str(i - 1)] = [VA_d_before_start, VA_d_before_end]

            # add by order

            split_line_set.add(VA_f_before_start)
            split_line_set.add(VA_f_before_end)

            split_line_set.add(VA_gt_before_start)
            split_line_set.add(VA_gt_before_end)

            split_line_set.add(VA_d_before_start)
            split_line_set.add(VA_d_before_end)

            split_line_set.add(VA_d_before_end)
            split_line_set.add(VA_f_now_start)

            VA_gt_before_len = VA_gt_before_end - VA_gt_before_start
            x1, x2 = cal_x1_x2(VA_gt_before_len, parameter_rho)

            # # print("area_id,area",area_id,area)
            # # # print("x1,x2",x1, x2)
            # # print("x1_ori,x2_ori",x1+VA_gt_before_start[0], x2+VA_gt_before_start[0])
            x1_ori, x2_ori = x1 + VA_gt_before_start, x2 + VA_gt_before_start
            x_mid_ori = 2 * x1 + VA_gt_before_start
            split_line_set.add(x1_ori)
            split_line_set.add(x_mid_ori)
            split_line_set.add(x2_ori)
        else:
            IA_dict["id_" + str(i)] = [0, VA_f_now_start]
            split_line_set.add(VA_f_now_start)

        VA_gt_before_start, VA_gt_before_end = VA_gt_now_start, VA_gt_now_end
        VA_f_before_start = VA_f_now_start
        VA_f_before_end = VA_f_now_end
        VA_d_before_start = VA_d_now_start
        VA_d_before_end = VA_d_now_end

    # last one VA and last two IA
    IA_dict["id_" + str(label_interval_num)] = [VA_d_before_end, window_length]
    VA_dict["id_" + str(label_interval_num - 1)] = [VA_f_before_start, VA_d_before_end]
    VA_f_dict["id_" + str(label_interval_num - 1)] = [VA_f_before_start, VA_f_before_end]
    VA_gt_dict["id_" + str(label_interval_num - 1)] = [VA_gt_before_start, VA_gt_before_end]
    VA_d_dict["id_" + str(label_interval_num - 1)] = [VA_d_before_start, VA_d_before_end]

    VA_gt_before_len = VA_gt_before_end - VA_gt_before_start
    x1, x2 = cal_x1_x2(VA_gt_before_len, parameter_rho)

    # # print("area_id,area",area_id,area)
    # # # print("x1,x2",x1, x2)
    # # print("x1_ori,x2_ori",x1+VA_gt_before_start[0], x2+VA_gt_before_start[0])
    x1_ori, x2_ori = x1 + VA_gt_before_start, x2 + VA_gt_before_start
    x_mid_ori = 2 * x1 + VA_gt_before_start
    split_line_set.add(x1_ori)
    split_line_set.add(x_mid_ori)
    split_line_set.add(x2_ori)

    # add by order

    split_line_set.add(VA_f_before_start)
    split_line_set.add(VA_f_before_end)

    split_line_set.add(VA_gt_before_start)
    split_line_set.add(VA_gt_before_end)

    split_line_set.add(VA_d_before_start)
    split_line_set.add(VA_d_before_end)

    if print_msg:
        # print area dict
        # # print("split area")
        print("IA_dict", IA_dict)
        # print("VA_dict",VA_dict)
        print("VA_f_dict", VA_f_dict)
        print("VA_gt_dict", VA_gt_dict)
        print("VA_d_dict", VA_d_dict)

    def dict_filer(dict_data: dict):
        for i, (area_id, area_range) in enumerate(dict_data.items()):
            if area_range != []:
                dict_data[area_id] = [] if area_range[0] == area_range[1] else area_range
        return dict_data

    IA_dict = dict_filer(IA_dict)
    VA_dict = dict_filer(VA_dict)
    VA_f_dict = dict_filer(VA_f_dict)
    VA_gt_dict = dict_filer(VA_gt_dict)
    VA_d_dict = dict_filer(VA_d_dict)

    if print_msg:
        print("dict adjust +++++++++++")
        print("IA_dict", IA_dict)
        # print("VA_dict",VA_dict)
        print("VA_f_dict", VA_f_dict)
        print("VA_gt_dict", VA_gt_dict)
        print("VA_d_dict", VA_d_dict)

        # split pred to basic interval
        print("split_line_set", split_line_set)

    split_IA1_VA1_IA2_relative_list = split_intervals(IA1_VA1_IA2_relative_list, split_line_set)
    # print("split_IA1_VA1_IA2_relative_list",split_IA1_VA1_IA2_relative_list)

    # forecasting_len=10
    # delay_len=10
    IA_pred_group_dict = {}
    VA_pred_group_dict = {}
    VA_f_pred_group_dict = {}
    VA_gt_pred_group_dict = {}
    VA_d_pred_group_dict = {}

    VA_gt1_pred_group_dict = {}
    VA_gt2_pred_group_dict = {}
    VA_gt3_pred_group_dict = {}
    VA_gt4_pred_group_dict = {}
    func_dict = {}

    # "id_"+str(label_interval_num)
    for i in range(label_interval_num):
        IA_pred_group_dict["id_" + str(i)] = []
        VA_pred_group_dict["id_" + str(i)] = []
        VA_f_pred_group_dict["id_" + str(i)] = []
        VA_gt_pred_group_dict["id_" + str(i)] = []
        VA_d_pred_group_dict["id_" + str(i)] = []
        VA_gt1_pred_group_dict["id_" + str(i)] = []
        VA_gt2_pred_group_dict["id_" + str(i)] = []
        VA_gt3_pred_group_dict["id_" + str(i)] = []
        VA_gt4_pred_group_dict["id_" + str(i)] = []
    IA_pred_group_dict["id_" + str(label_interval_num)] = []

    # for i, basic_interval in enumerate(IA1_VA1_IA2_relative_list):
    for i, basic_interval in enumerate(split_IA1_VA1_IA2_relative_list):
        # IA
        for j, (area_id, area) in enumerate(IA_dict.items()):
            if pred_in_area(basic_interval, area):
                if area_id not in IA_pred_group_dict.keys():
                    IA_pred_group_dict[area_id] = [basic_interval]
                else:
                    IA_pred_group_dict[area_id].append(basic_interval)

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

            # group by x1,x2
            VA_gt_len = area[1] - area[0]
            x1, x2 = cal_x1_x2(VA_gt_len, parameter_rho)

            # # print("area_id,area",area_id,area)
            # # # print("x1,x2",x1, x2)
            # # print("x1_ori,x2_ori",x1+area[0], x2+area[0])
            area_gt1 = [0, x1]
            area_gt2 = [x1, 2 * x1]
            area_gt3 = [2 * x1, x2]
            area_gt4 = [x2, area[1] - area[0]]
            area_gt1_ori = [area[0], x1 + area[0]]
            area_gt2_ori = [x1 + area[0], 2 * x1 + area[0]]
            area_gt3_ori = [2 * x1 + area[0], x2 + area[0]]
            area_gt4_ori = [x2 + area[0], area[1]]
            # # print("area_gt1",area_gt1)
            # # print("area_gt2",area_gt2)
            # # print("area_gt3",area_gt3)
            # # print("area_gt4",area_gt4)

            # # print("area_gt1_ori",area_gt1_ori)
            # # print("area_gt2_ori",area_gt2_ori)
            # # print("area_gt3_ori",area_gt3_ori)
            # # print("area_gt4_ori",area_gt4_ori)

            if pred_in_area(basic_interval, area_gt1_ori):
                # if area_id not in VA_gt1_pred_group_dict.keys():
                #     VA_gt1_pred_group_dict[area_id]=[basic_interval]
                # else:
                #     VA_gt1_pred_group_dict[area_id].append(basic_interval)
                VA_gt1_pred_group_dict[area_id].append(basic_interval)
            # if basic_interval == [135, 141] and area == [130, 150]:
            #     a =1
            if pred_in_area(basic_interval, area_gt2_ori):
                # if area_id not in VA_gt2_pred_group_dict.keys():
                #     VA_gt2_pred_group_dict[area_id]=[basic_interval]
                # else:
                #     VA_gt2_pred_group_dict[area_id].append(basic_interval)
                VA_gt2_pred_group_dict[area_id].append(basic_interval)
            if pred_in_area(basic_interval, area_gt3_ori):
                # if area_id not in VA_gt3_pred_group_dict.keys():
                #     VA_gt3_pred_group_dict[area_id]=[basic_interval]
                # else:
                VA_gt3_pred_group_dict[area_id].append(basic_interval)
            if pred_in_area(basic_interval, area_gt4_ori):
                # if area_id not in VA_gt4_pred_group_dict.keys():
                #     VA_gt4_pred_group_dict[area_id]=[basic_interval]
                # else:
                VA_gt4_pred_group_dict[area_id].append(basic_interval)

        # VA-d
        for j, (area_id, area) in enumerate(VA_d_dict.items()):
            if pred_in_area(basic_interval, area):
                if area_id not in VA_d_pred_group_dict.keys():
                    VA_d_pred_group_dict[area_id] = [basic_interval]
                else:
                    VA_d_pred_group_dict[area_id].append(basic_interval)
        # VA
        for j, (area_id, area) in enumerate(VA_dict.items()):
            if pred_in_area(basic_interval, area):
                if area_id not in VA_pred_group_dict.keys():
                    VA_pred_group_dict[area_id] = [basic_interval]
                else:
                    VA_pred_group_dict[area_id].append(basic_interval)
    if print_msg:
        # print pred dict
        print("IA_pred_group_dict", IA_pred_group_dict)
        print("VA_f_pred_group_dict", VA_f_pred_group_dict)
        # print("VA_gt_pred_group_dict",VA_gt_pred_group_dict)
        print("VA_gt1_pred_group_dict", VA_gt1_pred_group_dict)
        print("VA_gt2_pred_group_dict", VA_gt2_pred_group_dict)
        print("VA_gt3_pred_group_dict", VA_gt3_pred_group_dict)
        print("VA_gt4_pred_group_dict", VA_gt4_pred_group_dict)
        print("VA_d_pred_group_dict", VA_d_pred_group_dict)
        # print("VA_pred_group_dict",VA_pred_group_dict)

    d_ia_set_dict = {}
    d_ia_sign_set_dict = {}
    l_ia_set_dict = {}
    dl_score_ia_dict = {}

    dl_part_ia_score_dict = {}

    for i, (area_id, basic_interval_list) in enumerate(IA_pred_group_dict.items()):
        if basic_interval_list == []:
            dl_score_ia_dict[area_id] = 0
            l_ia_set_dict[area_id] = 0
            d_ia_set_dict[
                area_id] = 0  # only when basic_interval_list ！= [],the value are used,so set value to 0 have boe problem
            d_ia_sign_set_dict[
                area_id] = 0  # only when basic_interval_list ！= [],the value are used,so set value to 0 have boe problem
            continue
        l_ia = 0
        l_ia_set_sum = 0
        d_ia_set_sum = 0
        d_ia_sign_set_sum = 0
        ia_set_num = len(basic_interval_list)
        IA_mid = (IA_dict[area_id][0] + IA_dict[area_id][1]) / 2
        IA_len = IA_dict[area_id][1] - IA_dict[area_id][0]

        test_group = "group"
        # test_group = "no group"
        if test_group == "group":
            for basic_interval in basic_interval_list:
                l_ia += basic_interval[1] - basic_interval[0]
                # the length of fitst and last IA could be set a fixed value,min(一个周期的长度,ori len),or min(mean len,ori len)
                if i == 0:
                    base_position = IA_dict[area_id][1]
                    # d_ia_set_sum+= abs((basic_interval[1] + basic_interval[0])/2 -base_position)
                # last id of IA is label_interval_num,not label_interval_num -1
                elif i == label_interval_num:
                    base_position = IA_dict[area_id][0]
                else:
                    base_position = IA_mid
                # d_ia_set_sum+= abs((basic_interval[1] + basic_interval[0])/2 -IA_mid)
                d_ia_set_sum += abs((basic_interval[1] + basic_interval[0]) / 2 - base_position)
                # i =0 or i = label_interval_num,only have positive distance

                d_ia_sign_set_sum += (basic_interval[1] + basic_interval[0]) / 2 - base_position

            l_ia_set_dict[area_id] = l_ia
            d_ia = d_ia_set_sum / ia_set_num if ia_set_num != 0 else 0
            d_ia_set_dict[area_id] = d_ia
            d_ia_sign = d_ia_sign_set_sum / ia_set_num if ia_set_num != 0 else 0
            d_ia_sign_set_dict[area_id] = d_ia_sign

            dl_score_ia = 0
            #
            if l_ia != 0:
                if i == 0:
                    # 一边情况下就没有1了
                    # parameter_alpha1 = parameter_rate/(2+parameter_rate)
                    #
                    # debug1 = (1/2*IA_len-d_ia)
                    # debug2 = 2 / IA_len * (1/2*IA_len-d_ia)
                    # debug3 = parameter_alpha1 * 2 / IA_len * (1/2*IA_len-d_ia)
                    #
                    # debug4 = (1 - l_ia / IA_len)
                    # debug5 = (1 - parameter_alpha1) * (1 - l_ia / IA_len)

                    # dl_score_ia = parameter_alpha1 * 2 / IA_len * (1/2*IA_len-d_ia) + (1 - parameter_alpha1) * (1 - l_ia / IA_len) \
                    #     if IA_len != 0 else 0

                    parameter_alpha = parameter_rate / (1 + parameter_rate)
                    debug11 = (IA_len - d_ia)
                    debug21 = 1 / IA_len * (IA_len - d_ia)
                    debug31 = parameter_alpha / IA_len * (IA_len - d_ia)

                    debug41 = (1 - l_ia / IA_len)
                    debug51 = (1 - parameter_alpha) * (1 - l_ia / IA_len)
                    # 得分跟系数也有关系

                    dl_score_ia1 = parameter_alpha / IA_len * (IA_len - d_ia) + (1 - parameter_alpha) * (
                            1 - l_ia / IA_len) \
                        if IA_len != 0 else 0
                    # 对抗边界是2
                    if parameter_rate <= 2:
                        min_value = parameter_alpha / IA_len * (IA_len - 1 / 2 * IA_len)
                        # min_value = min_value + 0.0000001
                        dl_score_ia = map_value(dl_score_ia1, min_value, 1, 1e-10, 1)
                    else:
                        min_value = (1 - parameter_alpha) * (1 - 0 / IA_len)
                        dl_score_ia = map_value(dl_score_ia1, min_value, 1, 1e-10, 1)

                        # dl_score_ia = dl_score_ia1
                    # print("0 dl_score_ia",dl_score_ia)
                    # dl_part_ia_score_dict[area_id] = {
                    #     "parameter_alpha":parameter_alpha,
                    #     "d_score":1/IA_len * (IA_len - d_ia),
                    #     "(1 - parameter_alpha)":(1 - parameter_alpha),
                    #     "d_score_parameter":(1 - parameter_alpha) * (
                    #                 1 - l_ia / IA_len)
                    # }
                    if pred_case_id == 2:
                        debug = 1
                # last id of IA is label_interval_num,not label_interval_num -1
                elif i == label_interval_num:

                    # debug1 = (1 / 2 * IA_len - d_ia)
                    # debug2 = 2 / IA_len * (1 / 2 * IA_len - d_ia)
                    # debug3 = parameter_alpha * 2 / IA_len * (1 / 2 * IA_len - d_ia)
                    #
                    # debug4 = (1 - l_ia / IA_len)
                    # debug5 = (1 - parameter_alpha) * (1 - l_ia / IA_len)

                    # dl_score_ia = parameter_alpha * 2 / IA_len * (1/2*IA_len-d_ia) + (1 - parameter_alpha) * (1 - l_ia / IA_len) \
                    #     if IA_len != 0 else 0

                    parameter_alpha = parameter_rate / (1 + parameter_rate)
                    debug11 = (IA_len - d_ia)
                    debug21 = 1 / IA_len * (IA_len - d_ia)
                    debug31 = parameter_alpha / IA_len * (IA_len - d_ia)

                    debug41 = (1 - l_ia / IA_len)
                    debug51 = (1 - parameter_alpha) * (1 - l_ia / IA_len)
                    dl_score_ia1 = parameter_alpha / IA_len * (IA_len - d_ia) + (1 - parameter_alpha) * (
                                1 - l_ia / IA_len) \
                        if IA_len != 0 else 0
                    # x, a, b, c, d
                    if parameter_rate <= 2:
                        min_value = parameter_alpha / IA_len * (IA_len - 1 / 2 * IA_len)
                        # min_value = min_value + 0.0000001
                        dl_score_ia = map_value(dl_score_ia1, min_value, 1, 1e-10, 1)
                    else:
                        min_value = (1 - parameter_alpha) * (1 - 0 / IA_len)
                        dl_score_ia = map_value(dl_score_ia1, min_value, 1, 1e-10, 1)
                        # dl_score_ia = map_value(dl_score_ia,parameter_alpha,1,0,1)
                    # dl_part_ia_score_dict[area_id] = {
                    #     "parameter_alpha": parameter_alpha,
                    #     "d_score": 1 / IA_len * (IA_len - d_ia),
                    #     "(1 - parameter_alpha)": (1 - parameter_alpha),
                    #     "d_score_parameter": (1 - parameter_alpha) * (
                    #             1 - l_ia / IA_len)
                    # }
                    if pred_case_id == 2:
                        debug = 1
                else:

                    parameter_alpha = parameter_rate / (2 + parameter_rate)
                    dl_score_ia = parameter_alpha * 2 / IA_len * d_ia + (1 - parameter_alpha) * (1 - l_ia / IA_len) \
                        if IA_len != 0 else 0
                    if pred_case_id == 2:
                        print("  parameter_alpha", parameter_alpha)
                        print("  d_ia", d_ia)
                        print("  d_score", 2 / IA_len * d_ia)
                        print("  d_score_parameter", parameter_alpha * 2 / IA_len * d_ia)
                        print("  (1 - parameter_alpha)", (1 - parameter_alpha))
                        print("  l_ia", l_ia)
                        print("  l_score", (1 - l_ia / IA_len))
                        print("  l_score_parameter", (1 - parameter_alpha) * (1 - l_ia / IA_len))
                        print("  IA_len", IA_len)
                        debug = 1
                dl_part_ia_score_dict
        else:
            if print_msg:
                print("basic_interval_list num", len(basic_interval_list))
            dl_score_ia_sum = 0
            for basic_interval in basic_interval_list:
                if i == 0:
                    base_position = IA_dict[area_id][1]
                elif i == label_interval_num:
                    base_position = IA_dict[area_id][0]
                else:
                    base_position = IA_mid

                l_ia = basic_interval[1] - basic_interval[0]
                d_ia = abs((basic_interval[1] + basic_interval[0]) / 2 - base_position)
                if pred_case_id == 2:
                    debug = 1
                dl_score_ia = 0
                #
                if l_ia != 0:
                    if i == 0:
                        parameter_alpha = parameter_rate / (1 + parameter_rate)
                        dl_score_ia1 = parameter_alpha / IA_len * (IA_len - d_ia) + (1 - parameter_alpha) * (
                                1 - l_ia / IA_len) \
                            if IA_len != 0 else 0
                        # 对抗边界是2
                        if parameter_rate <= 2:
                            min_value = parameter_alpha / IA_len * (IA_len - 1 / 2 * IA_len)
                            dl_score_ia = map_value(dl_score_ia1, min_value, 1, 1e-10, 1)
                        else:
                            min_value = (1 - parameter_alpha) * (1 - 0 / IA_len)
                            dl_score_ia = map_value(dl_score_ia1, min_value, 1, 1e-10, 1)
                        if print_msg:
                            print("0 dl_score_ia", dl_score_ia)
                        if pred_case_id == 2:
                            debug = 1
                    elif i == label_interval_num:
                        parameter_alpha = parameter_rate / (1 + parameter_rate)
                        debug11 = (IA_len - d_ia)
                        debug21 = 1 / IA_len * (IA_len - d_ia)
                        debug31 = parameter_alpha / IA_len * (IA_len - d_ia)

                        debug41 = (1 - l_ia / IA_len)
                        debug51 = (1 - parameter_alpha) * (1 - l_ia / IA_len)
                        dl_score_ia1 = parameter_alpha / IA_len * (IA_len - d_ia) + (1 - parameter_alpha) * (
                                1 - l_ia / IA_len) \
                            if IA_len != 0 else 0
                        # x, a, b, c, d
                        if parameter_rate <= 2:
                            min_value = parameter_alpha / IA_len * (IA_len - 1 / 2 * IA_len)
                            # min_value = min_value + 0.0000001
                            dl_score_ia = map_value(dl_score_ia1, min_value, 1, 1e-10, 1)
                        else:
                            min_value = (1 - parameter_alpha) * (1 - 0 / IA_len)
                            dl_score_ia = map_value(dl_score_ia1, min_value, 1, 1e-10, 1)
                            # dl_score_ia = map_value(dl_score_ia,parameter_alpha,1,0,1)
                        if pred_case_id == 2:
                            debug = 1
                    else:

                        parameter_alpha = parameter_rate / (2 + parameter_rate)
                        dl_score_ia = parameter_alpha * 2 / IA_len * d_ia + (1 - parameter_alpha) * (1 - l_ia / IA_len) \
                            if IA_len != 0 else 0
                        if pred_case_id == 2:
                            if print_msg:
                                print("  parameter_alpha", parameter_alpha)
                                print("  d_ia", d_ia)
                                print("  d_score", 2 / IA_len * d_ia)
                                print("  d_score_parameter", parameter_alpha * 2 / IA_len * d_ia)
                                print("  (1 - parameter_alpha)", (1 - parameter_alpha))
                                print("  l_ia", l_ia)
                                print("  l_score", (1 - l_ia / IA_len))
                                print("  l_score_parameter", (1 - parameter_alpha) * (1 - l_ia / IA_len))
                                print("  IA_len", IA_len)
                                debug = 1

                # sum
                dl_score_ia_sum += dl_score_ia
                l_ia_set_sum += basic_interval[1] - basic_interval[0]
                d_ia_set_sum += abs((basic_interval[1] + basic_interval[0]) / 2 - base_position)
                d_ia_sign_set_sum += (basic_interval[1] + basic_interval[0]) / 2 - base_position

            # sum
            # test_no_group_type = "sum"
            test_no_group_type = "average"
            if test_no_group_type == "sum":
                dl_score_ia = dl_score_ia_sum
                l_ia_set_dict[area_id] = l_ia_set_sum
                d_ia_set_dict[area_id] = d_ia_set_sum
                d_ia_sign_set_dict[area_id] = d_ia_sign_set_sum
            else:
                dl_score_ia = dl_score_ia_sum / len(basic_interval_list) if len(basic_interval_list) != 0 else 0
                l_ia_set_dict[area_id] = l_ia_set_sum / len(basic_interval_list) if len(basic_interval_list) != 0 else 0
                d_ia_set_dict[area_id] = d_ia_set_sum / len(basic_interval_list) if len(basic_interval_list) != 0 else 0
                d_ia_sign_set_dict[area_id] = d_ia_sign_set_sum
        dl_score_ia_dict[area_id] = dl_score_ia
    #     deal empty and no area
    if print_msg:
        print("d_ia_set_dict", d_ia_set_dict)
        print("l_ia_set_dict", l_ia_set_dict)
    # print("dl_score_ia_dict",dl_score_ia_dict)

    # cal label_function_gt

    for i, (area_id, area) in enumerate(VA_gt_dict.items()):

        VA_gt_len = area[1] - area[0]
        x1, x2 = cal_x1_x2(VA_gt_len, parameter_rho)
        x1_ori, x2_ori = x1 + area[0], x2 + area[0]
        x_mid = 2 * x1
        x_mid_ori = x_mid + area[0]
        # x1_ori,x2_ori = x1+area[0], x2+area[0]
        area_gt1 = [0, x1]
        area_gt2 = [x1, 2 * x1]
        area_gt3 = [2 * x1, x2]
        area_gt4 = [x2, VA_gt_len]
        area_gt1_ori = [area[0], x1 + area[0]]
        area_gt2_ori = [x1 + area[0], 2 * x1 + area[0]]
        area_gt3_ori = [2 * x1 + area[0], x2 + area[0]]
        area_gt4_ori = [x2 + area[0], area[1]]
        if print_msg:
            print("VA_gt_len", VA_gt_len)

        if parameter_rho < 0:
            parameter_b1, parameter_gama1 = 1 / 2, 0
            parameter_b2, parameter_gama2 = 1 / 2, 0
        else:
            parameter_b1, parameter_gama1 = cal_power_function_coefficient(x1, parameter_lamda)
            parameter_b2, parameter_gama2 = cal_power_function_coefficient(x2 - x_mid, parameter_lamda)

        func_dict.update({
            area_id + "_gt1": {"area": area_gt1_ori,
                               "area_start": area_gt1_ori[0],
                               "parameter_a": parameter_b1,
                               "parameter_beta": parameter_gama1,
                               "func": "va_gt1_func",
                               "x_tp": x1,
                               "gt_len": VA_gt_len,
                               "area_len": None,
                               "parameter_rho": parameter_rho
                               },
            area_id + "_gt2": {"area": area_gt2_ori,
                               "area_start": area_gt1_ori[0],
                               "parameter_a": parameter_b1,
                               "parameter_beta": parameter_gama1,
                               "func": "va_gt2_func",
                               "x_tp": x1,
                               "gt_len": VA_gt_len,
                               "area_len": None,
                               "parameter_rho": parameter_rho
                               },
            area_id + "_gt3": {"area": area_gt3_ori,
                               "area_start": area_gt1_ori[0],
                               "parameter_a": parameter_b2,
                               "parameter_beta": parameter_gama2,
                               "func": "va_gt3_func",
                               "x_tp": x2,
                               "gt_len": VA_gt_len,
                               "area_len": None,
                               "parameter_rho": parameter_rho
                               },
            area_id + "_gt4": {"area": area_gt4_ori,
                               "area_start": area_gt1_ori[0],
                               "parameter_a": parameter_b2,
                               "parameter_beta": parameter_gama2,
                               "func": "va_gt4_func",
                               "x_tp": x2,
                               "gt_len": VA_gt_len,
                               "area_len": None,
                               "parameter_rho": parameter_rho
                               },

        })

        s_a_gt1 = cal_integral_in_range_gt_power_func1(area_gt1_ori, parameter_b1, parameter_gama1, x_tp=x1,
                                                       gt_len=VA_gt_len, area_start=area_gt1_ori[0],
                                                       parameter_rho=parameter_rho)
        s_a_gt2 = cal_integral_in_range_gt_power_func2(area_gt2_ori, parameter_b1, parameter_gama1, x_tp=x1,
                                                       gt_len=VA_gt_len, area_start=area_gt1_ori[0],
                                                       parameter_rho=parameter_rho)
        s_a_gt3 = cal_integral_in_range_gt_power_func3(area_gt3_ori, parameter_b2, parameter_gama2, x_tp=x2,
                                                       gt_len=VA_gt_len, area_start=area_gt1_ori[0],
                                                       parameter_rho=parameter_rho)
        s_a_gt4 = cal_integral_in_range_gt_power_func4(area_gt4_ori, parameter_b2, parameter_gama2, x_tp=x2,
                                                       gt_len=VA_gt_len, area_start=area_gt1_ori[0],
                                                       parameter_rho=parameter_rho)

        s_a_gt = s_a_gt1 + s_a_gt2 + s_a_gt3 + s_a_gt4

        func_dict[area_id + "_gt1"]["area_s"] = s_a_gt1
        func_dict[area_id + "_gt2"]["area_s"] = s_a_gt2
        func_dict[area_id + "_gt3"]["area_s"] = s_a_gt3
        func_dict[area_id + "_gt4"]["area_s"] = s_a_gt4

        method = 2

        va_f_area = VA_f_dict[area_id]
        s_a_f = 0
        if va_f_area == []:
            pass
        else:
            VA_f_len = va_f_area[1] - va_f_area[0]
            parameter_beta1 = 1
            parameter_a1 = 1 / (2 * VA_f_len)

            greater_than_th = False

            func_dict[area_id + "_f"] = {"area": va_f_area,
                                         "area_start": va_f_area[0],
                                         "parameter_a": parameter_a1,
                                         "parameter_beta": parameter_beta1,
                                         "func": "va_f_func",
                                         "x_tp": None,
                                         "gt_len": None,
                                         "area_len": VA_f_len,
                                         "greater_than_th": greater_than_th
                                         }

            # f
            s_max_f = cal_integral_in_range_f_power_func(va_f_area, parameter_a1, parameter_beta1, VA_f_len, VA_f_len,
                                                         area_start=va_f_area[0], greater_than_th=greater_than_th)
            func_dict[area_id + "_f"]["area_s"] = s_max_f

            s_rate = s_max_f / s_a_gt
            func_dict[area_id + "_f_a"] = func_dict[area_id + "_f"]
            s_a_f = s_max_f
            if s_rate > parameter_theta:
                greater_than_th = True

                # method1
                if method == 1:
                    parameter_beta1_a = VA_f_len / (2 * parameter_theta * s_a_gt) - 1
                    parameter_a1_a = 1 / (2 * pow(VA_f_len, parameter_beta1_a))
                # method2
                else:
                    parameter_beta1_a = VA_f_len / (2 * parameter_theta * s_a_gt) - 1  # same with method1
                    parameter_a1_a = VA_f_len / pow(1 / 2, parameter_beta1_a)

                s_a_f = cal_integral_in_range_f_power_func(va_f_area, parameter_a1_a, parameter_beta1_a, VA_f_len,
                                                           VA_f_len, area_start=va_f_area[0], method=method,
                                                           greater_than_th=greater_than_th)
                func_dict[area_id + "_f_a"] = {
                    "area": va_f_area,
                    "area_start": va_f_area[0],
                    "area_s": s_a_f,
                    "parameter_a": parameter_a1_a,
                    "parameter_beta": parameter_beta1_a,
                    # "func": "va_f_func",
                    # "func": "va_f_reverse_func",
                    "x_tp": None,
                    "gt_len": None,
                    "area_len": VA_f_len,
                    "greater_than_th": greater_than_th
                }
                if method == 1:
                    func_dict[area_id + "_f_a"]["func"] = "va_f_func"
                else:
                    func_dict[area_id + "_f_a"]["func"] = "va_f_reverse_func"

        # d
        va_d_area = VA_d_dict[area_id]
        s_a_d = 0
        if va_d_area == []:
            pass
        else:

            if pred_case_id == 7 and area_id == "id_2":
                debug = 1

            VA_d_len = va_d_area[1] - va_d_area[0]
            parameter_beta2 = 1
            parameter_a2 = 1 / (2 * VA_d_len)

            greater_than_th = False
            s_max_d = cal_integral_in_range_d_power_func(va_d_area, parameter_a2, parameter_beta2, VA_d_len, VA_d_len,
                                                         area_start=va_d_area[0], greater_than_th=greater_than_th)
            func_dict[area_id + "_d"] = {
                "area": va_d_area,
                "area_start": va_d_area[0],
                "parameter_a": parameter_a2,
                "parameter_beta": parameter_beta2,
                "func": "va_d_func",
                "x_tp": None,
                "gt_len": None,
                "area_len": VA_d_len,
                "greater_than_th": greater_than_th
            }

            func_dict[area_id + "_d"]["area_s"] = s_max_d

            func_dict[area_id + "_d_a"] = func_dict[area_id + "_d"]
            s_rate = s_max_d / s_a_gt
            s_a_d = s_max_d

            if s_rate > parameter_theta:
                greater_than_th = True

                # method1
                if method == 1:
                    parameter_beta2_a = VA_d_len / (2 * parameter_theta * s_a_gt) - 1
                    parameter_a2_a = 1 / (2 * pow(VA_d_len, parameter_beta2_a))
                else:
                    # method2
                    parameter_beta2_a = VA_d_len / (2 * parameter_theta * s_a_gt) - 1  # same with method1
                    parameter_a2_a = VA_d_len / pow(1 / 2, parameter_beta2_a)

                s_a_d = cal_integral_in_range_d_power_func(va_d_area, parameter_a2_a, parameter_beta2_a, VA_d_len,
                                                           VA_d_len,
                                                           area_start=va_d_area[0], method=method,
                                                           greater_than_th=greater_than_th)
                func_dict[area_id + "_d_a"] = {
                    "area": va_d_area,
                    "area_start": va_d_area[0],
                    "area_s": s_a_d,
                    "parameter_a": parameter_a2_a,
                    "parameter_beta": parameter_beta2_a,
                    # "func": "va_d_func",
                    # "func": "va_d_reverse_func",
                    "x_tp": None,
                    "gt_len": None,
                    "area_len": VA_d_len,
                    "greater_than_th": greater_than_th

                }

                if method == 1:
                    func_dict[area_id + "_d_a"]["func"] = "va_d_func"
                else:
                    func_dict[area_id + "_d_a"]["func"] = "va_d_reverse_func"

        s_a_func_dict[area_id] = {
            "s_a_va_gt_len": VA_gt_len,
            "s_a_va_gt": s_a_gt,
            "s_a_va": s_a_f + s_a_gt + s_a_d,
            "x1": x1,
            "x2": x2,
            "x_mid": x_mid,
            "x1_ori": x1_ori,
            "x2_ori": x2_ori,
            "x_mid_ori": x_mid_ori,
        }
    # plot func
    if pred_case_id == 7 and area_id == "id_2":
        debug = 1
        # plot_func_multi(func_dict)
    # plot_func_multi(func_dict)
    # plot_func_multi_paper(func_dict,window_length)

    import pprint

    # print("s_a_func_dict")
    # pprint(s_a_func_dict, indent=4)

    # cal recall_va,detection_score
    recall_va_list = []
    recall_va_dict = {}
    detection_score_ia_i_dict = {}
    precision_va_dict = {}

    gt_num = label_interval_num
    pred_func_dict = {}
    basic_interval_integral_dict = {}

    for i in range(gt_num):
        area_id = "id_" + str(i)
        # print("area_id",area_id)
        area_id_back = "id_" + str(i + 1)
        VA_f_i_pred_group = VA_f_pred_group_dict[area_id]
        VA_gt1_i_pred_group = VA_gt1_pred_group_dict[area_id]
        VA_gt2_i_pred_group = VA_gt2_pred_group_dict[area_id]
        VA_gt3_i_pred_group = VA_gt3_pred_group_dict[area_id]
        VA_gt4_i_pred_group = VA_gt4_pred_group_dict[area_id]
        VA_d_i_pred_group = VA_d_pred_group_dict[area_id]

        IA_pred_i_group = IA_pred_group_dict[area_id]
        IA_pred_i_back_group = IA_pred_group_dict[area_id_back]
        VA_pred_i_group = VA_pred_group_dict[area_id]

        pred_integral_all = 0

        # # f
        va_f_area = VA_f_dict[area_id]
        if va_f_area == []:
            pass
        else:
            VA_f_len = va_f_area[1] - va_f_area[0]
            parameter_a1 = func_dict[area_id + "_f_a"]["parameter_a"]
            parameter_beta1 = func_dict[area_id + "_f_a"]["parameter_beta"]
            greater_than_th = func_dict[area_id + "_f_a"]["greater_than_th"]

            for j, basic_interval in enumerate(VA_f_i_pred_group):
                if basic_interval != []:

                    pred_func_dict[area_id + "_f_a" + "_" + str(j)] = {
                        "area": basic_interval,
                        "area_start": va_f_area[0],
                        "parameter_a": parameter_a1,
                        "parameter_beta": parameter_beta1,
                        # "func":"va_f_func",
                        "x_tp": None,
                        "gt_len": None,
                        "area_len": VA_f_len
                    }
                    if not greater_than_th:
                        pred_func_dict[area_id + "_f_a" + "_" + str(j)]["func"] = "va_d_func"
                    else:
                        if method == 1:
                            pred_func_dict[area_id + "_f_a" + "_" + str(j)]["func"] = "va_f_func"
                        else:
                            pred_func_dict[area_id + "_f_a" + "_" + str(j)]["func"] = "va_f_reverse_func"

                    cal_integral_basic_interval_f = cal_integral_in_range_f_power_func(basic_interval, parameter_a1,
                                                                                       parameter_beta1, VA_f_len,
                                                                                       VA_f_len,
                                                                                       area_start=va_f_area[0],
                                                                                       method=method,
                                                                                       greater_than_th=greater_than_th)
                    pred_integral_all += cal_integral_basic_interval_f
                    basic_interval_integral_dict[area_id + "_f" + "_" + str(j)] = cal_integral_basic_interval_f

        # gt1-gt4
        s_a_gt_len = s_a_func_dict[area_id]["s_a_va_gt_len"]
        s_a_va = s_a_func_dict[area_id]["s_a_va"]
        parameter_b1 = func_dict[area_id + "_gt1"]["parameter_a"]
        parameter_gama1 = func_dict[area_id + "_gt1"]["parameter_beta"]
        parameter_b2 = func_dict[area_id + "_gt3"]["parameter_a"]
        parameter_gama2 = func_dict[area_id + "_gt3"]["parameter_beta"]
        x1 = s_a_func_dict[area_id]["x1"]
        x2 = s_a_func_dict[area_id]["x2"]
        area_gt = VA_gt_dict[area_id]
        VA_gt_len = area_gt[1] - area_gt[0]

        area_gt1 = func_dict[area_id + "_gt1"]["area"]
        area_gt2 = func_dict[area_id + "_gt2"]["area"]
        area_gt3 = func_dict[area_id + "_gt3"]["area"]
        area_gt4 = func_dict[area_id + "_gt4"]["area"]
        area_start = func_dict[area_id + "_gt1"]["area_start"]

        for j, basic_interval in enumerate(VA_gt1_i_pred_group):
            if basic_interval != []:
                pred_func_dict[area_id + "_gt1" + "_" + str(j)] = {
                    "area": basic_interval,
                    "area_start": area_start,
                    "parameter_a": parameter_b1,
                    "parameter_beta": parameter_gama1,
                    "func": "va_gt1_func",
                    "x_tp": None,
                    "gt_len": VA_gt_len,
                    "area_len": None,
                    "parameter_rho": parameter_rho
                }
                cal_integral_basic_interval_gt1 = cal_integral_in_range_gt_power_func1(basic_interval, parameter_b1,
                                                                                       parameter_gama1, x_tp=x1,
                                                                                       gt_len=VA_gt_len,
                                                                                       area_start=area_start,
                                                                                       parameter_rho=parameter_rho)
                pred_integral_all += cal_integral_basic_interval_gt1
                basic_interval_integral_dict[area_id + "_gt1" + "_" + str(j)] = cal_integral_basic_interval_gt1

        for j, basic_interval in enumerate(VA_gt2_i_pred_group):
            if basic_interval != []:
                pred_func_dict[area_id + "_gt2" + "_" + str(j)] = {
                    "area": basic_interval,
                    "area_start": area_start,
                    "parameter_a": parameter_b1,
                    "parameter_beta": parameter_gama1,
                    "func": "va_gt2_func",
                    "x_tp": x1,
                    "gt_len": VA_gt_len,
                    "area_len": None,
                    "parameter_rho": parameter_rho
                }
                cal_integral_basic_interval_gt2 = cal_integral_in_range_gt_power_func2(basic_interval, parameter_b1,
                                                                                       parameter_gama1, x_tp=x1,
                                                                                       gt_len=VA_gt_len,
                                                                                       area_start=area_start,
                                                                                       parameter_rho=parameter_rho)
                pred_integral_all += cal_integral_basic_interval_gt2
                basic_interval_integral_dict[area_id + "_gt2" + "_" + str(j)] = cal_integral_basic_interval_gt2

        for j, basic_interval in enumerate(VA_gt3_i_pred_group):
            if basic_interval != []:
                pred_func_dict[area_id + "_gt3" + "_" + str(j)] = {
                    "area": basic_interval,
                    "area_start": area_start,
                    "parameter_a": parameter_b2,
                    "parameter_beta": parameter_gama2,
                    "func": "va_gt3_func",
                    "x_tp": x2,
                    "gt_len": VA_gt_len,
                    "area_len": None,
                    "parameter_rho": parameter_rho

                }
                cal_integral_basic_interval_gt3 = cal_integral_in_range_gt_power_func3(basic_interval, parameter_b2,
                                                                                       parameter_gama2, x_tp=x2,
                                                                                       gt_len=VA_gt_len,
                                                                                       area_start=area_start,
                                                                                       parameter_rho=parameter_rho)
                pred_integral_all += cal_integral_basic_interval_gt3
                basic_interval_integral_dict[area_id + "_gt3" + "_" + str(j)] = cal_integral_basic_interval_gt3

        for j, basic_interval in enumerate(VA_gt4_i_pred_group):
            if basic_interval != []:
                pred_func_dict[area_id + "_gt4" + "_" + str(j)] = {
                    "area": basic_interval,
                    "area_start": area_start,
                    "parameter_a": parameter_b2,
                    "parameter_beta": parameter_gama2,
                    "func": "va_gt4_func",
                    "x_tp": x2,
                    "gt_len": VA_gt_len,
                    "area_len": None,
                    "parameter_rho": parameter_rho
                }
                cal_integral_basic_interval_gt4 = cal_integral_in_range_gt_power_func4(basic_interval, parameter_b2,
                                                                                       parameter_gama2, x_tp=x2,
                                                                                       gt_len=VA_gt_len,
                                                                                       area_start=area_start,
                                                                                       parameter_rho=parameter_rho)
                pred_integral_all += cal_integral_basic_interval_gt4
                basic_interval_integral_dict[area_id + "_gt4" + "_" + str(j)] = cal_integral_basic_interval_gt4

        # # d
        va_d_area = VA_d_dict[area_id]
        if va_d_area == []:
            pass
        else:
            VA_d_len = va_d_area[1] - va_d_area[0]
            parameter_a2 = func_dict[area_id + "_d_a"]["parameter_a"]
            parameter_beta2 = func_dict[area_id + "_d_a"]["parameter_beta"]
            greater_than_th = func_dict[area_id + "_d_a"]["greater_than_th"]
            if pred_case_id == 7 and area_id == "id_2":
                debug = 1

            for j, basic_interval in enumerate(VA_d_i_pred_group):
                if basic_interval != []:
                    pred_func_dict[area_id + "_d_a" + "_" + str(j)] = {
                        "area": basic_interval,
                        "area_start": va_d_area[0],
                        "parameter_a": parameter_a2,
                        "parameter_beta": parameter_beta2,
                        # "func":"va_d_func",
                        "x_tp": None,
                        "gt_len": None,
                        "area_len": VA_d_len
                    }
                    if not greater_than_th:
                        pred_func_dict[area_id + "_d_a" + "_" + str(j)]["func"] = "va_d_func"
                    else:
                        if method == 1:
                            pred_func_dict[area_id + "_d_a" + "_" + str(j)]["func"] = "va_d_func"
                        else:
                            pred_func_dict[area_id + "_d_a" + "_" + str(j)]["func"] = "va_d_reverse_func"

                    cal_integral_basic_interval_d = cal_integral_in_range_d_power_func(basic_interval, parameter_a2,
                                                                                       parameter_beta2, VA_d_len,
                                                                                       VA_d_len,
                                                                                       area_start=va_d_area[0],
                                                                                       method=method,
                                                                                       greater_than_th=greater_than_th)
                    pred_integral_all += cal_integral_basic_interval_d
                    basic_interval_integral_dict[area_id + "_d" + "_" + str(j)] = cal_integral_basic_interval_d

        # print("pred_integral_all",pred_integral_all)
        va_area = VA_dict[area_id]
        recall_va_i = pred_integral_all / s_a_va
        recall_va_list.append(recall_va_i)
        if pred_case_id == 7:
            debug = 1
        recall_va_dict[area_id] = recall_va_i

        detection_score_i = 1 if VA_pred_group_dict[area_id] != [] else 0
        detection_score_ia_i_dict[area_id] = detection_score_i
        # invalid part punish in is not enough
        # detection
        if cal_mode == "detection":
            precision_va_dict[area_id] = detection_score_i

        precision_i_group = IA_pred_i_group + IA_pred_i_back_group + VA_pred_i_group

        precision_i_all = 0
        precision_i_valid = 0
        for precision_i_item in precision_i_group:
            precision_i_all += precision_i_item[1] - precision_i_item[0]
        for precision_i_item in VA_pred_i_group:
            precision_i_valid += precision_i_item[1] - precision_i_item[0]
        if pred_case_id == 4:
            debug = 1
        precision_va_i = precision_i_valid / precision_i_all if precision_i_all != 0 else 0

        # precision also consider invalid part
        # proportion
        if cal_mode == "proportion":
            precision_va_dict[area_id] = precision_va_i

    if print_msg:
        # print basic_interval_integral
        print("basic_interval_integral_dict", basic_interval_integral_dict)

    # pot pred func
    # if pred_case_id == 7 and area_id == "id_2":
    #     debug = 1
    #     plot_func_multi(pred_func_dict)
    # plot_func_multi(pred_func_dict)

    if print_msg:
        # print recall_va_dict,precision_va_dict,dl_score_ia_dict
        print("recall_va_dict", recall_va_dict)
        print("precision_va_dict", precision_va_dict)
        print("dl_score_ia_dict", dl_score_ia_dict)

    adjusted_dl_score_ia_dict = {}
    adjusted_dl_score_ia_dict_ratio_contribution = {}

    if pred_case_id == 2:
        debug = 1
    area_id_0 = "id_" + str(0)
    area_id_num = "id_" + str(gt_num)
    # two logics
    logic = 1
    # logic = 3
    if logic == 1:
        # logic 1
        #     one_exist_value = 1/2
        one_exist_value = 1
        for i in range(gt_num + 1):
            area_id = "id_" + str(i)
            area_id_front = "id_" + str(i - 1)
            area_id_back = "id_" + str(i + 1)

            ia_area = IA_dict[area_id]
            l_ia = l_ia_set_dict[area_id]
            d_ia = d_ia_set_dict[area_id]
            d_ia_sign = d_ia_sign_set_dict[area_id]
            ia_len = ia_area[1] - ia_area[0] if ia_area != [] else 0

            if pred_case_id == 22:
                debug = 1

            if i == 0:
                if IA_dict[area_id] == []:
                    # if IA_dict[area_id_back] == [] and area_id_back == area_id_num:
                    if area_id_back == area_id_num:
                        if IA_dict[area_id_back] == []:  # 自己为空，后面一个为最后一个，后面一个为空
                            ratio_before = 0
                            ratio_after = 1 / 2
                        else:  # 自己为空，后面一个为最后一个，后面一个不为空
                            ratio_before = 0
                            ratio_after = 0
                    else:
                        ratio_before = 0
                        ratio_after = 1 / 2

                    if VA_pred_group_dict[area_id] != []:
                        dl_score_ia_i_contribution = 1
                    else:
                        dl_score_ia_i_contribution = 0
                else:
                    # 数据里至少有一个异常，至少有两个ia，如果没有异常，那么IA_dict=[]
                    if IA_pred_group_dict[area_id] != []:
                        if area_id_back == area_id_num:
                            if IA_dict[area_id_back] == []:  # 自己不为空，后面一个为最后一个，后面一个为空
                                ratio_before = 0
                                ratio_after = 1
                            else:  # 自己不为空，后面一个为最后一个，后面一个不为空
                                ratio_before = 0
                                ratio_after = 1 / 2
                        else:
                            ratio_before = 0
                            ratio_after = 1 / 2
                        dl_score_ia_i_contribution = dl_score_ia_dict[area_id]
                    else:
                        if VA_pred_group_dict[area_id] != []:
                            if area_id_back == area_id_num:
                                if IA_dict[area_id_back] == []:  # 自己不为空，后面一个为最后一个，后面一个为空
                                    ratio_before = 0
                                    ratio_after = 1
                                else:  # 自己不为空，后面一个为最后一个，后面一个不为空
                                    ratio_before = 0
                                    ratio_after = 1 / 2
                            else:
                                ratio_before = 0
                                ratio_after = 1 / 2
                            dl_score_ia_i_contribution = 1
                        else:
                            if area_id_back == area_id_num:
                                if IA_dict[area_id_back] == []:  # 自己不为空，后面一个为最后一个，后面一个为空
                                    ratio_before = 0
                                    ratio_after = 1
                                else:  # 自己不为空，后面一个为最后一个，后面一个不为空
                                    ratio_before = 0
                                    ratio_after = 1 / 2
                            else:
                                ratio_before = 0
                                ratio_after = 1 / 2
                            dl_score_ia_i_contribution = 0
            elif i == gt_num:
                if IA_dict[area_id] == []:
                    if area_id_front == area_id_0:
                        if IA_dict[area_id_front] == []:  # 自己为空，前面一个为最后一个，前面一个为空
                            ratio_before = 1 / 2
                            ratio_after = 0
                        else:  # 自己为空，前面一个为第一个，前面一个不为空
                            ratio_before = 0
                            ratio_after = 0
                    else:
                        ratio_before = 1 / 2
                        ratio_after = 0

                    if VA_pred_group_dict[area_id_front] != []:
                        dl_score_ia_i_contribution = 1
                    else:
                        dl_score_ia_i_contribution = 0
                else:
                    if IA_pred_group_dict[area_id] != []:
                        if area_id_front == area_id_0:
                            if IA_dict[area_id_front] == []:
                                ratio_before = 1
                                ratio_after = 0
                            else:
                                ratio_before = 1 / 2
                                ratio_after = 0
                        else:
                            ratio_before = 1 / 2
                            ratio_after = 0
                        dl_score_ia_i_contribution = dl_score_ia_dict[area_id]
                    else:
                        if VA_pred_group_dict[area_id_front] != []:
                            if area_id_front == area_id_0:
                                if IA_dict[area_id_front] == []:
                                    ratio_before = 1
                                    ratio_after = 0
                                else:
                                    ratio_before = 1 / 2
                                    ratio_after = 0
                            else:
                                ratio_before = 1 / 2
                                ratio_after = 0
                            dl_score_ia_i_contribution = 1
                        else:
                            if IA_dict[area_id_front] == []:
                                if area_id_front == area_id_0:
                                    if IA_dict[area_id_front] == []:
                                        ratio_before = 1
                                        ratio_after = 0
                                    else:
                                        ratio_before = 1 / 2
                                        ratio_after = 0
                                else:
                                    ratio_before = 1 / 2
                                    ratio_after = 0
                            dl_score_ia_i_contribution = 0
            else:
                if IA_pred_group_dict[area_id] != []:
                    ratio_before = 1 / 2
                    ratio_after = 1 / 2
                    # if d_ia_sign > 0:
                    #     ratio_before = (ia_len / 2 - d_ia) / ia_len
                    #     ratio_after = (ia_len / 2 + d_ia) / ia_len
                    # elif d_ia_sign < 0:
                    #     ratio_before = (ia_len / 2 + d_ia) / ia_len
                    #     ratio_after = (ia_len / 2 - d_ia) / ia_len
                    # else:
                    #     ratio_before = 1/2
                    #     ratio_after = 1/2
                    dl_score_ia_i_contribution = dl_score_ia_dict[area_id]
                else:
                    ratio_before = 1 / 2
                    ratio_after = 1 / 2
                    if VA_pred_group_dict[area_id_front] != [] and VA_pred_group_dict[area_id] != []:
                        # ratio_before = 1/2
                        # ratio_after = 1/2
                        dl_score_ia_i_contribution = 1
                    elif VA_pred_group_dict[area_id_front] == [] and VA_pred_group_dict[area_id] == []:
                        # ratio_before = 1/2
                        # ratio_after = 1/2
                        dl_score_ia_i_contribution = 0
                    else:
                        # if VA_pred_group_dict[area_id_front] != []:
                        #     ratio_before = 1
                        #     ratio_after = 0
                        # else:
                        #     ratio_before = 0
                        #     ratio_after = 1
                        dl_score_ia_i_contribution = one_exist_value

            adjusted_dl_score_ia_dict[area_id] = dl_score_ia_i_contribution
            dl_score_ia_i_before_contribution = dl_score_ia_i_contribution * ratio_before
            dl_score_ia_i_after_contribution = dl_score_ia_i_contribution * ratio_after
            adjusted_dl_score_ia_dict_ratio_contribution[area_id] = [dl_score_ia_i_before_contribution,
                                                                     dl_score_ia_i_after_contribution]

    debug = 1
    # logic 2
    # one_exist_value = 1/2
    # for i in range(gt_num + 1):
    #     area_id = "id_" + str(i)
    #     area_id_front = "id_" + str(i - 1)
    #     area_id_back = "id_" + str(i + 1)
    #
    #     ia_area = IA_dict[area_id]
    #     l_ia = l_ia_set_dict[area_id]
    #     d_ia = d_ia_set_dict[area_id]
    #     d_ia_sign = d_ia_sign_set_dict[area_id]
    #     ia_len = ia_area[1] - ia_area[0] if ia_area != [] else 0
    #
    #     if i == 0:
    #         ratio_before = 0
    #         ratio_after = 1
    #         if IA_pred_group_dict[area_id] != []:
    #             dl_score_ia_i_contribution = dl_score_ia_dict[area_id]
    #         else:
    #             if VA_pred_group_dict[area_id] != []:
    #                 dl_score_ia_i_contribution = 1 / 2
    #             else:
    #                 dl_score_ia_i_contribution = 0
    #     elif i == gt_num:
    #         ratio_before = 1
    #         ratio_after = 0
    #         if IA_pred_group_dict[area_id] != []:
    #             dl_score_ia_i_contribution = dl_score_ia_dict[area_id]
    #         else:
    #             if VA_pred_group_dict[area_id_front] != []:
    #                 dl_score_ia_i_contribution = 1 / 2
    #             else:
    #                 dl_score_ia_i_contribution = 0
    #     else:
    #         if IA_pred_group_dict[area_id] != []:
    #             ratio_before = 1 / 2
    #             ratio_after = 1 / 2
    #             # if d_ia_sign > 0:
    #             #     ratio_before = (ia_len / 2 - d_ia) / ia_len
    #             #     ratio_after = (ia_len / 2 + d_ia) / ia_len
    #             # elif d_ia_sign < 0:
    #             #     ratio_before = (ia_len / 2 + d_ia) / ia_len
    #             #     ratio_after = (ia_len / 2 - d_ia) / ia_len
    #             # else:
    #             #     ratio_before = 1/2
    #             #     ratio_after = 1/2
    #             dl_score_ia_i_contribution = dl_score_ia_dict[area_id]
    #         else:
    #             # ratio_before = 1 / 2
    #             # ratio_after = 1 / 2
    #             if VA_pred_group_dict[area_id_front] != [] and VA_pred_group_dict[area_id] != []:
    #                 ratio_before = 1/2
    #                 ratio_after = 1/2
    #                 dl_score_ia_i_contribution = 1
    #             elif VA_pred_group_dict[area_id_front] == [] and VA_pred_group_dict[area_id] == []:
    #                 ratio_before = 1/2
    #                 ratio_after = 1/2
    #                 dl_score_ia_i_contribution = 0
    #             else:
    #                 if VA_pred_group_dict[area_id_front] != []:
    #                     ratio_before = 1
    #                     ratio_after = 0
    #                 else:
    #                     ratio_before = 0
    #                     ratio_after = 1
    #                 dl_score_ia_i_contribution = one_exist_value

    # logic 3
    if logic == 3:
        # one_exist_value = 1/2
        one_exist_value = 1
        for i in range(gt_num + 1):
            area_id = "id_" + str(i)
            area_id_front = "id_" + str(i - 1)
            area_id_back = "id_" + str(i + 1)

            ia_area = IA_dict[area_id]
            l_ia = l_ia_set_dict[area_id]
            d_ia = d_ia_set_dict[area_id]
            d_ia_sign = d_ia_sign_set_dict[area_id]
            ia_len = ia_area[1] - ia_area[0] if ia_area != [] else 0

            if i == 0:
                ratio_before = 0
                ratio_after = 1
                if IA_dict[area_id] == []:
                    dl_score_ia_i_contribution = 0
                else:
                    if IA_pred_group_dict[area_id] != []:
                        dl_score_ia_i_contribution = dl_score_ia_dict[area_id]
                    else:
                        if VA_pred_group_dict[area_id] != []:
                            dl_score_ia_i_contribution = 1
                        else:
                            dl_score_ia_i_contribution = 0
            elif i == gt_num:
                ratio_before = 1
                ratio_after = 0
                if IA_dict[area_id] == []:
                    dl_score_ia_i_contribution = 0
                else:
                    if IA_pred_group_dict[area_id] != []:
                        dl_score_ia_i_contribution = dl_score_ia_dict[area_id]
                    else:
                        if VA_pred_group_dict[area_id_front] != []:
                            dl_score_ia_i_contribution = 1
                        else:
                            dl_score_ia_i_contribution = 0
            else:
                if IA_pred_group_dict[area_id] != []:
                    # ratio_before = 1 / 2
                    # ratio_after = 1 / 2
                    if d_ia_sign > 0:
                        ratio_before = (ia_len / 2 - d_ia) / ia_len
                        ratio_after = (ia_len / 2 + d_ia) / ia_len
                    elif d_ia_sign < 0:
                        ratio_before = (ia_len / 2 + d_ia) / ia_len
                        ratio_after = (ia_len / 2 - d_ia) / ia_len
                    else:
                        ratio_before = 1 / 2
                        ratio_after = 1 / 2
                    dl_score_ia_i_contribution = dl_score_ia_dict[area_id]
                else:
                    # ratio_before = 1 / 2
                    # ratio_after = 1 / 2
                    if VA_pred_group_dict[area_id_front] != [] and VA_pred_group_dict[area_id] != []:
                        ratio_before = 1 / 2
                        ratio_after = 1 / 2
                        dl_score_ia_i_contribution = 2
                    elif VA_pred_group_dict[area_id_front] == [] and VA_pred_group_dict[area_id] == []:
                        ratio_before = 1 / 2
                        ratio_after = 1 / 2
                        dl_score_ia_i_contribution = 0
                    else:
                        if VA_pred_group_dict[area_id_front] != []:
                            ratio_before = 1
                            ratio_after = 0
                        else:
                            ratio_before = 0
                            ratio_after = 1
                        dl_score_ia_i_contribution = one_exist_value

            adjusted_dl_score_ia_dict[area_id] = dl_score_ia_i_contribution
            dl_score_ia_i_before_contribution = dl_score_ia_i_contribution * ratio_before
            dl_score_ia_i_after_contribution = dl_score_ia_i_contribution * ratio_after
            adjusted_dl_score_ia_dict_ratio_contribution[area_id] = [dl_score_ia_i_before_contribution,
                                                                     dl_score_ia_i_after_contribution]

    if print_msg:
        print("adjusted_dl_score_ia_dict", adjusted_dl_score_ia_dict)
        print("adjusted_dl_score_ia_dict_ratio_contribution", adjusted_dl_score_ia_dict_ratio_contribution)
    event_recall_i_list = []
    event_recall_i_dict = {}
    event_precision_i_dict = {}
    event_f1_score_first_i_dict = {}
    event_f1_score_first_i_dict_w0 = {}
    event_f1_score_first_i_dict_w1 = {}
    for i in range(gt_num):
        area_id = "id_" + str(i)
        if print_msg:
            print(area_id)
        area_id_back = "id_" + str(i + 1)
        recall_va_i = recall_va_dict[area_id]
        # if VA_pred_group_dict[area_id] != [] and IA_pred_group_dict[area_id] == [] and IA_pred_group_dict[area_id_back] == [] :
        #     dl_score_ia_i = 1
        #     dl_score_ia_i_back = 1
        # else:
        #     dl_score_ia_i = dl_score_ia_dict[area_id]
        #     dl_score_ia_i_back = dl_score_ia_dict[area_id_back]

        dl_score_ia_i_after_contribution = adjusted_dl_score_ia_dict_ratio_contribution[area_id][1]
        dl_score_ia_i_back_before_contribution = adjusted_dl_score_ia_dict_ratio_contribution[area_id_back][0]
        # if print_msg:
        #     print(" dl_score_ia_i_after_contribution", dl_score_ia_i_after_contribution)
        #     print(" recall_va_i,", recall_va_i)
        #     print(" dl_score_ia_i_back_before_contribution", dl_score_ia_i_back_before_contribution)

        # event_recall_i = (1-2*parameter_eta)*recall_va_i + parameter_eta*(dl_score_ia_i_after_contribution+ dl_score_ia_i_back_before_contribution)
        if pred_case_id == 26:
            debug = 1

        precision_va_i = precision_va_dict[area_id]
        recall_va_i = recall_va_dict[area_id]
        meata_f1_first_i = compute_f1_score(precision_va_i, recall_va_i)
        # local auc
        # meata_f1_first_i = compute_f1_score(precision_va_i, recall_va_i)
        meata_f1_i = (1 - parameter_eta) * meata_f1_first_i + parameter_eta * (
                        dl_score_ia_i_after_contribution + dl_score_ia_i_back_before_contribution)
        meata_f1_i_w0 = (1 - 0) * meata_f1_first_i + 0 * (
                        dl_score_ia_i_after_contribution + dl_score_ia_i_back_before_contribution)
        meata_f1_i_w1 = (1 - 1) * meata_f1_first_i + 1 * (
                        dl_score_ia_i_after_contribution + dl_score_ia_i_back_before_contribution)
        if print_msg:
            print(" dl_score_ia_i_after_contribution", dl_score_ia_i_after_contribution)
            print(" precision_va_i,", precision_va_i)
            print(" recall_va_i,", recall_va_i)
            print(" meata_f1_first_i,", meata_f1_first_i)
            print(" meata_f1_i,", meata_f1_i)
            print(" dl_score_ia_i_back_before_contribution", dl_score_ia_i_back_before_contribution)

        event_f1_score_first_i_dict[area_id] = meata_f1_i
        event_f1_score_first_i_dict_w0[area_id] = meata_f1_i_w0
        event_f1_score_first_i_dict_w1[area_id] = meata_f1_i_w1

    if print_msg:
        print("event_f1_score_first_i_dict", event_f1_score_first_i_dict)

    if pred_case_id == 2:
        debug = 1

    meata_f1_first = np.array(list(event_f1_score_first_i_dict.values())).mean()
    meata_f1_first_w0 = np.array(list(event_f1_score_first_i_dict_w0.values())).mean()
    meata_f1_first_w1 = np.array(list(event_f1_score_first_i_dict_w1.values())).mean()

    # print(meata_f1_first)

    gt_label_point_array = convert_interval_to_point_array(label_interval_ranges, time_series_length=window_length)
    ont_count = np.count_nonzero(gt_label_point_array == 1)
    anomaly_rate = ont_count / window_length
    if anomaly_rate > 0.2:
        print("anomaly_rate exceeds threshold")
        non_random_coefficient = 1
    else:
        non_random_coefficient = cal_no_random_measure_coefficient(pred_label_point_array, window_length)

    coefficient_meata_f1_first = non_random_coefficient * meata_f1_first
    # coefficient_event_recall = non_random_coefficient * event_recall
    # coefficient_event_precision = non_random_coefficient * event_precision
    print("meata_f1_first", meata_f1_first)
    print("coefficient_meata_f1_first", coefficient_meata_f1_first)

    coefficient_meata_f1_first_w0 = non_random_coefficient * meata_f1_first_w0
    coefficient_meata_f1_first_w1 = non_random_coefficient * meata_f1_first_w1


    # return event_recall,event_precision,meata_f1,no_random_meata_f1
    return meata_f1_first, coefficient_meata_f1_first, VA_f_dict, VA_d_dict, \
        meata_f1_first_w0,coefficient_meata_f1_first_w0,meata_f1_first_w1,coefficient_meata_f1_first_w1

# 绝对距离
def meata_tp_merge_first_v2(label_interval_ranges, IA1_VA1_IA2_relative_list, window_length, parameter_dict, pred_label_point_array,
          pred_case_id=None, cal_mode="proportion"):
    print_msg = False
    # print_msg = True

    # parameter_alpha = parameter_dict["parameter_alpha"]
    parameter_rate = parameter_dict["parameter_rate"]

    parameter_theta = parameter_dict["parameter_theta"]
    # parameter_w = parameter_dict["parameter_w"]
    # parameter_switch_f =parameter_dict["parameter_switch_f"]
    # parameter_switch_d = parameter_dict["parameter_switch_d"]
    VA_f_window = parameter_dict["forecasting_window"]
    VA_d_window = parameter_dict["delay_window"]

    parameter_rho = parameter_dict["parameter_rho"]
    parameter_lamda = parameter_dict["parameter_lamda"]

    parameter_eta = parameter_dict["parameter_eta"]

    if print_msg:
        print("VA_f_window", VA_f_window)
        print("VA_d_window", VA_d_window)

    f_window_rate = 0
    d_window_rate = 0
    f_d_window_len = VA_f_window + VA_d_window
    if f_d_window_len != 0:
        f_window_rate = VA_f_window / f_d_window_len
        d_window_rate = 1 - f_window_rate

    IA_dict = {}
    VA_dict = {}
    VA_f_dict = {}
    VA_gt_dict = {}
    VA_d_dict = {}

    # pred generated is continuous range a,b(len) -> [a,a+b(len)),range of [a,b] -> [a,b) -> point of [a,b-1],point of [a,b] -> [a,b+1)

    # cal every f/d of gt,split IA and VA

    label_interval_num = len(label_interval_ranges)

    s_a_func_dict = {}

    for i, label_interval_range in enumerate(label_interval_ranges):
        # adjust now and before,add before
        VA_gt_now_start, VA_gt_now_end = label_interval_range
        # VA_f_now_start = max(VA_gt_now_start - VA_f_window,0) if parameter_switch_f == True else max(VA_gt_now_start - 3,0)
        VA_f_now_start = max(VA_gt_now_start - VA_f_window, 0)
        VA_f_now_end = VA_gt_now_start
        VA_d_now_start = VA_gt_now_end
        # VA_d_now_end = min(VA_gt_now_end + VA_f_window,window_length) if parameter_switch_d == True else min(VA_gt_now_end + 3,window_length)
        VA_d_now_end = min(VA_gt_now_end + VA_d_window, window_length)
        # # this way below is more fast
        # if i == 0:
        #     VA_f_now_start = max(VA_f_now_start,0)
        # if i == label_interval_num:
        #     VA_d_now_end = min(VA_d_now_end,window_length)
        if i > 0:
            # adjust now and before
            if VA_d_before_end > VA_f_now_start:
                intersection_range_start = max(VA_f_now_start, VA_gt_before_end)
                intersection_range_end = min(VA_d_before_end, VA_gt_now_start)
                intersection_range = [intersection_range_start, intersection_range_end]
                intersection_len = intersection_range_end - intersection_range_start
                #         if intersection_len == 1 and (parameter_switch_f == True and parameter_switch_d == True) and parameter_rho == 1/2:
                # #             f prior
                #             VA_d_before_end -=1
                #         else:
                #         VA_d_before_end = VA_f_now_start = intersection_range_start + round(parameter_w*intersection_len)
                #         VA_d_before_end = VA_f_now_start = intersection_range_start + round(parameter_w*intersection_len,2)
                #         VA_f_now_start = intersection_range_start + round(f_window_rate*intersection_len,2)
                #         VA_d_before_end = intersection_range_start + round(d_window_rate*intersection_len,2)
                VA_f_now_start = intersection_range_start + round(f_window_rate * intersection_len)
                VA_d_before_end = intersection_range_start + round(d_window_rate * intersection_len)
            #     add before
            IA_dict["id_" + str(i)] = [VA_d_before_end, VA_f_now_start]
            VA_dict["id_" + str(i - 1)] = [VA_f_before_start, VA_d_before_end]
            VA_f_dict["id_" + str(i - 1)] = [VA_f_before_start, VA_f_before_end]
            VA_gt_dict["id_" + str(i - 1)] = [VA_gt_before_start, VA_gt_before_end]
            VA_d_dict["id_" + str(i - 1)] = [VA_d_before_start, VA_d_before_end]

            # add by order

            split_line_set.add(VA_f_before_start)
            split_line_set.add(VA_f_before_end)

            split_line_set.add(VA_gt_before_start)
            split_line_set.add(VA_gt_before_end)

            split_line_set.add(VA_d_before_start)
            split_line_set.add(VA_d_before_end)

            split_line_set.add(VA_d_before_end)
            split_line_set.add(VA_f_now_start)

            VA_gt_before_len = VA_gt_before_end - VA_gt_before_start
            x1, x2 = cal_x1_x2(VA_gt_before_len, parameter_rho)

            # # print("area_id,area",area_id,area)
            # # # print("x1,x2",x1, x2)
            # # print("x1_ori,x2_ori",x1+VA_gt_before_start[0], x2+VA_gt_before_start[0])
            x1_ori, x2_ori = x1 + VA_gt_before_start, x2 + VA_gt_before_start
            x_mid_ori = 2 * x1 + VA_gt_before_start
            split_line_set.add(x1_ori)
            split_line_set.add(x_mid_ori)
            split_line_set.add(x2_ori)
        else:
            IA_dict["id_" + str(i)] = [0, VA_f_now_start]
            split_line_set.add(VA_f_now_start)

        VA_gt_before_start, VA_gt_before_end = VA_gt_now_start, VA_gt_now_end
        VA_f_before_start = VA_f_now_start
        VA_f_before_end = VA_f_now_end
        VA_d_before_start = VA_d_now_start
        VA_d_before_end = VA_d_now_end

    # last one VA and last two IA
    IA_dict["id_" + str(label_interval_num)] = [VA_d_before_end, window_length]
    VA_dict["id_" + str(label_interval_num - 1)] = [VA_f_before_start, VA_d_before_end]
    VA_f_dict["id_" + str(label_interval_num - 1)] = [VA_f_before_start, VA_f_before_end]
    VA_gt_dict["id_" + str(label_interval_num - 1)] = [VA_gt_before_start, VA_gt_before_end]
    VA_d_dict["id_" + str(label_interval_num - 1)] = [VA_d_before_start, VA_d_before_end]

    VA_gt_before_len = VA_gt_before_end - VA_gt_before_start
    x1, x2 = cal_x1_x2(VA_gt_before_len, parameter_rho)

    # # print("area_id,area",area_id,area)
    # # # print("x1,x2",x1, x2)
    # # print("x1_ori,x2_ori",x1+VA_gt_before_start[0], x2+VA_gt_before_start[0])
    x1_ori, x2_ori = x1 + VA_gt_before_start, x2 + VA_gt_before_start
    x_mid_ori = 2 * x1 + VA_gt_before_start
    split_line_set.add(x1_ori)
    split_line_set.add(x_mid_ori)
    split_line_set.add(x2_ori)

    # add by order

    split_line_set.add(VA_f_before_start)
    split_line_set.add(VA_f_before_end)

    split_line_set.add(VA_gt_before_start)
    split_line_set.add(VA_gt_before_end)

    split_line_set.add(VA_d_before_start)
    split_line_set.add(VA_d_before_end)

    if print_msg:
        # print area dict
        # # print("split area")
        print("IA_dict", IA_dict)
        # print("VA_dict",VA_dict)
        print("VA_f_dict", VA_f_dict)
        print("VA_gt_dict", VA_gt_dict)
        print("VA_d_dict", VA_d_dict)

    def dict_filer(dict_data: dict):
        for i, (area_id, area_range) in enumerate(dict_data.items()):
            if area_range != []:
                dict_data[area_id] = [] if area_range[0] == area_range[1] else area_range
        return dict_data

    IA_dict = dict_filer(IA_dict)
    VA_dict = dict_filer(VA_dict)
    VA_f_dict = dict_filer(VA_f_dict)
    VA_gt_dict = dict_filer(VA_gt_dict)
    VA_d_dict = dict_filer(VA_d_dict)

    if print_msg:
        print("dict adjust +++++++++++")
        print("IA_dict", IA_dict)
        # print("VA_dict",VA_dict)
        print("VA_f_dict", VA_f_dict)
        print("VA_gt_dict", VA_gt_dict)
        print("VA_d_dict", VA_d_dict)

        # split pred to basic interval
        print("split_line_set", split_line_set)

    split_IA1_VA1_IA2_relative_list = split_intervals(IA1_VA1_IA2_relative_list, split_line_set)
    # print("split_IA1_VA1_IA2_relative_list",split_IA1_VA1_IA2_relative_list)

    # forecasting_len=10
    # delay_len=10
    IA_pred_group_dict = {}
    VA_pred_group_dict = {}
    VA_f_pred_group_dict = {}
    VA_gt_pred_group_dict = {}
    VA_d_pred_group_dict = {}

    VA_gt1_pred_group_dict = {}
    VA_gt2_pred_group_dict = {}
    VA_gt3_pred_group_dict = {}
    VA_gt4_pred_group_dict = {}
    func_dict = {}

    # "id_"+str(label_interval_num)
    for i in range(label_interval_num):
        IA_pred_group_dict["id_" + str(i)] = []
        VA_pred_group_dict["id_" + str(i)] = []
        VA_f_pred_group_dict["id_" + str(i)] = []
        VA_gt_pred_group_dict["id_" + str(i)] = []
        VA_d_pred_group_dict["id_" + str(i)] = []
        VA_gt1_pred_group_dict["id_" + str(i)] = []
        VA_gt2_pred_group_dict["id_" + str(i)] = []
        VA_gt3_pred_group_dict["id_" + str(i)] = []
        VA_gt4_pred_group_dict["id_" + str(i)] = []
    IA_pred_group_dict["id_" + str(label_interval_num)] = []

    # for i, basic_interval in enumerate(IA1_VA1_IA2_relative_list):
    for i, basic_interval in enumerate(split_IA1_VA1_IA2_relative_list):
        # IA
        for j, (area_id, area) in enumerate(IA_dict.items()):
            if pred_in_area(basic_interval, area):
                if area_id not in IA_pred_group_dict.keys():
                    IA_pred_group_dict[area_id] = [basic_interval]
                else:
                    IA_pred_group_dict[area_id].append(basic_interval)

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

            # group by x1,x2
            VA_gt_len = area[1] - area[0]
            x1, x2 = cal_x1_x2(VA_gt_len, parameter_rho)

            # # print("area_id,area",area_id,area)
            # # # print("x1,x2",x1, x2)
            # # print("x1_ori,x2_ori",x1+area[0], x2+area[0])
            area_gt1 = [0, x1]
            area_gt2 = [x1, 2 * x1]
            area_gt3 = [2 * x1, x2]
            area_gt4 = [x2, area[1] - area[0]]
            area_gt1_ori = [area[0], x1 + area[0]]
            area_gt2_ori = [x1 + area[0], 2 * x1 + area[0]]
            area_gt3_ori = [2 * x1 + area[0], x2 + area[0]]
            area_gt4_ori = [x2 + area[0], area[1]]
            # # print("area_gt1",area_gt1)
            # # print("area_gt2",area_gt2)
            # # print("area_gt3",area_gt3)
            # # print("area_gt4",area_gt4)

            # # print("area_gt1_ori",area_gt1_ori)
            # # print("area_gt2_ori",area_gt2_ori)
            # # print("area_gt3_ori",area_gt3_ori)
            # # print("area_gt4_ori",area_gt4_ori)

            if pred_in_area(basic_interval, area_gt1_ori):
                # if area_id not in VA_gt1_pred_group_dict.keys():
                #     VA_gt1_pred_group_dict[area_id]=[basic_interval]
                # else:
                #     VA_gt1_pred_group_dict[area_id].append(basic_interval)
                VA_gt1_pred_group_dict[area_id].append(basic_interval)
            # if basic_interval == [135, 141] and area == [130, 150]:
            #     a =1
            if pred_in_area(basic_interval, area_gt2_ori):
                # if area_id not in VA_gt2_pred_group_dict.keys():
                #     VA_gt2_pred_group_dict[area_id]=[basic_interval]
                # else:
                #     VA_gt2_pred_group_dict[area_id].append(basic_interval)
                VA_gt2_pred_group_dict[area_id].append(basic_interval)
            if pred_in_area(basic_interval, area_gt3_ori):
                # if area_id not in VA_gt3_pred_group_dict.keys():
                #     VA_gt3_pred_group_dict[area_id]=[basic_interval]
                # else:
                VA_gt3_pred_group_dict[area_id].append(basic_interval)
            if pred_in_area(basic_interval, area_gt4_ori):
                # if area_id not in VA_gt4_pred_group_dict.keys():
                #     VA_gt4_pred_group_dict[area_id]=[basic_interval]
                # else:
                VA_gt4_pred_group_dict[area_id].append(basic_interval)

        # VA-d
        for j, (area_id, area) in enumerate(VA_d_dict.items()):
            if pred_in_area(basic_interval, area):
                if area_id not in VA_d_pred_group_dict.keys():
                    VA_d_pred_group_dict[area_id] = [basic_interval]
                else:
                    VA_d_pred_group_dict[area_id].append(basic_interval)
        # VA
        for j, (area_id, area) in enumerate(VA_dict.items()):
            if pred_in_area(basic_interval, area):
                if area_id not in VA_pred_group_dict.keys():
                    VA_pred_group_dict[area_id] = [basic_interval]
                else:
                    VA_pred_group_dict[area_id].append(basic_interval)
    if print_msg:
        # print pred dict
        print("IA_pred_group_dict", IA_pred_group_dict)
        print("VA_f_pred_group_dict", VA_f_pred_group_dict)
        # print("VA_gt_pred_group_dict",VA_gt_pred_group_dict)
        print("VA_gt1_pred_group_dict", VA_gt1_pred_group_dict)
        print("VA_gt2_pred_group_dict", VA_gt2_pred_group_dict)
        print("VA_gt3_pred_group_dict", VA_gt3_pred_group_dict)
        print("VA_gt4_pred_group_dict", VA_gt4_pred_group_dict)
        print("VA_d_pred_group_dict", VA_d_pred_group_dict)
        # print("VA_pred_group_dict",VA_pred_group_dict)

    d_ia_set_dict = {}
    d_ia_sign_set_dict = {}
    l_ia_set_dict = {}
    dl_score_ia_dict = {}

    dl_part_ia_score_dict = {}
    d_ia_both_sides_dict = {}
    dl_score = {}
    adjust_dl_score = {}

    for i, (area_id, basic_interval_list) in enumerate(IA_pred_group_dict.items()):
        if basic_interval_list == []:
            dl_score_ia_dict[area_id] = 0
            l_ia_set_dict[area_id] = 0
            d_ia_set_dict[
                area_id] = 0  # only when basic_interval_list ！= [],the value are used,so set value to 0 have boe problem
            d_ia_sign_set_dict[
                area_id] = 0  # only when basic_interval_list ！= [],the value are used,so set value to 0 have boe problem
            d_ia_both_sides_dict[area_id] = [0,0]
            dl_score[area_id] = [1,1]
            continue
        l_ia = 0
        l_ia_set_sum = 0
        d_ia_set_sum = 0
        d_ia_sign_set_sum = 0
        d_ia_set_before_side_sum = 0
        ia_set_num = len(basic_interval_list)
        IA_mid = (IA_dict[area_id][0] + IA_dict[area_id][1]) / 2
        IA_len = IA_dict[area_id][1] - IA_dict[area_id][0]
        IA_start = IA_dict[area_id][0]
        IA_end = IA_dict[area_id][1]
        gt_num = label_interval_num

        test_group = "group"
        # test_group = "no group"
        if test_group == "group":
            for basic_interval in basic_interval_list:
                l_ia += basic_interval[1] - basic_interval[0]
                # the length of fitst and last IA could be set a fixed value,min(一个周期的长度,ori len),or min(mean len,ori len)
                if i == 0:
                    base_position = IA_dict[area_id][1]
                    # d_ia_set_sum+= abs((basic_interval[1] + basic_interval[0])/2 -base_position)
                # last id of IA is label_interval_num,not label_interval_num -1
                elif i == label_interval_num:
                    base_position = IA_dict[area_id][0]
                else:
                    base_position = IA_mid
                # d_ia_set_sum+= abs((basic_interval[1] + basic_interval[0])/2 -IA_mid)
                d_ia_set_sum += abs((basic_interval[1] + basic_interval[0]) / 2 - base_position)
                # i =0 or i = label_interval_num,only have positive distance

                d_ia_sign_set_sum += (basic_interval[1] + basic_interval[0]) / 2 - base_position

                d_ia_before = (basic_interval[1] + basic_interval[0]) / 2 - IA_start
                d_ia_end = IA_end - (basic_interval[1] + basic_interval[0]) / 2
                if i == 0:
                    d_ia_set_before_side_sum += d_ia_end
                elif i == gt_num:
                    d_ia_set_before_side_sum += d_ia_before
                else:
                    d_ia_set_before_side_sum += min(d_ia_before,d_ia_end)

            l_ia_set_dict[area_id] = l_ia
            d_ia = d_ia_set_sum / ia_set_num if ia_set_num != 0 else 0
            d_ia_set_dict[area_id] = d_ia
            d_ia_sign = d_ia_sign_set_sum / ia_set_num if ia_set_num != 0 else 0
            d_ia_sign_set_dict[area_id] = d_ia_sign
            d_ia_before_side = d_ia_set_before_side_sum / ia_set_num if ia_set_num != 0 else 0
            # d_ia_min = min(d_ia_before_side,IA_len-d_ia_before_side)
            d_ia_both_sides_dict[area_id] = [d_ia_before_side,d_ia_before_side]


            # n = 0.01
            # n = 0.005
            # n = 0.001
            n = 0.0003
            dl_rate = 1/2
            # dl_rate = 2/3
            dl_score_before_d = dl_rate * math.e ** (-n * d_ia_both_sides_dict[area_id][0])
            dl_score_after_d = dl_rate * math.e ** (-n * d_ia_both_sides_dict[area_id][1])
            dl_score_after_l = (1-dl_rate) * math.e ** (-n * l_ia_set_dict[area_id])


            if i == 0:
                dl_score_before = 0
                dl_score_after = dl_score_after_d + dl_score_after_l
            elif i == gt_num:
                dl_score_before = dl_score_before_d + dl_score_after_l
                dl_score_after = 0
            else:
                dl_score_before = dl_score_before_d + dl_score_after_l
                dl_score_after = dl_score_after_d + dl_score_after_l
            dl_score[area_id] = [dl_score_before,dl_score_after]

            dl_score_ia = 0
            #
            if l_ia != 0:
                if i == 0:
                    # 一边情况下就没有1了
                    # parameter_alpha1 = parameter_rate/(2+parameter_rate)
                    #
                    # debug1 = (1/2*IA_len-d_ia)
                    # debug2 = 2 / IA_len * (1/2*IA_len-d_ia)
                    # debug3 = parameter_alpha1 * 2 / IA_len * (1/2*IA_len-d_ia)
                    #
                    # debug4 = (1 - l_ia / IA_len)
                    # debug5 = (1 - parameter_alpha1) * (1 - l_ia / IA_len)

                    # dl_score_ia = parameter_alpha1 * 2 / IA_len * (1/2*IA_len-d_ia) + (1 - parameter_alpha1) * (1 - l_ia / IA_len) \
                    #     if IA_len != 0 else 0

                    parameter_alpha = parameter_rate / (1 + parameter_rate)
                    debug11 = (IA_len - d_ia)
                    debug21 = 1 / IA_len * (IA_len - d_ia)
                    debug31 = parameter_alpha / IA_len * (IA_len - d_ia)

                    debug41 = (1 - l_ia / IA_len)
                    debug51 = (1 - parameter_alpha) * (1 - l_ia / IA_len)
                    # 得分跟系数也有关系

                    dl_score_ia1 = parameter_alpha / IA_len * (IA_len - d_ia) + (1 - parameter_alpha) * (
                            1 - l_ia / IA_len) \
                        if IA_len != 0 else 0
                    # 对抗边界是2
                    if parameter_rate <= 2:
                        min_value = parameter_alpha / IA_len * (IA_len - 1 / 2 * IA_len)
                        # min_value = min_value + 0.0000001
                        dl_score_ia = map_value(dl_score_ia1, min_value, 1, 1e-10, 1)
                    else:
                        min_value = (1 - parameter_alpha) * (1 - 0 / IA_len)
                        dl_score_ia = map_value(dl_score_ia1, min_value, 1, 1e-10, 1)

                        # dl_score_ia = dl_score_ia1
                    # print("0 dl_score_ia",dl_score_ia)
                    # dl_part_ia_score_dict[area_id] = {
                    #     "parameter_alpha":parameter_alpha,
                    #     "d_score":1/IA_len * (IA_len - d_ia),
                    #     "(1 - parameter_alpha)":(1 - parameter_alpha),
                    #     "d_score_parameter":(1 - parameter_alpha) * (
                    #                 1 - l_ia / IA_len)
                    # }
                    if pred_case_id == 2:
                        debug = 1
                # last id of IA is label_interval_num,not label_interval_num -1
                elif i == label_interval_num:

                    # debug1 = (1 / 2 * IA_len - d_ia)
                    # debug2 = 2 / IA_len * (1 / 2 * IA_len - d_ia)
                    # debug3 = parameter_alpha * 2 / IA_len * (1 / 2 * IA_len - d_ia)
                    #
                    # debug4 = (1 - l_ia / IA_len)
                    # debug5 = (1 - parameter_alpha) * (1 - l_ia / IA_len)

                    # dl_score_ia = parameter_alpha * 2 / IA_len * (1/2*IA_len-d_ia) + (1 - parameter_alpha) * (1 - l_ia / IA_len) \
                    #     if IA_len != 0 else 0

                    parameter_alpha = parameter_rate / (1 + parameter_rate)
                    debug11 = (IA_len - d_ia)
                    debug21 = 1 / IA_len * (IA_len - d_ia)
                    debug31 = parameter_alpha / IA_len * (IA_len - d_ia)

                    debug41 = (1 - l_ia / IA_len)
                    debug51 = (1 - parameter_alpha) * (1 - l_ia / IA_len)
                    dl_score_ia1 = parameter_alpha / IA_len * (IA_len - d_ia) + (1 - parameter_alpha) * (
                                1 - l_ia / IA_len) \
                        if IA_len != 0 else 0
                    # x, a, b, c, d
                    if parameter_rate <= 2:
                        min_value = parameter_alpha / IA_len * (IA_len - 1 / 2 * IA_len)
                        # min_value = min_value + 0.0000001
                        dl_score_ia = map_value(dl_score_ia1, min_value, 1, 1e-10, 1)
                    else:
                        min_value = (1 - parameter_alpha) * (1 - 0 / IA_len)
                        dl_score_ia = map_value(dl_score_ia1, min_value, 1, 1e-10, 1)
                        # dl_score_ia = map_value(dl_score_ia,parameter_alpha,1,0,1)
                    # dl_part_ia_score_dict[area_id] = {
                    #     "parameter_alpha": parameter_alpha,
                    #     "d_score": 1 / IA_len * (IA_len - d_ia),
                    #     "(1 - parameter_alpha)": (1 - parameter_alpha),
                    #     "d_score_parameter": (1 - parameter_alpha) * (
                    #             1 - l_ia / IA_len)
                    # }
                    if pred_case_id == 2:
                        debug = 1
                else:

                    parameter_alpha = parameter_rate / (2 + parameter_rate)
                    dl_score_ia = parameter_alpha * 2 / IA_len * d_ia + (1 - parameter_alpha) * (1 - l_ia / IA_len) \
                        if IA_len != 0 else 0
                    if pred_case_id == 2:
                        print("  parameter_alpha", parameter_alpha)
                        print("  d_ia", d_ia)
                        print("  d_score", 2 / IA_len * d_ia)
                        print("  d_score_parameter", parameter_alpha * 2 / IA_len * d_ia)
                        print("  (1 - parameter_alpha)", (1 - parameter_alpha))
                        print("  l_ia", l_ia)
                        print("  l_score", (1 - l_ia / IA_len))
                        print("  l_score_parameter", (1 - parameter_alpha) * (1 - l_ia / IA_len))
                        print("  IA_len", IA_len)
                        debug = 1
                dl_part_ia_score_dict
        else:
            if print_msg:
                print("basic_interval_list num", len(basic_interval_list))
            dl_score_ia_sum = 0
            for basic_interval in basic_interval_list:
                if i == 0:
                    base_position = IA_dict[area_id][1]
                elif i == label_interval_num:
                    base_position = IA_dict[area_id][0]
                else:
                    base_position = IA_mid

                l_ia = basic_interval[1] - basic_interval[0]
                d_ia = abs((basic_interval[1] + basic_interval[0]) / 2 - base_position)
                if pred_case_id == 2:
                    debug = 1
                dl_score_ia = 0
                #
                if l_ia != 0:
                    if i == 0:
                        parameter_alpha = parameter_rate / (1 + parameter_rate)
                        dl_score_ia1 = parameter_alpha / IA_len * (IA_len - d_ia) + (1 - parameter_alpha) * (
                                1 - l_ia / IA_len) \
                            if IA_len != 0 else 0
                        # 对抗边界是2
                        if parameter_rate <= 2:
                            min_value = parameter_alpha / IA_len * (IA_len - 1 / 2 * IA_len)
                            dl_score_ia = map_value(dl_score_ia1, min_value, 1, 1e-10, 1)
                        else:
                            min_value = (1 - parameter_alpha) * (1 - 0 / IA_len)
                            dl_score_ia = map_value(dl_score_ia1, min_value, 1, 1e-10, 1)
                        if print_msg:
                            print("0 dl_score_ia", dl_score_ia)
                        if pred_case_id == 2:
                            debug = 1
                    elif i == label_interval_num:
                        parameter_alpha = parameter_rate / (1 + parameter_rate)
                        debug11 = (IA_len - d_ia)
                        debug21 = 1 / IA_len * (IA_len - d_ia)
                        debug31 = parameter_alpha / IA_len * (IA_len - d_ia)

                        debug41 = (1 - l_ia / IA_len)
                        debug51 = (1 - parameter_alpha) * (1 - l_ia / IA_len)
                        dl_score_ia1 = parameter_alpha / IA_len * (IA_len - d_ia) + (1 - parameter_alpha) * (
                                1 - l_ia / IA_len) \
                            if IA_len != 0 else 0
                        # x, a, b, c, d
                        if parameter_rate <= 2:
                            min_value = parameter_alpha / IA_len * (IA_len - 1 / 2 * IA_len)
                            # min_value = min_value + 0.0000001
                            dl_score_ia = map_value(dl_score_ia1, min_value, 1, 1e-10, 1)
                        else:
                            min_value = (1 - parameter_alpha) * (1 - 0 / IA_len)
                            dl_score_ia = map_value(dl_score_ia1, min_value, 1, 1e-10, 1)
                            # dl_score_ia = map_value(dl_score_ia,parameter_alpha,1,0,1)
                        if pred_case_id == 2:
                            debug = 1
                    else:

                        parameter_alpha = parameter_rate / (2 + parameter_rate)
                        dl_score_ia = parameter_alpha * 2 / IA_len * d_ia + (1 - parameter_alpha) * (1 - l_ia / IA_len) \
                            if IA_len != 0 else 0
                        if pred_case_id == 2:
                            if print_msg:
                                print("  parameter_alpha", parameter_alpha)
                                print("  d_ia", d_ia)
                                print("  d_score", 2 / IA_len * d_ia)
                                print("  d_score_parameter", parameter_alpha * 2 / IA_len * d_ia)
                                print("  (1 - parameter_alpha)", (1 - parameter_alpha))
                                print("  l_ia", l_ia)
                                print("  l_score", (1 - l_ia / IA_len))
                                print("  l_score_parameter", (1 - parameter_alpha) * (1 - l_ia / IA_len))
                                print("  IA_len", IA_len)
                                debug = 1

                # sum
                dl_score_ia_sum += dl_score_ia
                l_ia_set_sum += basic_interval[1] - basic_interval[0]
                d_ia_set_sum += abs((basic_interval[1] + basic_interval[0]) / 2 - base_position)
                d_ia_sign_set_sum += (basic_interval[1] + basic_interval[0]) / 2 - base_position

            # sum
            # test_no_group_type = "sum"
            test_no_group_type = "average"
            if test_no_group_type == "sum":
                dl_score_ia = dl_score_ia_sum
                l_ia_set_dict[area_id] = l_ia_set_sum
                d_ia_set_dict[area_id] = d_ia_set_sum
                d_ia_sign_set_dict[area_id] = d_ia_sign_set_sum
            else:
                dl_score_ia = dl_score_ia_sum / len(basic_interval_list) if len(basic_interval_list) != 0 else 0
                l_ia_set_dict[area_id] = l_ia_set_sum / len(basic_interval_list) if len(basic_interval_list) != 0 else 0
                d_ia_set_dict[area_id] = d_ia_set_sum / len(basic_interval_list) if len(basic_interval_list) != 0 else 0
                d_ia_sign_set_dict[area_id] = d_ia_sign_set_sum
        dl_score_ia_dict[area_id] = dl_score_ia
    #     deal empty and no area
    if print_msg:
        print("d_ia_set_dict", d_ia_set_dict)
        print("d_ia_sign_set_dict", d_ia_sign_set_dict)
        print("l_ia_set_dict", l_ia_set_dict)
    # print("dl_score_ia_dict",dl_score_ia_dict)

    adjust_dl_score = dl_score

    # cal label_function_gt

    for i, (area_id, area) in enumerate(VA_gt_dict.items()):

        VA_gt_len = area[1] - area[0]
        x1, x2 = cal_x1_x2(VA_gt_len, parameter_rho)
        x1_ori, x2_ori = x1 + area[0], x2 + area[0]
        x_mid = 2 * x1
        x_mid_ori = x_mid + area[0]
        # x1_ori,x2_ori = x1+area[0], x2+area[0]
        area_gt1 = [0, x1]
        area_gt2 = [x1, 2 * x1]
        area_gt3 = [2 * x1, x2]
        area_gt4 = [x2, VA_gt_len]
        area_gt1_ori = [area[0], x1 + area[0]]
        area_gt2_ori = [x1 + area[0], 2 * x1 + area[0]]
        area_gt3_ori = [2 * x1 + area[0], x2 + area[0]]
        area_gt4_ori = [x2 + area[0], area[1]]
        if print_msg:
            print("VA_gt_len", VA_gt_len)

        if parameter_rho < 0:
            parameter_b1, parameter_gama1 = 1 / 2, 0
            parameter_b2, parameter_gama2 = 1 / 2, 0
        else:
            parameter_b1, parameter_gama1 = cal_power_function_coefficient(x1, parameter_lamda)
            parameter_b2, parameter_gama2 = cal_power_function_coefficient(x2 - x_mid, parameter_lamda)

        func_dict.update({
            area_id + "_gt1": {"area": area_gt1_ori,
                               "area_start": area_gt1_ori[0],
                               "parameter_a": parameter_b1,
                               "parameter_beta": parameter_gama1,
                               "func": "va_gt1_func",
                               "x_tp": x1,
                               "gt_len": VA_gt_len,
                               "area_len": None,
                               "parameter_rho": parameter_rho
                               },
            area_id + "_gt2": {"area": area_gt2_ori,
                               "area_start": area_gt1_ori[0],
                               "parameter_a": parameter_b1,
                               "parameter_beta": parameter_gama1,
                               "func": "va_gt2_func",
                               "x_tp": x1,
                               "gt_len": VA_gt_len,
                               "area_len": None,
                               "parameter_rho": parameter_rho
                               },
            area_id + "_gt3": {"area": area_gt3_ori,
                               "area_start": area_gt1_ori[0],
                               "parameter_a": parameter_b2,
                               "parameter_beta": parameter_gama2,
                               "func": "va_gt3_func",
                               "x_tp": x2,
                               "gt_len": VA_gt_len,
                               "area_len": None,
                               "parameter_rho": parameter_rho
                               },
            area_id + "_gt4": {"area": area_gt4_ori,
                               "area_start": area_gt1_ori[0],
                               "parameter_a": parameter_b2,
                               "parameter_beta": parameter_gama2,
                               "func": "va_gt4_func",
                               "x_tp": x2,
                               "gt_len": VA_gt_len,
                               "area_len": None,
                               "parameter_rho": parameter_rho
                               },

        })

        s_a_gt1 = cal_integral_in_range_gt_power_func1(area_gt1_ori, parameter_b1, parameter_gama1, x_tp=x1,
                                                       gt_len=VA_gt_len, area_start=area_gt1_ori[0],
                                                       parameter_rho=parameter_rho)
        s_a_gt2 = cal_integral_in_range_gt_power_func2(area_gt2_ori, parameter_b1, parameter_gama1, x_tp=x1,
                                                       gt_len=VA_gt_len, area_start=area_gt1_ori[0],
                                                       parameter_rho=parameter_rho)
        s_a_gt3 = cal_integral_in_range_gt_power_func3(area_gt3_ori, parameter_b2, parameter_gama2, x_tp=x2,
                                                       gt_len=VA_gt_len, area_start=area_gt1_ori[0],
                                                       parameter_rho=parameter_rho)
        s_a_gt4 = cal_integral_in_range_gt_power_func4(area_gt4_ori, parameter_b2, parameter_gama2, x_tp=x2,
                                                       gt_len=VA_gt_len, area_start=area_gt1_ori[0],
                                                       parameter_rho=parameter_rho)

        s_a_gt = s_a_gt1 + s_a_gt2 + s_a_gt3 + s_a_gt4

        func_dict[area_id + "_gt1"]["area_s"] = s_a_gt1
        func_dict[area_id + "_gt2"]["area_s"] = s_a_gt2
        func_dict[area_id + "_gt3"]["area_s"] = s_a_gt3
        func_dict[area_id + "_gt4"]["area_s"] = s_a_gt4

        method = 2

        va_f_area = VA_f_dict[area_id]
        s_a_f = 0
        if va_f_area == []:
            pass
        else:
            VA_f_len = va_f_area[1] - va_f_area[0]
            parameter_beta1 = 1
            parameter_a1 = 1 / (2 * VA_f_len)

            greater_than_th = False

            func_dict[area_id + "_f"] = {"area": va_f_area,
                                         "area_start": va_f_area[0],
                                         "parameter_a": parameter_a1,
                                         "parameter_beta": parameter_beta1,
                                         "func": "va_f_func",
                                         "x_tp": None,
                                         "gt_len": None,
                                         "area_len": VA_f_len,
                                         "greater_than_th": greater_than_th
                                         }

            # f
            s_max_f = cal_integral_in_range_f_power_func(va_f_area, parameter_a1, parameter_beta1, VA_f_len, VA_f_len,
                                                         area_start=va_f_area[0], greater_than_th=greater_than_th)
            func_dict[area_id + "_f"]["area_s"] = s_max_f

            s_rate = s_max_f / s_a_gt
            func_dict[area_id + "_f_a"] = func_dict[area_id + "_f"]
            s_a_f = s_max_f
            if s_rate > parameter_theta:
                greater_than_th = True

                # method1
                if method == 1:
                    parameter_beta1_a = VA_f_len / (2 * parameter_theta * s_a_gt) - 1
                    parameter_a1_a = 1 / (2 * pow(VA_f_len, parameter_beta1_a))
                # method2
                else:
                    parameter_beta1_a = VA_f_len / (2 * parameter_theta * s_a_gt) - 1  # same with method1
                    parameter_a1_a = VA_f_len / pow(1 / 2, parameter_beta1_a)

                s_a_f = cal_integral_in_range_f_power_func(va_f_area, parameter_a1_a, parameter_beta1_a, VA_f_len,
                                                           VA_f_len, area_start=va_f_area[0], method=method,
                                                           greater_than_th=greater_than_th)
                func_dict[area_id + "_f_a"] = {
                    "area": va_f_area,
                    "area_start": va_f_area[0],
                    "area_s": s_a_f,
                    "parameter_a": parameter_a1_a,
                    "parameter_beta": parameter_beta1_a,
                    # "func": "va_f_func",
                    # "func": "va_f_reverse_func",
                    "x_tp": None,
                    "gt_len": None,
                    "area_len": VA_f_len,
                    "greater_than_th": greater_than_th
                }
                if method == 1:
                    func_dict[area_id + "_f_a"]["func"] = "va_f_func"
                else:
                    func_dict[area_id + "_f_a"]["func"] = "va_f_reverse_func"

        # d
        va_d_area = VA_d_dict[area_id]
        s_a_d = 0
        if va_d_area == []:
            pass
        else:

            if pred_case_id == 7 and area_id == "id_2":
                debug = 1

            VA_d_len = va_d_area[1] - va_d_area[0]
            parameter_beta2 = 1
            parameter_a2 = 1 / (2 * VA_d_len)

            greater_than_th = False
            s_max_d = cal_integral_in_range_d_power_func(va_d_area, parameter_a2, parameter_beta2, VA_d_len, VA_d_len,
                                                         area_start=va_d_area[0], greater_than_th=greater_than_th)
            func_dict[area_id + "_d"] = {
                "area": va_d_area,
                "area_start": va_d_area[0],
                "parameter_a": parameter_a2,
                "parameter_beta": parameter_beta2,
                "func": "va_d_func",
                "x_tp": None,
                "gt_len": None,
                "area_len": VA_d_len,
                "greater_than_th": greater_than_th
            }

            func_dict[area_id + "_d"]["area_s"] = s_max_d

            func_dict[area_id + "_d_a"] = func_dict[area_id + "_d"]
            s_rate = s_max_d / s_a_gt
            s_a_d = s_max_d

            if s_rate > parameter_theta:
                greater_than_th = True

                # method1
                if method == 1:
                    parameter_beta2_a = VA_d_len / (2 * parameter_theta * s_a_gt) - 1
                    parameter_a2_a = 1 / (2 * pow(VA_d_len, parameter_beta2_a))
                else:
                    # method2
                    parameter_beta2_a = VA_d_len / (2 * parameter_theta * s_a_gt) - 1  # same with method1
                    parameter_a2_a = VA_d_len / pow(1 / 2, parameter_beta2_a)

                s_a_d = cal_integral_in_range_d_power_func(va_d_area, parameter_a2_a, parameter_beta2_a, VA_d_len,
                                                           VA_d_len,
                                                           area_start=va_d_area[0], method=method,
                                                           greater_than_th=greater_than_th)
                func_dict[area_id + "_d_a"] = {
                    "area": va_d_area,
                    "area_start": va_d_area[0],
                    "area_s": s_a_d,
                    "parameter_a": parameter_a2_a,
                    "parameter_beta": parameter_beta2_a,
                    # "func": "va_d_func",
                    # "func": "va_d_reverse_func",
                    "x_tp": None,
                    "gt_len": None,
                    "area_len": VA_d_len,
                    "greater_than_th": greater_than_th

                }

                if method == 1:
                    func_dict[area_id + "_d_a"]["func"] = "va_d_func"
                else:
                    func_dict[area_id + "_d_a"]["func"] = "va_d_reverse_func"

        s_a_func_dict[area_id] = {
            "s_a_va_gt_len": VA_gt_len,
            "s_a_va_gt": s_a_gt,
            "s_a_va": s_a_f + s_a_gt + s_a_d,
            "x1": x1,
            "x2": x2,
            "x_mid": x_mid,
            "x1_ori": x1_ori,
            "x2_ori": x2_ori,
            "x_mid_ori": x_mid_ori,
        }
    # plot func
    if pred_case_id == 7 and area_id == "id_2":
        debug = 1
        # plot_func_multi(func_dict)
    # plot_func_multi(func_dict)
    # plot_func_multi_paper(func_dict,window_length)

    import pprint

    # print("s_a_func_dict")
    # pprint(s_a_func_dict, indent=4)

    # cal recall_va,detection_score
    recall_va_list = []
    recall_va_dict = {}
    detection_score_ia_i_dict = {}
    precision_va_dict = {}

    gt_num = label_interval_num
    pred_func_dict = {}
    basic_interval_integral_dict = {}

    for i in range(gt_num):
        area_id = "id_" + str(i)
        # print("area_id",area_id)
        area_id_back = "id_" + str(i + 1)
        VA_f_i_pred_group = VA_f_pred_group_dict[area_id]
        VA_gt1_i_pred_group = VA_gt1_pred_group_dict[area_id]
        VA_gt2_i_pred_group = VA_gt2_pred_group_dict[area_id]
        VA_gt3_i_pred_group = VA_gt3_pred_group_dict[area_id]
        VA_gt4_i_pred_group = VA_gt4_pred_group_dict[area_id]
        VA_d_i_pred_group = VA_d_pred_group_dict[area_id]

        IA_pred_i_group = IA_pred_group_dict[area_id]
        IA_pred_i_back_group = IA_pred_group_dict[area_id_back]
        VA_pred_i_group = VA_pred_group_dict[area_id]

        pred_integral_all = 0

        # # f
        va_f_area = VA_f_dict[area_id]
        if va_f_area == []:
            pass
        else:
            VA_f_len = va_f_area[1] - va_f_area[0]
            parameter_a1 = func_dict[area_id + "_f_a"]["parameter_a"]
            parameter_beta1 = func_dict[area_id + "_f_a"]["parameter_beta"]
            greater_than_th = func_dict[area_id + "_f_a"]["greater_than_th"]

            for j, basic_interval in enumerate(VA_f_i_pred_group):
                if basic_interval != []:

                    pred_func_dict[area_id + "_f_a" + "_" + str(j)] = {
                        "area": basic_interval,
                        "area_start": va_f_area[0],
                        "parameter_a": parameter_a1,
                        "parameter_beta": parameter_beta1,
                        # "func":"va_f_func",
                        "x_tp": None,
                        "gt_len": None,
                        "area_len": VA_f_len
                    }
                    if not greater_than_th:
                        pred_func_dict[area_id + "_f_a" + "_" + str(j)]["func"] = "va_d_func"
                    else:
                        if method == 1:
                            pred_func_dict[area_id + "_f_a" + "_" + str(j)]["func"] = "va_f_func"
                        else:
                            pred_func_dict[area_id + "_f_a" + "_" + str(j)]["func"] = "va_f_reverse_func"

                    cal_integral_basic_interval_f = cal_integral_in_range_f_power_func(basic_interval, parameter_a1,
                                                                                       parameter_beta1, VA_f_len,
                                                                                       VA_f_len,
                                                                                       area_start=va_f_area[0],
                                                                                       method=method,
                                                                                       greater_than_th=greater_than_th)
                    pred_integral_all += cal_integral_basic_interval_f
                    basic_interval_integral_dict[area_id + "_f" + "_" + str(j)] = cal_integral_basic_interval_f

        # gt1-gt4
        s_a_gt_len = s_a_func_dict[area_id]["s_a_va_gt_len"]
        s_a_va = s_a_func_dict[area_id]["s_a_va"]
        parameter_b1 = func_dict[area_id + "_gt1"]["parameter_a"]
        parameter_gama1 = func_dict[area_id + "_gt1"]["parameter_beta"]
        parameter_b2 = func_dict[area_id + "_gt3"]["parameter_a"]
        parameter_gama2 = func_dict[area_id + "_gt3"]["parameter_beta"]
        x1 = s_a_func_dict[area_id]["x1"]
        x2 = s_a_func_dict[area_id]["x2"]
        area_gt = VA_gt_dict[area_id]
        VA_gt_len = area_gt[1] - area_gt[0]

        area_gt1 = func_dict[area_id + "_gt1"]["area"]
        area_gt2 = func_dict[area_id + "_gt2"]["area"]
        area_gt3 = func_dict[area_id + "_gt3"]["area"]
        area_gt4 = func_dict[area_id + "_gt4"]["area"]
        area_start = func_dict[area_id + "_gt1"]["area_start"]

        for j, basic_interval in enumerate(VA_gt1_i_pred_group):
            if basic_interval != []:
                pred_func_dict[area_id + "_gt1" + "_" + str(j)] = {
                    "area": basic_interval,
                    "area_start": area_start,
                    "parameter_a": parameter_b1,
                    "parameter_beta": parameter_gama1,
                    "func": "va_gt1_func",
                    "x_tp": None,
                    "gt_len": VA_gt_len,
                    "area_len": None,
                    "parameter_rho": parameter_rho
                }
                cal_integral_basic_interval_gt1 = cal_integral_in_range_gt_power_func1(basic_interval, parameter_b1,
                                                                                       parameter_gama1, x_tp=x1,
                                                                                       gt_len=VA_gt_len,
                                                                                       area_start=area_start,
                                                                                       parameter_rho=parameter_rho)
                pred_integral_all += cal_integral_basic_interval_gt1
                basic_interval_integral_dict[area_id + "_gt1" + "_" + str(j)] = cal_integral_basic_interval_gt1

        for j, basic_interval in enumerate(VA_gt2_i_pred_group):
            if basic_interval != []:
                pred_func_dict[area_id + "_gt2" + "_" + str(j)] = {
                    "area": basic_interval,
                    "area_start": area_start,
                    "parameter_a": parameter_b1,
                    "parameter_beta": parameter_gama1,
                    "func": "va_gt2_func",
                    "x_tp": x1,
                    "gt_len": VA_gt_len,
                    "area_len": None,
                    "parameter_rho": parameter_rho
                }
                cal_integral_basic_interval_gt2 = cal_integral_in_range_gt_power_func2(basic_interval, parameter_b1,
                                                                                       parameter_gama1, x_tp=x1,
                                                                                       gt_len=VA_gt_len,
                                                                                       area_start=area_start,
                                                                                       parameter_rho=parameter_rho)
                pred_integral_all += cal_integral_basic_interval_gt2
                basic_interval_integral_dict[area_id + "_gt2" + "_" + str(j)] = cal_integral_basic_interval_gt2

        for j, basic_interval in enumerate(VA_gt3_i_pred_group):
            if basic_interval != []:
                pred_func_dict[area_id + "_gt3" + "_" + str(j)] = {
                    "area": basic_interval,
                    "area_start": area_start,
                    "parameter_a": parameter_b2,
                    "parameter_beta": parameter_gama2,
                    "func": "va_gt3_func",
                    "x_tp": x2,
                    "gt_len": VA_gt_len,
                    "area_len": None,
                    "parameter_rho": parameter_rho

                }
                cal_integral_basic_interval_gt3 = cal_integral_in_range_gt_power_func3(basic_interval, parameter_b2,
                                                                                       parameter_gama2, x_tp=x2,
                                                                                       gt_len=VA_gt_len,
                                                                                       area_start=area_start,
                                                                                       parameter_rho=parameter_rho)
                pred_integral_all += cal_integral_basic_interval_gt3
                basic_interval_integral_dict[area_id + "_gt3" + "_" + str(j)] = cal_integral_basic_interval_gt3

        for j, basic_interval in enumerate(VA_gt4_i_pred_group):
            if basic_interval != []:
                pred_func_dict[area_id + "_gt4" + "_" + str(j)] = {
                    "area": basic_interval,
                    "area_start": area_start,
                    "parameter_a": parameter_b2,
                    "parameter_beta": parameter_gama2,
                    "func": "va_gt4_func",
                    "x_tp": x2,
                    "gt_len": VA_gt_len,
                    "area_len": None,
                    "parameter_rho": parameter_rho
                }
                cal_integral_basic_interval_gt4 = cal_integral_in_range_gt_power_func4(basic_interval, parameter_b2,
                                                                                       parameter_gama2, x_tp=x2,
                                                                                       gt_len=VA_gt_len,
                                                                                       area_start=area_start,
                                                                                       parameter_rho=parameter_rho)
                pred_integral_all += cal_integral_basic_interval_gt4
                basic_interval_integral_dict[area_id + "_gt4" + "_" + str(j)] = cal_integral_basic_interval_gt4

        # # d
        va_d_area = VA_d_dict[area_id]
        if va_d_area == []:
            pass
        else:
            VA_d_len = va_d_area[1] - va_d_area[0]
            parameter_a2 = func_dict[area_id + "_d_a"]["parameter_a"]
            parameter_beta2 = func_dict[area_id + "_d_a"]["parameter_beta"]
            greater_than_th = func_dict[area_id + "_d_a"]["greater_than_th"]
            if pred_case_id == 7 and area_id == "id_2":
                debug = 1

            for j, basic_interval in enumerate(VA_d_i_pred_group):
                if basic_interval != []:
                    pred_func_dict[area_id + "_d_a" + "_" + str(j)] = {
                        "area": basic_interval,
                        "area_start": va_d_area[0],
                        "parameter_a": parameter_a2,
                        "parameter_beta": parameter_beta2,
                        # "func":"va_d_func",
                        "x_tp": None,
                        "gt_len": None,
                        "area_len": VA_d_len
                    }
                    if not greater_than_th:
                        pred_func_dict[area_id + "_d_a" + "_" + str(j)]["func"] = "va_d_func"
                    else:
                        if method == 1:
                            pred_func_dict[area_id + "_d_a" + "_" + str(j)]["func"] = "va_d_func"
                        else:
                            pred_func_dict[area_id + "_d_a" + "_" + str(j)]["func"] = "va_d_reverse_func"

                    cal_integral_basic_interval_d = cal_integral_in_range_d_power_func(basic_interval, parameter_a2,
                                                                                       parameter_beta2, VA_d_len,
                                                                                       VA_d_len,
                                                                                       area_start=va_d_area[0],
                                                                                       method=method,
                                                                                       greater_than_th=greater_than_th)
                    pred_integral_all += cal_integral_basic_interval_d
                    basic_interval_integral_dict[area_id + "_d" + "_" + str(j)] = cal_integral_basic_interval_d

        # print("pred_integral_all",pred_integral_all)
        va_area = VA_dict[area_id]
        recall_va_i = pred_integral_all / s_a_va
        recall_va_list.append(recall_va_i)
        if pred_case_id == 7:
            debug = 1
        recall_va_dict[area_id] = recall_va_i

        detection_score_i = 1 if VA_pred_group_dict[area_id] != [] else 0
        detection_score_ia_i_dict[area_id] = detection_score_i
        # invalid part punish in is not enough
        # detection
        if cal_mode == "detection":
            precision_va_dict[area_id] = detection_score_i

        # precision_i_group = IA_pred_i_group + IA_pred_i_back_group + VA_pred_i_group
        #
        # precision_i_all = 0
        # precision_i_valid = 0
        # for precision_i_item in precision_i_group:
        #     precision_i_all += precision_i_item[1] - precision_i_item[0]
        # for precision_i_item in VA_pred_i_group:
        #     precision_i_valid += precision_i_item[1] - precision_i_item[0]

        precision_i_all = pred_integral_all
        precision_i_valid = pred_integral_all
        precision_i_group = IA_pred_i_group + IA_pred_i_back_group

        for precision_i_item in precision_i_group:
            precision_i_all += parameter_eta*(precision_i_item[1] - precision_i_item[0])

        if pred_case_id == 4:
            debug = 1
        precision_va_i = precision_i_valid / precision_i_all if precision_i_all != 0 else 0

        # precision also consider invalid part
        # proportion
        if cal_mode == "proportion":
            precision_va_dict[area_id] = precision_va_i

    if print_msg:
        # print basic_interval_integral
        print("basic_interval_integral_dict", basic_interval_integral_dict)

    # pot pred func
    # if pred_case_id == 7 and area_id == "id_2":
    #     debug = 1
    #     plot_func_multi(pred_func_dict)
    # plot_func_multi(pred_func_dict)

    if print_msg:
        # print recall_va_dict,precision_va_dict,dl_score_ia_dict
        print("recall_va_dict", recall_va_dict)
        print("precision_va_dict", precision_va_dict)
        print("dl_score_ia_dict", dl_score_ia_dict)
        print("d_ia_both_sides_dict+++++++++", d_ia_both_sides_dict)
        print("l_ia_set_dict+++++++++", l_ia_set_dict)
        print("dl_score+++++++++", dl_score)

    adjusted_dl_score_ia_dict = {}
    adjusted_dl_score_ia_dict_ratio_contribution = {}

    if pred_case_id == 2:
        debug = 1
    area_id_0 = "id_" + str(0)
    area_id_num = "id_" + str(gt_num)
    # two logics
    logic = 1
    # logic = 3
    if logic == 1:
        # logic 1
        #     one_exist_value = 1/2
        one_exist_value = 1
        for i in range(gt_num + 1):
            area_id = "id_" + str(i)
            area_id_front = "id_" + str(i - 1)
            area_id_back = "id_" + str(i + 1)

            ia_area = IA_dict[area_id]
            l_ia = l_ia_set_dict[area_id]
            d_ia = d_ia_set_dict[area_id]
            d_ia_sign = d_ia_sign_set_dict[area_id]
            ia_len = ia_area[1] - ia_area[0] if ia_area != [] else 0

            if pred_case_id == 22:
                debug = 1

            if i == 0:
                if IA_dict[area_id] == []:
                    # if IA_dict[area_id_back] == [] and area_id_back == area_id_num:
                    if area_id_back == area_id_num:
                        if IA_dict[area_id_back] == []:  # 自己为空，后面一个为最后一个，后面一个为空
                            ratio_before = 0
                            ratio_after = 1 / 2
                        else:  # 自己为空，后面一个为最后一个，后面一个不为空
                            ratio_before = 0
                            ratio_after = 0
                    else:
                        ratio_before = 0
                        ratio_after = 1 / 2

                    if VA_pred_group_dict[area_id] != []:
                        dl_score_ia_i_contribution = 1
                    else:
                        dl_score_ia_i_contribution = 0
                else:
                    # 数据里至少有一个异常，至少有两个ia，如果没有异常，那么IA_dict=[]
                    if IA_pred_group_dict[area_id] != []:
                        if area_id_back == area_id_num:
                            if IA_dict[area_id_back] == []:  # 自己不为空，后面一个为最后一个，后面一个为空
                                ratio_before = 0
                                ratio_after = 1
                            else:  # 自己不为空，后面一个为最后一个，后面一个不为空
                                ratio_before = 0
                                ratio_after = 1 / 2
                        else:
                            ratio_before = 0
                            ratio_after = 1 / 2
                        dl_score_ia_i_contribution = dl_score_ia_dict[area_id]
                    else:
                        if VA_pred_group_dict[area_id] != []:
                            if area_id_back == area_id_num:
                                if IA_dict[area_id_back] == []:  # 自己不为空，后面一个为最后一个，后面一个为空
                                    ratio_before = 0
                                    ratio_after = 1
                                else:  # 自己不为空，后面一个为最后一个，后面一个不为空
                                    ratio_before = 0
                                    ratio_after = 1 / 2
                            else:
                                ratio_before = 0
                                ratio_after = 1 / 2
                            dl_score_ia_i_contribution = 1
                        else:
                            if area_id_back == area_id_num:
                                if IA_dict[area_id_back] == []:  # 自己不为空，后面一个为最后一个，后面一个为空
                                    ratio_before = 0
                                    ratio_after = 1
                                else:  # 自己不为空，后面一个为最后一个，后面一个不为空
                                    ratio_before = 0
                                    ratio_after = 1 / 2
                            else:
                                ratio_before = 0
                                ratio_after = 1 / 2
                            dl_score_ia_i_contribution = 0
            elif i == gt_num:
                if IA_dict[area_id] == []:
                    if area_id_front == area_id_0:
                        if IA_dict[area_id_front] == []:  # 自己为空，前面一个为最后一个，前面一个为空
                            ratio_before = 1 / 2
                            ratio_after = 0
                        else:  # 自己为空，前面一个为第一个，前面一个不为空
                            ratio_before = 0
                            ratio_after = 0
                    else:
                        ratio_before = 1 / 2
                        ratio_after = 0

                    if VA_pred_group_dict[area_id_front] != []:
                        dl_score_ia_i_contribution = 1
                    else:
                        dl_score_ia_i_contribution = 0
                else:
                    if IA_pred_group_dict[area_id] != []:
                        if area_id_front == area_id_0:
                            if IA_dict[area_id_front] == []:
                                ratio_before = 1
                                ratio_after = 0
                            else:
                                ratio_before = 1 / 2
                                ratio_after = 0
                        else:
                            ratio_before = 1 / 2
                            ratio_after = 0
                        dl_score_ia_i_contribution = dl_score_ia_dict[area_id]
                    else:
                        if VA_pred_group_dict[area_id_front] != []:
                            if area_id_front == area_id_0:
                                if IA_dict[area_id_front] == []:
                                    ratio_before = 1
                                    ratio_after = 0
                                else:
                                    ratio_before = 1 / 2
                                    ratio_after = 0
                            else:
                                ratio_before = 1 / 2
                                ratio_after = 0
                            dl_score_ia_i_contribution = 1
                        else:
                            if IA_dict[area_id_front] == []:
                                if area_id_front == area_id_0:
                                    if IA_dict[area_id_front] == []:
                                        ratio_before = 1
                                        ratio_after = 0
                                    else:
                                        ratio_before = 1 / 2
                                        ratio_after = 0
                                else:
                                    ratio_before = 1 / 2
                                    ratio_after = 0
                            dl_score_ia_i_contribution = 0
            else:
                if IA_pred_group_dict[area_id] != []:
                    ratio_before = 1 / 2
                    ratio_after = 1 / 2
                    # if d_ia_sign > 0:
                    #     ratio_before = (ia_len / 2 - d_ia) / ia_len
                    #     ratio_after = (ia_len / 2 + d_ia) / ia_len
                    # elif d_ia_sign < 0:
                    #     ratio_before = (ia_len / 2 + d_ia) / ia_len
                    #     ratio_after = (ia_len / 2 - d_ia) / ia_len
                    # else:
                    #     ratio_before = 1/2
                    #     ratio_after = 1/2
                    dl_score_ia_i_contribution = dl_score_ia_dict[area_id]
                else:
                    ratio_before = 1 / 2
                    ratio_after = 1 / 2
                    if VA_pred_group_dict[area_id_front] != [] and VA_pred_group_dict[area_id] != []:
                        # ratio_before = 1/2
                        # ratio_after = 1/2
                        dl_score_ia_i_contribution = 1
                    elif VA_pred_group_dict[area_id_front] == [] and VA_pred_group_dict[area_id] == []:
                        # ratio_before = 1/2
                        # ratio_after = 1/2
                        dl_score_ia_i_contribution = 0
                    else:
                        # if VA_pred_group_dict[area_id_front] != []:
                        #     ratio_before = 1
                        #     ratio_after = 0
                        # else:
                        #     ratio_before = 0
                        #     ratio_after = 1
                        dl_score_ia_i_contribution = one_exist_value


            if i == 0:
                adjust_dl_score[area_id][0] = 0
                if IA_pred_group_dict[area_id] == []:
                    if VA_pred_group_dict[area_id] == []:
                        adjust_dl_score[area_id][1] = 0
            elif i == gt_num:
                adjust_dl_score[area_id][1] = 0
                if IA_pred_group_dict[area_id] == []:
                    if VA_pred_group_dict[area_id_front] == []:
                        adjust_dl_score[area_id][0] = 0
            else:
                if IA_pred_group_dict[area_id] == []:
                    if VA_pred_group_dict[area_id_front] == []:
                        adjust_dl_score[area_id][0] = 0
                    if VA_pred_group_dict[area_id] == []:
                        adjust_dl_score[area_id][1] = 0

            adjusted_dl_score_ia_dict[area_id] = dl_score_ia_i_contribution
            dl_score_ia_i_before_contribution = dl_score_ia_i_contribution * ratio_before
            dl_score_ia_i_after_contribution = dl_score_ia_i_contribution * ratio_after
            adjusted_dl_score_ia_dict_ratio_contribution[area_id] = [dl_score_ia_i_before_contribution,
                                                                     dl_score_ia_i_after_contribution]

    debug = 1
    # logic 2
    # one_exist_value = 1/2
    # for i in range(gt_num + 1):
    #     area_id = "id_" + str(i)
    #     area_id_front = "id_" + str(i - 1)
    #     area_id_back = "id_" + str(i + 1)
    #
    #     ia_area = IA_dict[area_id]
    #     l_ia = l_ia_set_dict[area_id]
    #     d_ia = d_ia_set_dict[area_id]
    #     d_ia_sign = d_ia_sign_set_dict[area_id]
    #     ia_len = ia_area[1] - ia_area[0] if ia_area != [] else 0
    #
    #     if i == 0:
    #         ratio_before = 0
    #         ratio_after = 1
    #         if IA_pred_group_dict[area_id] != []:
    #             dl_score_ia_i_contribution = dl_score_ia_dict[area_id]
    #         else:
    #             if VA_pred_group_dict[area_id] != []:
    #                 dl_score_ia_i_contribution = 1 / 2
    #             else:
    #                 dl_score_ia_i_contribution = 0
    #     elif i == gt_num:
    #         ratio_before = 1
    #         ratio_after = 0
    #         if IA_pred_group_dict[area_id] != []:
    #             dl_score_ia_i_contribution = dl_score_ia_dict[area_id]
    #         else:
    #             if VA_pred_group_dict[area_id_front] != []:
    #                 dl_score_ia_i_contribution = 1 / 2
    #             else:
    #                 dl_score_ia_i_contribution = 0
    #     else:
    #         if IA_pred_group_dict[area_id] != []:
    #             ratio_before = 1 / 2
    #             ratio_after = 1 / 2
    #             # if d_ia_sign > 0:
    #             #     ratio_before = (ia_len / 2 - d_ia) / ia_len
    #             #     ratio_after = (ia_len / 2 + d_ia) / ia_len
    #             # elif d_ia_sign < 0:
    #             #     ratio_before = (ia_len / 2 + d_ia) / ia_len
    #             #     ratio_after = (ia_len / 2 - d_ia) / ia_len
    #             # else:
    #             #     ratio_before = 1/2
    #             #     ratio_after = 1/2
    #             dl_score_ia_i_contribution = dl_score_ia_dict[area_id]
    #         else:
    #             # ratio_before = 1 / 2
    #             # ratio_after = 1 / 2
    #             if VA_pred_group_dict[area_id_front] != [] and VA_pred_group_dict[area_id] != []:
    #                 ratio_before = 1/2
    #                 ratio_after = 1/2
    #                 dl_score_ia_i_contribution = 1
    #             elif VA_pred_group_dict[area_id_front] == [] and VA_pred_group_dict[area_id] == []:
    #                 ratio_before = 1/2
    #                 ratio_after = 1/2
    #                 dl_score_ia_i_contribution = 0
    #             else:
    #                 if VA_pred_group_dict[area_id_front] != []:
    #                     ratio_before = 1
    #                     ratio_after = 0
    #                 else:
    #                     ratio_before = 0
    #                     ratio_after = 1
    #                 dl_score_ia_i_contribution = one_exist_value

    # logic 3
    if logic == 3:
        # one_exist_value = 1/2
        one_exist_value = 1
        for i in range(gt_num + 1):
            area_id = "id_" + str(i)
            area_id_front = "id_" + str(i - 1)
            area_id_back = "id_" + str(i + 1)

            ia_area = IA_dict[area_id]
            l_ia = l_ia_set_dict[area_id]
            d_ia = d_ia_set_dict[area_id]
            d_ia_sign = d_ia_sign_set_dict[area_id]
            ia_len = ia_area[1] - ia_area[0] if ia_area != [] else 0

            if i == 0:
                ratio_before = 0
                ratio_after = 1
                if IA_dict[area_id] == []:
                    dl_score_ia_i_contribution = 0
                else:
                    if IA_pred_group_dict[area_id] != []:
                        dl_score_ia_i_contribution = dl_score_ia_dict[area_id]
                    else:
                        if VA_pred_group_dict[area_id] != []:
                            dl_score_ia_i_contribution = 1
                        else:
                            dl_score_ia_i_contribution = 0
            elif i == gt_num:
                ratio_before = 1
                ratio_after = 0
                if IA_dict[area_id] == []:
                    dl_score_ia_i_contribution = 0
                else:
                    if IA_pred_group_dict[area_id] != []:
                        dl_score_ia_i_contribution = dl_score_ia_dict[area_id]
                    else:
                        if VA_pred_group_dict[area_id_front] != []:
                            dl_score_ia_i_contribution = 1
                        else:
                            dl_score_ia_i_contribution = 0
            else:
                if IA_pred_group_dict[area_id] != []:
                    # ratio_before = 1 / 2
                    # ratio_after = 1 / 2
                    if d_ia_sign > 0:
                        ratio_before = (ia_len / 2 - d_ia) / ia_len
                        ratio_after = (ia_len / 2 + d_ia) / ia_len
                    elif d_ia_sign < 0:
                        ratio_before = (ia_len / 2 + d_ia) / ia_len
                        ratio_after = (ia_len / 2 - d_ia) / ia_len
                    else:
                        ratio_before = 1 / 2
                        ratio_after = 1 / 2
                    dl_score_ia_i_contribution = dl_score_ia_dict[area_id]
                else:
                    # ratio_before = 1 / 2
                    # ratio_after = 1 / 2
                    if VA_pred_group_dict[area_id_front] != [] and VA_pred_group_dict[area_id] != []:
                        ratio_before = 1 / 2
                        ratio_after = 1 / 2
                        dl_score_ia_i_contribution = 2
                    elif VA_pred_group_dict[area_id_front] == [] and VA_pred_group_dict[area_id] == []:
                        ratio_before = 1 / 2
                        ratio_after = 1 / 2
                        dl_score_ia_i_contribution = 0
                    else:
                        if VA_pred_group_dict[area_id_front] != []:
                            ratio_before = 1
                            ratio_after = 0
                        else:
                            ratio_before = 0
                            ratio_after = 1
                        dl_score_ia_i_contribution = one_exist_value

            adjusted_dl_score_ia_dict[area_id] = dl_score_ia_i_contribution
            dl_score_ia_i_before_contribution = dl_score_ia_i_contribution * ratio_before
            dl_score_ia_i_after_contribution = dl_score_ia_i_contribution * ratio_after
            adjusted_dl_score_ia_dict_ratio_contribution[area_id] = [dl_score_ia_i_before_contribution,
                                                                     dl_score_ia_i_after_contribution]

    if print_msg:
        print("adjusted_dl_score_ia_dict", adjusted_dl_score_ia_dict)
        print("adjusted_dl_score_ia_dict_ratio_contribution", adjusted_dl_score_ia_dict_ratio_contribution)
        print("adjust_dl_score", adjust_dl_score)

    event_recall_i_list = []
    event_recall_i_dict = {}
    event_precision_i_dict = {}
    event_f1_score_first_i_dict = {}
    event_f1_score_first_i_dict_w0 = {}
    event_f1_score_first_i_dict_w1 = {}
    for i in range(gt_num):
        area_id = "id_" + str(i)
        if print_msg:
            print(area_id)
        area_id_back = "id_" + str(i + 1)
        recall_va_i = recall_va_dict[area_id]
        # if VA_pred_group_dict[area_id] != [] and IA_pred_group_dict[area_id] == [] and IA_pred_group_dict[area_id_back] == [] :
        #     dl_score_ia_i = 1
        #     dl_score_ia_i_back = 1
        # else:
        #     dl_score_ia_i = dl_score_ia_dict[area_id]
        #     dl_score_ia_i_back = dl_score_ia_dict[area_id_back]

        dl_score_ia_i_after_contribution = adjusted_dl_score_ia_dict_ratio_contribution[area_id][1]
        dl_score_ia_i_back_before_contribution = adjusted_dl_score_ia_dict_ratio_contribution[area_id_back][0]
        adjusted_dl_score_absolute_i_after = adjust_dl_score[area_id][1]
        adjusted_dl_score_absolute_i_back_before = adjust_dl_score[area_id_back][0]
        if print_msg:
            print(" adjusted_dl_score_absolute_i_after", adjusted_dl_score_absolute_i_after)
            # print(" dl_score_ia_i_after_contribution", dl_score_ia_i_after_contribution)
            print(" recall_va_i,", recall_va_i)
            # print(" dl_score_ia_i_back_before_contribution", dl_score_ia_i_back_before_contribution)
            print(" adjusted_dl_score_absolute_i_back_before", adjusted_dl_score_absolute_i_back_before)

        # event_recall_i = (1-2*parameter_eta)*recall_va_i + parameter_eta*(dl_score_ia_i_after_contribution+ dl_score_ia_i_back_before_contribution)
        if pred_case_id == 26:
            debug = 1

        precision_va_i = precision_va_dict[area_id]
        recall_va_i = recall_va_dict[area_id]
        meata_f1_first_i = compute_f1_score(precision_va_i, recall_va_i)
        # local auc
        # meata_f1_first_i = compute_f1_score(precision_va_i, recall_va_i)

        if i == 0 and IA_dict[area_id] == [] and IA_dict[area_id_back] != []:
            adjusted_dl_score_absolute_i_after = 0
            adjusted_dl_score_absolute_i_back_before = adjusted_dl_score_absolute_i_back_before * 2
        if i == (gt_num - 1) and IA_dict[area_id] != [] and IA_dict[area_id_back] == []:
            adjusted_dl_score_absolute_i_after = adjusted_dl_score_absolute_i_after * 2
            adjusted_dl_score_absolute_i_back_before = 0

        meata_f1_i = (1 - parameter_eta) * meata_f1_first_i + parameter_eta * (
                        adjusted_dl_score_absolute_i_after + adjusted_dl_score_absolute_i_back_before)/2
        meata_f1_i_w0 = (1 - 0) * meata_f1_first_i + 0 * (
                        adjusted_dl_score_absolute_i_after + adjusted_dl_score_absolute_i_back_before)/2
        meata_f1_i_w1 = (1 - 1) * meata_f1_first_i + 1 * (
                        adjusted_dl_score_absolute_i_after + adjusted_dl_score_absolute_i_back_before)/2
        if print_msg:
            # print(" dl_score_ia_i_after_contribution", dl_score_ia_i_after_contribution)
            print(" adjusted_dl_score_absolute_i_after", adjusted_dl_score_absolute_i_after)
            print(" precision_va_i,", precision_va_i)
            print(" recall_va_i,", recall_va_i)
            print(" meata_f1_first_i,", meata_f1_first_i)
            print(" meata_f1_i,", meata_f1_i)
            # print(" dl_score_ia_i_back_before_contribution", dl_score_ia_i_back_before_contribution)
            print(" adjusted_dl_score_absolute_i_back_before", adjusted_dl_score_absolute_i_back_before)


        event_f1_score_first_i_dict[area_id] = meata_f1_i
        event_f1_score_first_i_dict_w0[area_id] = meata_f1_i_w0
        event_f1_score_first_i_dict_w1[area_id] = meata_f1_i_w1

    if print_msg:
        print("event_f1_score_first_i_dict", event_f1_score_first_i_dict)

    if pred_case_id == 2:
        debug = 1

    meata_f1_first = np.array(list(event_f1_score_first_i_dict.values())).mean()
    meata_f1_first_w0 = np.array(list(event_f1_score_first_i_dict_w0.values())).mean()
    meata_f1_first_w1 = np.array(list(event_f1_score_first_i_dict_w1.values())).mean()

    # print(meata_f1_first)

    gt_label_point_array = convert_interval_to_point_array(label_interval_ranges, time_series_length=window_length)
    ont_count = np.count_nonzero(gt_label_point_array == 1)
    anomaly_rate = ont_count / window_length
    if anomaly_rate > 0.2:
        print("anomaly_rate exceeds threshold")
        non_random_coefficient = 1
    else:
        non_random_coefficient = cal_no_random_measure_coefficient(pred_label_point_array, window_length)

    coefficient_meata_f1_first = non_random_coefficient * meata_f1_first
    # coefficient_event_recall = non_random_coefficient * event_recall
    # coefficient_event_precision = non_random_coefficient * event_precision
    print("meata_f1_first", meata_f1_first)
    print("coefficient_meata_f1_first", coefficient_meata_f1_first)

    coefficient_meata_f1_first_w0 = non_random_coefficient * meata_f1_first_w0
    coefficient_meata_f1_first_w1 = non_random_coefficient * meata_f1_first_w1


    # return event_recall,event_precision,meata_f1,no_random_meata_f1
    return meata_f1_first, coefficient_meata_f1_first, VA_f_dict, VA_d_dict, \
        meata_f1_first_w0,coefficient_meata_f1_first_w0,meata_f1_first_w1,coefficient_meata_f1_first_w1


def plotFigures_systhetic_data(label_ranges,label_array_list,slidingWindow=100, forecasting_len=3,delay_len=3,color_box=0.4, plotRange=None, save_plot=False,
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
            ax.axvspan(r[0], r[1]+1, color='red', alpha=color_box)
            # ax.axvspan(r[0]-forecasting_len, r[0], color='green', alpha=color_box)
            # ax.axvspan(r[1], r[1]+delay_len, color='green', alpha=color_box)
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



def auc_pr_meata(y_true,y_score,parameter_dict=parameter_dict,pos_label=1,sample_weight=None,drop_intermediate=True,Big_Data=False,num_desired_thresholds=250,cal_mode="proportion"):
    window_length =len(y_true)

    # Generate thresholds
    fps_orig, tps_orig, thresholds = _binary_clf_curve(y_true, y_score, pos_label=pos_label,
                                                       sample_weight=sample_weight)

    if drop_intermediate and len(tps_orig) > 2:
        # Drop unnecessary thresholds for simplification
        optimal_idxs = np.where(np.concatenate([[True], np.logical_or(np.diff(tps_orig[:-1]), np.diff(tps_orig[1:])), [True]]))[0]
        fps_orig, tps_orig, thresholds = fps_orig[optimal_idxs], tps_orig[optimal_idxs], thresholds[optimal_idxs]

    if Big_Data:
        # Reduce the number of thresholds for large datasets
        percentiles = np.linspace(100, 0, num_desired_thresholds)
        thresholds = np.percentile(thresholds, percentiles)

    # array -> interval_ranges
    gt_label_interval_ranges = convert_vector_to_events(y_true)

    event_recall_list = []
    event_precision_list = []
    event_f1_list = []

    coefficient_event_recall_list = []
    coefficient_event_precision_list = []
    coefficient_event_f1_list = []

    for i, threshold in enumerate(thresholds):
        # score->binary array
        binary_predicted = (y_score >= threshold).astype(int)
        # # binary array->point ranges
        # pred_ranges = convert_vector_to_events_PATE(binary_predicted)
        #
        # pred_label_point_array = convert_events_to_array_PATE(pred_ranges, time_series_length=window_length)
        # # binary array->interval ranges
        pred_label_interval_ranges = convert_vector_to_events(binary_predicted)
        event_recall, event_precision, meata_f1,coefficient_meata_f1,VA_f_dict,VA_d_dict,coefficient_event_recall,coefficient_event_precision = meata(gt_label_interval_ranges, pred_label_interval_ranges,
                                                        window_length, parameter_dict,binary_predicted,cal_mode=cal_mode)

        print()
        print("coefficient_event_recall",coefficient_event_recall)
        print("coefficient_event_precision",coefficient_event_precision)

        # plot

        # label_ranges = [gt_label_interval_ranges]
        # label_ranges.append(pred_label_interval_ranges)
        # label_array_list= [y_true.tolist()]
        #
        # label_array_list.append(binary_predicted.tolist())
        #
        # plotFigures_systhetic_data(label_ranges,label_array_list,forecasting_len=0,delay_len= 0)


        event_recall_list.append(event_recall)
        event_precision_list.append(event_precision)

        coefficient_event_recall_list.append(coefficient_event_recall)
        coefficient_event_precision_list.append(coefficient_event_precision)

        event_f1_list.append(meata_f1)
        coefficient_event_f1_list.append(coefficient_meata_f1)


    # ori meata
    precision = np.array(event_precision_list)
    recall = np.array(event_recall_list)
    precision = np.hstack(([1], precision))
    recall = np.hstack(([0], recall))
    # auc_pr = clean_and_compute_auc_pr(recall, precision)

    precision_mean = precision.mean()
    recall_mean = recall.mean()
    mean_pr_f1 = compute_f1_score(precision_mean, recall_mean)
    

    # ori meata with no random
    coefficient_precision = np.array(coefficient_event_precision_list)
    coefficient_recall = np.array(coefficient_event_recall_list)
    coefficient_precision = np.hstack(([1], coefficient_precision))
    coefficient_recall = np.hstack(([0], coefficient_recall))
    # coefficient_auc_pr = clean_and_compute_auc_pr(coefficient_recall, coefficient_precision)
    
    coefficient_precision_mean = coefficient_precision.mean()
    coefficient_recall_mean = coefficient_recall.mean()
    coefficient_mean_pr_f1 = compute_f1_score(coefficient_precision_mean, coefficient_recall_mean)

    event_f1_array = np.array(event_f1_list)
    coefficient_event_f1_array = np.array(coefficient_event_f1_list)

    event_f1_mean = event_f1_array.mean()
    coefficient_event_f1_mean = coefficient_event_f1_array.mean()

    return mean_pr_f1,coefficient_mean_pr_f1,event_f1_mean,coefficient_event_f1_mean







def auc_pr_meata_f1_first(y_true, y_score, parameter_dict=parameter_dict, pos_label=1, sample_weight=None,
                 drop_intermediate=True, Big_Data=False, num_desired_thresholds=250, cal_mode="proportion"):
    window_length = len(y_true)

    # Generate thresholds
    fps_orig, tps_orig, thresholds = _binary_clf_curve(y_true, y_score, pos_label=pos_label,
                                                       sample_weight=sample_weight)

    if drop_intermediate and len(tps_orig) > 2:
        # Drop unnecessary thresholds for simplification
        optimal_idxs = \
        np.where(np.concatenate([[True], np.logical_or(np.diff(tps_orig[:-1]), np.diff(tps_orig[1:])), [True]]))[0]
        fps_orig, tps_orig, thresholds = fps_orig[optimal_idxs], tps_orig[optimal_idxs], thresholds[optimal_idxs]

    if Big_Data:
        # Reduce the number of thresholds for large datasets
        percentiles = np.linspace(100, 0, num_desired_thresholds)
        thresholds = np.percentile(thresholds, percentiles)

    # array -> interval_ranges
    gt_label_interval_ranges = convert_vector_to_events(y_true)

    event_recall_list = []
    event_precision_list = []
    event_f1_list = []

    coefficient_event_recall_list = []
    coefficient_event_precision_list = []
    coefficient_event_f1_list = []

    for i, threshold in enumerate(thresholds):
        # score->binary array
        binary_predicted = (y_score >= threshold).astype(int)
        # # binary array->point ranges
        # pred_ranges = convert_vector_to_events_PATE(binary_predicted)
        #
        # pred_label_point_array = convert_events_to_array_PATE(pred_ranges, time_series_length=window_length)
        # # binary array->interval ranges
        pred_label_interval_ranges = convert_vector_to_events(binary_predicted)
        meata_f1, coefficient_meata_f1, VA_f_dict, VA_d_dict= meata_tp_merge_first(
            gt_label_interval_ranges, pred_label_interval_ranges,
            window_length, parameter_dict, binary_predicted, cal_mode=cal_mode)

        # plot

        # label_ranges = [gt_label_interval_ranges]
        # label_ranges.append(pred_label_interval_ranges)
        # label_array_list= [y_true.tolist()]
        #
        # label_array_list.append(binary_predicted.tolist())
        #
        # plotFigures_systhetic_data(label_ranges,label_array_list,forecasting_len=0,delay_len= 0)

        event_recall_list.append(event_recall)
        event_precision_list.append(event_precision)

        coefficient_event_recall_list.append(coefficient_event_recall)
        coefficient_event_precision_list.append(coefficient_event_precision)

        event_f1_list.append(meata_f1)
        coefficient_event_f1_list.append(coefficient_meata_f1)

    # ori meata
    precision = np.array(event_precision_list)
    recall = np.array(event_recall_list)
    precision = np.hstack(([1], precision))
    recall = np.hstack(([0], recall))
    # auc_pr = clean_and_compute_auc_pr(recall, precision)

    precision_mean = precision.mean()
    recall_mean = recall.mean()
    mean_pr_f1 = compute_f1_score(precision_mean, recall_mean)

    # ori meata with no random
    coefficient_precision = np.array(coefficient_event_precision_list)
    coefficient_recall = np.array(coefficient_event_recall_list)
    coefficient_precision = np.hstack(([1], coefficient_precision))
    coefficient_recall = np.hstack(([0], coefficient_recall))
    # coefficient_auc_pr = clean_and_compute_auc_pr(coefficient_recall, coefficient_precision)

    coefficient_precision_mean = coefficient_precision.mean()
    coefficient_recall_mean = coefficient_recall.mean()
    coefficient_mean_pr_f1 = compute_f1_score(coefficient_precision_mean, coefficient_recall_mean)

    event_f1_array = np.array(event_f1_list)
    coefficient_event_f1_array = np.array(coefficient_event_f1_list)

    event_f1_mean = event_f1_array.mean()
    coefficient_event_f1_mean = coefficient_event_f1_array.mean()

    return mean_pr_f1, coefficient_mean_pr_f1, event_f1_mean, coefficient_event_f1_mean



def triangle_weights(n: int):
    """
    n : 把 [0,1] 分成 n 段（即得到 n+1 个点：0, 1/n, 2/n, ..., 1）
    返回与这些点一一对应的权重数组，满足：
        - 以 0.5 为中心对称
        - 左半边从 0 线性增加到 2/n
        - 右半边从 2/n 线性减小到 0
        - 所有权值之和为 1
    """
    if n <= 0:
        raise ValueError("n 必须为正整数")

    # n 段 -> n+1 个点
    x = np.linspace(0, 1, n + 1)

    # 三角形底宽 = 1
    base = 1.0
    # 三角形高，使得面积 = 1（底 × 高 / 2 = 1）
    height = 2.0 / base

    # 构造三角形权重
    weights = np.zeros_like(x)
    mid = 0.5
    for i, val in enumerate(x):
        if val <= mid:
            weights[i] = height * (val / mid)   # 左半边
        else:
            weights[i] = height * ((1 - val) / (1 - mid))  # 右半边

    # 归一化（理论上已经是 1，但用浮点计算后可能略有误差，再保险归一一次）
    weights /= weights.sum()
    return x, weights

def triangle_weights_add(n: int,values):
    """
    n : 把 [0,1] 分成 n 段（即得到 n+1 个点：0, 1/n, 2/n, ..., 1）
    返回与这些点一一对应的权重数组，满足：
        - 以 0.5 为中心对称
        - 左半边从 0 线性增加到 2/n
        - 右半边从 2/n 线性减小到 0
        - 所有权值之和为 1
    """
    if n <= 0:
        raise ValueError("n 必须为正整数")

    # n 段 -> n+1 个点
    x = np.linspace(0, 1, n + 1)

    # 三角形底宽 = 1
    base = 1.0
    # 三角形高，使得面积 = 1（底 × 高 / 2 = 1）
    height = 2.0 / base

    # 构造三角形权重
    weights = np.zeros_like(x)
    mid = 0.5
    for i, val in enumerate(x):
        if val <= mid:
            weights[i] = height * (val / mid)   # 左半边
        else:
            weights[i] = height * ((1 - val) / (1 - mid))  # 右半边

    # 归一化（理论上已经是 1，但用浮点计算后可能略有误差，再保险归一一次）
    weigh_sum = weights.sum()
    weights /= weights.sum()
    weigh_sum1 = weights.sum()
    # try:
    weight_sum = weights @ values
    # except:
    #     d=1
    return weight_sum,weights


def triangle_weights_add_v1(x: np.ndarray,values):
    """
    对任意给定的 1-D 数组 x（长度不限，也不必等距），
    返回一个与之等长的权重向量 w，满足：
      1. 以 0.5 为中心对称；
      2. 在 0→0.5 区间线性增加到峰值，0.5→1 区间线性减小到 0；
      3. 所有权重之和为 1。

    参数
    ----
    x : np.ndarray
        任意实数向量，元素需在 [0,1] 区间内（函数不做越界检查）。

    返回
    ----
    w : np.ndarray
        归一化后的权重向量，形状与 x 相同。
    """
    # 1. 原始权重：三角形，底宽 1，高 2（保证面积=1）
    height = 2.0
    weights = np.where(x <= 0.5,
                 height * (x / 0.5),  # 左半边
                 height * ((1 - x) / 0.5))  # 右半边
    # 2. 归一化
    weights /= weights.sum()
    
    weight_sum = weights @ values

    return weight_sum,weights

# 演示
# if __name__ == "__main__":
#     n = 8
#     x, w = triangle_weights(n)
#     print("x  :", x)
#     print("w  :", w)
#     print("sum:", w.sum())



def meata_auc_pr(y_true, y_score, output,parameter_dict=parameter_dict, max_ia_distant_length=None,pos_label=1, \
                 sample_weight=None, drop_intermediate=True, Big_Data=False, \
                 num_desired_thresholds=250, ype="row_auc_add", find_type="ts_section",cal_mode="proportion"):
    y_score = output
    # print_msg = True
    print_msg = False
    window_length = len(y_true)

    # no random
    # no_random_coefficient = cal_no_random_measure_coefficient_method2(output,window_length)
    thresh_num = 101
    # thresh_num = 251
    thresh_method = 1
    if thresh_method == 1:
        # thresholds = np.linspace(output.max(), output.min(), thresh_num)
        thresholds = np.linspace(1, 0, thresh_num)
    else:
        # Generate thresholds
        fps_orig, tps_orig, thresholds = _binary_clf_curve(y_true, y_score, pos_label=pos_label,
                                                           sample_weight=sample_weight)

        if drop_intermediate and len(tps_orig) > 2:
            # Drop unnecessary thresholds for simplification
            optimal_idxs = \
            np.where(np.concatenate([[True], np.logical_or(np.diff(tps_orig[:-1]), np.diff(tps_orig[1:])), [True]]))[0]
            fps_orig, tps_orig, thresholds = fps_orig[optimal_idxs], tps_orig[optimal_idxs], thresholds[optimal_idxs]

        if Big_Data:
            # Reduce the number of thresholds for large datasets
            percentiles = np.linspace(100, 0, num_desired_thresholds)
            thresholds = np.percentile(thresholds, percentiles)

    # array -> interval_ranges
    gt_label_interval_ranges = convert_vector_to_events(y_true)

    gt_num = len(gt_label_interval_ranges)
    if ("max_ia_distant_length" in parameter_dict
            and isinstance(parameter_dict["max_ia_distant_length"], numbers.Number)
            and parameter_dict["max_ia_distant_length"] > 0):
        max_ia_distant_length = parameter_dict["max_ia_distant_length"]
    # if max_ia_distant_length == None:
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

    # thresh_num = thresholds.__len__()

    for i, threshold in enumerate(thresholds):
        # score->binary array
        binary_predicted = (y_score >= threshold).astype(int)
        # # binary array->point ranges
        # pred_ranges = convert_vector_to_events_PATE(binary_predicted)
        #
        # pred_label_point_array = convert_events_to_array_PATE(pred_ranges, time_series_length=window_length)
        # # binary array->interval ranges
        pred_label_interval_ranges = convert_vector_to_events(binary_predicted)
        local_recall_list, local_precision_list,local_f1_list, \
            local_near_fq_list, local_distant_fq_list, gt_detected_rate = meata_v5(gt_label_interval_ranges,
                                                                                       pred_label_interval_ranges,
                                                                                       window_length,
                                                                                       parameter_dict,
                                                                                       binary_predicted,
                                                                                       pred_case_id=i,
                                                                                       max_ia_distant_length=max_ia_distant_length,
                                                                                       output=output,
                                                                                       thresh_id=i,
                                                                                       cal_mode=cal_mode)

        # print()
        # print("coefficient_event_recall",coefficient_event_recall)
        # print("coefficient_event_precision",coefficient_event_precision)

        # plot

        label_ranges = [gt_label_interval_ranges]
        label_ranges.append(pred_label_interval_ranges)
        label_array_list= [y_true.tolist()]

        label_array_list.append(binary_predicted.tolist())

        # plotFigures_systhetic_data(label_ranges,label_array_list,forecasting_len=0,delay_len= 0)

        local_recall_matrix.append(local_recall_list)
        local_precision_matrix.append(local_precision_list)
        local_f1_matrix.append(local_f1_list)
        gt_detection_list.append(gt_detected_rate)


        if threshold > 0:
            # local_recall_matrix.append(local_recall_list)
            # local_precision_matrix.append(local_precision_list)
            # local_f1_matrix.append(local_f1_list)
            # gt_detection_list.append(gt_detected_rate)

            local_near_fq_matrix.append(local_near_fq_list)
            local_distant_fq_matrix.append(local_distant_fq_list)

    # # calc auc
    # # f1_type = "row_add"
    # # f1_type = "add_row"
    local_recall_matrix_np = np.array(local_recall_matrix)
    local_precision_matrix_np = np.array(local_precision_matrix)
    local_f1_matrix_np = np.array(local_f1_matrix)
    local_near_fq_matrix_np = np.array(local_near_fq_matrix)
    local_distant_fq_matrix_np = np.array(local_distant_fq_matrix)
    gt_detection_list_np = np.array(gt_detection_list)

    # auc调f1算local p,r and local fq matrix
    parameter_w_gt = parameter_dict["parameter_w_gt"]
    parameter_w_near_ngt = parameter_dict["parameter_w_near_ngt"]
    parameter_distant_direction = parameter_dict["parameter_distant_direction"]

    # auc_type = "row_auc_add"
    row_mean_recall_list = []
    row_mean_precision_list = []

    row_mean_near_fp_list = []
    row_mean_distant_fp_list = []

    # gt_num = local_recall_matrix_np.shape[1]
    gt_detection_mean = np.mean(gt_detection_list_np)

    # auc_type = "auc_add_row"
    local_tq_auc_pr_list = []

    col_local_f1_mean_list = []
    col_local_f1_add_mean_list = []
    row_local_f1_list = []

    local_meata_list = []
    local_meata_list_w_gt = []
    local_meata_list_w_near_ngt = []
    local_meata_list_w_distant_ngt = []

    # 有两个二维数组a,b,分别代表三列的recall和precision，形状都是100*3的，我这里有两种计算方式，第一种，先把a和b的每一列对应位置取出来算f1-score，再取平均，再对三个auc取平均得到最后的平均的auc，第二种，先把a,b的每一行的三个值取平均得到平均的recall和precision，再用平均的recall和precision算auc，这两种计算的结果哪个更大，哪个更小，为什么
    # auc_type = "auc_row_add"

    f1_type = "mean_pr_f1"
    # f1_type = "mean_f1"

    # auc_type = "row_auc_add" # 这个配row
    auc_type = "row_add_auc" #
    # auc_type = "auc_row_add"# 1和2基本一样，1和auc_add_row的1一样， # 这个配global
    # auc_type = "auc_add_row"# 只有1,惩罚是对的

    # auc_type = "add_row_auc"
    # auc_type = "add_auc_row"

    detection_rate_method = "row" # FFT,DQE,DQE_TQ
    # detection_rate_method = "global"
    # detection_rate_method = "ori" # ori


    weigh_sum_method = "equal"
    # weigh_sum_method = "triangle" # SR,KmeansAD_U,DQE,DQE_TQ,DQE_FQ

    # if_plot = True
    if_plot = False

    # x_point_list, weights_np = triangle_weights(99)



    # if auc_type == "row_auc_add":
    if auc_type.split("_")[0] == "row":
        # 这个可以放到最后一个for循环
        for i, threshold in enumerate(thresholds):
            if threshold <=0:
                continue
            row_mean_near_fq = np.mean(local_near_fq_matrix_np[i])
            row_mean_distant_fq = np.mean(local_distant_fq_matrix_np[i])
            if detection_rate_method == "row":
                row_mean_recall = np.mean(local_recall_matrix_np[i])*gt_detection_list[i]
                row_mean_precision = np.mean(local_precision_matrix_np[i])*gt_detection_list[i]
            elif detection_rate_method == "global":
                row_mean_recall = np.mean(local_recall_matrix_np[i])
                row_mean_precision = np.mean(local_precision_matrix_np[i])
            else:
                row_mean_recall = np.mean(local_recall_matrix_np[i])
                row_mean_precision = np.mean(local_precision_matrix_np[i])
            row_mean_recall_list.append(row_mean_recall)
            row_mean_precision_list.append(row_mean_precision)

            row_mean_near_fp_list.append(row_mean_near_fq)
            row_mean_distant_fp_list.append(row_mean_distant_fq)

            if f1_type == "mean_pr_f1":
                row_local_f1 = compute_f1_score(row_mean_precision,row_mean_recall)
            else:
                row_local_f1_np = local_f1_matrix_np[i]
                row_local_f1 = np.mean(row_local_f1_np)
            row_local_f1_list.append(row_local_f1)

        row_mean_recall_np = np.array(row_mean_recall_list)
        row_mean_precision_np = np.array(row_mean_precision_list)

        row_local_f1_list_np = np.array(row_local_f1_list)



        if auc_type == "row_auc_add":  # row_col_add
            # idx = np.where(row_mean_recall_np >= 1)[0]  # 所有等于 1 的下标
            # first_pos = idx[0] if idx.size else None  # 第一个下标，没有则为 None
            #
            # print()
            # print("i", i)
            # print("first_pos", first_pos)

            # add first point
            # row_mean_recall_np = np.insert(row_mean_recall_np, 0, 0)
            # row_mean_precision_np = np.insert(row_mean_precision_np, 0, 1)
            #
            # row_mean_recall_np = np.insert(row_mean_recall_np, len(row_mean_recall_np), 1)
            # row_mean_precision_np = np.insert(row_mean_precision_np, len(row_mean_precision_np), 0)

            pr_auc = auc(row_mean_recall_np, row_mean_precision_np)
            # tq_auc_pr = clean_and_compute_auc_pr(row_mean_recall_np, row_mean_precision_np)

            # pr_auc=tq_auc_pr
            if weigh_sum_method == "equal":
                row_f1 = np.mean(row_local_f1_list_np)
            else:
                row_f1 = triangle_weights_add(thresh_num-2,row_local_f1_list_np)



            if if_plot:
                # 2) 画图
                plt.figure(figsize=(5, 4))
                plt.plot(row_mean_recall_np, row_mean_precision_np, lw=2, color='darkorange',
                         label=f'PR curve (AUC = {pr_auc:.3f})')

                # 把面积涂出来
                plt.fill_between(row_mean_recall_np, row_mean_precision_np, alpha=0.2, color='darkorange')

                plt.xlabel('Recall')
                plt.ylabel('Precision')
                plt.ylim([0.0, 1.05])
                plt.xlim([0.0, 1.0])
                plt.title('Precision-Recall Curve')
                plt.legend(loc="lower left")
                plt.grid(True, linestyle='--', alpha=0.4)
                plt.tight_layout()
                plt.show()


            mean_local_near_fq = np.mean(local_near_fq_matrix_np)
            mean_distant_fq = np.mean(local_distant_fq_matrix_np)
            gt_detection_mean = np.mean(gt_detection_list_np)
            if detection_rate_method == "row":
                tq_auc_pr_final = pr_auc
                row_f1_final = row_f1
            elif detection_rate_method == "global":
                tq_auc_pr_final = pr_auc*gt_detection_mean
                row_f1_final = row_f1*gt_detection_mean
            else:
                tq_auc_pr_final = pr_auc
                row_f1_final = row_f1


            # meata = cal_dqe(mean_distant_fq, mean_local_near_fq, parameter_w_gt, parameter_w_near_ngt, tq_auc_pr_final)
            # meata_w_gt = cal_dqe(mean_distant_fq, mean_local_near_fq, 1, 0, tq_auc_pr_final)
            # meata_w_near_ngt = cal_dqe(mean_distant_fq, mean_local_near_fq, 0, 1, tq_auc_pr_final)
            # meata_w_distant_ngt = cal_dqe(mean_distant_fq, mean_local_near_fq, 0, 0, tq_auc_pr_final)

            meata = cal_dqe(mean_distant_fq, mean_local_near_fq, parameter_w_gt, parameter_w_near_ngt, row_f1_final)
            meata_w_gt = cal_dqe(mean_distant_fq, mean_local_near_fq, 1, 0, row_f1_final)
            meata_w_near_ngt = cal_dqe(mean_distant_fq, mean_local_near_fq, 0, 1, row_f1_final)
            meata_w_distant_ngt = cal_dqe(mean_distant_fq, mean_local_near_fq, 0, 0, row_f1_final)
        else: # auc_type == "row_add_auc"
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

            triangle_weights = np.full((thresh_num-1), 1.0 / (thresh_num-1))
            equal_weights = np.full((thresh_num-1), 1.0 / (thresh_num-1))

            if weigh_sum_method == "equal":
                # col_local_recall_mean = np.mean(col_local_recall)
                # col_local_precision_mean = np.mean(col_local_precision)

                meata = np.mean(local_meata_value_list)
                meata_w_gt = np.mean(local_meata_value_list_w_gt)
                meata_w_near_ngt = np.mean(local_meata_value_list_w_near_ngt)
                meata_w_distant_ngt = np.mean(local_meata_value_list_w_distant_ngt)
            else:
                # col_local_recall_mean = triangle_weights_add(thresh_num-1,col_local_recall)
                # col_local_precision_mean = triangle_weights_add(thresh_num-1,col_local_precision)
                # debug_flag
                # try:
                # meata,triangle_weights = triangle_weights_add(thresh_num-2,np.array(local_meata_value_list))
                # meata_w_gt,_ = triangle_weights_add(thresh_num-2,np.array(local_meata_value_list_w_gt))
                # meata_w_near_ngt,_ = triangle_weights_add(thresh_num-2,np.array(local_meata_value_list_w_near_ngt))
                # meata_w_distant_ngt,_ = triangle_weights_add(thresh_num-2,np.array(local_meata_value_list_w_distant_ngt))

                meata, triangle_weights = triangle_weights_add_v1(thresholds[:-1], np.array(local_meata_value_list))
                meata_w_gt, _ = triangle_weights_add_v1(thresholds[:-1], np.array(local_meata_value_list_w_gt))
                meata_w_near_ngt, _ = triangle_weights_add_v1(thresholds[:-1], np.array(local_meata_value_list_w_near_ngt))
                meata_w_distant_ngt, _ = triangle_weights_add_v1(thresholds[:-1],
                                                              np.array(local_meata_value_list_w_distant_ngt))

                # except:
                #     d=1


            # meata = np.mean(local_meata_value_list)
            # meata_w_gt = np.mean(local_meata_value_list_w_gt)
            # meata_w_near_ngt = np.mean(local_meata_value_list_w_near_ngt)
            # meata_w_distant_ngt = np.mean(local_meata_value_list_w_distant_ngt)


            # plot_flag = True
            plot_flag = False
            thresholds_plot = thresholds[:-1]
            local_meata_value_list_np = np.array(local_meata_value_list)

            if plot_flag:
                # 2. 画图
                # figsize = (12,8)
                figsize = (9,6)
                fontsize = 25
                fontsize1 = 30
                plt.figure(figsize=figsize)
                plt.scatter(thresholds_plot, local_meata_value_list_np, c='steelblue', s=20)

                # 3. 坐标轴说明
                plt.xlabel('Threshold',fontsize=fontsize1)
                plt.ylabel('Score',fontsize=fontsize1)
                plt.title("F1-score across thresholds",fontsize=fontsize1)
                plt.xlim(0, 1)
                plt.ylim(0, 1)
                plt.tick_params(axis='both', labelsize=fontsize)  # 同时改 x、y


                plt.grid(alpha=0.3)
                plt.tight_layout()

                file_name = "weight_sum_F1-score"
                # plt.savefig(save_dir+file_name.split(".")[0]+".png")  # 保存为PNG文件

                # save_path_svg = "paper/src/figures/" + file_name + ".svg"
                save_path_pdf = "paper/src/figures/" + file_name + ".pdf"
                save_path_png = "paper/src/figures/" + file_name + ".png"
                # plt.savefig(save_path_svg, format='svg')
                # 保存为 PDF 文件
                plt.savefig(save_path_pdf, format="pdf", bbox_inches="tight")
                plt.savefig(save_path_png)


                # weight equal
                plt.figure(figsize=figsize)
                plt.scatter(thresholds_plot, equal_weights, c='steelblue', s=20)

                # 3. 坐标轴说明
                plt.xlabel('Threshold',fontsize=fontsize1)
                plt.ylabel('Weight',fontsize=fontsize1)
                plt.title("Equal weights across thresholds",fontsize=fontsize1)
                plt.xlim(0, 1)
                plt.ylim(0, max(triangle_weights))
                plt.tick_params(axis='both', labelsize=fontsize)  # 同时改 x、y

                loc, lab = plt.yticks()  # 拿到当前刻度位置和标签
                print(loc)  # 数组，如 [0.  0.2 0.4 0.6 0.8 1. ]

                # 原刻度位置不变，只改标签文字
                plt.yticks(loc,
                           ['0', '', '1/M', '', '2/M'])

                plt.grid(alpha=0.3)
                plt.tight_layout()

                file_name = "weight_sum_Equal_weights"
                # plt.savefig(save_dir+file_name.split(".")[0]+".png")  # 保存为PNG文件

                # save_path_svg = "paper/src/figures/" + file_name + ".svg"
                save_path_pdf = "paper/src/figures/" + file_name + ".pdf"
                save_path_png = "paper/src/figures/" + file_name + ".png"
                # plt.savefig(save_path_svg, format='svg')
                # 保存为 PDF 文件
                plt.savefig(save_path_pdf, format="pdf", bbox_inches="tight")
                plt.savefig(save_path_png)

                # weight triangle
                plt.figure(figsize=figsize)
                plt.scatter(thresholds_plot, triangle_weights, c='steelblue', s=20)

                # 3. 坐标轴说明
                plt.xlabel('Threshold',fontsize=fontsize1)
                plt.ylabel('Weight',fontsize=fontsize1)
                plt.title("Triangle weights across thresholds",fontsize=fontsize1)
                plt.xlim(0, 1)
                plt.ylim(0, max(triangle_weights))
                plt.tick_params(axis='both', labelsize=fontsize)  # 同时改 x、y


                loc, lab = plt.yticks()  # 拿到当前刻度位置和标签
                print(loc)  # 数组，如 [0.  0.2 0.4 0.6 0.8 1. ]

                # 原刻度位置不变，只改标签文字
                plt.yticks(loc,
                           ['0', '', '1/M', '', '2/M'])

                plt.grid(alpha=0.3)
                plt.tight_layout()



                file_name = "weight_sum_Triangle_weights"
                # plt.savefig(save_dir+file_name.split(".")[0]+".png")  # 保存为PNG文件

                # save_path_svg = "paper/src/figures/" + file_name + ".svg"
                save_path_pdf = "paper/src/figures/" + file_name + ".pdf"
                save_path_png = "paper/src/figures/" + file_name + ".png"
                # plt.savefig(save_path_svg, format='svg')
                # 保存为 PDF 文件
                plt.savefig(save_path_pdf, format="pdf", bbox_inches="tight")
                plt.savefig(save_path_png)


                plt.show()

    elif auc_type.split("_")[0] == "auc":
        for i in range(gt_num):
            # col_local_recall = local_recall_matrix_np[:, i]
            # col_local_precision = local_precision_matrix_np[:, i]
            # col_local_f1 = local_f1_matrix_np[:, i]
            col_local_recall = local_recall_matrix_np[:-1, i]
            col_local_precision = local_precision_matrix_np[:-1, i]
            col_local_f1 = local_f1_matrix_np[:-1, i]
            # add first point
            # col_local_recall = np.insert(col_local_recall, 0, 0)
            # col_local_precision = np.insert(col_local_precision, 0, 1)
            #
            # col_local_recall = np.insert(col_local_recall, len(col_local_recall), 1)
            # col_local_precision = np.insert(col_local_precision, len(col_local_precision), 0)

            # col_local_recall = np.insert(col_local_recall, -1, 1)
            # col_local_precision = np.insert(col_local_precision, -1, 0)
            #
            local_tq_auc_pr = clean_and_compute_auc_pr(col_local_recall, col_local_precision)
            # precision, recall, thresholds = precision_recall_curve(y_true, y_score)
            local_tq_auc_pr1 = auc(col_local_recall, col_local_precision)
            
            if f1_type == "mean_pr_f1":
                if weigh_sum_method == "equal":
                    col_local_recall_mean = np.mean(col_local_recall)
                    col_local_precision_mean = np.mean(col_local_precision)
                else:
                    col_local_recall_mean = triangle_weights_add(thresh_num-2,col_local_recall)
                    col_local_precision_mean = triangle_weights_add(thresh_num-2,col_local_precision)

                col_local_f1_mean = compute_f1_score(col_local_precision_mean,col_local_recall_mean)
            else: # f1_type == "mean_f1"
                if weigh_sum_method == "equal":
                    col_local_f1_mean = np.mean(col_local_f1)
                else:
                    col_local_f1_mean = triangle_weights_add(thresh_num-1,col_local_f1)




            idx = np.where(col_local_recall >= 1)[0]  # 所有等于 1 的下标
            first_pos = idx[0] if idx.size else None  # 第一个下标，没有则为 None

            print()
            print("local_tq_auc_pr",local_tq_auc_pr)
            print("local_tq_auc_pr1",local_tq_auc_pr1)
            print("col_local_f1_mean",col_local_f1_mean)
            print("first_pos",first_pos)

            # KmeansAD_U
            # 6,2,19,thresh_method=1
            # 7,3,17,thresh_method=1

            if if_plot:
                # 2) 画图
                plt.figure(figsize=(5, 4))
                plt.plot(col_local_recall, col_local_precision, lw=2, color='darkorange',
                         label=f'PR curve (AUC = {local_tq_auc_pr:.3f})')

                # 把面积涂出来
                plt.fill_between(col_local_recall, col_local_precision, alpha=0.2, color='darkorange')

                plt.xlabel('Recall')
                plt.ylabel('Precision')
                plt.ylim([0.0, 1.05])
                plt.xlim([0.0, 1.0])
                plt.title('Precision-Recall Curve')
                plt.legend(loc="lower left")
                plt.grid(True, linestyle='--', alpha=0.4)
                plt.tight_layout()
                plt.show()

            if auc_type == "auc_row_add":
                if detection_rate_method == "row":
                    local_tq_auc_pr_new = gt_detection_mean * local_tq_auc_pr
                    col_local_f1_mean_new = gt_detection_mean * col_local_f1_mean
                elif detection_rate_method == "global":
                    local_tq_auc_pr_new = local_tq_auc_pr
                    col_local_f1_mean_new = col_local_f1_mean
                else:
                    local_tq_auc_pr_new = local_tq_auc_pr
                    col_local_f1_mean_new = col_local_f1_mean
                local_tq_auc_pr_list.append(local_tq_auc_pr_new)
                col_local_f1_mean_list.append(col_local_f1_mean_new)
            else:
                # if auc_type == "auc_add_row":
                if detection_rate_method == "row":
                    local_tq_auc_pr_new = gt_detection_mean * local_tq_auc_pr
                    col_local_f1_mean_new = gt_detection_mean * col_local_f1_mean
                elif detection_rate_method == "global": # 没有这种情况
                    local_tq_auc_pr_new = local_tq_auc_pr
                    col_local_f1_mean_new = col_local_f1_mean
                else:
                    local_tq_auc_pr_new = local_tq_auc_pr
                    col_local_f1_mean_new = col_local_f1_mean


                # col_local_near_fq_list = local_near_fq_matrix_np[:, i]
                # col_local_near_fq_list1 = local_near_fq_matrix_np[:, i][:-1]
                col_mean_local_near_fq = np.mean(local_near_fq_matrix_np[:, i])
                col_mean_distant_fq = np.mean(local_distant_fq_matrix_np[:, i])

                # col_mean_local_near_fq = np.mean(local_near_fq_matrix_np[:, i][:-1])
                # col_mean_distant_fq = np.mean(local_distant_fq_matrix_np[:, i][:-1])

                # local_meata = cal_meata_auc_add_row(col_mean_distant_fq, col_mean_local_near_fq, local_tq_auc_pr_new,
                #                                     parameter_w_gt, parameter_w_near_ngt)
                # local_meata_w_gt = cal_meata_auc_add_row(col_mean_distant_fq, col_mean_local_near_fq, local_tq_auc_pr_new,
                #                                          1, 0)
                # local_meata_w_near_ngt = cal_meata_auc_add_row(col_mean_distant_fq, col_mean_local_near_fq, local_tq_auc_pr_new,
                #                                                0, 1)
                # local_meata_w_distant_ngt = cal_meata_auc_add_row(col_mean_distant_fq, col_mean_local_near_fq, local_tq_auc_pr_new,
                #                                                   0, 0)


                local_meata = cal_meata_auc_add_row(col_mean_distant_fq, col_mean_local_near_fq, col_local_f1_mean_new,
                                                    parameter_w_gt, parameter_w_near_ngt)
                local_meata_w_gt = cal_meata_auc_add_row(col_mean_distant_fq, col_mean_local_near_fq,
                                                         col_local_f1_mean_new,
                                                         1, 0)
                local_meata_w_near_ngt = cal_meata_auc_add_row(col_mean_distant_fq, col_mean_local_near_fq,
                                                               col_local_f1_mean_new,
                                                               0, 1)
                local_meata_w_distant_ngt = cal_meata_auc_add_row(col_mean_distant_fq, col_mean_local_near_fq,
                                                                  col_local_f1_mean_new,
                                                                  0, 0)
                
                
                local_meata_list.append(local_meata)

                local_meata_list_w_gt.append(local_meata_w_gt)
                local_meata_list_w_near_ngt.append(local_meata_w_near_ngt)
                local_meata_list_w_distant_ngt.append(local_meata_w_distant_ngt)
        # auc_type = "auc_row_add"
        if auc_type == "auc_row_add":
            mean_tq_auc = np.mean(np.array(local_tq_auc_pr_list))
            mean_f1 = np.mean(np.array(col_local_f1_mean_list))
            gt_detection_mean = np.mean(gt_detection_list_np)

            mean_local_near_fq = np.mean(local_near_fq_matrix_np)
            mean_distant_fq = np.mean(local_distant_fq_matrix_np)
            if detection_rate_method == "row":
                tq_auc_pr_final = mean_tq_auc
                mean_f1_final = mean_f1
            elif detection_rate_method == "global":
                tq_auc_pr_final = mean_tq_auc * gt_detection_mean
                mean_f1_final = mean_f1 * gt_detection_mean
            else:
                tq_auc_pr_final = mean_tq_auc
                mean_f1_final = mean_f1

            # 0.6529527576853527
            # 0.6620463381555154
            # meata = cal_meata_auc_row_add(mean_distant_fq, mean_local_near_fq, parameter_w_gt, parameter_w_near_ngt,
            #                               tq_auc_pr_final)
            # meata_w_gt = cal_meata_auc_row_add(mean_distant_fq, mean_local_near_fq, 1, 0,tq_auc_pr_final)
            # meata_w_near_ngt = cal_meata_auc_row_add(mean_distant_fq, mean_local_near_fq, 0, 1,tq_auc_pr_final)
            # meata_w_distant_ngt = cal_meata_auc_row_add(mean_distant_fq, mean_local_near_fq, 0, 0,tq_auc_pr_final)

            meata = cal_meata_auc_row_add(mean_distant_fq, mean_local_near_fq, parameter_w_gt, parameter_w_near_ngt,
                                          mean_f1_final)
            meata_w_gt = cal_meata_auc_row_add(mean_distant_fq, mean_local_near_fq, 1, 0,mean_f1_final)
            meata_w_near_ngt = cal_meata_auc_row_add(mean_distant_fq, mean_local_near_fq, 0, 1,mean_f1_final)
            meata_w_distant_ngt = cal_meata_auc_row_add(mean_distant_fq, mean_local_near_fq, 0, 0,mean_f1_final)
            a = 1
        # auc_type = "auc_add_row"
        else:
            meata = np.mean(np.array(local_meata_list))
            meata_w_gt = np.mean(np.array(local_meata_list_w_gt))
            meata_w_near_ngt = np.mean(np.array(local_meata_list_w_near_ngt))
            meata_w_distant_ngt = np.mean(np.array(local_meata_list_w_distant_ngt))
    else: # auc_type.split("_")[0] == "add"
        local_f1_add_fp_matrix = copy.deepcopy(local_f1_matrix_np)
        local_f1_add_fp_matrix_w_gt = copy.deepcopy(local_f1_matrix_np)
        local_f1_add_fp_matrix_w_near_ngt = copy.deepcopy(local_f1_matrix_np)
        local_f1_add_fp_matrix_w_distant_ngt = copy.deepcopy(local_f1_matrix_np)
        for id_thresh, threshold in enumerate(thresholds):
            if threshold <= 0:
                continue
            # row_local_f1_np = local_f1_matrix_np[id_thresh]
            # row_local_near_fq_np = local_near_fq_matrix_np[id_thresh]
            # row_local_distant_fq_np = local_distant_fq_matrix_np[id_thresh]
            # row_local_distant_fq_np = local_distant_fq_matrix_np[id_thresh]

            for id_gt in range(gt_num):
                if detection_rate_method == "row":
                    gt_detection_rate = gt_detection_list[id_gt]
                elif detection_rate_method == "global":
                    gt_detection_rate = gt_detection_mean

                else:
                    gt_detection_rate = 1
                cal_local_meata_value(gt_detection_rate, id_gt, id_thresh, local_distant_fq_matrix_np,
                                      local_f1_add_fp_matrix, local_f1_matrix_np, local_near_fq_matrix_np,
                                      parameter_w_gt, parameter_w_near_ngt)
                cal_local_meata_value(gt_detection_rate, id_gt, id_thresh, local_distant_fq_matrix_np,
                                      local_f1_add_fp_matrix_w_gt, local_f1_matrix_np, local_near_fq_matrix_np,
                                      1, 0)
                cal_local_meata_value(gt_detection_rate, id_gt, id_thresh, local_distant_fq_matrix_np,
                                      local_f1_add_fp_matrix_w_near_ngt, local_f1_matrix_np, local_near_fq_matrix_np,
                                      0, 1)
                cal_local_meata_value(gt_detection_rate, id_gt, id_thresh, local_distant_fq_matrix_np,
                                      local_f1_add_fp_matrix_w_distant_ngt, local_f1_matrix_np, local_near_fq_matrix_np,
                                      0, 0)

        if auc_type == "add_row_auc":
            row_mean_meata_list = []
            row_mean_meata_list_w_gt = []
            row_mean_meata_list_w_near_ngt = []
            row_mean_meata_list_w_distant_ngt = []
            for id_thresh, threshold in enumerate(thresholds):
                if threshold <=0:
                    continue
                # row_local_f1_np = local_f1_matrix_np[id_thresh]
                # row_local_near_fq_np = local_near_fq_matrix_np[id_thresh]
                # row_local_distant_fq_np = local_distant_fq_matrix_np[id_thresh]
                # row_local_distant_fq_np = local_distant_fq_matrix_np[id_thresh]
                row_mean_meata = np.mean(local_f1_add_fp_matrix[id_thresh])
                row_mean_meata_w_gt = np.mean(local_f1_add_fp_matrix_w_gt[id_thresh])
                row_mean_meata_w_near_ngt = np.mean(local_f1_add_fp_matrix_w_near_ngt[id_thresh])
                row_mean_meata_w_distant_ngt = np.mean(local_f1_add_fp_matrix_w_distant_ngt[id_thresh])

                row_mean_meata_list.append(row_mean_meata)
                row_mean_meata_list_w_gt.append(row_mean_meata_w_gt)
                row_mean_meata_list_w_near_ngt.append(row_mean_meata_w_near_ngt)
                row_mean_meata_list_w_distant_ngt.append(row_mean_meata_w_distant_ngt)
            
            if weigh_sum_method == "equal":
                meata = np.mean(row_mean_meata_list)
                meata_w_gt = np.mean(row_mean_meata_list_w_gt)
                meata_w_near_ngt = np.mean(row_mean_meata_list_w_near_ngt)
                meata_w_distant_ngt = np.mean(row_mean_meata_list_w_distant_ngt)
            else:
                meata = triangle_weights_add(thresh_num-2,np.array(row_mean_meata_list))
                meata_w_gt = triangle_weights_add(thresh_num-2,np.array(row_mean_meata_list_w_gt))
                meata_w_near_ngt = triangle_weights_add(thresh_num-2,np.array(row_mean_meata_list_w_near_ngt))
                meata_w_distant_ngt = triangle_weights_add(thresh_num-2,np.array(row_mean_meata_list_w_distant_ngt))

        else: # auc_type == "add_auc_row"
            col_mean_meata_list = []
            col_mean_meata_list_w_gt = []
            col_mean_meata_list_w_near_ngt = []
            col_mean_meata_list_w_distant_ngt = []
            for id_gt in range(gt_num):
                # row_local_f1_np = local_f1_matrix_np[id_thresh]
                # row_local_near_fq_np = local_near_fq_matrix_np[id_thresh]
                # row_local_distant_fq_np = local_distant_fq_matrix_np[id_thresh]
                # row_local_distant_fq_np = local_distant_fq_matrix_np[id_thresh]
                if weigh_sum_method == "equal":
                    col_mean_meata = np.mean(local_f1_add_fp_matrix[:-1,id_gt])
                    col_mean_meata_w_gt = np.mean(local_f1_add_fp_matrix_w_gt[:-1,id_gt])
                    col_mean_meata_w_near_ngt = np.mean(local_f1_add_fp_matrix_w_near_ngt[:-1,id_gt])
                    col_mean_meata_w_distant_ngt = np.mean(local_f1_add_fp_matrix_w_distant_ngt[:-1,id_gt])
                else:
                    col_mean_meata = triangle_weights_add(thresh_num - 2, local_f1_add_fp_matrix[:-1,id_gt])
                    col_mean_meata_w_gt = triangle_weights_add(thresh_num - 2, local_f1_add_fp_matrix_w_gt[:-1,id_gt])
                    col_mean_meata_w_near_ngt = triangle_weights_add(thresh_num - 2, local_f1_add_fp_matrix_w_near_ngt[:-1,id_gt])
                    col_mean_meata_w_distant_ngt = triangle_weights_add(thresh_num - 2,
                                                               local_f1_add_fp_matrix_w_distant_ngt[:-1,id_gt])
                col_mean_meata_list.append(col_mean_meata)
                col_mean_meata_list_w_gt.append(col_mean_meata_w_gt)
                col_mean_meata_list_w_near_ngt.append(col_mean_meata_w_near_ngt)
                col_mean_meata_list_w_distant_ngt.append(col_mean_meata_w_distant_ngt)

            meata = np.mean(col_mean_meata_list)
            meata_w_gt = np.mean(col_mean_meata_list_w_gt)
            meata_w_near_ngt = np.mean(col_mean_meata_list_w_near_ngt)
            meata_w_distant_ngt = np.mean(col_mean_meata_list_w_distant_ngt)

    # final_meata = no_random_coefficient*meata
    final_meata = meata

    if print_msg:
        print()
        print(" meata", meata)
        print(" meata_w_gt", meata_w_gt)
        print(" meata_w_near_ngt", meata_w_near_ngt)
        print(" meata_w_distant_ngt", meata_w_distant_ngt)
        print(" final_meata", final_meata)
    meata_w_ngt = (meata_w_near_ngt+meata_w_distant_ngt) / 2
    return final_meata, meata,meata_w_gt,meata_w_near_ngt,meata_w_distant_ngt,meata_w_ngt


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


def meata_f1(y_true,y_score,output=None,parameter_dict=parameter_dict,thresh_id=None,binary_preds=False,pred_case_id=None,max_ia_distant_length=None,f1_type = "row_add",cal_mode="proportion",find_type="ts_section"):
    print_msg = True
    # print_msg = False
    window_length =len(y_true)



    if binary_preds:
        output=y_score


    # no random
    # no_random_coefficient = cal_no_random_measure_coefficient_method2(output,window_length)

    # array -> interval_ranges
    gt_label_interval_ranges = convert_vector_to_events(y_true)
    gt_num = len(gt_label_interval_ranges)

    # if max_ia_distant_length == None:
    #     max_ia_distant_length = find_max_ia_distant_length_in_ts(gt_label_interval_ranges,
    #                                                              window_length,
    #                                                              parameter_dict=parameter_dict,
    #                                                              find_type=find_type)
    if ("max_ia_distant_length" in parameter_dict
            and isinstance(parameter_dict["max_ia_distant_length"], numbers.Number)
            and parameter_dict["max_ia_distant_length"] > 0):
        max_ia_distant_length = parameter_dict["max_ia_distant_length"]
    # if max_ia_distant_length == None:
    else:
        parameter_near_single_side_range = parameter_dict["parameter_near_single_side_range"]
        parameter_distant_method = parameter_dict["parameter_distant_method"]
        max_ia_distant_length =  find_max_ia_distant_length_in_single_ts(gt_label_interval_ranges,window_length,gt_num,parameter_near_single_side_range,parameter_distant_method,max_ia_distant_length=0,parameter_dict=parameter_dict)


    # score->binary array
    # binary_predicted = (y_score >= threshold).astype(int)
    # y_score是binary prediction
    binary_predicted = y_score
    # # binary array->point ranges
    # pred_ranges = convert_vector_to_events_PATE(binary_predicted)
    #
    # pred_label_point_array = convert_events_to_array_PATE(pred_ranges, time_series_length=window_length)
    # # binary array->interval ranges
    pred_label_interval_ranges = convert_vector_to_events(binary_predicted)

    # find max distant ia length
    # find first and realize in final calculation



    # event_recall, event_precision, meata_f1,meata_multiply,VA_f_dict,VA_d_dict,no_random_event_recall,no_random_event_precision = meata(gt_label_interval_ranges,
    local_recall_matrix, local_precision_matrix,local_f1_matrix, \
        local_fq_near_matrix, local_fq_distant_matrix,gt_detected_rate = meata_v5(gt_label_interval_ranges,
                           pred_label_interval_ranges,
                           window_length,
                           parameter_dict,
                           binary_predicted,
                           pred_case_id=pred_case_id,
                           max_ia_distant_length=max_ia_distant_length,
                           output=output,
                           thresh_id=thresh_id,
                           cal_mode=cal_mode)

    # todo:临时改变
    gt_detected_rate = 1
    local_recall_matrix_np = np.array(local_recall_matrix)
    local_precision_matrix_np = np.array(local_precision_matrix)
    local_near_fq_list_np = np.array(local_fq_near_matrix)
    local_distant_fq_list_np = np.array(local_fq_distant_matrix)
    # gt_num = local_recall_matrix_np.shape[0]

    parameter_w_gt = parameter_dict["parameter_w_gt"]
    parameter_w_near_ngt = parameter_dict["parameter_w_near_ngt"]


    local_meata_recall_list = []
    local_meata_precision_list = []
    local_meata_f1_list = []

    local_meata_recall_list_w_gt = []
    local_meata_precision_list_w_gt = []
    local_meata_f1_list_w_gt = []
    
    local_meata_recall_list_w_near_ngt = []
    local_meata_precision_list_w_near_ngt = []
    local_meata_f1_list_w_near_ngt = []
    
    local_meata_recall_list_w_distant_ngt = []
    local_meata_precision_list_w_distant_ngt = []
    local_meata_f1_list_w_distant_ngt = []

    #  meata_f1 0.901010101010101
    #  meata_f1_w_gt 0.7363636363636364
    #  meata_f1_w_near_ngt 0.9666666666666667
    #  meata_f1_w_distant_ngt 1.0
    #  final_meata_f1 0.901010101010101
    # f1_type = "add_row"
    # f1_type = "row_add"
    # f1_type = "row(pr)_add"
    f1_type = "row(f1)_add"
    if f1_type == "add_row":
        # 这个可以在最后一次循环求和
        for i in range(gt_num):
            local_recall_i = local_recall_matrix_np[i]
            local_precision_i = local_precision_matrix_np[i]
            local_near_fq_i = local_near_fq_list_np[i]
            local_distant_fq_i = local_distant_fq_list_np[i]

            local_meata_recall_i,local_meata_precision_i,local_meata_f1_i = cal_meata_f1(local_distant_fq_i, local_near_fq_i, local_precision_i, local_recall_i,
                                      parameter_w_gt, parameter_w_near_ngt,gt_detected_rate)

            local_meata_recall_i_w_gt,local_meata_precision_i_w_gt,local_meata_f1_i_w_gt = cal_meata_f1(local_distant_fq_i, local_near_fq_i, local_precision_i, local_recall_i,
                                      parameter_w_gt=1, parameter_w_near_ngt=0,gt_detected_rate=gt_detected_rate)

            local_meata_recall_i_w_near_ngt,local_meata_precision_i_w_near_ngt,local_meata_f1_i_w_near_ngt = cal_meata_f1(local_distant_fq_i, local_near_fq_i, local_precision_i, local_recall_i,
                                      parameter_w_gt=0, parameter_w_near_ngt=1,gt_detected_rate=gt_detected_rate)
            
            local_meata_recall_i_w_distant_ngt,local_meata_precision_i_w_distant_ngt,local_meata_f1_i_w_distant_ngt = cal_meata_f1(local_distant_fq_i, local_near_fq_i, local_precision_i, local_recall_i,
                                      parameter_w_gt=0, parameter_w_near_ngt=0,gt_detected_rate=gt_detected_rate)

            local_meata_recall_list.append(local_meata_recall_i)
            local_meata_precision_list.append(local_meata_precision_i)
            local_meata_f1_list.append(local_meata_f1_i)

            local_meata_recall_list_w_gt.append(local_meata_recall_i_w_gt)
            local_meata_precision_list_w_gt.append(local_meata_precision_i_w_gt)
            local_meata_f1_list_w_gt.append(local_meata_f1_i_w_gt)

            local_meata_recall_list_w_near_ngt.append(local_meata_recall_i_w_near_ngt)
            local_meata_precision_list_w_near_ngt.append(local_meata_precision_i_w_near_ngt)
            local_meata_f1_list_w_near_ngt.append(local_meata_f1_i_w_near_ngt)

            local_meata_recall_list_w_distant_ngt.append(local_meata_recall_i_w_distant_ngt)
            local_meata_precision_list_w_distant_ngt.append(local_meata_precision_i_w_distant_ngt)
            local_meata_f1_list_w_distant_ngt.append(local_meata_f1_i_w_distant_ngt)
            
        print()
        print(" local_meata_recall_list", local_meata_recall_list)
        print(" local_meata_precision_list", local_meata_precision_list)
        print(" local_meata_f1_list", local_meata_f1_list)
        print("w_gt")
        print(" local_meata_recall_list_w_gt", local_meata_recall_list_w_gt)
        print(" local_meata_precision_list_w_gt", local_meata_precision_list_w_gt)
        print(" local_meata_f1_list_w_gt", local_meata_f1_list_w_gt)
        print("w_near_ngt")
        print(" local_meata_recall_list_w_near_ngt", local_meata_recall_list_w_near_ngt)
        print(" local_meata_precision_list_w_near_ngt", local_meata_precision_list_w_near_ngt)
        print(" local_meata_f1_list_w_near_ngt", local_meata_f1_list_w_near_ngt)
        print("w_distant_ngt")
        print(" local_meata_recall_list_w_distant_ngt", local_meata_recall_list_w_distant_ngt)
        print(" local_meata_precision_list_w_distant_ngt", local_meata_precision_list_w_distant_ngt)
        print(" local_meata_f1_list_w_distant_ngt", local_meata_f1_list_w_distant_ngt)

        
        meata_f1 = np.mean(np.array(local_meata_f1_list))
        meata_f1_w_gt = np.mean(np.array(local_meata_f1_list_w_gt))
        meata_f1_w_near_ngt = np.mean(np.array(local_meata_f1_list_w_near_ngt))
        meata_f1_w_distant_ngt = np.mean(np.array(local_meata_f1_list_w_distant_ngt))


    # f1_type = "row_add"
    elif f1_type == "row(pr)_add":
        local_recall_mean = local_recall_matrix_np.mean()
        local_precision_mean = local_precision_matrix_np.mean()
        meata_f1_pr = compute_f1_score(local_precision_mean, local_recall_mean)
        
        local_near_fq_mean = local_near_fq_list_np.mean()
        local_distant_fq_mean = local_distant_fq_list_np.mean()
        meata_f1 = row_add_weight_sum(local_distant_fq_mean, local_near_fq_mean, meata_f1_pr, parameter_w_gt,
                                      parameter_w_near_ngt,gt_detected_rate=gt_detected_rate)

        meata_f1_w_gt = row_add_weight_sum(local_distant_fq_mean, local_near_fq_mean,
                                           meata_f1_pr, 1,0,
                                           gt_detected_rate=gt_detected_rate)
        meata_f1_w_near_ngt = row_add_weight_sum(local_distant_fq_mean, local_near_fq_mean,
                                                 meata_f1_pr, 0,1,
                                                 gt_detected_rate=gt_detected_rate)
        meata_f1_w_distant_ngt = row_add_weight_sum(local_distant_fq_mean, local_near_fq_mean,
                                                    meata_f1_pr, 0,0,
                                                    gt_detected_rate=gt_detected_rate)

        print()
        print(" meata_f1_pr", meata_f1_pr)
    else:
        # local_recall_mean = local_recall_matrix_np.mean()
        # local_precision_mean = local_precision_matrix_np.mean()
        local_f1_matrix = []
        for idx, local_recall in enumerate(local_recall_matrix_np):
            local_precision = local_precision_matrix_np[idx]
            local_f1 = compute_f1_score(local_precision, local_recall)
            local_f1_matrix.append(local_f1)
        local_near_fq_mean = local_near_fq_list_np.mean()
        local_distant_fq_mean = local_distant_fq_list_np.mean()
        meata_local_f1 = np.array(local_f1_matrix).mean()
        # gt_detected_rate=1
        meata_f1 = row_add_weight_sum(local_distant_fq_mean, local_near_fq_mean, meata_local_f1, parameter_w_gt,
                                      parameter_w_near_ngt, gt_detected_rate=gt_detected_rate)

        meata_f1_w_gt = row_add_weight_sum(local_distant_fq_mean, local_near_fq_mean,
                                           meata_local_f1, 1, 0,
                                           gt_detected_rate=gt_detected_rate)
        meata_f1_w_near_ngt = row_add_weight_sum(local_distant_fq_mean, local_near_fq_mean,
                                                 meata_local_f1, 0, 1,
                                                 gt_detected_rate=gt_detected_rate)
        meata_f1_w_distant_ngt = row_add_weight_sum(local_distant_fq_mean, local_near_fq_mean,
                                                    meata_local_f1, 0, 0,
                                                    gt_detected_rate=gt_detected_rate)

        print()
        print(" meata_local_f1", meata_local_f1)


    # final_meata_f1 = no_random_coefficient*meata_f1
    final_meata_f1 = meata_f1

    print()
    print(" meata_f1", meata_f1)
    print(" meata_f1_w_gt", meata_f1_w_gt)
    print(" meata_f1_w_near_ngt", meata_f1_w_near_ngt)
    print(" meata_f1_w_distant_ngt", meata_f1_w_distant_ngt)
    print(" final_meata_f1", meata_f1)


    # plot
    print("gt_label_interval_ranges",gt_label_interval_ranges)
    print("pred_label_interval_ranges",pred_label_interval_ranges)
    label_ranges = [gt_label_interval_ranges]
    label_ranges.append(pred_label_interval_ranges)
    label_array_list= [y_true.tolist()]

    label_array_list.append(binary_predicted.tolist())
    # Sub_OCSVM
    # max_f1_idx 57
    # TimesNet
    # max_f1_idx 5

    # CNN
    # max_f1_idx 6
    # FFT
    # max_f1_idx 25
    # if thresh_id == 15:
    # plotFigures_systhetic_data(label_ranges,label_array_list,forecasting_len=0,delay_len= 0)
    meata_f1_w_ngt = (meata_f1_w_near_ngt+meata_f1_w_distant_ngt) / 2


    return final_meata_f1,meata_f1, \
            meata_f1_w_gt,meata_f1_w_near_ngt,meata_f1_w_distant_ngt,meata_f1_w_ngt


def row_add_weight_sum(local_distant_fq_mean, local_near_fq_mean, \
                       meata_f1_pr, parameter_w_gt, parameter_w_near_ngt,gt_detected_rate):
    meata_f1 = parameter_w_gt * meata_f1_pr*gt_detected_rate + \
               parameter_w_near_ngt * local_near_fq_mean + \
               (1 - parameter_w_gt - parameter_w_near_ngt) * local_distant_fq_mean
    return meata_f1


def cal_meata_f1(local_distant_fq_i, local_near_fq_i, local_precision_i, local_recall_i, parameter_w_gt,
                 parameter_w_near_ngt,gt_detected_rate):
    print_msg = True
    # print_msg = False
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

def meata_f1_first(y_true,y_score,parameter_dict=parameter_dict,cal_mode="proportion",pred_case_id=None):
    window_length =len(y_true)

    # array -> interval_ranges
    gt_label_interval_ranges = convert_vector_to_events(y_true)

    # score->binary array
    binary_predicted = y_score
    # # binary array->point ranges
    # pred_ranges = convert_vector_to_events_PATE(binary_predicted)
    #
    # pred_label_point_array = convert_events_to_array_PATE(pred_ranges, time_series_length=window_length)
    # # binary array->interval ranges
    pred_label_interval_ranges = convert_vector_to_events(binary_predicted)
    meata_f1_first_value,meata_f1_first_value_multiply_no_coefficient,VA_f_dict,VA_d_dict,\
        meata_f1_first_w0,coefficient_meata_f1_first_w0,meata_f1_first_w1,coefficient_meata_f1_first_w1 = meata_tp_merge_first(gt_label_interval_ranges,
                                                                                       pred_label_interval_ranges,
                                                                                       window_length,
                                                                                       parameter_dict,
                                                                                       binary_predicted,
                                                                                       pred_case_id=pred_case_id,
                                                                                       cal_mode=cal_mode)

    # plot
    print("gt_label_interval_ranges",gt_label_interval_ranges)
    print("pred_label_interval_ranges",pred_label_interval_ranges)
    label_ranges = [gt_label_interval_ranges]
    label_ranges.append(pred_label_interval_ranges)
    label_array_list= [y_true.tolist()]

    label_array_list.append(binary_predicted.tolist())

    # plotFigures_systhetic_data(label_ranges,label_array_list,forecasting_len=0,delay_len= 0)

    return meata_f1_first_value,meata_f1_first_value_multiply_no_coefficient, \
            meata_f1_first_w0, coefficient_meata_f1_first_w0, meata_f1_first_w1, coefficient_meata_f1_first_w1

def find_max_ia_distant_length_in_single_ts(label_interval_ranges,ts_length,gt_num,near_single_side_range,parameter_distant_method,max_ia_distant_length,parameter_dict):    # print_msg = False
    print_msg = True
    parameter_w_near_2_ngt_gt_left = parameter_dict["parameter_w_near_2_ngt_gt_left"]
    parameter_w_near_0_ngt_gt_right = parameter_dict["parameter_w_near_0_ngt_gt_right"]

    # for i, label_data in enumerate(label_data_list):
    #     ts_length = len(label_data)
    #     label_interval_ranges = convert_vector_to_events(label_data)
    #     gt_num = len(label_interval_ranges)

    for i, label_interval_range in enumerate(label_interval_ranges):
        if i == 0:
            VA_gt_now_start, VA_gt_now_end = label_interval_range
            if parameter_w_near_2_ngt_gt_left <= 0:  # 左为空
                ia_distant_length = max(VA_gt_now_start - 0, 0)
            elif parameter_w_near_0_ngt_gt_right <= 0:
                ia_distant_length = max(VA_gt_now_start - 0 -near_single_side_range,0)
            else:
                ia_distant_length = max(VA_gt_now_start - 0 -near_single_side_range,0)
            if gt_num == 1:
                if parameter_w_near_2_ngt_gt_left <= 0:  # 左为空
                    ia_distant_length = max(ia_distant_length, ts_length-VA_gt_now_end-near_single_side_range)
                elif parameter_w_near_0_ngt_gt_right <= 0:
                    ia_distant_length = max(ia_distant_length, ts_length-VA_gt_now_end)
                else:
                    ia_distant_length = max(ia_distant_length, ts_length-VA_gt_now_end-near_single_side_range)
        elif i == gt_num-1:
            VA_gt_now_start, VA_gt_now_end = label_interval_range
            VA_gt_before_start, VA_gt_before_end = label_interval_ranges[i - 1]
            if parameter_distant_method == 1:
                if parameter_w_near_2_ngt_gt_left <= 0:  # 左为空
                    ia_distant_length = max(VA_gt_now_start - VA_gt_before_end - near_single_side_range * 1, 0)
                elif parameter_w_near_0_ngt_gt_right <= 0:
                    ia_distant_length = max(VA_gt_now_start - VA_gt_before_end - near_single_side_range * 1, 0)
                else:
                    ia_distant_length = max(VA_gt_now_start - VA_gt_before_end - near_single_side_range * 2, 0)
            else:
                if parameter_w_near_2_ngt_gt_left <= 0:  # 左为空
                    ia_distant_length = max((VA_gt_now_start - VA_gt_before_end - near_single_side_range * 1)/2, 0)
                elif parameter_w_near_0_ngt_gt_right <= 0:
                    ia_distant_length = max((VA_gt_now_start - VA_gt_before_end - near_single_side_range * 1)/2, 0)
                else:
                    ia_distant_length = max((VA_gt_now_start - VA_gt_before_end - near_single_side_range * 2)/2, 0)

            if parameter_w_near_2_ngt_gt_left <= 0:  # 左为空
                ia_distant_length = max(ia_distant_length, ts_length-VA_gt_now_end-near_single_side_range)
            elif parameter_w_near_0_ngt_gt_right <= 0:
                ia_distant_length = max(ia_distant_length, ts_length-VA_gt_now_end)
            else:
                ia_distant_length = max(ia_distant_length, ts_length-VA_gt_now_end-near_single_side_range)
        else:
            VA_gt_now_start, VA_gt_now_end = label_interval_range
            VA_gt_before_start, VA_gt_before_end = label_interval_ranges[i-1]
            if parameter_distant_method == 1:
                if parameter_w_near_2_ngt_gt_left <= 0:  # 左为空
                    ia_distant_length = max(VA_gt_now_start - VA_gt_before_end-near_single_side_range*1,0)
                elif parameter_w_near_0_ngt_gt_right <= 0:
                    ia_distant_length = max(VA_gt_now_start - VA_gt_before_end-near_single_side_range*1,0)
                else:
                    ia_distant_length = max(VA_gt_now_start - VA_gt_before_end-near_single_side_range*2,0)
            else:
                if parameter_w_near_2_ngt_gt_left <= 0:  # 左为空
                    ia_distant_length = max((VA_gt_now_start - VA_gt_before_end-near_single_side_range*1)/2,0)
                elif parameter_w_near_0_ngt_gt_right <= 0:
                    ia_distant_length = max((VA_gt_now_start - VA_gt_before_end-near_single_side_range*1)/2,0)
                else:
                    ia_distant_length = max((VA_gt_now_start - VA_gt_before_end-near_single_side_range*2)/2,0)


        if ia_distant_length > max_ia_distant_length:
            max_ia_distant_length = ia_distant_length

    return max_ia_distant_length

def find_max_ia_distant_length_in_ts(ts_label_data, parameter_dict=parameter_dict, find_type="ts_section"):
    # 这里右四种策略，dataset,multi_datasets,ts_section,max_value_cut
    parameter_distant_method = parameter_dict["parameter_distant_method"]

    if isinstance(ts_label_data, list):
        ts_label_data = np.array(ts_label_data)
    ts_label_data_list = []
    # if find_type=="multi_datasets":
    #     # check,numpy or list
    #     for single_dataset in ts_data:
    #         for single_ts_data in single_dataset:
    #             ts_data_list.append(single_ts_data)
    # if find_type == "ts_section_list":
    if ts_label_data.ndim == 2:
        ts_label_data_list = ts_label_data.tolist()
    #     # check,numpy or list
    #     for single_ts_data in ts_data:
    #         ts_data_list.append(single_ts_data)
    # if find_type == "ts_section":
    else:
        # check
        ts_label_data_list.append(ts_label_data)
    # else:
    #     # check
    #     if max_value_cut != None:
    #         return max(max_value_cut,0)
    #     else:
    #         pass

    max_ia_distant_length = 0
    near_single_side_range = parameter_dict["parameter_near_single_side_range"]

    for single_ts_label_data in ts_label_data_list:
        # ts_length = len(single_ts_label_data)
        # for i, label_data in enumerate(label_data_list):
        ts_length = len(single_ts_label_data)
        # if isinstance(single_ts_label_data, np.ndarray):
        #     single_ts_label_data = single_ts_label_data.tolist()
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