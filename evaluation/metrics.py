import time

from .basic_metrics import basic_metricor, generate_curve
from pate.PATE_metric import PATE
from pate.PATE_utils import convert_events_to_array_PATE, categorize_predicted_ranges_with_ids, \
    convert_vector_to_events_PATE
from meata import meata,auc_pr_meata
from config.meata_config import parameter_dict
from metrics.metrics_pa import PointAdjustKPercent


def get_metrics(score, labels, slidingWindow=100, pred=None, version='opt', thre=250):
    metrics = {}
    case_analyze = [
        # "meata_f1_proportion_no_random",
        "meata_auc",
        # "PA-K"
    ]
    th_50_100_set1_exp_list = [
    'Standard-F1',
    'AUC-ROC',
    'AUC-PR',

    # 'PA-F1',
    "PA-K",

    'VUS-ROC',
    'VUS-PR',
    'PATE',
    # 'PATE_F1',

    'R-based-F1',
    # 'Event-based-F1',
    'eTaPR_F1',
    'Affiliation-F',
    "meata_auc",

    ]

    random_100_100_exp_list = [
        'AUC-PR'
        , 'AUC-ROC'
        , 'VUS-PR'
        , 'VUS-ROC'

        , 'PATE'

        # , 'meata_f1_proportion'
          ,"meata_auc"

    , 'Standard-F1'
        , 'PA-F1'
        , 'Event-based-F1'
        , 'R-based-F1'
        , 'eTaPR_F1'
        , 'Affiliation-F'

        ,"PA-K"

    ]
    th_100_100_exp_list = [
    'AUC-PR'
    ,'AUC-ROC'
    ,'VUS-PR'
    ,'VUS-ROC'

    ,'PATE'
     ]

    # exp_list = random_100_100_exp_list
    # exp_list = th_50_100_set1_exp_list
    exp_list = case_analyze


    '''
    Threshold Independent
    '''
    grader = basic_metricor()
    # AUC_ROC, Precision, Recall, PointF1, PointF1PA, Rrecall, ExistenceReward, OverlapReward, Rprecision, RF, Precision_at_k = grader.metric_new(labels, score, pred, plot_ROC=False)
    if "AUC-ROC" in exp_list:
        time_start = time.time()
        AUC_ROC = grader.metric_ROC(labels, score)
        time_end = time.time()
        print("AUC_ROC time_end - time_start", time_end - time_start)
        metrics['AUC-ROC'] = AUC_ROC

    if "AUC-PR" in exp_list:

        time_start = time.time()
        AUC_PR = grader.metric_PR(labels, score)
        time_end = time.time()
        print("AUC_PR time_end - time_start", time_end - time_start)
        metrics['AUC-PR'] = AUC_PR

    if "VUS-PR"  in exp_list and "VUS-ROC" in exp_list:

        # R_AUC_ROC, R_AUC_PR, _, _, _ = grader.RangeAUC(labels=labels, score=score, window=slidingWindow, plot_ROC=True)
        time_start = time.time()
        _, _, _, _, _, _,VUS_ROC, VUS_PR = generate_curve(labels.astype(int), score, slidingWindow, version, thre)
        time_end = time.time()
        print("VUS_ROC VUS_PR time_end - time_start", time_end - time_start)
        metrics['VUS-PR'] = VUS_PR
        metrics['VUS-ROC'] = VUS_ROC

    if "PATE" in exp_list:

        time_start = time.time()
        e_buffer = d_buffer = slidingWindow//2
        # pate = PATE(labels, score, e_buffer, d_buffer, Big_Data=False, n_jobs=1, include_zero=False)
        pate = PATE(labels, score, e_buffer, d_buffer, Big_Data=True, n_jobs=1, include_zero=False,num_desired_thresholds=thre)
        time_end = time.time()
        print("pate time_end - time_start", time_end - time_start)
        # 029_WSD_id_1_WebService_tr_4559_1st_10201_name_list
        metrics['PATE'] = pate


    # need thresh

    if "PATE_F1" in exp_list:
        time_start = time.time()
        e_buffer = d_buffer = slidingWindow//2
        pate_f1 = grader.metric_PATE_F1(labels, score, preds=pred,e_buffer=e_buffer,d_buffer=d_buffer)
        time_end = time.time()
        print("pate_f1 time_end - time_start", time_end - time_start)
        metrics['PATE_F1'] = pate_f1

    # if "meata_final_proportion" in exp_list or "meata_final_proportion_no_random" in exp_list:
    #     time_start = time.time()
    #     meata_final_proportion,meata_final_proportion_no_random = auc_pr_meata(labels, score, Big_Data=True,num_desired_thresholds=thre,cal_mode="proportion")
    #     # meata_final_proportion = auc_pr_meata(labels, score,cal_mode="proportion")
    #     time_end = time.time()
    #     print("meata_final_proportion time_end - time_start", time_end - time_start)
    #     metrics['meata_final_proportion'] = meata_final_proportion
    #     metrics['meata_final_proportion_no_random'] = meata_final_proportion_no_random
    # 
    # 
    # if "meata_final_detection" in exp_list or "meata_final_detection_no_random" in exp_list:
    # 
    #     time_start = time.time()
    #     meata_final_detection,meata_final_detection_no_random = auc_pr_meata(labels, score, Big_Data=True,num_desired_thresholds=thre,cal_mode="detection")
    #     # meata_final_detection = auc_pr_meata(labels, score,cal_mode="detection")
    #     time_end = time.time()
    #     print("meata_final_detection time_end - time_start", time_end - time_start)
    #     metrics['meata_final_detection'] = meata_final_detection
    #     metrics['meata_final_detection_no_random'] = meata_final_detection_no_random

    if "meata_f1_proportion" in exp_list or "meata_f1_proportion_no_random" in exp_list:
        time_start = time.time()
        # meata_f1_proportion,meata_f1_proportion_no_random = grader.metric_meata_F1(labels, score, preds=pred,cal_mode="proportion")
        import copy
        parameter_dict_copy = copy.deepcopy(parameter_dict)
        parameter_dict_copy["parameter_near_single_side_range"] = slidingWindow

        meata_f1_proportion,meata_f1_proportion_no_random,\
            meata_F1_w_gt,meata_F1_w_near_ngt,meata_F1_w_distant_ngt,meata_F1_w_ngt = grader.metric_meata_F1(labels, score, preds=pred,parameter_dict=parameter_dict_copy,cal_mode="proportion")

        time_end = time.time()
        print("meata_f1_proportion time_end - time_start", time_end - time_start)
        metrics['meata_f1_proportion'] = meata_f1_proportion
        metrics['meata_f1_proportion_no_random'] = meata_f1_proportion_no_random
        metrics['meata_F1_w_gt'] = meata_F1_w_gt
        metrics['meata_F1_w_near_ngt'] = meata_F1_w_near_ngt
        metrics['meata_F1_w_distant_ngt'] = meata_F1_w_distant_ngt
        metrics['meata_F1_w_ngt'] = meata_F1_w_ngt

    if "meata_auc" in exp_list:
        time_start = time.time()
        import copy
        parameter_dict_copy = copy.deepcopy(parameter_dict)
        parameter_dict_copy["parameter_near_single_side_range"] = slidingWindow

        final_meata, meata,meata_w_gt,meata_w_near_ngt,meata_w_distant_ngt,meata_w_ngt = grader.metric_meata_AUC_PR(labels, score, preds=pred,parameter_dict=parameter_dict_copy,cal_mode="proportion")

        time_end = time.time()
        print("meata_auc time_end - time_start", time_end - time_start)
        metrics['final_meata'] = final_meata
        metrics['meata'] = meata
        metrics['meata_w_gt'] = meata_w_gt
        metrics['meata_w_near_ngt'] = meata_w_near_ngt
        metrics['meata_w_distant_ngt'] = meata_w_distant_ngt
        metrics['meata_w_ngt'] = meata_w_ngt
    # if "meata_f1_proportion_no_random_w0" in exp_list:
    #     import copy
    #     parameter_dict_copy = copy.deepcopy(parameter_dict)
    #     parameter_dict_copy["parameter_eta"] = 0
    #     time_start = time.time()
    #     meata_f1_proportion_w0,meata_f1_proportion_no_random_w0 = grader.metric_meata_F1(labels, score, preds=pred,parameter_dict=parameter_dict_copy,cal_mode="proportion")
    #
    #     time_end = time.time()
    #     print("meata_f1_proportion time_end - time_start", time_end - time_start)
    #     metrics['meata_f1_proportion_w0'] = meata_f1_proportion_w0
    #     metrics['meata_f1_proportion_no_random_w0'] = meata_f1_proportion_no_random_w0
    #
    # if "meata_f1_proportion_no_random_w1" in exp_list:
    #     import copy
    #     parameter_dict_copy = copy.deepcopy(parameter_dict)
    #     parameter_dict_copy["parameter_eta"] = 1
    #     time_start = time.time()
    #     meata_f1_proportion_w1,meata_f1_proportion_no_random_w1 = grader.metric_meata_F1(labels, score, preds=pred,parameter_dict=parameter_dict_copy,cal_mode="proportion")
    #
    #     time_end = time.time()
    #     print("meata_f1_proportion time_end - time_start", time_end - time_start)
    #     metrics['meata_f1_proportion_w1'] = meata_f1_proportion_w1
    #     metrics['meata_f1_proportion_no_random_w1'] = meata_f1_proportion_no_random_w1
    #
    # if "meata_f1_proportion_no_random_w1_2" in exp_list:
    #     import copy
    #     parameter_dict_copy = copy.deepcopy(parameter_dict)
    #     parameter_dict_copy["parameter_eta"] = 1/2
    #     time_start = time.time()
    #     meata_f1_proportion_w1_2,meata_f1_proportion_no_random_w1_2 = grader.metric_meata_F1(labels, score, preds=pred,parameter_dict=parameter_dict_copy,cal_mode="proportion")
    #
    #     time_end = time.time()
    #     print("meata_f1_proportion time_end - time_start", time_end - time_start)
    #     metrics['meata_f1_proportion_w1_2'] = meata_f1_proportion_w1_2
    #     metrics['meata_f1_proportion_no_random_w1_2'] = meata_f1_proportion_no_random_w1_2
    # # "meata_f1_score_first_p",
    # # "meata_f1_score_first_p_no_random",
    # # "meata_f1_score_first_p_no_random_w0",
    # # "meata_f1_score_first_p_no_random_w1",
    #
    # if "meata_f1_score_first_p" or "meata_f1_score_first_p_no_random" in exp_list:
    #     import copy
    #     parameter_dict_copy = copy.deepcopy(parameter_dict)
    #     parameter_dict_copy["parameter_eta"] = 1/2
    #     time_start = time.time()
    #     meata_f1_score_first_p,meata_f1_score_first_p_no_random,meata_F1_first_w0,meata_F1_first_multiply_w0,meata_F1_first_w1,meata_F1_first_multiply_w1 = grader.metric_meata_F1_score_first(labels, score, preds=pred,parameter_dict=parameter_dict_copy,cal_mode="proportion")
    #
    #     time_end = time.time()
    #     print("meata_f1_proportion time_end - time_start", time_end - time_start)
    #     metrics['meata_f1_score_first_p'] = meata_f1_score_first_p
    #     metrics['meata_f1_score_first_p_no_random'] = meata_f1_score_first_p_no_random
    #     metrics['meata_F1_first_w0'] = meata_F1_first_w0
    #     # metrics['meata_F1_first_multiply_w0'] = meata_F1_first_multiply_w0
    #     metrics['meata_F1_first_w1'] = meata_F1_first_w1
    #     # metrics['meata_F1_first_multiply_w1'] = meata_F1_first_multiply_w1
    #
    # if "meata_f1_score_first_p_no_random_w0" in exp_list:
    #     import copy
    #     parameter_dict_copy = copy.deepcopy(parameter_dict)
    #     parameter_dict_copy["parameter_eta"] = 0
    #     time_start = time.time()
    #     meata_f1_score_first_p_w0,meata_f1_score_first_p_no_random_w0 = grader.metric_meata_F1_score_first(labels, score, preds=pred,parameter_dict=parameter_dict_copy,cal_mode="proportion")
    #
    #     time_end = time.time()
    #     print("meata_f1_proportion time_end - time_start", time_end - time_start)
    #     metrics['meata_f1_score_first_p_w0'] = meata_f1_score_first_p_w0
    #     metrics['meata_f1_score_first_p_no_random_w0'] = meata_f1_score_first_p_no_random_w0
    #
    # if "meata_f1_score_first_p_no_random_w1" in exp_list:
    #     import copy
    #     parameter_dict_copy = copy.deepcopy(parameter_dict)
    #     parameter_dict_copy["parameter_eta"] = 1
    #     time_start = time.time()
    #     meata_f1_score_first_p_w1,meata_f1_score_first_p_no_random_w1 = grader.metric_meata_F1_score_first(labels, score, preds=pred,parameter_dict=parameter_dict_copy,cal_mode="proportion")
    #
    #     time_end = time.time()
    #     print("meata_f1_proportion time_end - time_start", time_end - time_start)
    #     metrics['meata_f1_score_first_p_w1'] = meata_f1_score_first_p_w1
    #     metrics['meata_f1_score_first_p_no_random_w1'] = meata_f1_score_first_p_no_random_w1
    #
    # if "meata_f1_detection" in exp_list or "meata_f1_detection_no_random" in exp_list:
    #
    #     time_start = time.time()
    #     meata_f1_detection,meata_f1_detection_no_random = grader.metric_meata_F1(labels, score, preds=pred,cal_mode="detection")
    #
    #     time_end = time.time()
    #     print("meata_f1_detection time_end - time_start", time_end - time_start)
    #     metrics['meata_f1_detection'] = meata_f1_detection
    #     metrics['meata_f1_detection_no_random'] = meata_f1_detection_no_random
    #
    # # "meata_f1_score_first_d",
    # # "meata_f1_score_first_d_no_random",
    #
    # if "meata_f1_score_first_d" in exp_list or "meata_f1_score_first_d_no_random" in exp_list:
    #     import copy
    #     parameter_dict_copy = copy.deepcopy(parameter_dict)
    #     parameter_dict_copy["parameter_eta"] = 1/2
    #     time_start = time.time()
    #     meata_f1_score_first_d,meata_f1_score_first_d_no_random = grader.metric_meata_F1_score_first(labels, score, preds=pred,parameter_dict=parameter_dict_copy,cal_mode="detection")
    #
    #     time_end = time.time()
    #     print("meata_f1_detection time_end - time_start", time_end - time_start)
    #     metrics['meata_f1_score_first_d'] = meata_f1_score_first_d
    #     metrics['meata_f1_score_first_d_no_random'] = meata_f1_score_first_d_no_random

    '''
    Threshold Dependent
    if pred is None --> use the oracle threshold
    '''

    if "Standard-F1" in exp_list:
        PointF1 = grader.metric_PointF1(labels, score, preds=pred)
        metrics['Standard-F1'] = PointF1

    if "PA-F1" in exp_list:
        PointF1PA = grader.metric_PointF1PA(labels, score, preds=pred)
        metrics['PA-F1'] = PointF1PA

    if "PA-K" in exp_list:
        PointF1PA_K = grader.metric_PointF1PA_K(labels, score, preds=pred)
        metrics['PA-K'] = PointF1PA_K



    # EventF1PA = grader.metric_EventF1PA(labels, score, preds=pred)
    if "R-based-F1" in exp_list:
        RF1 = grader.metric_RF1(labels, score, preds=pred)
        metrics['R-based-F1'] = RF1
    if "Affiliation-F" in exp_list:
        Affiliation_F = grader.metric_Affiliation(labels, score, preds=pred)
        metrics['Affiliation-F'] = Affiliation_F
    if "eTaPR_F1" in exp_list:
        eTaPR_F1 = grader.metric_eTaPR_F1(labels, score, preds=pred)
        metrics['eTaPR_F1'] = eTaPR_F1


    # metrics['AUC-PR'] = AUC_PR
    # metrics['AUC-ROC'] = AUC_ROC
    # metrics['VUS-PR'] = VUS_PR
    # metrics['VUS-ROC'] = VUS_ROC
    #
    # metrics['PATE'] = pate
    # metrics['PATE_F1'] = pate_f1
    #
    # metrics['meata_final_proportion'] = meata_final_proportion
    # metrics['meata_final_detection'] = meata_final_detection
    #
    #
    # metrics['Standard-F1'] = PointF1
    # metrics['PA-F1'] = PointF1PA
    # # metrics['Event-based-F1'] = EventF1PA
    # metrics['R-based-F1'] = RF1
    # metrics['eTaPR_F1'] = eTaPR_F1
    # metrics['Affiliation-F'] = Affiliation_F


    return metrics


def get_metrics_pred(score, labels, pred, slidingWindow=100):
    metrics = {}

    grader = basic_metricor()

    PointF1 = grader.metric_PointF1(labels, score, preds=pred)
    PointF1PA = grader.metric_PointF1PA(labels, score, preds=pred)
    EventF1PA = grader.metric_EventF1PA(labels, score, preds=pred)
    RF1 = grader.metric_RF1(labels, score, preds=pred)
    Affiliation_F = grader.metric_Affiliation(labels, score, preds=pred)
    VUS_R, VUS_P, VUS_F = grader.metric_VUS_pred(labels, preds=pred, windowSize=slidingWindow)

    metrics['Standard-F1'] = PointF1
    metrics['PA-F1'] = PointF1PA
    metrics['Event-based-F1'] = EventF1PA
    metrics['R-based-F1'] = RF1
    metrics['Affiliation-F'] = Affiliation_F

    metrics['VUS-Recall'] = VUS_R
    metrics['VUS-Precision'] = VUS_P
    metrics['VUS-F'] = VUS_F

    return metrics
