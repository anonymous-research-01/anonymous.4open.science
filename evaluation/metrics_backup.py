import time

from .basic_metrics import basic_metricor, generate_curve
from pate.PATE_metric import PATE
from pate.PATE_utils import convert_events_to_array_PATE, categorize_predicted_ranges_with_ids
from meata import meata,auc_pr_meata


def get_metrics(score, labels, slidingWindow=100, pred=None, version='opt', thre=250):
    metrics = {}
    th_50_100_set1_exp_list = [
    'AUC-PR',
    'AUC-ROC',
    'VUS-PR',
    'VUS-ROC',

    'PATE',
    # 'PATE_F1',

    'meata_f1_proportion_coefficient',
    'meata_f1_proportion',
    # 'meata_f1_detection_coefficient',
    # 'meata_f1_detection',
    # 'mean_pr_f1_p',
    # 'coefficient_mean_pr_f1_p',
    # 'event_f1_mean_p',
    # 'coefficient_event_f1_mean_p',
    # 'mean_pr_f1_d',
    # 'coefficient_mean_pr_f1_d',
    # 'event_f1_mean_d',
    # 'coefficient_event_f1_mean_d',

    # ,'meata_final_detection'

    'PA-F1',
    'Standard-F1',
    'Event-based-F1',
    'R-based-F1',
    'eTaPR_F1',
    'Affiliation-F',
     ]
    th_50_100_meata_f1_exp_list = [
    # 'meata_final_proportion_coefficient',
    # 'meata_final_detection_coefficient',
    # 'meata_f1_proportion',
    # 'meata_f1_detection',
    'meata_f1_proportion_coefficient',
    'meata_f1_proportion',
    'mean_pr_f1_p',
    'coefficient_mean_pr_f1_p',
    'event_f1_mean_p',
    'coefficient_event_f1_mean_p',
    # 'mean_pr_f1_d',
    # 'coefficient_mean_pr_f1_d',
    # 'event_f1_mean_d',
    # 'coefficient_event_f1_mean_d',
     ]

    test_list = [
    # 'meata_final_proportion_coefficient',
    # 'PATE',
        'meata_f1_proportion_coefficient',
        'meata_f1_proportion',
        # 'meata_f1_detection_coefficient',
        # 'meata_f1_detection',
        # 'mean_pr_f1_p',
        # 'coefficient_mean_pr_f1_p',
        # 'event_f1_mean_p',
        # 'coefficient_event_f1_mean_p',
        # 'mean_pr_f1_d',
        # 'coefficient_mean_pr_f1_d',
        # 'event_f1_mean_d',
        # 'coefficient_event_f1_mean_d',

     ]

    exp_list = th_50_100_set1_exp_list


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

    if "mean_pr_f1_p" in exp_list:
        time_start = time.time()
        mean_pr_f1_p, coefficient_mean_pr_f1_p, event_f1_mean_p, coefficient_event_f1_mean_p = auc_pr_meata(labels, score, Big_Data=True,num_desired_thresholds=thre,cal_mode="proportion")
        # meata_final_proportion,meata_final_proportion_coefficient = auc_pr_meata(labels, score, Big_Data=True,num_desired_thresholds=thre,cal_mode="proportion")
        # meata_final_proportion = auc_pr_meata(labels, score,cal_mode="proportion")
        time_end = time.time()
        print("meata_final_proportion time_end - time_start", time_end - time_start)
        # metrics['meata_final_proportion'] = meata_final_proportion
        # metrics['meata_final_proportion_coefficient'] = meata_final_proportion_coefficient

        metrics['mean_pr_f1_p'] = mean_pr_f1_p
        metrics['coefficient_mean_pr_f1_p'] = coefficient_mean_pr_f1_p
        metrics['event_f1_mean_p'] = event_f1_mean_p
        metrics['coefficient_event_f1_mean_p'] = coefficient_event_f1_mean_p


    if "mean_pr_f1_d" in exp_list:

        time_start = time.time()
        mean_pr_f1_d, coefficient_mean_pr_f1_d, event_f1_mean_d, coefficient_event_f1_mean_d = auc_pr_meata(labels, score, Big_Data=True,num_desired_thresholds=thre,cal_mode="detection")
        # meata_final_detection,meata_final_detection_coefficient = auc_pr_meata(labels, score, Big_Data=True,num_desired_thresholds=thre,cal_mode="detection")
        # meata_final_detection = auc_pr_meata(labels, score,cal_mode="detection")
        time_end = time.time()
        print("meata_final_detection time_end - time_start", time_end - time_start)
        # metrics['meata_final_detection'] = meata_final_detection
        # metrics['meata_final_detection_coefficient'] = meata_final_detection_coefficient
        
        metrics['mean_pr_f1_d'] = mean_pr_f1_d
        metrics['coefficient_mean_pr_f1_d'] = coefficient_mean_pr_f1_d
        metrics['event_f1_mean_d'] = event_f1_mean_d
        metrics['coefficient_event_f1_mean_d'] = coefficient_event_f1_mean_d

    if "meata_f1_proportion" in exp_list or "meata_f1_proportion_coefficient" in exp_list:

        time_start = time.time()
        meata_f1_proportion,meata_f1_proportion_coefficient = grader.metric_meata_F1(labels, score, preds=pred,cal_mode="proportion")

        time_end = time.time()
        print("meata_f1_proportion time_end - time_start", time_end - time_start)
        metrics['meata_f1_proportion'] = meata_f1_proportion
        metrics['meata_f1_proportion_coefficient'] = meata_f1_proportion_coefficient


    if "meata_f1_detection" in exp_list or "meata_f1_detection_coefficient" in exp_list:

        time_start = time.time()
        meata_f1_detection,meata_f1_detection_coefficient = grader.metric_meata_F1(labels, score, preds=pred,cal_mode="detection")

        time_end = time.time()
        print("meata_f1_detection time_end - time_start", time_end - time_start)
        metrics['meata_f1_detection'] = meata_f1_detection
        metrics['meata_f1_detection_coefficient'] = meata_f1_detection_coefficient

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
