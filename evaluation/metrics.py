import time

from .basic_metrics import basic_metricor, generate_curve
from pate.PATE_metric import PATE

from config.meata_config import parameter_dict


def get_metrics(score, labels, slidingWindow=100, pred=None, version='opt', thre=250):
    metrics = {}
    case_analyze = [
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

    # exp_list = th_50_100_set1_exp_list
    exp_list = case_analyze


    '''
    Threshold Independent
    '''
    grader = basic_metricor()
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

        time_start = time.time()
        _, _, _, _, _, _,VUS_ROC, VUS_PR = generate_curve(labels.astype(int), score, slidingWindow, version, thre)
        time_end = time.time()
        print("VUS_ROC VUS_PR time_end - time_start", time_end - time_start)
        metrics['VUS-PR'] = VUS_PR
        metrics['VUS-ROC'] = VUS_ROC

    if "PATE" in exp_list:

        time_start = time.time()
        e_buffer = d_buffer = slidingWindow//2
        pate = PATE(labels, score, e_buffer, d_buffer, Big_Data=True, n_jobs=1, include_zero=False,num_desired_thresholds=thre)
        time_end = time.time()
        print("pate time_end - time_start", time_end - time_start)
        metrics['PATE'] = pate


    # need thresh

    if "PATE_F1" in exp_list:
        time_start = time.time()
        e_buffer = d_buffer = slidingWindow//2
        pate_f1 = grader.metric_PATE_F1(labels, score, preds=pred,e_buffer=e_buffer,d_buffer=d_buffer)
        time_end = time.time()
        print("pate_f1 time_end - time_start", time_end - time_start)
        metrics['PATE_F1'] = pate_f1

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



    if "R-based-F1" in exp_list:
        RF1 = grader.metric_RF1(labels, score, preds=pred)
        metrics['R-based-F1'] = RF1
    if "Affiliation-F" in exp_list:
        Affiliation_F = grader.metric_Affiliation(labels, score, preds=pred)
        metrics['Affiliation-F'] = Affiliation_F
    if "eTaPR_F1" in exp_list:
        eTaPR_F1 = grader.metric_eTaPR_F1(labels, score, preds=pred)
        metrics['eTaPR_F1'] = eTaPR_F1



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
