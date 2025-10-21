#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from prts import ts_precision, ts_recall # Ranged based Precision and Recall metric
from metrics.f1_score_f1_pa import * # used for PA-F1 and general F1 calculation


def get_F1Score_RangeBased(y_true, scores):
    # bias = "middle"
    bias = "flat"
    precision_flat = ts_precision(y_true, scores, alpha=0.2, cardinality="one", bias=bias)
    recall_flat = ts_recall(y_true, scores, alpha=0.2, cardinality="one", bias=bias)
    
    f1_score_flat = get_f_score(precision_flat, recall_flat)
    return precision_flat, recall_flat, f1_score_flat

