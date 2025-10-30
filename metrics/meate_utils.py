from dqe import get_meate_gt_area
from metrics.affiliation.generics import convert_vector_to_events
from metrics.pate.PATE_utils import convert_events_to_array_PATE, convert_events_to_point_array
import copy

class_name = "ori"
# class_name = "ia"
# class_name = "va"
# class_name = "hybrid"
class_name_list = ["ia","va","hybrid"]

def get_class_pred_with_window(labels_original_len,gt_point_ranges,pred_array,parameter_dict):
    if class_name in class_name_list:
        # class pred
        # modify gt_ranges by range
        gt_label_point_array = convert_events_to_array_PATE(gt_point_ranges,time_series_length=labels_original_len)
        gt_label_interval_ranges = convert_vector_to_events(gt_label_point_array)


        pred_interval_ranges = convert_vector_to_events(pred_array)

        VA_f_dict_gt, VA_d_dict_gt, IA_dict, VA_dict, VA_gt_dict = get_meate_gt_area(gt_label_interval_ranges,
                                                                                     labels_original_len,
                                                                                     parameter_dict)

        # get class pred
        pred_interval_ranges_copy = copy.deepcopy(pred_interval_ranges)

        class_cal_res_dict = {}
        ia_pred_ranges = []
        for idx, (area_id,area) in enumerate(IA_dict.items()):
            area_start = area[0]
            area_end = area[1]
            for idy,pred_interval  in enumerate(pred_interval_ranges_copy):
                pred_interval_start = pred_interval[0]
                pred_interval_end = pred_interval[1]
                if pred_interval_start >= area_start and pred_interval_end <= area_end:
                    ia_pred_ranges.append(pred_interval)
                    pred_interval_ranges_copy.remove(pred_interval)

        va_pred_ranges = []
        for idx, (area_id, area) in enumerate(VA_dict.items()):
            area_start = area[0]
            area_end = area[1]
            for idy, pred_interval in enumerate(pred_interval_ranges_copy):
                pred_interval_start = pred_interval[0]
                pred_interval_end = pred_interval[1]
                if pred_interval_start >= area_start and pred_interval_end <= area_end:
                    va_pred_ranges.append(pred_interval)
                    pred_interval_ranges_copy.remove(pred_interval)


        hybrid_pred_ranges = pred_interval_ranges_copy


        # get class TPR, FPR, Precision
        # class_pred_list = [ia_pred_ranges,va_pred_ranges,hybrid_pred_ranges]
        # class_pred_name_list = ["ia_pred_ranges","va_pred_ranges","hybrid_pred_ranges"]
        if class_name == "ia":
            # interval_range -> point array,[20,40]->[20,...,39]
            class_pred_array = convert_events_to_point_array(ia_pred_ranges,time_series_length=labels_original_len)
        elif class_name == "va":
            class_pred_array = convert_events_to_point_array(va_pred_ranges,time_series_length=labels_original_len)
        else:
            class_pred_array = convert_events_to_point_array(hybrid_pred_ranges,time_series_length=labels_original_len)
        return class_pred_array
    else:
        return pred_array
