# DQE
parameter_w_gt = 1/3 # tq weight
parameter_w_near_ngt = 1/3 # fq_near weight
parameter_w_distant_ngt = 1 - parameter_w_gt- parameter_w_near_ngt# fq_distant weight
parameter_w_near_0_ngt_gt_right = 1 / 2 # fq_near_delayed weight
parameter_w_near_2_ngt_gt_left = 1-parameter_w_near_0_ngt_gt_right # fq_near_early weight

# DQE_delay
# parameter_w_gt = 1/3
# parameter_w_near_ngt = 1/3
# parameter_w_distant_ngt = 1 - parameter_w_gt- parameter_w_near_ngt
# parameter_w_near_0_ngt_gt_right = 1
# parameter_w_near_2_ngt_gt_left = 1-parameter_w_near_0_ngt_gt_right

# DQE_early
# parameter_w_gt = 1/3
# parameter_w_near_ngt = 1/3
# parameter_w_distant_ngt = 1 - parameter_w_gt- parameter_w_near_ngt
# parameter_w_near_0_ngt_gt_right = 0
# parameter_w_near_2_ngt_gt_left = 1-parameter_w_near_0_ngt_gt_right

# DQE_TQ
# parameter_w_gt = 1
# parameter_w_near_ngt = 0
# parameter_w_distant_ngt = 1 - parameter_w_gt- parameter_w_near_ngt
# parameter_w_near_0_ngt_gt_right = 1 / 2
# parameter_w_near_2_ngt_gt_left = 1-parameter_w_near_0_ngt_gt_right

# DQE_FQ
# parameter_w_gt = 0
# parameter_w_near_ngt = 1/2
# parameter_w_distant_ngt = 1 - parameter_w_gt- parameter_w_near_ngt
# parameter_w_near_0_ngt_gt_right = 1 / 2
# parameter_w_near_2_ngt_gt_left = 1-parameter_w_near_0_ngt_gt_right

# # DQE_FQ_near
# parameter_w_gt = 0
# parameter_w_near_ngt = 1
# parameter_w_distant_ngt = 1 - parameter_w_gt- parameter_w_near_ngt
# parameter_w_near_0_ngt_gt_right = 1 / 2 
# parameter_w_near_2_ngt_gt_left = 1-parameter_w_near_0_ngt_gt_right 

# # DQE_FQ_0
# parameter_w_gt = 0
# parameter_w_near_ngt = 1
# parameter_w_distant_ngt = 1 - parameter_w_gt- parameter_w_near_ngt
# parameter_w_near_0_ngt_gt_right = 1 
# parameter_w_near_2_ngt_gt_left = 1-parameter_w_near_0_ngt_gt_right 

# # DQE_FQ_2
# parameter_w_gt = 0
# parameter_w_near_ngt = 1
# parameter_w_distant_ngt = 1 - parameter_w_gt- parameter_w_near_ngt
# parameter_w_near_0_ngt_gt_right = 0 
# parameter_w_near_2_ngt_gt_left = 1-parameter_w_near_0_ngt_gt_right 

# # DQE_FQ_distant
# parameter_w_gt = 0
# parameter_w_near_ngt = 0
# parameter_w_distant_ngt = 1 - parameter_w_gt- parameter_w_near_ngt
# parameter_w_near_0_ngt_gt_right = 1/2 
# parameter_w_near_2_ngt_gt_left = 1-parameter_w_near_0_ngt_gt_right 


parameter_near_single_side_range = 125 # fq_near range distance

parameter_gama = 2 # beta in paper
# parameter_gama = 3

parameter_distant_method = 1 # fq_distant strategy_1
# parameter_distant_method = 2 # fq_distant strategy_2


parameter_distant_direction = "both" # fq_distant direction
# parameter_distant_direction = "left"


parameter_dict = {
    "parameter_w_gt" : parameter_w_gt,
    "parameter_w_near_ngt" : parameter_w_near_ngt,
    "parameter_w_near_2_ngt_gt_left" : parameter_w_near_2_ngt_gt_left,
    "parameter_w_near_0_ngt_gt_right" : parameter_w_near_0_ngt_gt_right,

    "parameter_near_single_side_range" : parameter_near_single_side_range,

    "parameter_gama" : parameter_gama,
    "parameter_distant_method" : parameter_distant_method,
    "parameter_distant_direction" : parameter_distant_direction,
}