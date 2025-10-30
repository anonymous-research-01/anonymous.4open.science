# DQE
w_tq = 1 / 3  # tq weight
w_fq_near = 1 / 3  # fq_near weight
w_fq_dis = 1 - w_tq - w_fq_near  # fq_distant weight
w_fq_near_delay = 1 / 2  # fq_near_delayed weight
w_fq_near_early = 1 - w_fq_near_delay  # fq_near_early weight

# DQE_delay
# w_tq = 1/3
# w_fq_near = 1/3
# w_fq_dis = 1 - w_tq- w_fq_near
# w_fq_near_delay = 1
# w_fq_near_early = 1-w_fq_near_delay

# DQE_early
# w_tq = 1/3
# w_fq_near = 1/3
# w_fq_dis = 1 - w_tq- w_fq_near
# w_fq_near_delay = 0
# w_fq_near_early = 1-w_fq_near_delay

# DQE_TQ
# w_tq = 1
# w_fq_near = 0
# w_fq_dis = 1 - w_tq- w_fq_near
# w_fq_near_delay = 1 / 2
# w_fq_near_early = 1-w_fq_near_delay

# DQE_FQ
# w_tq = 0
# w_fq_near = 1/2
# w_fq_dis = 1 - w_tq- w_fq_near
# w_fq_near_delay = 1 / 2
# w_fq_near_early = 1-w_fq_near_delay

# # DQE_FQ_near
# w_tq = 0
# w_fq_near = 1
# w_fq_dis = 1 - w_tq- w_fq_near
# w_fq_near_delay = 1 / 2
# w_fq_near_early = 1-w_fq_near_delay

# # DQE_FQ_0
# w_tq = 0
# w_fq_near = 1
# w_fq_dis = 1 - w_tq- w_fq_near
# w_fq_near_delay = 1
# w_fq_near_early = 1-w_fq_near_delay

# # DQE_FQ_2
# w_tq = 0
# w_fq_near = 1
# w_fq_dis = 1 - w_tq- w_fq_near
# w_fq_near_delay = 0
# w_fq_near_early = 1-w_fq_near_delay

# # DQE_FQ_distant
# w_tq = 0
# w_fq_near = 0
# w_fq_dis = 1 - w_tq- w_fq_near
# w_fq_near_delay = 1/2
# w_fq_near_early = 1-w_fq_near_delay


beta = 2  # beta in paper
# beta = 3

fq_near_mean_proximity_gama = 1  # gama of mean proximity in near fq
fq_near_closest_onset_gama = 1  # gama of closest onset response in near fq
fq_near_total_duration_gama = 1  # gama of total duration in near fq


fq_distant_mean_proximity_gama = 1  # gama of mean proximity in distant fq
fq_distant_closest_onset_gama = 1  # gama of closest onset response in distant fq
fq_distant_total_duration_gama = 2  # gama of total duration in distant fq


distant_method = "whole"  # fq_distant strategy_1 whole
# distant_method = "split" # fq_distant strategy_2 split

distant_direction = "both"  # fq_distant direction
# distant_direction = "delay"

use_detection_rate = True
# use_detection_rate = False

weight_sum_method = "triangle"
# weight_sum_method = "equal"

parameter_dict = {
    # weight
    "w_tq": w_tq,
    "w_fq_near": w_fq_near,
    "w_fq_near_early": w_fq_near_early,
    "w_fq_near_delay": w_fq_near_delay,

    "beta": beta,
    "distant_direction": distant_direction,

    # strategy configuration
    "distant_method": distant_method,
    "use_detection_rate": use_detection_rate,
    "weight_sum_method": weight_sum_method,

    # gama
    "fq_near_mean_proximity_gama": fq_near_mean_proximity_gama,
    "fq_near_closest_onset_gama": fq_near_closest_onset_gama,
    "fq_near_total_duration_gama": fq_near_total_duration_gama,
    "fq_distant_mean_proximity_gama": fq_distant_mean_proximity_gama,
    "fq_distant_closest_onset_gama": fq_distant_closest_onset_gama,
    "fq_distant_total_duration_gama": fq_distant_total_duration_gama,

}