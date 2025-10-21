
# IA
# parameter_alpha = 1/3 # refine
# parameter_alpha = 1/2 # refine

# parameter_rate = d/l,指导图

# 单侧移动
# 0-2，l起决定作用
# 2，边界
# 2-正无穷，d起决定作用

# fd同时移动相同距离等价性
# 0-1，
# 1，边界
# 1-正无穷，

# parameter_rate = 1/10 # l起决定作用,极限思想，只看l最差，l最好，l最好有多重情况，再看d

# parameter_rate = 1/2 # 单侧移动fd的自然比例

parameter_rate = 1 # fd移动相同距离，fd谁是决定作用的边界

# parameter_rate = 2 # 单侧移动fd谁是决定作用的边界，对抗单侧移动的自然比例

# parameter_rate = 10 # d起决定作用,极限思想，只看d最差，d最好



# VA_f,VA_d
# parameter_theta = 1/10 # demand
# parameter_theta = 1/10 # demand
# parameter_theta = 1/10 # demand,指导图
# parameter_theta = 1/18 # demand,指导图
parameter_theta = 0.5 # 为了让precision的得分在短异常的时候不过于低,也就是让临近区域的fp的空间和tp的空间取得平衡，否则得分会偏小
# 这个值也有意义，可以用示例图指导它设置，把每个parameter_theta的值等价的中间的预测给出来，但是还有gt面积参数，不好控制，
# 只能给出默认配置的parameter_theta的调整情况，之后可以给出参数调整的可视化计算工具，
# 也就是parameter_theta，parameter_rho，parameter_lamda的变化关系的图
# 指导1
# parameter_theta，parameter_rho，parameter_lamda
# parameter_rho是等价的，以c+f+d为例，parameter_lamda不影响gt总面积，但是会影响区间范围，所以就是parameter_lamda和parameter_theta的balance
# 给定一个 parameter_theta，中间面积就定了，再给定一个 parameter_lamda，区间就定了
# parameter_lamda = 0 的时候，parameter_theta 就对应一个距离，就用这个指导 parameter_theta大概取多少
# 通过改变 parameter_lamda，产生的面积也是确定的，也对应一个parameter_lamda = 0 的时候的距离
# 每个gt里面的距离不一样，所以指导不了

# 也就是parameter_theta，forecasting_window，delay_window的变化关系的图，也就是幂函数逼近，长度变化，不同的幂函数逼近结果变化的图
# parameter_theta = 1/4 # demand
# parameter_w = 1/2 # demand

# VA_gt
parameter_rho = 0 # demand,是个需要根据实际需要指定的参数,演示图
# parameter_rho = 1/2 # demand,是个需要根据实际需要指定的参数,演示图
# parameter_rho = -1 # demand,是个需要根据实际需要指定的参数,演示图

# parameter_lamda = 0# refine
# parameter_lamda = 1/4# refine,演示图,因为标注问题,其实这样1/2更好一些,可以缓解边界标注误差，标注误差不清楚，那就取个中间值1/2
parameter_lamda = 1/2# refine,演示图,因为标注问题,其实这样1/2更好一些,可以缓解边界标注误差，标注误差不清楚，那就取个中间值1/2
# parameter_lamda = 3/4# refine,演示图,因为标注问题,其实这样1/2更好一些,可以缓解边界标注误差，标注误差不清楚，那就取个中间值1/2
# 这里可以做个实验,证明1/2比0更好的缓解了标注误差,找误差数据集

# parameter_switch_f = True # demand
# parameter_switch_d = True # demand

# forecasting_window = 10 # demand,是个需要根据实际需要指定的参数数,演示图
# delay_window = 10 # demand,是个需要根据实际需要指定的参数数,演示图
forecasting_window = 125# demand,是个需要根据实际需要指定的参数数,演示图
delay_window = 125 # demand,是个需要根据实际需要指定的参数数,演示图

# parameter_eta = 1/4 # find
# parameter_eta = 1/2 # find,指导图
parameter_eta = 2**(1/2)-1 # find,指导图

# DQE
parameter_w_gt = 1/3 # gt weight
parameter_w_near_ngt = 1/3 # 近ngt weight
parameter_w_distant_ngt = 1 - parameter_w_gt- parameter_w_near_ngt# 近ngt weight
parameter_w_near_0_ngt_gt_right = 1 / 2 # 近ngt weight
parameter_w_near_2_ngt_gt_left = 1-parameter_w_near_0_ngt_gt_right # 近ngt weight

# DQE_delay
# parameter_w_gt = 1/3 # gt weight
# parameter_w_near_ngt = 1/3 # 近ngt weight
# parameter_w_distant_ngt = 1 - parameter_w_gt- parameter_w_near_ngt# 近ngt weight
# parameter_w_near_0_ngt_gt_right = 1 # 近ngt weight
# parameter_w_near_2_ngt_gt_left = 1-parameter_w_near_0_ngt_gt_right # 近ngt weight

# DQE_early
# parameter_w_gt = 1/3 # gt weight
# parameter_w_near_ngt = 1/3 # 近ngt weight
# parameter_w_distant_ngt = 1 - parameter_w_gt- parameter_w_near_ngt# 近ngt weight
# parameter_w_near_0_ngt_gt_right = 0 # 近ngt weight
# parameter_w_near_2_ngt_gt_left = 1-parameter_w_near_0_ngt_gt_right # 近ngt weight

# DQE_TQ
# parameter_w_gt = 1 # gt weight
# parameter_w_near_ngt = 0 # 近ngt weight
# parameter_w_distant_ngt = 1 - parameter_w_gt- parameter_w_near_ngt# 近ngt weight
# parameter_w_near_0_ngt_gt_right = 1 / 2 # 近ngt weight
# parameter_w_near_2_ngt_gt_left = 1-parameter_w_near_0_ngt_gt_right # 近ngt weight

# DQE_FQ
# parameter_w_gt = 0 # gt weight
# parameter_w_near_ngt = 1/2 # 近ngt weight
# parameter_w_distant_ngt = 1 - parameter_w_gt- parameter_w_near_ngt# 近ngt weight
# parameter_w_near_0_ngt_gt_right = 1 / 2 # 近ngt weight
# parameter_w_near_2_ngt_gt_left = 1-parameter_w_near_0_ngt_gt_right # 近ngt weight

# # DQE_FQ_near
# parameter_w_gt = 0 # gt weight
# parameter_w_near_ngt = 1# 近ngt weight
# parameter_w_distant_ngt = 1 - parameter_w_gt- parameter_w_near_ngt# 近ngt weight
# parameter_w_near_0_ngt_gt_right = 1 / 2 # 近ngt weight
# parameter_w_near_2_ngt_gt_left = 1-parameter_w_near_0_ngt_gt_right # 近ngt weight

# # DQE_FQ_0
# parameter_w_gt = 0 # gt weight
# parameter_w_near_ngt = 1# 近ngt weight
# parameter_w_distant_ngt = 1 - parameter_w_gt- parameter_w_near_ngt# 近ngt weight
# parameter_w_near_0_ngt_gt_right = 1 # 近ngt weight
# parameter_w_near_2_ngt_gt_left = 1-parameter_w_near_0_ngt_gt_right # 近ngt weight

# # DQE_FQ_2
# parameter_w_gt = 0 # gt weight
# parameter_w_near_ngt = 1# 近ngt weight
# parameter_w_distant_ngt = 1 - parameter_w_gt- parameter_w_near_ngt# 近ngt weight
# parameter_w_near_0_ngt_gt_right = 0 # 近ngt weight
# parameter_w_near_2_ngt_gt_left = 1-parameter_w_near_0_ngt_gt_right # 近ngt weight

# # DQE_FQ_distant
# parameter_w_gt = 0 # gt weight
# parameter_w_near_ngt = 0# 近ngt weight
# parameter_w_distant_ngt = 1 - parameter_w_gt- parameter_w_near_ngt# 近ngt weight
# parameter_w_near_0_ngt_gt_right = 1/2 # 近ngt weight
# parameter_w_near_2_ngt_gt_left = 1-parameter_w_near_0_ngt_gt_right # 近ngt weight

# parameter_w_gt = 2/5 # gt weight
# parameter_w_near_ngt = 2/5 # 近ngt weight
# parameter_w_gt = 0.5 # gt weight
# parameter_w_near_ngt = 0.5 # 近ngt weight

# d_onset:d_mean:l = 2:2:1
parameter_near_r_onset = 1 # near d_onset
parameter_near_r_mean = 1 # near d_mean
parameter_near_l = 1/parameter_near_r_onset/parameter_near_r_mean # near l


# print(2**(1/2))
# print(2**(1/3))
# print(2**(1/4))
# print(2**(1/5))
# print(2**(1/5)*2**(1/5))
# print(2**(1/2)*2**(1/2)*2**(-1))
# d_onset:d_mean:l = 1:1:2
parameter_distant_r_onset = 1 # distant d_onset
parameter_distant_r_mean = 1 # distant d_mean
parameter_distant_l = 1/parameter_distant_r_onset/parameter_distant_r_mean # distant l
# 1/2**(1/2)/2**(1/2)

parameter_near_single_side_range = 125 # near_single_side_range,默认用周期，如果有经验，可以设一个经验值

parameter_gama = 2 # gama,太大的画，对detection的临近性要求越高,检测越具有临近性的，得分越高，没有临近性的，得分降低,DQE,DQE_TQ
# parameter_gama = 3 # gama,太大的画，对detection的临近性要求越高,检测越具有临近性的，得分越高，没有临近性的，得分降低

parameter_distant_method = 1 # Sub_MCD distant fp，实验，画mcd的图,DQE,DQE_FQ_distant
# parameter_distant_method = 2 # Sub_MCD distant fp，实验

parameter_ddl_parameter_strategy = 3 # KmeansAD_U,Sub_MCD distant fp,DQE,DQE_FQ_distant
# parameter_ddl_parameter_strategy = 1 # KmeansAD_U,Sub_MCD distant fp

parameter_distant_direction = "both" # Sub_MCD distant fp，实验
# parameter_distant_direction = "left" # Sub_MCD distant fp，实验


parameter_dict = {
    # "parameter_alpha":parameter_alpha,
    "parameter_rate":parameter_rate,

    "parameter_theta":parameter_theta,
    # "parameter_w":parameter_w,
    # "parameter_switch_f": parameter_switch_f,
    # "parameter_switch_d": parameter_switch_d,
    "forecasting_window": forecasting_window,
    "delay_window": delay_window,

    "parameter_rho":parameter_rho,
    "parameter_lamda":parameter_lamda,

    "parameter_eta":parameter_eta,
    
    "parameter_w_gt" : parameter_w_gt, # gt weight
    "parameter_w_near_ngt" : parameter_w_near_ngt, # 近ngt weight
    "parameter_w_near_2_ngt_gt_left" : parameter_w_near_2_ngt_gt_left, # 近ngt weight
    "parameter_w_near_0_ngt_gt_right" : parameter_w_near_0_ngt_gt_right, # 近ngt weight

    "parameter_near_r_onset" : parameter_near_r_onset, # near d_onset
    "parameter_near_r_mean" : parameter_near_r_mean, # near d_mean
    "parameter_near_l" : parameter_near_l, # near d_mean

    "parameter_distant_r_onset" : parameter_distant_r_onset, # distant d_onset
    "parameter_distant_r_mean" : parameter_distant_r_mean, # distant d_mean
    "parameter_distant_l" : parameter_distant_l, # distant d_mean

    "parameter_near_single_side_range" : parameter_near_single_side_range, # near_single_side_range

    "parameter_gama" : parameter_gama, # gama
    "parameter_distant_method" : parameter_distant_method, # gama
    "parameter_distant_direction" : parameter_distant_direction, # gama
    "parameter_ddl_parameter_strategy" : parameter_ddl_parameter_strategy # gama
}