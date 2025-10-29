from TSB_AD.model_wrapper import *
from TSB_AD.HP_list import *

# list_a = ["a","b","c"]
# list_b = ["a"]
# set1 = set(list_a)
# set2 = set(list_b)
# difference_set = set1 - set2
# print(list(difference_set))

print(len(Unsupervise_AD_Pool))
print(Unsupervise_AD_Pool)
filter_set = set(Unsupervise_AD_Pool) - set(exclude_ad_pool)
filter_pool = list(filter_set)
print(len(filter_pool))
print(filter_pool)

not_have_optimal_set = filter_set - set(Optimal_Uni_algo_HP_dict.keys())
print(not_have_optimal_set)
print(len(Optimal_Multi_algo_HP_dict.keys()))
print(len(filter_set - not_have_optimal_set))

# dict_a = {"a":"A","b":"B"}
# print(dict_a.items())
# if "a" in dict_a.keys():
#     print(True)
