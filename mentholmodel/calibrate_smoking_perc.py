from hypothesis import target
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


savedir = "/Users/nick/Documents/Gillings_work/nick_calibrated_population"
uncalibrated_population_file = "/Users/nick/Documents/Gillings_work/population_files_Feb8/population_file_sent_Feb8.xlsx"
target_smoking_percentage = 0.15

pop_df = pd.read_excel(uncalibrated_population_file)
# print(pop_df.columns)

pop_arr = pop_df.to_numpy(dtype=np.float64)

# print(np.int64(pop_arr[:5,:]))

state_3_arr = pop_arr[:,4]
weights_arr = pop_arr[:,8]

smokers_arr = np.logical_or(state_3_arr == 3, state_3_arr == 4) # smokers only in states 3 and 4, not ecig/dual
non_smokers_arr = (1 - smokers_arr) == 1 # convert from int arr to bool arr

assert(np.sum(smokers_arr) + np.sum(non_smokers_arr) == len(pop_arr))

print(np.sum(weights_arr[smokers_arr]))
print(np.sum(weights_arr[non_smokers_arr]))
print(np.sum(weights_arr[smokers_arr]) / np.sum(weights_arr))

total_weight = np.sum(weights_arr)

smoker_weight_factor = total_weight * target_smoking_percentage / np.sum(weights_arr[smokers_arr])
non_smoker_weight_factor = (total_weight - total_weight * target_smoking_percentage) / np.sum(weights_arr[non_smokers_arr])

print("---------------")
print(smoker_weight_factor)
print(non_smoker_weight_factor)

new_weights_arr = np.copy(weights_arr)
new_weights_arr[smokers_arr] *= smoker_weight_factor
new_weights_arr[non_smokers_arr] *= non_smoker_weight_factor

print('-------------')
print(np.sum(new_weights_arr[smokers_arr]))
print(int(np.sum(new_weights_arr[non_smokers_arr])))
print(np.sum(new_weights_arr))
print(np.sum(new_weights_arr[smokers_arr]) / np.sum(weights_arr))
