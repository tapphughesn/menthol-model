import numpy as np
import pandas as pd
from simulation_helpers import path_to_indicator_form
from calibrate_smoking_perc import calibrate_smoking_percentage

#######
# SETUP
#######

uncalibrated_population_file = "/Users/nick/Documents/Gillings_work/population_files_Feb8/population_file_sent_Feb8.xlsx"
target_smoking_percentage = 0.15

pop_df = pd.read_excel(uncalibrated_population_file)
pop_arr = pop_df.to_numpy(dtype = np.float64)


arr2345 = np.asarray([row for row in pop_arr 
            if (row[4] == 2 or row[4] == 3 or row[4] == 4 or row[4] == 5
            or row[3] == 2 or row[3] == 3 or row[3] == 4 or row[3] == 5)], dtype=np.float64)
arr1 = np.asarray([row for row in pop_arr 
            if (row[4] == 1 and row[3] == 1)], dtype=np.float64)

arr2345 = path_to_indicator_form(arr2345)
arr1 = path_to_indicator_form(arr1)

start_year = 2016
age_last_smoked_for_ia1 = 17

# Here we figure out the year_last_smoked variable for all cases

# for people whose last state is 3,4 the year last smoked is self.start_year - 1
arr2345[np.logical_or(arr2345[:,3],arr2345[:,4]),16] = start_year - 1

# for people currently in groups 3,4 the year last smoked is self.start_year
arr2345[np.logical_or(arr2345[:,6],arr2345[:,7]),16] = start_year

# for people whose last state is 5, the year last smoked is self.start_year - 1
# we are treating ecig users the same as smokers here
arr2345[np.sum(arr2345[:,1:5], axis=1) == 0,16] = start_year

# for people whose current state is 5, the year last smoked is self.start_year
arr2345[np.sum(arr2345[:,5:8], axis=1) == 0,16] = start_year

# for people in group 2 last state AND this state
# if initialization age is 1 then year last smoked is self.year_last_smoked_for_ia1 + self.start_year - age
ind = np.logical_and(arr2345[:,2], arr2345[:,5], arr2345[:,8]).astype(np.bool_)
arr2345[ind,16] = age_last_smoked_for_ia1 + start_year - arr2345[ind, 11]

# if initialization age is 2 for former smokers then year last smoked is randomly chosen between start_age and current age
ind = np.logical_and(arr2345[:,2], arr2345[:,5], arr2345[:,9]).astype(np.bool_)
age_started = np.maximum(18, arr2345[ind,14]) # use starting age if available, otherwise use 18
to_multiply_rand = arr2345[ind, 11] - age_started + 1 - 1e-8
to_add_after_multiply = start_year - arr2345[ind, 11] - 0.5 + 1e-8
arr2345[ind ,16] = np.round(np.random.rand(np.sum(ind)) * to_multiply_rand + to_add_after_multiply)

##########
# NOW TEST
##########

arr1_calib, arr2345_calib = calibrate_smoking_percentage(arr1, arr2345, 0.15)

smoker_ind = np.sum(arr2345[:,6:8], axis=1, dtype=bool)
new_smoker_weight = np.sum(arr2345_calib[:,15][smoker_ind])
new_nonsmoker_weight = np.sum(arr1[:,15]) + np.sum(arr2345[:,15][np.logical_not(smoker_ind)])
new_total_weight = np.sum(arr1[:,15]) + np.sum(arr2345[:,15])

print("smoker weight", new_smoker_weight)
print("nonsmoker weight", new_nonsmoker_weight)
print("proportion", new_smoker_weight / new_total_weight)
