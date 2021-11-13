from simulation import Simulation
import pandas as pd
import numpy as np

life_table_dict = {}
life_table_dict[2016] = {}
life_table_dict[2017] = {}
life_table_dict[2018] = {}
life_table_dict[2016][0] = pd.read_excel("../../life_tables/2016/Males/life_table_2016_male.xlsx").to_numpy()[2:-1,1]
life_table_dict[2017][0] = pd.read_excel("../../life_tables/2017/Male/life_table_2017_male.xlsx").to_numpy()[2:-1,1]
life_table_dict[2018][0] = pd.read_excel("../../life_tables/2018/Males/life_table_2018_male.xlsx").to_numpy()[2:-1,1]
life_table_dict[2016][1] = pd.read_excel("../../life_tables/2016/Females/life_table_2016_female.xlsx").to_numpy()[2:-1,1]
life_table_dict[2017][1] = pd.read_excel("../../life_tables/2017/Female/life_table_2017_female.xlsx").to_numpy()[2:-1,1]
life_table_dict[2018][1] = pd.read_excel("../../life_tables/2018/Females/life_table_2018_female.xlsx").to_numpy()[2:-1,1]

pop_file_name = "../../data/age_individual_October21_renamed.xlsx"
pop_df = pd.read_excel(pop_file_name)

beta234_f = "../../beta_estimates/beta_estimates_234.xlsx"
beta15_f = "../../beta_estimates/beta_estimates_15.xlsx"
beta234_arr = pd.read_excel(beta234_f).to_numpy()[:,2:]
beta15_arr = pd.read_excel(beta15_f).to_numpy()[:,2:]

# print("beta234", beta234_arr)
# print("beta15", beta15_arr)

s = Simulation(pop_df=pop_df, 
               beta234=beta234_arr, 
               beta15=beta15_arr, 
               life_tables=life_table_dict,
               save_xl_fname='simulation_status_quo',
               save_np_fname='simulation_status_quo')

s.simulate()
