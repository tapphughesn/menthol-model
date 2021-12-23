from simulation import Simulation
import pandas as pd
import numpy as np
import argparse
import os

def main(args):
    # 0 = male
    # 1 = female

    # Get life tables
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

    # Get smoking prevalences by wave for ages 55-90
    smoking_prevalence_dict = {}
    smoking_prevalence_dict[2016] = pd.read_excel(os.path.join("..", "..", "smoking_prevalence", "Wave345_age55up_smokers_SM Dec21.xlsx")).to_numpy()[:,3]
    smoking_prevalence_dict[2017] = pd.read_excel(os.path.join("..", "..", "smoking_prevalence", "Wave345_age55up_smokers_SM Dec21.xlsx")).to_numpy()[:,6]
    smoking_prevalence_dict[2018] = pd.read_excel(os.path.join("..", "..", "smoking_prevalence", "Wave345_age55up_smokers_SM Dec21.xlsx")).to_numpy()[:,9]

    # Get Releative Risks for current smokers vs nonsmoker and former smoker vs current smoker
    csvnsRR = pd.read_excel(os.path.join("..", "..", "smoking_prevalence", "current_smoker_mortality_vs_nonsmoker.xlsx")).to_numpy()[:,1:]
    fsvcsRR = pd.read_excel(os.path.join("..", "..", "smoking_prevalence", "former_smoker_mortality_vs_current_smoker.xlsx")).to_numpy()[:,1:]

    # print(smoking_prevalence_dict[2016].shape)
    # print(csvnsRR)
    # print(fsvcsRR)
    # quit()

    # Get population data
    pop_file_name = "../../path_data/age_individual_October21_renamed.xlsx"
    pop_df = pd.read_excel(pop_file_name)

    # Get logistic regression betas
    beta234_f = "../../beta_estimates/beta_estimates_234.xlsx"
    beta15_f = "../../beta_estimates/beta_estimates_15.xlsx"
    beta234_arr = pd.read_excel(beta234_f).to_numpy()[:,2:]
    beta15_arr = pd.read_excel(beta15_f).to_numpy()[:,2:]

    for i in range(args.number_replications):
        s = Simulation(pop_df=pop_df, 
                    beta234=beta234_arr, 
                    beta15=beta15_arr, 
                    life_tables=life_table_dict,
                    smoking_prevalences=smoking_prevalence_dict,
                    current_smoker_RR=csvnsRR,
                    former_smoker_RR=fsvcsRR,
                    save_xl_fname='transitions',
                    save_np_fname='transitions',
                    save_transition_np_fname='transitions',
                    )

        s.simulate()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Specify simulation parameters')
    parser.add_argument('number_replications', 
                        type=int,
                        default=1,
                        help='the number of relplications to do')
    main(parser.parse_args())