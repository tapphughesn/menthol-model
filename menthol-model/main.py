from simulation import Simulation
import pandas as pd
import numpy as np
import argparse
import os

def main(args):
    # 0 = male
    # 1 = female
    print(args)

    # Get life tables
    life_table_dict = {}
    life_table_dict[2016] = {}
    life_table_dict[2017] = {}
    life_table_dict[2018] = {}
    life_table_dict[2016][0] = pd.read_excel(os.path.join("..","..","life_tables","2016","Males","life_table_2016_male.xlsx")).to_numpy()[2:-1,1]
    life_table_dict[2017][0] = pd.read_excel(os.path.join("..","..","life_tables","2017","Males","life_table_2017_male.xlsx")).to_numpy()[2:-1,1]
    life_table_dict[2018][0] = pd.read_excel(os.path.join("..","..","life_tables","2018","Males","life_table_2018_male.xlsx")).to_numpy()[2:-1,1]
    life_table_dict[2016][1] = pd.read_excel(os.path.join("..","..","life_tables","2016","Females","life_table_2016_female.xlsx")).to_numpy()[2:-1,1]
    life_table_dict[2017][1] = pd.read_excel(os.path.join("..","..","life_tables","2017","Females","life_table_2017_female.xlsx")).to_numpy()[2:-1,1]
    life_table_dict[2018][1] = pd.read_excel(os.path.join("..","..","life_tables","2018","Females","life_table_2018_female.xlsx")).to_numpy()[2:-1,1]

    # Get smoking prevalences by age and sex
    smoking_prevalence_dict = {}
    smoking_prevalence_dict[2016] = {}
    smoking_prevalence_dict[2017] = {}
    smoking_prevalence_dict[2018] = {}
    smoking_prevalence_dict[2016][0] = pd.read_excel(os.path.join("..","..","smoking_prevalences_Feb9","Smoker_percentage16_M.xlsx")).to_numpy()[:,4::3]
    smoking_prevalence_dict[2017][0] = pd.read_excel(os.path.join("..","..","smoking_prevalences_Feb9","Smoker_percentage17_M.xlsx")).to_numpy()[:,4::3]
    smoking_prevalence_dict[2018][0] = pd.read_excel(os.path.join("..","..","smoking_prevalences_Feb9","Smoker_percentage18_M.xlsx")).to_numpy()[:,4::3]
    smoking_prevalence_dict[2016][1] = pd.read_excel(os.path.join("..","..","smoking_prevalences_Feb9","Smoker_percentage16_F.xlsx")).to_numpy()[:,4::3]
    smoking_prevalence_dict[2017][1] = pd.read_excel(os.path.join("..","..","smoking_prevalences_Feb9","Smoker_percentage17_F.xlsx")).to_numpy()[:,4::3]
    smoking_prevalence_dict[2018][1] = pd.read_excel(os.path.join("..","..","smoking_prevalences_Feb9","Smoker_percentage18_F.xlsx")).to_numpy()[:,4::3]

    # Get Releative Risks for current smokers vs nonsmoker and former smoker vs current smoker
    # According to a published review
    csvnsRR = pd.read_excel(os.path.join("..", "..", "smoking_prevalence", "current_smoker_mortality_vs_nonsmoker.xlsx")).to_numpy()[:,1:]
    fsvcsRR = pd.read_excel(os.path.join("..", "..", "smoking_prevalence", "former_smoker_mortality_vs_current_smoker.xlsx")).to_numpy()[:,1:]

    # Get population data
    pop_file_name = os.path.join("..","..","population_files_Feb8","population_file_sent_Feb8.xlsx")
    # Use calibration population data
    # pop_file_name = os.path.join("..","..","Calibrated Population","Calibrated Population","PATH_Calibrate_18_64.xlsx")

    pop_df = pd.read_excel(pop_file_name)

    # Get cohorts of 18 year olds
    cohorts_18_dict = {}

    # cohorts_18_dict[2016] = pd.read_excel(os.path.join("..", "..", "Output_SM", "Cohort 18 years", "Wave 2 fresh population profile.xlsx")).to_numpy()
    # cohorts_18_dict[2017] = pd.read_excel(os.path.join("..", "..", "Output_SM", "Cohort 18 years", "Wave 3 fresh population profile.xlsx")).to_numpy()
    # cohorts_18_dict[2018] = pd.read_excel(os.path.join("..", "..", "Output_SM", "Cohort 18 years", "Wave 4 fresh population profile.xlsx")).to_numpy()

    cohorts_18_dict[2016] = pd.read_excel(os.path.join("..", "..", "corrected_18yo_cohorts", "Wave 2 fresh population profile.xlsx")).to_numpy()
    cohorts_18_dict[2017] = pd.read_excel(os.path.join("..", "..", "corrected_18yo_cohorts", "Wave 3 fresh population profile.xlsx")).to_numpy()
    cohorts_18_dict[2018] = pd.read_excel(os.path.join("..", "..", "corrected_18yo_cohorts", "Wave 4 fresh population profile.xlsx")).to_numpy()

    # cohorts_18_dict[2016] = pd.read_excel(os.path.join("..", "..", "Calibrated Population", "Calibrated Population", "Wave2_Calibrate_18.xlsx")).to_numpy()
    # cohorts_18_dict[2017] = pd.read_excel(os.path.join("..", "..", "Calibrated Population", "Calibrated Population", "Wave3_Calibrate_18.xlsx")).to_numpy()
    # cohorts_18_dict[2018] = pd.read_excel(os.path.join("..", "..", "Calibrated Population", "Calibrated Population", "Wave4_Calibrate_18.xlsx")).to_numpy()

    cohort_adding_pattern = [2,1,1,1,1,1,1,1,1,1]

    # Get logistic regression betas
    beta2345_f = os.path.join("..","..","Output_SM","Betas","Beta_Estimates_2345.xlsx")
    beta1_f = os.path.join("..","..","Output_SM","Betas","Beta_Estimates_1.xlsx")
    beta2345_arr = pd.read_excel(beta2345_f).to_numpy()[:,2:]
    beta1_arr = pd.read_excel(beta1_f).to_numpy()[:,2:]

    # for i in range(args.number_replications):
    for i in range(4):
        for j in range(4):
            print(i+1, j+1)
            s = Simulation(pop_df=pop_df, 
                        beta2345=beta2345_arr, 
                        beta1=beta1_arr, 
                        life_tables=life_table_dict,
                        cohorts=cohorts_18_dict,
                        cohort_adding_pattern=cohort_adding_pattern,
                        smoking_prevalences=smoking_prevalence_dict,
                        current_smoker_RR=csvnsRR,
                        former_smoker_RR=fsvcsRR,
                        save_xl_fname=f'xl_output{2021}',
                        # save_xl_fname='xl_output_calibrated',
                        save_np_fname=f'np_output_ban{2021}',
                        # save_np_fname='np_output_calibrated',
                        save_transition_np_fname=f'transitions_ban{2021}',
                        # save_transition_np_fname='transitions_calibrated',
                        use_adjusted_death_rates=args.complex_death_rates,
                        end_year = 2066,
                        menthol_ban=args.menthol_ban,
                        short_term_option=i+1,
                        # short_term_option=1,
                        long_term_option=j+1,
                        # long_term_option=1,
                        menthol_ban_year = 2021,
                        )

            s.simulate()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Specify simulation parameters')
    parser.add_argument('number_replications', 
                        type=int,
                        default=1,
                        help='the number of relplications to do')
    parser.add_argument('--complex_death_rates', 
                        # type=bool,
                        default=False,
                        action='store_true',
                        help='whether or not to use separate death rates for smokers, nonsmokers, and former smokers')
    parser.add_argument('--menthol_ban', 
                        # type=bool,
                        default=False,
                        action='store_true',
                        help='whether or not to implement a menthol ban at year 10')
    main(parser.parse_args())