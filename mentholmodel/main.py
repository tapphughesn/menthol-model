from simulation import Simulation
import pandas as pd
import numpy as np
import argparse
import os

def main(args):
    # 0 = male
    # 1 = female
    print("args:")
    print(args)

    # Get life tables
    # Used for death rates
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
    # Used for death rates
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
    # UNCALIBRATED
    pop_file_name = os.path.join("..","..","population_files_Feb8","population_file_sent_Feb8.xlsx")
    # CALIBRATED
    # pop_file_name = os.path.join("..","..","Calibrated Population","Calibrated Population","PATH_Calibrate_18_64.xlsx")

    pop_df = pd.read_excel(pop_file_name)

    # Get cohorts of 18 year olds
    cohorts_18_dict = {}

    """
    The cohorts dict will take the year corresponding to PATH waves 1, 2, 3
    (2015, 2016, 2017) as an index and return the cohort of 18 yearolds
    for that wave. 
    """

    # UNCALIBRATED
    cohorts_18_dict[2015] = pd.read_excel(os.path.join("..", "..", "corrected_18yo_cohorts", "Wave 2 fresh population profile.xlsx")).to_numpy()
    cohorts_18_dict[2016] = pd.read_excel(os.path.join("..", "..", "corrected_18yo_cohorts", "Wave 3 fresh population profile.xlsx")).to_numpy()
    cohorts_18_dict[2017] = pd.read_excel(os.path.join("..", "..", "corrected_18yo_cohorts", "Wave 4 fresh population profile.xlsx")).to_numpy()

    # CALIBRATED
    # cohorts_18_dict[2015] = pd.read_excel(os.path.join("..", "..", "Calibrated Population", "Calibrated Population", "Wave2_Calibrate_18.xlsx")).to_numpy()
    # cohorts_18_dict[2016] = pd.read_excel(os.path.join("..", "..", "Calibrated Population", "Calibrated Population", "Wave3_Calibrate_18.xlsx")).to_numpy()
    # cohorts_18_dict[2017] = pd.read_excel(os.path.join("..", "..", "Calibrated Population", "Calibrated Population", "Wave4_Calibrate_18.xlsx")).to_numpy()

    # Get logistic regression betas
    beta2345_f = os.path.join("..","..","Output_SM","Betas","Beta_Estimates_2345.xlsx")
    beta1_f = os.path.join("..","..","Output_SM","Betas","Beta_Estimates_1.xlsx")
    beta2345_arr = pd.read_excel(beta2345_f).to_numpy()[:,2:]
    beta1_arr = pd.read_excel(beta1_f).to_numpy()[:,2:]

    # check initial smoking rate

    # print(pop_df.columns)
    # pop_arr = pop_df.to_numpy()
    # print(np.unique(pop_arr[:,4]))
    # print(
    #     # np.sum(np.int64(((pop_arr[:,4] == 3) + (pop_arr[:,4] == 4) + (pop_arr[:,4] == 5)) > 0) * pop_arr[:,8]) / np.sum(pop_arr[:,8])
    #     np.sum(np.int64(((pop_arr[:,4] == 3) + (pop_arr[:,4] == 4)) > 0) * pop_arr[:,8]) / np.sum(pop_arr[:,8])
    # )
    # print(pop_arr[0,:])
    # quit()

    # our magic smoking percentage for calibration
    # path to the magic file I'm using:
    # /Users/nick/Documents/Gillings_work/nhis_data/NHIS_smoker_proportions./NHIS_State_age/NHIS_State_age/NHIS_state_age18_64.xlsx

    NHIS_smoking_percentage = 0.151316

    for _ in range(args.number_replications):
        s = Simulation(pop_df=pop_df, 
                    beta2345=beta2345_arr, 
                    beta1=beta1_arr, 
                    life_tables=life_table_dict,
                    cohorts=cohorts_18_dict,
                    smoking_prevalences=smoking_prevalence_dict,
                    current_smoker_RR=csvnsRR,
                    former_smoker_RR=fsvcsRR,
                    save_xl_fname='xl_output_calibrated',
                    save_np_fname='np_output_calibrated',
                    save_transition_np_fname='transitions_calibrated',
                    use_adjusted_death_rates=args.complex_death_rates,
                    end_year = 2066,
                    menthol_ban=args.menthol_ban,
                    short_term_option=1,
                    long_term_option=1,
                    menthol_ban_year = 2021,
                    target_initial_smoking_proportion=NHIS_smoking_percentage,
                    initiation_rate_decrease=0.0,
                    )
        s.simulate()
    
    # for i in range(101):
    # # for i in range(4):
    # #     for j in range(4):
    #         # print(i+1, j+1)
    #     print(i)
    #     i_str = str(i)
    #     while len(i_str) < 3:
    #         i_str = "0" + i_str
    #     assert(len(i_str) == 3)
    #     s = Simulation(pop_df=pop_df, 
    #                 beta2345=beta2345_arr, 
    #                 beta1=beta1_arr, 
    #                 life_tables=life_table_dict,
    #                 cohorts=cohorts_18_dict,
    #                 # cohort_adding_pattern=cohort_adding_pattern,
    #                 smoking_prevalences=smoking_prevalence_dict,
    #                 current_smoker_RR=csvnsRR,
    #                 former_smoker_RR=fsvcsRR,
    #                 # save_xl_fname=f'xl_output{2021}',
    #                 # save_xl_fname='xl_output_calibrated',
    #                 save_xl_fname=f'xl_output_calibrated_to_NHIS_' + i_str,
    #                 # save_np_fname=f'np_output_ban{2021}',
    #                 # save_np_fname='np_output_calibrated',
    #                 save_np_fname=f'np_output_calibrated_to_NHIS_' + i_str,
    #                 # save_transition_np_fname=f'transitions_ban{2021}',
    #                 # save_transition_np_fname='transitions_calibrated',
    #                 save_transition_np_fname=f'transitions_calibrated_' + i_str,
    #                 use_adjusted_death_rates=args.complex_death_rates,
    #                 end_year = 2066,
    #                 menthol_ban=args.menthol_ban,
    #                 # short_term_option=i+1,
    #                 short_term_option=1,
    #                 # long_term_option=j+1,
    #                 long_term_option=1,
    #                 menthol_ban_year = 2021,
    #                 initiation_rate_decrease=i/100,
    #                 )
    #     s.simulate()

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