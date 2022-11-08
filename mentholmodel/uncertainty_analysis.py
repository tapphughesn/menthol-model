from simulation import Simulation
import pandas as pd
import numpy as np
import argparse
import os
from datetime import datetime

def draw_mort_params():

    

    return

def main(args):

    now_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    results_dir = f'/Users/nick/Documents/Gillings_work/uncertainty_analysis_data/uncertainty_analysis_{now_str}'

    os.mkdir(results_dir)

    mort_sets_dir = os.path.join(results_dir, 'mortality_parameter_sets')
    init_pop_dir = os.path.join(results_dir, 'initial_populations')
    shortban_param_dir = os.path.join(results_dir, 'short_term_menthol_ban_parameter_sets')
    
    os.mkdir(mort_sets_dir)
    os.mkdir(init_pop_dir)
    os.mkdir(shortban_param_dir)

    print("args:")
    print(args)

    # Get life tables
    # Used for death rates
    # 0 = male
    # 1 = female
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
    smoking_prevalence_dict[2016][0] = pd.read_excel(os.path.join("..","..","smoking_prevalences_Sep13","Smoker_percentage16_M.xlsx")).to_numpy()[:,4::3]
    smoking_prevalence_dict[2017][0] = pd.read_excel(os.path.join("..","..","smoking_prevalences_Sep13","Smoker_percentage17_M.xlsx")).to_numpy()[:,4::3]
    smoking_prevalence_dict[2018][0] = pd.read_excel(os.path.join("..","..","smoking_prevalences_Sep13","Smoker_percentage18_M.xlsx")).to_numpy()[:,4::3]
    smoking_prevalence_dict[2016][1] = pd.read_excel(os.path.join("..","..","smoking_prevalences_Sep13","Smoker_percentage16_F.xlsx")).to_numpy()[:,4::3]
    smoking_prevalence_dict[2017][1] = pd.read_excel(os.path.join("..","..","smoking_prevalences_Sep13","Smoker_percentage17_F.xlsx")).to_numpy()[:,4::3]
    smoking_prevalence_dict[2018][1] = pd.read_excel(os.path.join("..","..","smoking_prevalences_Sep13","Smoker_percentage18_F.xlsx")).to_numpy()[:,4::3]

    # Get Releative Risks for current smokers vs nonsmoker and former smoker vs current smoker
    # According to a published review
    csvnsRR = pd.read_excel(os.path.join("..", "..", "smoking_prevalence", "current_smoker_mortality_vs_nonsmoker.xlsx")).to_numpy()[:,1:]
    fsvcsRR = pd.read_excel(os.path.join("..", "..", "smoking_prevalence", "former_smoker_mortality_vs_current_smoker.xlsx")).to_numpy()[:,1:]

    # Get population data
    # UNCALIBRATED
    pop_file_name = os.path.join("..","..","population_files_Feb8","population_file_sent_Feb8.xlsx")

    pop_df = pd.read_excel(pop_file_name)

    # Get cohorts of 18 year olds
    cohorts_18_dict = {}

    """
    The cohorts dict will take the year corresponding to PATH waves 1, 2, 3
    (2015, 2016, 2017) as an index and return the cohort of 18 yearolds
    for that wave. 
    """

    cohorts_18_dict[2015] = pd.read_excel(os.path.join("..", "..", "corrected_18yo_cohorts", "Wave 2 fresh population profile.xlsx")).to_numpy()
    cohorts_18_dict[2016] = pd.read_excel(os.path.join("..", "..", "corrected_18yo_cohorts", "Wave 3 fresh population profile.xlsx")).to_numpy()
    cohorts_18_dict[2017] = pd.read_excel(os.path.join("..", "..", "corrected_18yo_cohorts", "Wave 4 fresh population profile.xlsx")).to_numpy()

    # Get logistic regression betas
    beta2345_f = os.path.join("..","..","Output_SM","Betas","Beta_Estimates_2345.xlsx")
    beta1_f = os.path.join("..","..","Output_SM","Betas","Beta_Estimates_1.xlsx")
    beta2345_arr = pd.read_excel(beta2345_f).to_numpy()[:,2:]
    beta1_arr = pd.read_excel(beta1_f).to_numpy()[:,2:]

    # our magic smoking percentage for calibration
    # path to the magic file I'm using:
    # the goal is to have a "starting population" with the same smoking rate as that in NHIS
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
                    use_adjusted_death_rates=not args.simple_death_rates,
                    end_year = 2066,
                    menthol_ban=args.menthol_ban,
                    short_term_option=1,
                    long_term_option=5,
                    menthol_ban_year = 2021,
                    target_initial_smoking_proportion=NHIS_smoking_percentage,
                    initiation_rate_decrease=0.055,
                    continuation_rate_decrease=0.055,
                    )
        s.simulate()
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Specify simulation parameters')
    parser.add_argument('num_mortparams', 
                        type=int,
                        default=1,
                        help='the number of sets of mortality parameters (relative risks) to draw')
    parser.add_argument('num_initpops', 
                        type=int,
                        default=1,
                        help='the number of initial populations to create for each mortality parameter draw')
    parser.add_argument('num_shortbanparams', 
                        type=int,
                        default=1,
                        help='the number of short term menthol ban parameters to draw in the case of a menthol ban')
    parser.add_argument('--simple_death_rates', 
                        default=False,
                        action='store_true',
                        help='whether or not to use separate death rates for smokers, nonsmokers, and former smokers')
    parser.add_argument('--menthol_ban', 
                        default=False,
                        action='store_true',
                        help='whether or not to implement a menthol ban at year 10')
    main(parser.parse_args())