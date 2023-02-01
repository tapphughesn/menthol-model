from simulation import Simulation
import pandas as pd
import numpy as np
import argparse
import os
from datetime import datetime
import math


def main(args):

    start = datetime.now()
    print(f"analysing timestamp: {args.timestamp}")
    results_dir = f'../../uncertainty_analysis_data/uncertainty_analysis_{args.timestamp}'

    print("args:")
    print(args)

    if not os.path.isdir(results_dir):
        raise NotADirectoryError

    mort_sets_dir = os.path.join(results_dir, 'mortality_parameter_sets')
    init_pop_dir = os.path.join(results_dir, 'initial_populations')
    shortban_param_dir = os.path.join(results_dir, 'short_term_menthol_ban_parameter_sets')
    longban_param_dir = os.path.join(results_dir, 'long_term_menthol_ban_parameter_sets')
    output_dir = os.path.join(results_dir, 'outputs')

    # create longban parameter directories for each long-term scenario (of which there are 4)
    longban_options_dirs = [os.path.join(longban_param_dir, f'option_{i}') for i in range(1,5)]

    # create initial population directory for each of the mortality parameter sets
    init_pop_dirs = []
    for i in range(args.num_mortparams):
        num_i_digits = math.ceil(math.log10(args.num_mortparams))
        i_str = str(i)
        while len(i_str) < num_i_digits:
            i_str = "0" + i_str
        path = os.path.join(init_pop_dir, f"mortparam_set_{i_str}")
        init_pop_dirs.append(path)

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    # create output dir for this option
    output_dir = os.path.join(output_dir, f"option_{args.ban_option}")

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    
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

    for i in range(args.num_mortparams):
        num_i_digits = math.ceil(math.log10(args.num_mortparams))
        i_str = str(i)
        while len(i_str) < num_i_digits:
            i_str = "0" + i_str
        
        # get the mortality parameters
        this_csvns_sampling = np.load(os.path.join(mort_sets_dir, f"set_{i_str}_csvns.npy"))
        this_fsvcs_sampling = np.load(os.path.join(mort_sets_dir, f"set_{i_str}_fsvcs.npy"))

        # now for each mortality parameter set, get the initial population
        this_init_pop_dir = init_pop_dirs[i]
        for j in range(args.num_initpops):
            num_j_digits = math.ceil(math.log10(args.num_initpops))
            j_str = str(j)
            while len(j_str) < num_j_digits:
                j_str = "0" + j_str

            arr1 = np.load(os.path.join(this_init_pop_dir, f'mort_{i_str}_pop_{j_str}_arr1.npy'))
            arr6 = np.load(os.path.join(this_init_pop_dir, f'mort_{i_str}_pop_{j_str}_arr6.npy'))
            arr2345 = np.load(os.path.join(this_init_pop_dir, f'mort_{i_str}_pop_{j_str}_arr2345.npy'))

            for k in range(args.num_banparams):
                if args.second_half:
                    k += args.num_banparams // 2
                    if k >= args.num_banparams:
                        break

                num_k_digits = math.ceil(math.log10(args.num_banparams))
                k_str = str(k)
                while len(k_str) < num_k_digits:
                    k_str = "0" + k_str

                savename = os.path.join(output_dir, f'mort_{i_str}_pop_{j_str}_banparams_{k_str}_output.npy')
                if os.path.isfile(savename):
                    continue

                if args.ban_option == 0:
                    # status quo scenario, do simulations without menthol ban
                    t = Simulation(pop_df=pop_df, 
                        beta2345=beta2345_arr, 
                        beta1=beta1_arr, 
                        life_tables=life_table_dict,
                        cohorts=cohorts_18_dict,
                        smoking_prevalences=smoking_prevalence_dict,
                        current_smoker_RR=this_csvns_sampling,
                        former_smoker_RR=this_fsvcs_sampling,
                        save_xl_fname='xl_output_calibrated',
                        save_np_fname='np_output_calibrated',
                        save_transition_np_fname='transitions_calibrated',
                        use_adjusted_death_rates=not args.simple_death_rates,
                        end_year = 2066,
                        menthol_ban=False,
                        menthol_ban_year = 2021,
                        target_initial_smoking_proportion=NHIS_smoking_percentage,
                        initiation_rate_decrease=0.055,
                        continuation_rate_decrease=0.055,
                        )
                    
                    beta_2345_aug, beta_1_aug = t.get_augmented_betas()
                    t.arr1, t.arr2345, t.arr6 = np.copy(arr1), np.copy(arr2345), np.copy(arr6)

                    t.simulation_loop(beta_1_aug, beta_2345_aug)

                    np.save(savename, t.output_numpy)
                    
                    progress = i/args.num_mortparams + j/args.num_initpops/args.num_mortparams + k/args.num_banparams/args.num_initpops/args.num_mortparams
                    seconds_since_start = int((datetime.now() - start).total_seconds())
                    print(f"mort: {i_str}, initpop: {j_str}, ban params: {k_str}, {np.around(progress * 100, decimals=3)}% done, {seconds_since_start} seconds elapsed.")
                else:
                    # ban scenario, so load the ban parameters

                    # load ban params
                    shortbanparams = np.load(os.path.join(shortban_param_dir, f"set_{k_str}_shortbanparams.npy"))
                    longbanparams = np.load(os.path.join(longban_options_dirs[args.ban_option - 1], f"option_{args.ban_option}_set_{k_str}_longbanparams.npy"))

                    t = Simulation(pop_df=pop_df, 
                        beta2345=beta2345_arr, 
                        beta1=beta1_arr, 
                        life_tables=life_table_dict,
                        cohorts=cohorts_18_dict,
                        smoking_prevalences=smoking_prevalence_dict,
                        current_smoker_RR=this_csvns_sampling,
                        former_smoker_RR=this_fsvcs_sampling,
                        save_xl_fname='xl_output_calibrated',
                        save_np_fname='np_output_calibrated',
                        save_transition_np_fname='transitions_calibrated',
                        use_adjusted_death_rates=not args.simple_death_rates,
                        end_year = 2066,
                        menthol_ban=True,
                        menthol_ban_year = 2021,
                        target_initial_smoking_proportion=NHIS_smoking_percentage,
                        initiation_rate_decrease=0.055,
                        continuation_rate_decrease=0.055,
                        )
                    
                    beta_2345_aug, beta_1_aug = t.get_augmented_betas()
                    t.arr1, t.arr2345, t.arr6 = np.copy(arr1), np.copy(arr2345), np.copy(arr6)

                    t.simulation_loop(beta_1_aug, beta_2345_aug, shortbanparams=shortbanparams, longbanparams=longbanparams)

                    np.save(savename, t.output_numpy)
                    
                    progress = i/args.num_mortparams + j/args.num_initpops/args.num_mortparams + k/args.num_banparams/args.num_initpops/args.num_mortparams
                    seconds_since_start = int((datetime.now() - start).total_seconds())
                    print(f"mort: {i_str}, initpop: {j_str}, ban params: {k_str}, {np.around(progress * 100, decimals=3)}% done, {seconds_since_start} seconds elapsed.")
                 #endif
            #endfor
        #endfor
    #endfor

    print("finished!")


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
    parser.add_argument('num_banparams', 
                        type=int,
                        default=1,
                        help='the number of short term menthol ban parameters to draw in the case of a menthol ban')
    parser.add_argument('ban_option', 
                        type=int,
                        default=0,
                        help='which long-term ban option to use (0 = status quo)')
    parser.add_argument('timestamp', 
                        type=str,
                        default='',
                        help='timestamp of uncertainty analysis directory we are working in')
    parser.add_argument('--second_half', 
                        default=False,
                        action='store_true',
                        help='whether or not to just focus on the second half of the runs (start with ban paramset 50/100)')
    parser.add_argument('--simple_death_rates', 
                        default=False,
                        action='store_true',
                        help='whether or not to use separate death rates for smokers, nonsmokers, and former smokers')
    main(parser.parse_args())