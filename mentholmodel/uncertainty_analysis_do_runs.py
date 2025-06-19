"""
This script takes the uncertainty analysis params that were generated in stage 1 and
uses them to do many runs of the simulation with different combinations of parameters.

These are the categories of parameters:
    1. Mortality params: relative risks of death for different smoking states
    2. Short-term ban params: one-year effects of the menthol ban
    3. Long-term ban params: year-over-year effects of the menthol ban
    4. Initial populations: the starting population (2016 population)

During the creation of the the parameters, these sets of starting parameters were
randomly sampled from an appropriate distribution.

We also have 6 different ban scenarios (5 sets of long-term parameters and 1 status quo scenario).

For a given ban scenario, the uncertainty analysis might look like this:
    1. Generate 25 sets of mort params, 25 initial pops, 25 

"""
from simulation import Simulation
import pandas as pd
import numpy as np
import argparse
import os
from datetime import datetime
import math
from glob import glob
import debugpy
from time import sleep

def int_to_str(i: int, max: int) -> str:
    """
    Takes an integer and converts it into a string.
    Length of the string depends on max,
    so that all integers will have the same string length.
    """

    num_i_digits = math.ceil(math.log10(max) + np.finfo(float).eps)
    i_str = str(i)
    if len(i_str) > num_i_digits: raise Exception("integer is too big for given max, need to increase the max")
    while len(i_str) < num_i_digits:
        i_str = "0" + i_str
    return i_str


def main(args):
    # debugpy.listen(("localhost", 5678))
    # print("Waiting for debugger to attach...")
    # debugpy.wait_for_client()
    
    start = datetime.now()
    print(f"analysing timestamp: {args.timestamp}")
    results_dir = f'../../uncertainty_analysis_data/uncertainty_analysis_{args.timestamp}' # params are here

    print("args:")
    print(args)

    if not os.path.isdir(results_dir):
        print(results_dir)
        raise NotADirectoryError

    mort_sets_dir = os.path.join(results_dir, 'mortality_parameter_sets')
    init_pop_dir = os.path.join(results_dir, 'initial_populations')
    shortban_param_dir = os.path.join(results_dir, 'short_term_menthol_ban_parameter_sets')
    longban_param_dir = os.path.join(results_dir, 'long_term_menthol_ban_parameter_sets')
    output_dir = os.path.join(results_dir, 'outputs')
    disease_output_dir = os.path.join(results_dir, 'disease_modeling_outputs')
    LYL_output_dir = os.path.join(results_dir, 'LYL_outputs')

    # create longban parameter directories for each long-term scenario
    longban_options_dirs = sorted(glob(os.path.join(longban_param_dir, f'option_*')))

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    if not os.path.isdir(disease_output_dir):
        os.mkdir(disease_output_dir)
    if not os.path.isdir(LYL_output_dir):
        os.mkdir(LYL_output_dir)

    # create output dir for this option, rewriting the variable output_dir
    output_dir = os.path.join(output_dir, f"option_{args.ban_option}")
    disease_output_dir = os.path.join(disease_output_dir, f"option_{args.ban_option}")
    LYL_output_dir = os.path.join(LYL_output_dir, f"option_{args.ban_option}")

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    if not os.path.isdir(disease_output_dir):
        os.mkdir(disease_output_dir)
    if not os.path.isdir(LYL_output_dir):
        os.mkdir(LYL_output_dir)
    
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
    postban_18yo_cohort = pd.read_excel(os.path.join("..", "..", "corrected_18yo_cohorts", "postban population profile.xlsx")).to_numpy()

    # Get logistic regression betas
    beta2345_f = os.path.join("..","..","Output_SM","Betas","Beta_Estimates_2345.xlsx")
    beta1_f = os.path.join("..","..","Output_SM","Betas","Beta_Estimates_1.xlsx")
    beta2345_arr = pd.read_excel(beta2345_f).to_numpy()[:,2:]
    beta1_arr = pd.read_excel(beta1_f).to_numpy()[:,2:]

    # our magic smoking percentage for calibration
    # path to the magic file I'm using:
    # the goal is to have a "starting population" with the same smoking rate as that in NHIS
    # /Users/nick/Documents/Gillings_work/nhis_data/NHIS_smoker_proportions./NHIS_State_age/NHIS_State_age/NHIS_state_age18_64.xlsx

    NHIS_smoking_percentage = 0.151316 # this smoking rate comes from the NHIS

    for i in range(args.num_mortparams):
        i_str = int_to_str(i, args.num_mortparams)
        
        # get the mortality parameters
        this_csvns_sampling = np.load(os.path.join(mort_sets_dir, f"set_{i_str}_csvns.npy"))
        this_fsvcs_sampling = np.load(os.path.join(mort_sets_dir, f"set_{i_str}_fsvcs.npy"))

        # now for each mortality parameter set, get the initial population
        for j in range(args.num_initpops):
            j_str = int_to_str(j, args.num_initpops)

            arr1 = np.load(os.path.join(init_pop_dir, f'pop_{j_str}_arr1.npy'))
            arr2345 = np.load(os.path.join(init_pop_dir, f'pop_{j_str}_arr2345.npy'))
            arr6 = np.load(os.path.join(init_pop_dir, f'pop_{j_str}_arr6.npy'))
            arr6_noncohort = np.load(os.path.join(init_pop_dir, f'pop_{j_str}_arr6_noncohort.npy'))

            for k in range(args.num_banparams):
                # do second half 
                if args.second_half:
                    k += args.num_banparams // 2
                    if k >= args.num_banparams:
                        break

                k_str = int_to_str(k, args.num_banparams)

                savename = os.path.join(output_dir, f'mort_{i_str}_pop_{j_str}_banparams_{k_str}_output.npy')

                sleep(np.random.rand() / 1000) # avoids race conditions?
                if os.path.isfile(savename):
                    continue

                disease_savename_cvd = os.path.join(disease_output_dir, f'mort_{i_str}_pop_{j_str}_banparams_{k_str}_disease_output_cvd.npy' )
                disease_savename_lc = os.path.join(disease_output_dir, f'mort_{i_str}_pop_{j_str}_banparams_{k_str}_disease_output_lc.npy' )
                disease_savename_total = os.path.join(disease_output_dir, f'mort_{i_str}_pop_{j_str}_banparams_{k_str}_disease_output_total.npy' )

                LYL_savename = os.path.join(LYL_output_dir, f'mort_{i_str}_pop_{j_str}_banparams_{k_str}_LYL_output.npy' )

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
                        save_disease_np_fname='disease_incidence_output',
                        save_LYL_np_fname='live_years_output',
                        use_adjusted_death_rates=not args.simple_death_rates,
                        end_year = 2126,
                        menthol_ban=False,
                        menthol_ban_year = 2024,
                        target_initial_smoking_proportion=NHIS_smoking_percentage,
                        initiation_rate_decrease=0.055,
                        continuation_rate_decrease=0.055,
                        simulate_disease=args.simple_death_rates, # we want to simulate disease only when doing simple death rates
                        postban_18yo_cohort=postban_18yo_cohort,
                        )
                    
                    beta_2345_aug, beta_1_aug = t.get_augmented_betas()
                    t.arr1, t.arr2345, t.arr6, t.arr6_noncohort = np.copy(arr1), np.copy(arr2345), np.copy(arr6), np.copy(arr6_noncohort)

                    t.simulation_loop(beta_1_aug, beta_2345_aug)

                    np.save(savename, t.output_numpy)
                    
                    np.save(disease_savename_cvd, t.output_cvd)
                    np.save(disease_savename_lc, t.output_lc)
                    np.save(disease_savename_total, t.output_65yos)

                    np.save(LYL_savename, t.output_LYL)

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
                        save_disease_np_fname='disease_incidence_output',
                        save_LYL_np_fname='live_years_output',
                        use_adjusted_death_rates=not args.simple_death_rates,
                        end_year = 2126,
                        menthol_ban=True,
                        menthol_ban_year = 2024,
                        target_initial_smoking_proportion=NHIS_smoking_percentage,
                        initiation_rate_decrease=0.055,
                        continuation_rate_decrease=0.055,
                        simulate_disease=args.simple_death_rates,
                        postban_18yo_cohort=postban_18yo_cohort,
                        )
                    
                    beta_2345_aug, beta_1_aug = t.get_augmented_betas()
                    t.arr1, t.arr2345, t.arr6, t.arr6_noncohort = np.copy(arr1), np.copy(arr2345), np.copy(arr6), np.copy(arr6_noncohort)

                    t.simulation_loop(beta_1_aug, beta_2345_aug, shortbanparams=shortbanparams, longbanparams=longbanparams)

                    np.save(savename, t.output_numpy)

                    np.save(disease_savename_cvd, t.output_cvd)
                    np.save(disease_savename_lc, t.output_lc)
                    np.save(disease_savename_total, t.output_65yos)

                    np.save(LYL_savename, t.output_LYL)
                    
                    progress = i/args.num_mortparams + j/args.num_initpops/args.num_mortparams + k/args.num_banparams/args.num_initpops/args.num_mortparams
                    seconds_since_start = int((datetime.now() - start).total_seconds())
                    print(f"mort: {i_str}, initpop: {j_str}, ban params: {k_str}, {np.around(progress * 100, decimals=3)}% done, {seconds_since_start} seconds elapsed.")
                 #endif
            #endfor
        #endfor
    #endfor

    print(f"Finished! Results at {str(output_dir)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Specify simulation parameters')
    parser.add_argument('num_mortparams', 
                        type=int,
                        default=1,
                        help='the number of sets of mortality parameters (relative risks) that were drawn')
    parser.add_argument('num_initpops', 
                        type=int,
                        default=1,
                        help='the number of initial populations that were created for each mortality parameter draw')
    parser.add_argument('num_banparams', 
                        type=int,
                        default=1,
                        help='the number of short term menthol ban parameters that were drawn in the case of a menthol ban')
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