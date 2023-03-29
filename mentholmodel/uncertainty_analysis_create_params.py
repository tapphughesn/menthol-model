"""
Uncertainty analysis is done in two stages

1. Generate all the parameters that are going to be used for all the runs of the simulation
    Three main groups: mortality params (relative risks), initial populations, ban parameters (short- and long-term).
2. Iterate over all parameter sets, running a simulation for each, and recording all output

This script is responsible for stage 1.

The following acronyms are used throughout my code.
They refer to the relative risk ratios used in mortality computation.
csvns = Current Smoker Versus Never Smoker
fsvcs = Former Smoker Versus Current Smoker
"""
from simulation import Simulation
import pandas as pd
import numpy as np
import argparse
import os
from datetime import datetime
from scipy.stats import truncnorm
import math

from uncertainty_analysis_do_runs import int_to_str

def main(args):

    start = datetime.now()
    now_str = start.strftime("%Y-%m-%d_%H-%M-%S-%f")
    print(f"uncertainty analysis timestamp: {now_str}")
    results_dir = f'../../uncertainty_analysis_data/uncertainty_analysis_{now_str}' # The parameters get saved here

    if not os.path.isdir(results_dir):
        os.mkdir(results_dir)

    mort_sets_dir = os.path.join(results_dir, 'mortality_parameter_sets')
    init_pop_dir = os.path.join(results_dir, 'initial_populations')
    shortban_param_dir = os.path.join(results_dir, 'short_term_menthol_ban_parameter_sets')
    longban_param_dir = os.path.join(results_dir, 'long_term_menthol_ban_parameter_sets')

    os.mkdir(mort_sets_dir)
    os.mkdir(init_pop_dir)
    os.mkdir(shortban_param_dir)
    os.mkdir(longban_param_dir)

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

    # Get Releative Risks for current smokers vs nonsmoker and former smoker vs current smoker
    # According to a published review
    # Here we are also getting 95% CI and stddevs

    csvns_arr = pd.read_excel(os.path.join("..", "..", "smoking_mortality", "csvns_bounds.xlsx")).to_numpy()[:,1:]
    fsvcs_arr = pd.read_excel(os.path.join("..", "..", "smoking_mortality", "fsvcs_bounds.xlsx")).to_numpy()[:,1:]

    # create unit truncated normal to be used for confidence interval sampling
    unit_truncnorm = truncnorm(-1.96, 1.96) # 1.96 is z score for 95% confidence interval

    # MORT PARAMS

    base_mort_params = [] # used for initial populations
    base_mort_params.append(np.array([np.array([row[0], row[1]]) for row in csvns_arr]))
    base_mort_params.append(np.array([np.array([row[0], row[1]]) for row in fsvcs_arr]))

    # now determine mortality parameter samplings to be used in runs
    mortparamsset = []
    for i in range(args.num_mortparams):
        this_csvns_sampling = []
        this_fsvcs_sampling = []

        for row in csvns_arr:
            men_mean = row[0]
            women_mean = row[1]
            men_lower = row[2]
            women_lower = row[3]
            men_upper = row[4]
            women_upper = row[5]

            # get to log space then find stddev from CI
            men_stddev = (np.log(men_upper) - np.log(men_lower)) / (2*1.96) 
            women_stddev = (np.log(women_upper) - np.log(women_lower)) / (2*1.96) 

            # sample the ratio in log space then take exp to get back to the ratio
            men_RR = unit_truncnorm.rvs(size=1)[0] * men_stddev + np.log(men_mean)
            women_RR = unit_truncnorm.rvs(size=1)[0] * women_stddev + np.log(women_mean)
            men_RR = np.exp(men_RR)
            women_RR = np.exp(women_RR)

            this_csvns_sampling.append([men_RR, women_RR])

        for row in fsvcs_arr:
            men_mean = row[0]
            women_mean = row[1]
            men_lower = row[2]
            women_lower = row[3]
            men_upper = row[4]
            women_upper = row[5]

            # get to log space then find stddev from CI
            men_stddev = (np.log(men_upper) - np.log(men_lower)) / (2*1.96) 
            women_stddev = (np.log(women_upper) - np.log(women_lower)) / (2*1.96) 

            # sample the ratio in log space then take exp to get back to the ratio
            men_RR = unit_truncnorm.rvs(size=1)[0] * men_stddev + np.log(men_mean)
            women_RR = unit_truncnorm.rvs(size=1)[0] * women_stddev + np.log(women_mean)
            men_RR = np.exp(men_RR)
            women_RR = np.exp(women_RR)

            this_fsvcs_sampling.append([men_RR, women_RR])

        this_csvns_sampling = np.array(this_csvns_sampling)
        this_fsvcs_sampling = np.array(this_fsvcs_sampling)
        
        # save these mortality parameters for later analysis
        i_str = int_to_str(i, args.num_mortparams)

        np.save(os.path.join(mort_sets_dir, f"set_{i_str}_csvns.npy"), this_csvns_sampling)
        np.save(os.path.join(mort_sets_dir, f"set_{i_str}_fsvcs.npy"), this_fsvcs_sampling)

        mortparamsset.append(
            (this_csvns_sampling, this_fsvcs_sampling)
        )
    # endfor

    # SHORTBAN PARAMS

    """
    create the set of menthol ban parameters to be used for all combinations of mortality params and initpops
    ahead of time, that is before we actually use them in simulation
    """
    shortbanparams_25minus = np.array([0.,0.28,0.17,0.29,0.26])
    shortbanparams_25plus = np.array([0.,0.24,0.20,0.42,0.14])

    # this is needed to reduce the variance of the dirichlet sampling
    alpha_multiplier = 1000

    # use dirichlet dist to sample short-term ban params
    sample_25minus = np.random.dirichlet(
            alpha=shortbanparams_25minus[1:] * alpha_multiplier,
            size=args.num_banparams,
            )
    sample_25plus = np.random.dirichlet(
            alpha=shortbanparams_25plus[1:] * alpha_multiplier,
            size=args.num_banparams,
            )

    # save the individual short-term ban parameter sets
    for i in range(args.num_banparams):
        i_str = int_to_str(i, args.num_banparams)

        shortbanparams = np.concatenate([
            np.zeros((2,1)),
            np.concatenate([
                sample_25minus[i][np.newaxis, :],
                sample_25plus[i][np.newaxis, :],
            ], axis = 0),
        ], axis=1)

        np.save(os.path.join(shortban_param_dir, f"set_{i_str}_shortbanparams.npy"), shortbanparams)

    # LONGBAN PARAMS

    """
    Encode the longban params as a multinomial probability distribution
    over the following groups:

    1. former smoker
    2. menthol smoker
    3. nonmenthol smoker
    4. ecig/dual user

    These are the probabilities of menthol smokers switching 
    to another smoking state due to the ban. They sum to 1.
    Of course, we don't have a category for never smoker
    because menthol smokers cannot become never smokers.
    """
    # using machine epsilon instead of zeros to avoid divide by zero errors
    longban_options = np.array([
        [0.2, 0.5, 0.2, 0.1],
        [0.4, 0.5, np.finfo(float).eps, 0.1],
        [np.finfo(float).eps, 0.5, 0.4, 0.1],
        [np.finfo(float).eps, 0.75, 0.25, 0.05],
        [0.1, 0.5, 0.1, 0.3],
    ])

    # create longban parameter directories for each long-term scenario 
    longban_options_dirs = [os.path.join(longban_param_dir, f'option_{i+1}') for i in range(len(longban_options))]

    for dir in longban_options_dirs:
        os.mkdir(dir)

    for i, opt in enumerate(longban_options):
        option_num = i + 1
        sample = np.random.dirichlet(
            alpha=opt * alpha_multiplier,
            size=args.num_banparams,
        )
        # save these where they should go
        for j in range(args.num_banparams):
            longbanparams = sample[j]
            j_str = int_to_str(j, args.num_banparams)

            np.save(os.path.join(longban_options_dirs[i], f"option_{option_num}_set_{j_str}_longbanparams.npy"), longbanparams)

    # INITPOPS

    # Now get the initial populations. (25 total)
    for j in range(args.num_initpops):
        j_str = int_to_str(j, args.num_initpops)
        
        # create simulation obj just to make initial populations
        s = Simulation(pop_df=pop_df, 
            beta2345=beta2345_arr, 
            beta1=beta1_arr, 
            life_tables=life_table_dict,
            cohorts=cohorts_18_dict,
            smoking_prevalences=smoking_prevalence_dict,
            current_smoker_RR=base_mort_params[0],
            former_smoker_RR=base_mort_params[1],
            save_xl_fname='xl_output_calibrated',
            save_np_fname='np_output_calibrated',
            save_transition_np_fname='transitions_calibrated',
            use_adjusted_death_rates=not args.simple_death_rates,
            end_year = 2066,
            target_initial_smoking_proportion=NHIS_smoking_percentage,
            initiation_rate_decrease=0.055,
            continuation_rate_decrease=0.055,
            )

        # create initial population
        s.format_population()
        beta_2345_aug, beta_1_aug = s.get_augmented_betas()
        s.arr1, s.arr2345, s.arr6 = s.calibrate_initial_population(s.arr1, s.arr2345, s.arr6, beta_1_aug, beta_2345_aug)

        np.save(os.path.join(init_pop_dir, f'pop_{j_str}_arr1.npy'), s.arr1)
        np.save(os.path.join(init_pop_dir, f'pop_{j_str}_arr6.npy'), s.arr6)
        np.save(os.path.join(init_pop_dir, f'pop_{j_str}_arr2345.npy'), s.arr2345)
    #endfor


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
                        help='the number of short term and long term menthol ban parameter sets to draw in the case of a menthol ban')
    parser.add_argument('--simple_death_rates', 
                        default=False,
                        action='store_true',
                        help='whether or not to use separate death rates for smokers, nonsmokers, and former smokers')
    main(parser.parse_args())