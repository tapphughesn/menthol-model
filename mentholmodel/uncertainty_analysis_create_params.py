from simulation import Simulation
import pandas as pd
import numpy as np
import argparse
import os
from datetime import datetime
from scipy.stats import truncnorm
import math


def main(args):

    start = datetime.now()
    now_str = start.strftime("%Y-%m-%d_%H-%M-%S-%f")
    print(f"uncertainty analysis timestamp: {now_str}")
    results_dir = f'../../uncertainty_analysis_data/uncertainty_analysis_{now_str}'

    if not os.path.isdir(results_dir):
        os.mkdir(results_dir)

    mort_sets_dir = os.path.join(results_dir, 'mortality_parameter_sets')
    init_pop_dir = os.path.join(results_dir, 'initial_populations')
    shortban_param_dir = os.path.join(results_dir, 'short_term_menthol_ban_parameter_sets')
    output_dir = os.path.join(results_dir, 'outputs_numpy')
    
    os.mkdir(mort_sets_dir)
    os.mkdir(init_pop_dir)
    os.mkdir(shortban_param_dir)
    os.mkdir(output_dir)

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

    csvns_arr = pd.read_excel(os.path.join("..", "..", "smoking_mortality", "csvns_stddevs.xlsx")).to_numpy()[:,1:]
    fsvcs_arr = pd.read_excel(os.path.join("..", "..", "smoking_mortality", "fsvcs_stddevs.xlsx")).to_numpy()[:,1:]

    # create unit truncated normal to be used for confidence interval sampling
    unit_truncnorm = truncnorm(-1.96, 1.96) # 1.96 is z score for 95% confidence interval

    # now determine mortality parameters to be used ahead of time
    mortparamsset = []
    for i in range(args.num_mortparams):
        this_csvns_sampling = []
        this_fsvcs_sampling = []

        for row in csvns_arr:
            men_mean = row[0]
            women_mean = row[1]
            men_stddev = row[4]
            women_stddev = row[5]
            men_RR = unit_truncnorm.rvs(size=1)[0] * men_stddev + men_mean
            women_RR = unit_truncnorm.rvs(size=1)[0] * women_stddev + women_mean
            this_csvns_sampling.append([men_RR, women_RR])

        for row in fsvcs_arr:
            men_mean = row[0]
            women_mean = row[1]
            men_stddev = row[4]
            women_stddev = row[5]
            men_RR = unit_truncnorm.rvs(size=1)[0] * men_stddev + men_mean
            women_RR = unit_truncnorm.rvs(size=1)[0] * women_stddev + women_mean
            this_fsvcs_sampling.append([men_RR, women_RR])

        this_csvns_sampling = np.array(this_csvns_sampling)
        this_fsvcs_sampling = np.array(this_fsvcs_sampling)
        
        # save these mortality parameters for later analysis
        num_i_digits = math.floor(math.log10(args.num_mortparams))
        i_str = str(i)
        while len(i_str) < num_i_digits:
            i_str = "0" + i_str
        np.save(os.path.join(mort_sets_dir, f"set_{i_str}_csvns.npy"), this_csvns_sampling)
        np.save(os.path.join(mort_sets_dir, f"set_{i_str}_fsvcs.npy"), this_fsvcs_sampling)

        mortparamsset.append(
            (this_csvns_sampling, this_fsvcs_sampling)
        )
    # endfor

    # create the set of menthol ban parameters to be used for all combinations of mortality params and initpops
    # ahead of time, that is before we actually use them in simulation
    if args.menthol_ban:
        # these numbers specify the short term option from which we perturb to get randomness
        # based on values sent to me on 11/23/22
        shortbanparams_25minus = np.array([0.,0.28,0.17,0.31,0.24])
        shortbanparams_25plus = np.array([0.,0.22,0.24,0.42,0.12])
        # this is needed to reduce the variance of the dirichlet sampling
        alpha_multiplier = 1000

        shortbanparamset = []
        for i in range(args.num_shortbanparams):
            sample_25minus = np.random.dirichlet(
                    alpha=shortbanparams_25minus[1:] * alpha_multiplier,
                    size=args.num_shortbanparams,
                    )
            sample_25plus = np.random.dirichlet(
                    alpha=shortbanparams_25plus[1:] * alpha_multiplier,
                    size=args.num_shortbanparams,
                    )

            shortbanparams = np.concatenate([
                np.zeros((2,1)),
                np.concatenate([
                    sample_25minus[i][np.newaxis, :],
                    sample_25plus[i][np.newaxis, :],
                ], axis = 0),
            ], axis=1)

            shortbanparamset.append(shortbanparams)
        # endfor
    # endif

    for i in range(args.num_mortparams):
        # get the mortality parameters
        this_csvns_sampling, this_fsvcs_sampling = mortparamsset[i]

        i_str = str(i)
        while len(i_str) < num_i_digits:
            i_str = "0" + i_str

        # now for each mortality parameter set, make an initial population
        for j in range(args.num_initpops):
            num_j_digits = math.floor(math.log10(args.num_initpops))
            j_str = str(j)
            while len(j_str) < num_j_digits:
                j_str = "0" + j_str
            
            # create simulation obj just to make initial populations
            s = Simulation(pop_df=pop_df, 
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
                menthol_ban=args.menthol_ban,
                short_term_option=1,
                long_term_option=5,
                menthol_ban_year = 2021,
                target_initial_smoking_proportion=NHIS_smoking_percentage,
                initiation_rate_decrease=0.055,
                continuation_rate_decrease=0.055,
                )

            # create initial population
            s.format_population()
            beta_2345_aug, beta_1_aug = s.get_augmented_betas()
            s.arr1, s.arr2345, s.arr6 = s.calibrate_initial_population(s.arr1, s.arr2345, s.arr6, beta_1_aug, beta_2345_aug)

            np.save(os.path.join(init_pop_dir, f'mort_{i_str}_pop_{j_str}_arr1.npy'), s.arr1)
            np.save(os.path.join(init_pop_dir, f'mort_{i_str}_pop_{j_str}_arr6.npy'), s.arr6)
            np.save(os.path.join(init_pop_dir, f'mort_{i_str}_pop_{j_str}_arr2345.npy'), s.arr2345)


            if args.menthol_ban:
                #for each initial population, sample menthol ban params k times
                shortbanparams_25minus = np.array([0.,0.27,0.19,0.42,0.12])
                shortbanparams_25plus = np.array([0.,0.23,0.20,0.44,0.13])

                sample_25minus = np.random.dirichlet(alpha=shortbanparams_25minus[1:], size=args.num_shortbanparams)
                sample_25plus = np.random.dirichlet(alpha=shortbanparams_25plus[1:], size=args.num_shortbanparams)

                # do simulation for each i,j,k combo
                for k in range(args.num_shortbanparams):
                    # just creating a string to represent k
                    num_k_digits = math.floor(math.log10(args.num_shortbanparams))
                    k_str = str(k)
                    while len(k_str) < num_k_digits:
                        k_str = "0" + k_str
                    
                    shortbanparams = shortbanparamset[k]

                    # create simulation obj and set its initial pop to previous one
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
                        menthol_ban=args.menthol_ban,
                        short_term_option=1,
                        long_term_option=5,
                        menthol_ban_year = 2021,
                        target_initial_smoking_proportion=NHIS_smoking_percentage,
                        initiation_rate_decrease=0.055,
                        continuation_rate_decrease=0.055,
                        )
                    
                    t.arr1, t.arr2345, t.arr6 = np.copy(s.arr1), np.copy(s.arr2345), np.copy(s.arr6)

                    t.simulation_loop(beta_1_aug, beta_2345_aug, shortbanparams=shortbanparams)

                    savename = os.path.join(output_dir, f'mort_{i_str}_pop_{j_str}_banparams_{k_str}_output.npy')
                    np.save(savename, t.output_numpy)
                    
                    progress = i/args.num_mortparams + j/args.num_initpops/args.num_mortparams + k/args.num_shortbanparams/args.num_initpops/args.num_mortparams
                    seconds_since_start = int((datetime.now() - start).total_seconds())
                    print(f"mort: {i_str}, initpop: {j_str}, ban params: {k_str}, {np.around(progress * 100, decimals=3)}% done, {seconds_since_start} seconds elapsed.")
                 #endfor
            else:
                raise NotImplementedError

            #endif
        #endfor
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