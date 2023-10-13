"""
Script for computing Juan's probabilities from age 50 to 80.
"""

from simulation import Simulation
import pandas as pd
import numpy as np
import argparse
import os
from datetime import datetime
from scipy.stats import truncnorm

def main(args):

    print("args:")
    print(args)

    start = datetime.now()
    now_str = start.strftime("%Y-%m-%d_%H-%M-%S-%f")

    # read input
    inputfile = os.path.join("..","..","JuansWork","Inputs","Copy of Input JYEv5_raw_06222023.xlsx")
    input_arr = pd.read_excel(inputfile).to_numpy()
    input_arr = np.array([row for row in input_arr if row[5]==2]) # dont care about final state at age 80 column
    """
    input file columns:
    0. agegrp (nan)
    1. sex
    2. black
    3. prev_state
    4. current state
    5. final state (meaningless)
    6. init_age grp
    7. pov
    8. set (nan)
    9. weight (nan)
    10. age (50)
    11. start age (not needed)
    12. weighted count (nan)
    """
    input_list_nonsmoker=[]
    input_list_smoker=[]
    for row in input_arr:
        if (row[3]==1) & (row[4]==1):
            input_list_nonsmoker.append(row)
        else:
            input_list_smoker.append(row)

    input_arr_nonsmoker = np.array(input_list_nonsmoker)
    input_arr_smoker = np.array(input_list_smoker)
    
    input_arr_nonsmoker = to_indicator_form(input_arr_nonsmoker)
    input_arr_smoker = to_indicator_form(input_arr_smoker)

    # specify output descriptors
    output_columns = [
        "sex",
        "black",
        "poverty",
        "state age 49",
        "state age 50",
        "count state age 80 = 1",
        "count state age 80 = 2",
        "count state age 80 = 3",
        "count state age 80 = 4",
        "count state age 80 = 5",
    ]

    # Get logistic regression betas
    beta2345_f = os.path.join("..","..","Output_SM","Betas","Beta_Estimates_2345.xlsx")
    beta1_f = os.path.join("..","..","Output_SM","Betas","Beta_Estimates_1.xlsx")
    beta2345_arr = pd.read_excel(beta2345_f).to_numpy()[:,2:]
    beta1_arr = pd.read_excel(beta1_f).to_numpy()[:,2:]
    # get augmented betas
    s = Simulation(pop_df=None, 
        beta2345=beta2345_arr, 
        beta1=beta1_arr, 
        use_adjusted_death_rates=False,
        start_year=2016,
        end_year = 2016+30, # age 50 to 80
        save_dir=None,
        initiation_rate_decrease=0.055, # calibration 
        continuation_rate_decrease=0.055, # calibration
        )
    beta_2345_aug, beta_1_aug = s.get_augmented_betas()

    all_outputs = []

    # do nonsmokers
    for i,row in enumerate(input_arr_nonsmoker):
        print(i, "out of", len(input_arr))

        # prepare simulation with appropriate number of replications
        pop = np.tile(row,(args.number_replications,1))

        s = Simulation(pop_df=None, 
            beta2345=beta2345_arr, 
            beta1=beta1_arr, 
            use_adjusted_death_rates=False,
            start_year=2016,
            end_year = 2016+30, # age 50 to 80
            save_dir=None,
            initiation_rate_decrease=0.055, # calibration 
            continuation_rate_decrease=0.055, # calibration
            )
        
        s.arr1 = pop
        s.arr2345 = None
        s.arr6 = None

        s.simulation_loop_juan(beta_1_aug, beta_2345_aug)
        output = s.output_numpy[-1,:,:,:,:]
        output = np.sum(output,axis=(0,1,2))

        output = [
            row[12], # sex
            row[10], # black
            row[13], # pov
            input_list_nonsmoker[i][3], # state age 49
            input_list_nonsmoker[i][4], # state age 50
            output[0],
            output[1],
            output[2],
            output[3],
            output[4],
        ]
        all_outputs.append(output)

    # do smokers
    for i,row in enumerate(input_arr_smoker):
        print(i + len(input_arr_nonsmoker), "out of", len(input_arr))

        # prepare simulation with appropriate number of replications
        pop = np.tile(row,(args.number_replications,1))

        s = Simulation(pop_df=None, 
            beta2345=beta2345_arr, 
            beta1=beta1_arr, 
            use_adjusted_death_rates=False,
            start_year=2016,
            end_year = 2016+30, # age 50 to 80
            save_dir=None,
            initiation_rate_decrease=0.055, # calibration 
            continuation_rate_decrease=0.055, # calibration
            )
        
        s.arr1 = None
        s.arr2345 = pop
        s.arr6 = None

        s.simulation_loop_juan(beta_1_aug, beta_2345_aug)
        output = s.output_numpy[-1,:,:,:,:]
        output = np.sum(output,axis=(0,1,2))

        output = [
            row[12], # sex
            row[10], # black
            row[13], # pov
            input_list_smoker[i][3], # state age 49
            input_list_smoker[i][4], # state age 50
            output[0],
            output[1],
            output[2],
            output[3],
            output[4],
        ]
        all_outputs.append(output)
    
    # save outputs
    out = pd.DataFrame(all_outputs, columns=output_columns)
    fname = os.path.join("..","..","JuansWork","Outputs",f"output_{now_str}_num_replications_{args.number_replications}.xlsx")
    print("Saving to ", fname)
    out.to_excel(fname)

def to_indicator_form(a):
    """
    futher processing to make things into indicators that I need
    desired indexing is 
        0. one
        1. prev state = 1
        2. prev state = 2
        3. prev state = 3
        4. prev state = 4
        5. current state = 2
        6. current state = 3
        7. current state = 4
        8. initial age = 1
        9. initial age = 2
        10. black
        11. age
        12. sex
        13. poverty
        14. start_age
        15. weight
        16. year_last_smoked
    current indexing is this:
        0. agegrp (nan)
        1. sex
        2. black
        3. prev_state
        4. current state
        5. final state (meaningless)
        6. init_age grp
        7. pov
        8. set (nan)
        9. weight (nan)
        10. age (50)
        11. start age (not needed)
        12. weighted count (nan)
    """

    s2 = a[:,3]
    s3 = a[:,4]
    ia = np.logical_not(np.logical_and(s2==1,s3==1)).astype(int) # all smokers get ia=1, nonsmokers get ia=0
    a = np.concatenate([
        np.ones((a.shape[0], 1)),
        (s2 == 1)[:,np.newaxis],
        (s2 == 2)[:,np.newaxis],
        (s2 == 3)[:,np.newaxis],
        (s2 == 4)[:,np.newaxis],
        (s3 == 2)[:,np.newaxis],
        (s3 == 3)[:,np.newaxis],
        (s3 == 4)[:,np.newaxis],
        (ia == 1)[:,np.newaxis],
        (ia == 2)[:,np.newaxis],
        a[:,2][:,np.newaxis], # black
        a[:,10][:,np.newaxis], # age
        a[:,1][:,np.newaxis], # sex in {0,1}
        a[:,7][:,np.newaxis],  # poverty in {0,1} 
        np.zeros_like(a[:,10])[:,np.newaxis], # start age (irrelevant for juan)
        np.ones_like(a[:,8])[:,np.newaxis], # weight (irrelevant for juan)
        -1 * np.ones((a.shape[0],1)), # year last smoked (irrelevant for juan)
    ], axis=1)


    return a

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Specify simulation parameters')
    parser.add_argument('number_replications', 
                        type=int,
                        default=10,
                        help='the number of relplications to do per profile')
    main(parser.parse_args())