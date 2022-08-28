import numpy as np
import pandas as pd
from typing import Tuple

# Needs more refactoring before this can be used
# Need to get rid of all references to "self"
# Probably not worth the effort

def person_to_death_rate(p, ever_smoker: bool, current_year: int):
    """
    Takes a person array as encoded in Simulation.simulate() and returns their chance of dying using adjusted death rates

    p: 1d array of shape (n,)

    Need to take into account relative risk of death for smokers (state 3,4,5), never smokers (state 1), and former smokers (state 2)

    ps = proportion current smokers
    pf = proportion former smokers
    pn = proportion nonsmokers

    adr = average death rate 
    sdr = current smoker death rate
    fdr = former smoker death rate
    ndr = nonsmoker death rate

    RRfc = Relative Risk of mortality for former smokers  vs current smokers
    RRsn = Relative Risk of mortality for current smokers vs nonsmokers

    For a fixed age and sex, the following equations hold:

    adr == ps*sdr + pf*fdr + pn*ndr
    RRfc == fdr / sdr
    RRsn == sdr/ndr

    The solutions (due to mathematica) are:
    sdr == (adr * RRsn)/(pn + ps * RRsn + pf * RRfc * RRsn)
    fdr -> (adr RRfc * RRsn)/(pn + ps * RRsn + pf * RRfc * RRsn)
    ndr -> adr/(pn + ps RRsn + pf RRfc RRsn)
    """

    life_table_year = min(current_year, 2018)
    life_table_year = max(life_table_year, 2016)

    age = min(int(p[11]), 100)
    sex = int(p[12])
    adr = self.life_tables[life_table_year][sex].astype(np.float64)[age]
    # if the person is age < 55 then we can use average death rates
    if (age < 55):
        return adr

    # grab smoking status percentages for this age and sex
    pn, ps, pf = self.smoking_prevalences[life_table_year][sex].astype(np.float64)[min(age - 55, 29), :] / 100

    # grab relative risks
    RRsn = self.current_smoker_RR[min((age - 55) // 5, 6), sex]
    RRfc = self.former_smoker_RR[3, sex] # use the RR for former smokers who have not smoked in 10-19 years by default

    # separate into cases depending on the smoking status of the person
    if p[5]:
        # former smoker
        # need to update RRfc
        years_since_smoked = current_year - int(p[16])
        try:
            assert(years_since_smoked >= 0)
        except AssertionError:
            print(years_since_smoked)
            print(p[16])
            print(current_year)
            raise
        assert(isinstance(years_since_smoked, int))

        if years_since_smoked < 2:
            RRfc = self.former_smoker_RR[0, sex] # < 2 years since smoked
        elif years_since_smoked < 5:
            RRfc = self.former_smoker_RR[1, sex] # 2-4 years since smoked
        elif years_since_smoked < 10:
            RRfc = self.former_smoker_RR[2, sex] # 5-9 years since smoked
        elif years_since_smoked < 20:
            RRfc = self.former_smoker_RR[3, sex] # 10-19 years since smoked
        elif years_since_smoked < 30:
            RRfc = self.former_smoker_RR[4, sex] # 20-29 years since smoked
        elif years_since_smoked < 40:
            RRfc = self.former_smoker_RR[5, sex] # 30-39 years since smoked
        elif years_since_smoked < 50:
            RRfc = self.former_smoker_RR[6, sex] # 40-49 years since smoked
        else:
            RRfc = self.former_smoker_RR[7, sex] # >= 50 years since smoked

        # fdr -> (adr RRfc * RRsn)/(pn + ps * RRsn + pf * RRfc * RRsn)
        
        return (adr * RRfc * RRsn) / (pn + ps * RRsn + pf * RRfc * RRsn)
    elif p[6] or p[7] or ever_smoker:
        # current smoker
        # sdr == (adr * RRsn)/(pn + ps * RRsn + pf * RRfc * RRsn)
        
        res = min((adr * RRsn) / (pn + ps * RRsn + pf * RRfc * RRsn), 1.0)
        return res
    elif not ever_smoker:
        # never smoker
        # ndr -> adr/(pn + ps RRsn + pf RRfc RRsn)

        return adr / (pn + ps * RRsn + pf * RRfc * RRsn)

    print("While trying to determine person's death chance, they didn't fit into any smoking category")
    raise Exception

    return None

def path_to_indicator_form(a):
    """
    futher processing to make things into indicators that I need
    desired indexing is above
    current indexing is this:
    0 agegrp
    1 sex
    2 black
    3 state2
    4 state3
    5 ia
    6 pov
    7 set
    8 weight
    9 age
    10 start_age
    """

    s2 = a[:,3]
    s3 = a[:,4]
    ia = a[:,5]
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
        a[:,9][:,np.newaxis], # age
        a[:,1][:,np.newaxis] - 1, # change sex from {1,2} to {0,1}
        a[:,6][:,np.newaxis],  # poverty is already {0,1} now, not {1,2} like before
        a[:,10][:,np.newaxis], # start age
        a[:,8][:,np.newaxis], # weight
        -1 * np.ones((a.shape[0],1)), # year last smoked initialize to -1 for nonsmokers
    ], axis=1, dtype=np.float64)
    return a

def cohort_to_indicator_form(c):
    # get it in path form (each row a person)
    # then use path_to_indicator_form

    path_form_arr = np.concatenate([ 
        np.tile(np.array([
            0,         # agegrp
            row[0],    # sex
            row[1],    # black
            row[2],    # previous state
            row[3],    # current state
            row[4],    # ia
            row[5],    # pov
            0,    # set
            row[6] / row[7],    # weight
            18,    # age
            0 + 17 * (int(row[4]) == 1) + 18 * (int(row[4]) == 2),    # start_age
        ]), (int(row[7]), 1))
    for row in c], axis=0)

    path_form_arr = path_form_arr.astype(np.float64)

    arr2345 = np.asarray([row for row in path_form_arr 
                if (row[4] == 2 or row[4] == 3 or row[4] == 4 or row[4] == 5
                or row[3] == 2 or row[3] == 3 or row[3] == 4 or row[3] == 5)], dtype=np.float64)
    arr1 = np.asarray([row for row in path_form_arr
                if (row[4] == 1 and row[3] == 1)], dtype=np.float64)

    arr2345 = self.path_to_indicator_form(arr2345)
    arr1 = self.path_to_indicator_form(arr1)

    # for people whose last state is 3,4 the year last smoked is self.start_year - 1
    arr2345[np.logical_or(arr2345[:,3],arr2345[:,4]),16] = self.start_year - 1

    # for people currently in groups 3,4 the year last smoked is self.start_year
    arr2345[np.logical_or(arr2345[:,6],arr2345[:,7]),16] = self.start_year

    # for people whose last state is 5, the year last smoked is self.start_year - 1
    # we are treating ecig users the same as smokers here
    arr2345[np.sum(arr2345[:,1:5], axis=1) == 0,16] = self.start_year

    # for people whose current state is 5, the year last smoked is self.start_year
    arr2345[np.sum(arr2345[:,5:8], axis=1) == 0,16] = self.start_year

    # for people in group 2 last state AND this state
    # if initialization age is 1 then year last smoked is self.year_last_smoked_for_ia1 + self.start_year - age
    ind = np.logical_and(arr2345[:,2], arr2345[:,5], arr2345[:,8]).astype(np.bool_)
    arr2345[ind,16] = self.age_last_smoked_for_ia1 + self.start_year - arr2345[ind, 11]

    # if initialization age is 2 for former smokers then year last smoked is randomly chosen between start_age and current age
    ind = np.logical_and(arr2345[:,2], arr2345[:,5], arr2345[:,9]).astype(np.bool_)
    age_started = np.maximum(18, arr2345[ind,14]) # use starting age if available, otherwise use 18
    to_multiply_rand = arr2345[ind, 11] - age_started + 1 - 1e-8
    to_add_after_multiply = self.start_year - arr2345[ind, 11] - 0.5 + 1e-8
    arr2345[ind ,16] = np.round(np.random.rand(np.sum(ind)) * to_multiply_rand + to_add_after_multiply)

    return arr2345, arr1

def zero_a_prob(probs, idx):
    probs[:,idx] = 0
    probs *= np.sum(probs, axis=1)

def write_data(cy, arr2345, arr1, arr6, out_list, out_np):
    """
    Given the current year, arrays with the current state,
    and output destination arrays, write data accordingly
    """
    # probably a way to do this without loops but idk
    for black in [0,1]:
        for pov in [0,1]:
            for smoking_state in [1,2,3,4,5,6]: 
                # determine count of people which fit the descriptors
                # note smoking state == 6 means dead
                count = None
                if smoking_state == 5 and arr2345 is None:
                    count = 0
                elif smoking_state == 5:
                    count = np.sum(
                        (arr2345[:,10] == black) *
                        (arr2345[:,13] == pov) *
                        (arr2345[:,5] == 0) * 
                        (arr2345[:,6] == 0) * 
                        (arr2345[:,7] == 0) * 
                        (arr2345[:,15])
                    )
                elif smoking_state == 6 and arr6 is not None:
                    count = np.sum(
                        (arr6[:,10] == black) *
                        (arr6[:,13] == pov) *
                        (arr6[:,15])
                    )
                elif smoking_state == 6 and arr6 is None:
                    count = 0
                elif smoking_state == 1 and arr1 is None:
                    count = 0
                elif smoking_state == 1:
                    count = np.sum(
                        (arr1[:,10] == black) *
                        (arr1[:,13] == pov) *
                        (arr1[:,15])
                    )
                elif arr2345 is None and arr1 is None:
                    count = 0
                elif arr2345 is None:
                    count=0
                elif smoking_state in [2, 3, 4]:
                    # smoking state must be 2, 3, or 4
                    count = np.sum(
                        (arr2345[:,10] == black) *
                        (arr2345[:,13] == pov) *
                        arr2345[:, 4 + smoking_state - 1] * 
                        (arr2345[:,15])
                    )
                else:
                    raise Exception
                
                # write list and numpy arr
                out_list.append([
                    cy + self.start_year,
                    black,
                    pov,
                    smoking_state,
                    count,
                ])

                out_np[cy,black,pov,smoking_state - 1] = count

def random_select_arg_multinomial(probs):
    """"
    Takes in probs
    returns indicator for next state
    in a format like: [0,0,1,0,0]
    return array has same length as input array "probs"
    """
    chance = np.random.rand(probs.shape[0], 1)
    forward = np.concatenate([chance < np.sum(probs[:,:i], axis=1)[:,np.newaxis] for i in range(1, probs.shape[1] + 1)], axis=1)
    backward = np.concatenate([(1 - chance) < np.sum(probs[:,i:], axis=1)[:,np.newaxis] for i in range(probs.shape[1])], axis=1)
    arg_selection = forward * backward
    return arg_selection

def calibrate_smoking_percentage(in_arr1: np.ndarray, in_arr2345: np.ndarray, target_smoker_percentage: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calibrates the weights of a population so that the percentage of smokers
    is equal to some target percentage.

    Takes the indicator-form numpy arrays to represent the population.

    Returns a tuple of the modified indicator-form numpy arrays.

    Input arrays are not mutated!
    """

    smokers_arr = np.sum(in_arr2345[:,6:8], axis=1, dtype=bool)
    nonsmokers_2345_arr = np.logical_not(smokers_arr)

    # sanity check
    assert(np.sum(smokers_arr) + np.sum(nonsmokers_2345_arr) == len(in_arr2345))
    
    smoker_weight = np.sum(in_arr2345[:,15][smokers_arr])
    nonsmoker_weight = np.sum(in_arr1[:,15]) + np.sum(in_arr2345[:,15][nonsmokers_2345_arr])
    total_weight = np.sum(in_arr2345[:,15]) + np.sum(in_arr1[:,15])

    # sanity check
    # some tolerance, these might not be exact due to rounding error
    assert(abs(smoker_weight + nonsmoker_weight - total_weight) < 1e-2)

    smoker_weight_factor = total_weight * target_smoker_percentage / smoker_weight
    nonsmoker_weight_factor = (total_weight - total_weight * target_smoker_percentage) / nonsmoker_weight

    out_arr1 = np.copy(in_arr1)
    out_arr2345 = np.copy(in_arr2345)

    out_arr1[:,15] *= nonsmoker_weight_factor
    out_arr2345[:,15][nonsmokers_2345_arr] *= nonsmoker_weight_factor
    out_arr2345[:,15][smokers_arr] *= smoker_weight_factor

    return out_arr1, out_arr2345



