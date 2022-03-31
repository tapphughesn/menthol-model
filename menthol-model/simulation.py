from re import I
from matplotlib import use
import pandas as pd
import numpy as np
import os
from datetime import date, datetime

class Simulation(object):

    """
    The array output of a simulation need to have 4 dimensions:
        year
        race
        poverty
        smoking state
    where each number in the array is the count of people who belong to these categories

    This will also be written out as a dataframe with columns
        year, race, poverty, smoking state, count

    There are two logistic regression models:
    The old specification was:
        one for people in state 2,3,4 (former smoker, nonmenthol, menthol)
            in wave 2 OR wave 3
        another for people in state 1,5 (nonsmoker, ecig)
            in wave 2 AND wave 3
    The new specification is:
        one for people in state 2,3,4,5 (former smoker, nonmentol, menthol, ecig) (ever smokers)
            in wave 2 OR wave 3
        one for people in state 1 in wave 2 AND wave 3 (never smokers or ecig users)

    Need to transform ints into indicators (booleans)

    We have 6 states:
        1 -> never smoker
        2 -> former smoker
        3 -> menthol smoker
        4 -> nonmenthol smoker
        5 -> ecig
        6 -> dead
    
    Here are the independent variables we need to track 
    for logistic regression (mostly indicators):
    OLD:
        prev state = 1
        prev state = 2
        prev state = 3
        prev state = 4
        current state = 1
        current state = 2
        current state = 3
        current state = 4
        initial age = 1
        initial age = 2
        black
        age
        sex
        poverty
    NEW:
        prev state = 1
        prev state = 2
        prev state = 3
        prev state = 4
        current state = 2
        current state = 3
        current state = 4
        initial age = 1
        initial age = 2
        black
        age
        sex
        poverty

    The simulation population arrays will keep track of the following things
    at the following indices:
    NEW:
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
        16. year last smoked

    """

    def __init__(self, 
                 pop_df: pd.DataFrame, 
                 beta2345: np.ndarray, 
                 beta1: np.ndarray, 
                 life_tables: dict,
                 cohorts: dict=None,
                 cohort_adding_pattern: list=[1,1,1,1,1,1,1,1,1,1],
                 smoking_prevalences: dict=None,
                 current_smoker_RR: np.ndarray=None,
                 former_smoker_RR: np.ndarray=None,
                 use_adjusted_death_rates: bool=True,
                 save_xl_fname: str=None, 
                 save_np_fname: str=None, 
                 save_transition_np_fname: str=None,
                 save_dir: str= '../../outputs/',
                 end_year: int=2066, 
                 start_year: int=2016,
                 menthol_ban: bool=False):
        
        self.pop_df = pop_df
        self.life_tables = life_tables # dict int (year), int (sex) -> array
        self.cohorts = cohorts # dict int (year) -> array
        self.cohort_adding_pattern = cohort_adding_pattern # list specifying how many cohorts to add for each year
        self.smoking_prevalences = smoking_prevalences # dict int (year) -> dict int (sex) -> 2darray (age 55+ X (never, current, former))
        self.current_smoker_RR = current_smoker_RR # Relative Risk of all cause mortality vs nonsmokers
        self.former_smoker_RR = former_smoker_RR # Relative Risk of all cause mortality vs current smokers
        self.end_year=end_year 
        self.start_year=start_year
        self.beta2345 = np.asarray(beta2345, dtype=np.float64) # arr
        self.beta1 = np.asarray(beta1, dtype=np.float64) # arr
        self.save_xl_fname = save_xl_fname
        self.save_np_fname = save_np_fname
        self.save_transition_np_fname = save_transition_np_fname
        self.save_dir = save_dir
        self.output_columns = [
            "year", 
            "black", 
            "poverty", 
            "smoking state",
            "count"
        ]
        self.input_columns = pop_df.columns
        self.output_list_to_df = []
        self.output_numpy = np.zeros((end_year - start_year + 1, 2, 2, 6))
        self.now_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
        self.menthol_ban = menthol_ban
        if self.menthol_ban:
            self.save_xl_fname += "_menthol_ban"
            self.save_np_fname += "_menthol_ban"
            self.save_transition_np_fname += "_menthol_ban"
        self.age_last_smoked_for_ia1 = 17

        self.use_adjusted_death_rates = use_adjusted_death_rates
        if (self.use_adjusted_death_rates):
            try:
                assert(self.current_smoker_RR is not None and self.former_smoker_RR is not None and self.smoking_prevalences is not None)
            except AssertionError:
                print("use_adjusted_death_rates was set to True but not all of the following were provided: current_smoker_RR, former_smoker_RR, smoking_prevalences")
                raise
        
        return
    
    def person_to_death_rate(self, p, ever_smoker: bool, current_year: int):
        """
        Takes a person array as encoded in self.simulate() and returns their chance of dying using adjusted death rates

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
    
    def path_to_indicator_form(self, a):
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
    
    def cohort_to_indicator_form(self, c):
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

    def simulate(self):

        """
        Calling this function causes 1 run of the simulation to happen.
        Results are written according to save_xl_fname and save_np_fname.
        Optionally, transition numbers are written to save_transition_np_fname.

        Args:
            None
        
        Output:
            self.output: the data written out from the simulation
        """
        pop_arr = self.pop_df.to_numpy(dtype=np.float64)

        # make a list to record transition probabilities
        transition_numbers = []

        # now we need to construct 3 arrays which get updated 
        # during the course of the simulation
        # one for state {2,3,4} called arr2345
        # another for state {1,5} called arr1
        # another for state 6 = death called arr6

        arr2345 = np.asarray([row for row in pop_arr 
                  if (row[4] == 2 or row[4] == 3 or row[4] == 4 or row[4] == 5
                  or row[3] == 2 or row[3] == 3 or row[3] == 4 or row[3] == 5)], dtype=np.float64)
        arr1 = np.asarray([row for row in pop_arr 
                  if (row[4] == 1 and row[3] == 1)], dtype=np.float64)
        arr6 = None

        arr2345 = self.path_to_indicator_form(arr2345)
        arr1 = self.path_to_indicator_form(arr1)

        # print(np.sum(arr2345[:,15]))
        # print(np.sum(arr1[:,15]))
        # print(np.sum(arr1[:,15]) + np.sum(arr2345[:,15]))
        # print("-------------")

        # Here we figure out the year_last_smoked variable for all cases

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
                            
        # now the population arrays are in the right format for matrix mult
        # TODO: For experimentation, lets keep only the people that are in 
        # a specified group

        # next step is to format the betas
        beta_2345_aug = np.concatenate([
            self.beta2345,
            np.zeros((len(self.beta2345), 3)),
        ], axis=1, dtype=np.float64)

        beta_1_aug = np.concatenate([
            self.beta1[:,0][:,np.newaxis],
            np.zeros((len(self.beta1), 9)),
            self.beta1[:,1:],
            np.zeros((len(self.beta1), 3)),
        ], axis=1, dtype=np.float64)

        beta_2345_aug = np.transpose(beta_2345_aug)
        beta_1_aug = np.transpose(beta_1_aug)

        # define a function for writing out data for a current year      
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

        # Next step is to loop over years, updating the pop each year
        # and writing out the stats
        # cy means current year
        for cy in range(self.end_year - self.start_year):
            """
            Main loop and crux of the program.
            Steps:
                0. add cohorts of 18 yearolds if needed
                1. write data to appropriate structures to be saved for later analysis
                2. kill people according to life tables
                3. update people's smoking statuses
                    a. make sure to take care of hassmoked flag
                4. update people's ages
            """

            # print(cy)
            # pops = None
            # if arr6 is None:
            #     pops = (np.sum(arr2345[:,15]), np.sum(arr1[:,15]))
            # else:
            #     pops = (np.sum(arr2345[:,15]), np.sum(arr1[:,15]), np.sum(arr6[:,15]))
            # print(pops)
            # print(sum(pops))
            # print("-------------")
            # print("num state=1", np.sum(arr1[:,15]))
            # print("num state=2", np.sum(arr2345[:,15] * arr2345[:,5]))
            # print("num state=3", np.sum(arr2345[:,15] * arr2345[:,6]))
            # print("num state=4", np.sum(arr2345[:,15] * arr2345[:,7]))
            # print("num state=5", np.sum(arr2345[:,15] * (arr2345[:,5] == 0) * (arr2345[:,6] == 0) * (arr2345[:,7] == 0)))
            # print("num state=3 and black == sex == 0", np.sum(arr2345[:,15] * arr2345[:,6] * (arr2345[:,14] == 0) * (arr2345[:,11] == 0)))
            # print("num state=3 and black == sex == 0", np.sum(arr2345[:,15] * arr2345[:,6] * (arr2345[:,14] == 1) * (arr2345[:,11] == 0)))
            # print("num state=3 and black == sex == 0", np.sum(arr2345[:,15] * arr2345[:,6] * (arr2345[:,14] == 0) * (arr2345[:,11] == 1)))
            # print("num state=3 and black == sex == 0", np.sum(arr2345[:,15] * arr2345[:,6] * (arr2345[:,14] == 1) * (arr2345[:,11] == 1)))

            # print(np.count_nonzero(arr2345[:,14]))


            # insert new cohort(s) of 18yearolds
            if self.cohorts is not None and cy < len(self.cohort_adding_pattern):
                # print("adding cohort")
                cohort_idx = max(self.start_year + cy, 2016)
                cohort_idx = min(cohort_idx, 2018)
                cohort_arr = self.cohorts[cohort_idx]
                c2345, c1 = self.cohort_to_indicator_form(cohort_arr)
                for _i in range(self.cohort_adding_pattern[cy]):
                    arr2345 = np.concatenate([arr2345, c2345], axis=0)
                    arr1 = np.concatenate([arr1, c1], axis=0)

            # print("after cohorts", cy)
            # pops = None
            # if arr6 is None:
            #     pops = (np.sum(arr2345[:,15]), np.sum(arr1[:,15]))
            # else:
            #     pops = (np.sum(arr2345[:,15]), np.sum(arr1[:,15]), np.sum(arr6[:,15]))
            # print(pops)
            # print(sum(pops))
            # print("-------------")
            # start by writing out the appropriate data

            write_data(cy, arr2345, arr1, arr6, self.output_list_to_df, self.output_numpy)

            # continue by randomly determining if people
            # will die this year
            # first we will define some variables to use later
            
            # male = 0
            # female = 1

            # probability of death for each person
            arr2345_death_rates = None
            arr1_death_rates = None

            # the following conditional affects the "death_chances" arrays
            if self.use_adjusted_death_rates:
                arr2345_death_rates = np.array([self.person_to_death_rate(row, ever_smoker=True, current_year=cy+self.start_year) for row in arr2345]).astype(np.float64)
                arr1_death_rates = np.array([self.person_to_death_rate(row, ever_smoker=False, current_year=cy+self.start_year) for row in arr1]).astype(np.float64)

                # print(np.min(arr2345_death_rates), np.max(arr2345_death_rates))
                assert(np.max(arr2345_death_rates) <= 1)
                assert(np.min(arr2345_death_rates) >= 0)
                assert(np.max(arr1_death_rates) <= 1)
                assert(np.min(arr1_death_rates) >= 0)
            else:
                print("Not using death rates adjusted for smokers, formersmokers, nonsmokers.")

                life_table_year = min(self.start_year + cy, 2018)
                life_table_year = max(life_table_year, 2016)

                adr_male = self.life_tables[life_table_year][0].astype(np.float64)
                adr_female = self.life_tables[life_table_year][1].astype(np.float64)

                arr2345_ages = arr2345[:, 11].astype(np.int32)
                arr2345_ages = list(arr2345_ages.clip(min=0, max = 100)) # does not overwrite arr2345
                arr2345_sex = arr2345[:, 12] # 0 = male, 1 = female

                arr1_ages = arr1[:, 11].astype(np.int32)
                arr1_ages = list(arr1_ages.clip(min=0, max = 100))
                arr1_sex = arr1[:, 12] # 0 = male, 1 = female

                arr2345_death_chances_male = adr_male[arr2345_ages] 
                arr2345_death_chances_female = adr_female[arr2345_ages]

                arr1_death_chances_male = adr_male[arr1_ages] 
                arr1_death_chances_female = adr_female[arr1_ages]

                arr2345_death_rates = arr2345[:,12] * arr2345_death_chances_female + (1 - arr2345[:,12]) * arr2345_death_chances_male
                arr1_death_rates = arr1[:,12] * arr1_death_chances_female + (1 - arr1[:,12]) * arr1_death_chances_male

                assert(np.max(arr2345_death_rates) <= 1)
                assert(np.min(arr2345_death_rates) >= 0)
                assert(np.max(arr1_death_rates) <= 1)
                assert(np.min(arr1_death_rates) >= 0)

            # determine deaths in arr2345
            # print("before deaths", cy)
            # pops = None
            # if arr6 is None:
            #     pops = (np.sum(arr2345[:,15]), np.sum(arr1[:,15]))
            # else:
            #     pops = (np.sum(arr2345[:,15]), np.sum(arr1[:,15]), np.sum(arr6[:,15]))
            # print(pops)
            # print(sum(pops))
            # print("-------------")

            chance_2345 = np.random.rand(len(arr2345)).astype(np.float64)
            deaths_2345 = arr2345_death_rates > chance_2345

            if arr6 is None:
                arr6 = np.copy(arr2345)[deaths_2345]
            else:
                arr6 = np.concatenate([arr6, np.copy(arr2345)[deaths_2345]], axis=0, dtype=np.float64)

            arr2345 = arr2345[np.logical_not(deaths_2345)]

            # determine deaths in arr1
            chance_1 = np.random.rand(len(arr1)).astype(np.float64)
            deaths_1 = arr1_death_rates > chance_1

            if arr6 is None:
                arr6 = np.copy(arr1)[deaths_1]
            else:
                arr6 = np.concatenate([arr6, np.copy(arr1)[deaths_1]], axis=0, dtype=np.float64)
            
            arr1 = arr1[np.logical_not(deaths_1)]


            # next we update the smoking status of people
            logits_2345 = np.matmul(arr2345, beta_2345_aug).astype(np.float64)
            assert(logits_2345.shape[1] == 3)

            logits_1 = np.matmul(arr1, beta_1_aug).astype(np.float64)
            assert(logits_1.shape[1] == 4)

            # convert logits to probabilities

            exps = np.exp(logits_2345)
            p4 = 1 / (1 + np.sum(exps, axis=1))
            probs2345 = np.asarray([
                np.zeros(arr2345.shape[0]), # p1
                p4*exps[:,0], # p2
                p4*exps[:,1], # p3
                p4,           # p4
                p4*exps[:,2], # p5
            ], dtype=np.float64).transpose()

            exps = np.exp(logits_1)
            p4 = 1 / (1 + np.sum(exps, axis=1))
            probs1 = np.asarray([
                p4*exps[:,0], # p1
                p4*exps[:,1], # p2
                p4*exps[:,2], # p3
                p4,           # p4
                p4*exps[:,3], # p5
            ], dtype=np.float64).transpose()

            """
            Instantaneous menthol ban effects at year 10:

            Among those 25+ years, 
                23% of menthol cigarette smokers quit smoking, 
                44% switch of menthol cigarette smokers switch to non-menthol cigarettes (state 4), 
                20% continue using menthol cigarettes (state 3), 
                and 13% switch to e-cigs (state 5). 
            Among 18-24 year olds post-ban, 
                39% of menthol cigarette smokers quit, 
                3% continue using menthol cigarettes, 
                18% switch to e-cigs, 
                and 40% switch to non-menthol cigarettes. 

            I am going to implement this by overridding the transistion probabilities
            of people this year. I will take the probability of becoming a menthol smoker
            and distribute it among the 5 probabilities according to the proportions
            given above
            """

            if self.menthol_ban and cy == 0:

                probs_25minus = np.array([0.,0.39,0.03,0.40,0.18])
                probs_25plus = np.array([0.,0.23,0.20,0.44,0.13])

                assert(sum(probs_25minus) == 1.)
                assert(sum(probs_25plus) == 1.)

                are_25plus_2345 = arr2345[:,11] >= 25
                are_25minus_2345 = np.logical_not(are_25plus_2345)
                are_25plus_1 = arr1[:,11] >= 25
                are_25minus_1 = np.logical_not(are_25plus_1)

                for i in [1, 3, 4]:
                    probs2345[are_25plus_2345,i] += probs_25plus[i] * probs2345[are_25plus_2345,2]
                    probs2345[are_25minus_2345,i] += probs_25minus[i] * probs2345[are_25minus_2345,2]

                    probs1[are_25plus_1,i] += probs_25plus[i] * probs1[are_25plus_1,2]
                    probs1[are_25minus_1,i] += probs_25minus[i] * probs1[are_25minus_1,2]
                
                probs2345[are_25plus_2345,2] *= probs_25plus[2]
                probs2345[are_25minus_2345,2] *= probs_25minus[2]
                probs1[are_25plus_1,2] *= probs_25plus[2]
                probs1[are_25minus_1,2] *= probs_25minus[2]

            # update current state, old state

            # need to think of a better name for this function
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

            new_states2345 = random_select_arg_multinomial(probs2345)[:,:] # (s1, s2, s3, s4, s5)
            # print(new_states2345.shape) # (9508, 5)
            new_states1 = random_select_arg_multinomial(probs1)[:,:].astype(np.float64) # (s1, s2, s3, s4, s5)

            # leaving_1 is True for each row in new_states1 which has chosen to transition to 2, 3, 4, or 5
            leaving_1 = np.sum(new_states1[:,1:], axis=1).astype(np.bool_)

            # move current states to last years states and
            # the new states into the current states

            arr2345[:,2:5] = arr2345[:,5:8]
            arr2345[:,1] = np.zeros(arr2345.shape[0]) # no more previous never smokers
            # dont need to move arr1 stuff because arr1's previous state has not changed at all

            arr2345[:,5:8] = new_states2345[:,1:-1] 
            arr1[:,5:8] = new_states1[:,1:-1]
            
            # record the state transition numbers :)

            # we can calculate the number who died also from these numbers
            # Here's what the list means
            # list index | number of people in transition
            # 0 1->1
            # 1 1->2
            # 2 1->3
            # 3 1->4
            # 4 1->5
            # 5 2->2
            # 6 2->3
            # 7 2->4
            # 8 2->5
            # 9 3->2
            # 10 3->3
            # 11 3->4
            # 12 3->5
            # 13 4->2
            # 14 4->3
            # 15 4->4
            # 16 4->5
            # 17 5->1
            # 18 5->2
            # 19 5->3
            # 20 5->4
            # 21 5->5

            if self.save_transition_np_fname is not None:
                # transition_numbers.append([
                to_append = [
                    np.sum(arr1[:,15][np.logical_not(leaving_1)]),
                    np.sum(arr1[:,15][np.logical_and(arr1[:,1], arr1[:,5])]),
                    np.sum(arr1[:,15][np.logical_and(arr1[:,1], arr1[:,6])]),
                    np.sum(arr1[:,15][np.logical_and(arr1[:,1], arr1[:,7])]),
                    0, # placeholder -- fill in after this 
                    np.sum(arr2345[:,15][np.logical_and(arr2345[:,2], arr2345[:,5])]),
                    np.sum(arr2345[:,15][np.logical_and(arr2345[:,2], arr2345[:,6])]),
                    np.sum(arr2345[:,15][np.logical_and(arr2345[:,2], arr2345[:,7])]),
                    np.sum(arr2345[:,15][np.logical_and(arr2345[:,2], np.sum(arr2345[:,5:8], axis=1) == 0)]), #state 5 indicator not explictly tracked
                    np.sum(arr2345[:,15][np.logical_and(arr2345[:,3], arr2345[:,5])]),
                    np.sum(arr2345[:,15][np.logical_and(arr2345[:,3], arr2345[:,6])]),
                    np.sum(arr2345[:,15][np.logical_and(arr2345[:,3], arr2345[:,7])]),
                    np.sum(arr2345[:,15][np.logical_and(arr2345[:,3], np.sum(arr2345[:,5:8], axis=1) == 0)]), #state 5 indicator not explictly tracked
                    np.sum(arr2345[:,15][np.logical_and(arr2345[:,4], arr2345[:,5])]),
                    np.sum(arr2345[:,15][np.logical_and(arr2345[:,4], arr2345[:,6])]),
                    np.sum(arr2345[:,15][np.logical_and(arr2345[:,4], arr2345[:,7])]),
                    np.sum(arr2345[:,15][np.logical_and(arr2345[:,4], np.sum(arr2345[:,5:8], axis=1) == 0)]), #state 5 indicator not explictly tracked
                    0, # 5 -> 1 not allowed
                    np.sum(arr2345[:,15][np.logical_and(np.sum(arr2345[:,1:5], axis=1) == 0, arr2345[:,5])]),
                    np.sum(arr2345[:,15][np.logical_and(np.sum(arr2345[:,1:5], axis=1) == 0, arr2345[:,6])]),
                    np.sum(arr2345[:,15][np.logical_and(np.sum(arr2345[:,1:5], axis=1) == 0, arr2345[:,7])]),
                    np.sum(arr2345[:,15][np.logical_and(np.sum(arr2345[:,1:5], axis=1) == 0, np.sum(arr2345[:,5:8], axis=1) == 0)]), #state 5 indicator not explictly tracked
                ]
                to_append[4] = np.sum(arr1[:,15]) - np.sum(to_append[:4])
                transition_numbers.append([to_append])

            # move people from arr1 to arr2345 and vice versa as needed

            tmp_to_2345 = arr1[leaving_1]
            arr1 = arr1[np.logical_not(leaving_1)]

            arr2345 = np.concatenate([arr2345, tmp_to_2345], axis=0, dtype=np.float64)

            # update year_last_smoked variable
            # smokers currently in state 3,4 get their last year updated
            arr2345[np.logical_or(arr2345[:,6],arr2345[:,7]),16] = cy

            # smokers currently in state 5 get their last year updated
            arr2345[np.sum(arr2345[:,5:8], axis=1) == 0,16] = cy

            # people who made the transition 1->2 get their last year updated
            # this is after switching, so anybody who made that transition will be in arr2345
            arr2345[np.logical_and(arr2345[:,1], arr2345[:,5]),16] = cy

            # update agegrp and age params as needed

            arr2345[:,11] += 1
            arr1[:,11] += 1

            # here is where agegrp should be updated but I'm not 
            # going to do it just yet since
            # we don't write it out and it doesn't matter in the simulation

            # update inital age for people in arr2345 (ever smokers)
            # rules:
            # if ia=1 == 0 and and age >= 18 then ia = 2
            # if hassmoked == 1 and age < 18 then ia = 1

            arr2345[:,9] = (arr2345[:,8] == 0) * (arr2345[:,12] >= 18)
            arr2345[:,8] = (arr2345[:,12] < 18).astype(int).astype(np.float64)

            # print("endfor", cy)
            # pops = (np.sum(arr2345[:,15]), np.sum(arr1[:,15]), np.sum(arr6[:,15]))
            # print(pops)
            # print(sum(pops))
            # print("-------------")

            # endfor 

        # write data one last time for the final year

        write_data(self.end_year - self.start_year, arr2345, arr1, arr6, self.output_list_to_df, self.output_numpy)

        # writeout the results of the simulation to disk
        if self.save_xl_fname:
            out = pd.DataFrame(self.output_list_to_df, columns=self.output_columns)
            fname = os.path.join(self.save_dir, 'excel_files/', os.path.basename(self.save_xl_fname) + '_' + self.now_str + '.xlsx')
            out.to_excel(fname)

        if self.save_np_fname:
            fname = os.path.join(self.save_dir, 'numpy_arrays/', os.path.basename(self.save_np_fname) + '_' + self.now_str + '.npy')
            np.save(fname, self.output_numpy)
        
        if self.save_transition_np_fname:
            fname = os.path.join(self.save_dir, 'transition_numbers/', os.path.basename(self.save_transition_np_fname) + '_' + self.now_str + '.npy')
            np.save(fname, np.asarray(transition_numbers))

        return self.output_list_to_df, self.output_numpy
