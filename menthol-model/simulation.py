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
                 smoking_prevalences: dict=None,
                 current_smoker_RR: np.ndarray=None,
                 former_smoker_RR: np.ndarray=None,
                 save_xl_fname: str=None, 
                 save_np_fname: str=None, 
                 save_transition_np_fname: str=None,
                 save_dir: str= '../../outputs/',
<<<<<<< HEAD
                 end_year: int=2066, 
                 start_year: int=2016,
=======
                 end_year: int=2068, 
                 start_year: int=2018,
>>>>>>> 370f1bbab88447a1db8048fffb46c7388f62d2a8
                 menthol_ban: bool=False):
        
        self.pop_df = pop_df
        self.life_tables = life_tables # dict int (year), int (sex) -> array
        self.smoking_prevalences = smoking_prevalences # dict int (year) -> array
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
        self.age_last_smoked_for_ia1 = 17

        self.use_adjusted_death_rates = False
        self.use_adjusted_death_rates = self.current_smoker_RR is not None \
        and self.former_smoker_RR is not None \
        and self.smoking_prevalences is not None

        return
    
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
<<<<<<< HEAD
        # one for state {2,3,4} called arr2345
        # another for state {1,5} called arr1
        # another for state 6 = death called arr6
=======
        # one for state {2,3,4} called arr234
        # another for state {1,5} called arr15
        # another for state 6 = death called arr6

        # print("initial count:", np.sum(pop_arr[:,8])) # 188430785.536...
>>>>>>> 370f1bbab88447a1db8048fffb46c7388f62d2a8

        arr2345 = np.asarray([row for row in pop_arr 
                  if (row[4] == 2 or row[4] == 3 or row[4] == 4
                  or row[3] == 2 or row[3] == 3 or row[3] == 4)], dtype=np.float64)
        arr1 = np.asarray([row for row in pop_arr 
                  if (row[4] == 1 or row[4] == 5)
                  and (row[3] == 1 or row[3] == 5)], dtype=np.float64)
        arr6 = None

        # futher processing to make things into indicators that I need
        # desired indexing is above
        # current indexing is this:
        # 0 agegrp
        # 1 sex
        # 2 black
        # 3 state2
        # 4 state3
        # 5 ia
        # 6 pov
        # 7 set
        # 8 weight
        # 9 age
        # 10 start_age

        def path_to_indicator_form(a):
            s2 = a[:,3]
            s3 = a[:,4]
            ia = a[:,5]
            a = np.concatenate([
                np.ones((len(a), 1)),
                (s2 == 1)[:,np.newaxis],
                (s2 == 2)[:,np.newaxis],
                (s2 == 3)[:,np.newaxis],
                (s2 == 4)[:,np.newaxis],
                (s3 == 1)[:,np.newaxis],
                (s3 == 2)[:,np.newaxis],
                (s3 == 3)[:,np.newaxis],
                (s3 == 4)[:,np.newaxis],
                (ia == 1)[:,np.newaxis],
                (ia == 2)[:,np.newaxis],
                a[:,2][:,np.newaxis], # black
                a[:,9][:,np.newaxis], # age
                a[:,1][:,np.newaxis] - 1, # change sex from {1,2} to {0,1}
                a[:,6][:,np.newaxis],  # dont change poverty from {1,2} to {0,1}
                a[:,10][:,np.newaxis], # start age
                a[:,8][:,np.newaxis], # weight
                a[:,0][:,np.newaxis], # agegrp
                np.zeros((a.shape[0],1)), # hassmoked flag
                -1 * np.ones((a.shape[0],1)), # year last smoked initialize to -1 for nonsmokers
            ], axis=1, dtype=np.float64)
            return a

<<<<<<< HEAD
        arr2345 = path_to_indicator_form(arr2345)
        arr1 = path_to_indicator_form(arr1)

        # TODO: put this stuff in the path_to_indicator_form function
=======
        arr234 = path_to_indicator_form(arr234)
        arr234[:,-1] = np.ones((arr234.shape[0])) # hassmoked flag = 1 for people in 234
        arr15 = path_to_indicator_form(arr15)

        # now the population arrays are in the right format for matrix mult

        # test = np.sum(arr234[:,1:9], axis=1)
        # print(test[test < 1])
        # quit()

        # For experimentation, lets keep only the people that are in 
        # a specified group



        # next step is to format the betas
>>>>>>> 370f1bbab88447a1db8048fffb46c7388f62d2a8

        arr2345[:,18] = np.ones((arr2345.shape[0])) # hassmoked flag = 1 for people in 2345

        # for people whose last state is 3,4 the year last smoked is self.start_year - 1
        arr2345[np.logical_or(arr2345[:,3],arr2345[:,4]),19] = self.start_year - 1

        # for people currently in groups 3,4 the year last smoked is self.start_year
        arr2345[np.logical_or(arr2345[:,7],arr2345[:,8]),19] = self.start_year

        # for people whose last state is 5, the year last smoked is self.start_year - 1
        arr2345[np.sum(arr2345[:,1:5], axis=1) == 0,19] = self.start_year
        arr1[np.sum(arr1[:,1:5], axis=1) == 0,19] = self.start_year

        # for people whose current state is 5, the year last smoked is self.start_year
        arr2345[np.sum(arr2345[:,5:9], axis=1) == 0,19] = self.start_year
        arr1[np.sum(arr1[:,5:9], axis=1) == 0,19] = self.start_year

        # for people in group 2 last state AND this state
        # if initialization age is 1 then year last smoked is self.year_last_smoked_for_ia1
        ind = np.logical_and(arr2345[:,2], arr2345[:,6], arr2345[:,9]).astype(np.bool_)
        arr2345[ind,19] = self.age_last_smoked_for_ia1 + self.start_year - arr2345[ind, 12]

        # if initialization age is 2 then year last smoked is randomly chosen between start_age and current age
        ind = np.logical_and(arr2345[:,2], arr2345[:,6], arr2345[:,10]).astype(np.bool_)
        age_started = np.maximum(18, arr2345[ind,1]) # use starting age if available, otherwise use 18
        # print(arr2345[ind][arr2345[ind][:,1] == 0][0])
        # assert(np.all(arr2345[ind, 1])) # check all "former smokers" have nonzero start age -- returns 0 which is interesting
        to_multiply_rand = arr2345[ind, 12] - age_started + 1 - 1e-8
        to_add_after_multiply = self.start_year - arr2345[ind, 12] - 0.5 + 1e-8
        arr2345[ind ,19] = np.round(np.random.rand(np.sum(ind)) * to_multiply_rand + to_add_after_multiply)
                            
        # now the population arrays are in the right format for matrix mult
        # TODO: For experimentation, lets keep only the people that are in 
        # a specified group

        # next step is to format the betas
        beta_2345_aug = np.concatenate([
            self.beta2345[:,:5],
            np.zeros((len(self.beta2345), 1)),
            self.beta2345[:,5:],
            np.zeros((len(self.beta2345), 5)),
        ], axis=1, dtype=np.float64)

        beta_1_aug = np.concatenate([
            self.beta1[:,:2],
            np.zeros((len(self.beta1), 3)),
            self.beta1[:,2][:,np.newaxis],
            np.zeros((len(self.beta1), 3)),
            self.beta1[:,3:],
            np.zeros((len(self.beta1), 5)),
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
            # TODO: make this a function of arr234, arr15, arr6, and out arrays
            for black in [0,1]:
                for pov in [1,2]:
                    for smoking_state in [1,2,3,4,5,6]: 
                        # determine count of people which fit the descriptors
                        # note smoking state == 6 means dead
                        count = None
<<<<<<< HEAD
                        if smoking_state == 5 and arr2345 is None and arr1 is None:
                            count = 0
                        elif smoking_state == 5 and arr2345 is None:
                            count += np.sum(
                                (arr1[:,11] == black) *
                                (arr1[:,14] == pov) *
                                (arr1[:,4 + 1] == 0) * 
                                (arr1[:,4 + 2] == 0) * 
                                (arr1[:,4 + 3] == 0) * 
                                (arr1[:,4 + 4] == 0) * 
                                (arr1[:,16])
                            )
                        elif smoking_state == 5 and arr1 is None:
=======
                        if smoking_state == 5 and arr234 is None and arr15 is None:
                            count = 0
                        elif smoking_state == 5 and arr234 is None:
                            count += np.sum(
                                (arr15[:,11] == black) *
                                (arr15[:,14] == pov) *
                                (arr15[:,4 + 1] == 0) * 
                                (arr15[:,4 + 2] == 0) * 
                                (arr15[:,4 + 3] == 0) * 
                                (arr15[:,4 + 4] == 0) * 
                                (arr15[:,16])
                            )
                        elif smoking_state == 5 and arr15 is None:
                            count = np.sum(
                                (arr234[:,11] == black) *
                                (arr234[:,14] == pov) *
                                (arr234[:,4 + 1] == 0) * 
                                (arr234[:,4 + 2] == 0) * 
                                (arr234[:,4 + 3] == 0) * 
                                (arr234[:,4 + 4] == 0) * 
                                (arr234[:,16])
                            )
                        elif smoking_state == 5:
>>>>>>> 370f1bbab88447a1db8048fffb46c7388f62d2a8
                            count = np.sum(
                                (arr2345[:,11] == black) *
                                (arr2345[:,14] == pov) *
                                (arr2345[:,4 + 1] == 0) * 
                                (arr2345[:,4 + 2] == 0) * 
                                (arr2345[:,4 + 3] == 0) * 
                                (arr2345[:,4 + 4] == 0) * 
                                (arr2345[:,16])
                            )
                        elif smoking_state == 5:
                            count = np.sum(
                                (arr2345[:,11] == black) *
                                (arr2345[:,14] == pov) *
                                (arr2345[:,4 + 1] == 0) * 
                                (arr2345[:,4 + 2] == 0) * 
                                (arr2345[:,4 + 3] == 0) * 
                                (arr2345[:,4 + 4] == 0) * 
                                (arr2345[:,16])
                            )
                            count += np.sum(
                                (arr1[:,11] == black) *
                                (arr1[:,14] == pov) *
                                (arr1[:,4 + 1] == 0) * 
                                (arr1[:,4 + 2] == 0) * 
                                (arr1[:,4 + 3] == 0) * 
                                (arr1[:,4 + 4] == 0) * 
                                (arr1[:,16])
                            )
                        elif smoking_state == 6 and arr6 is not None:
                            count = np.sum(
                                (arr6[:,11] == black) *
                                (arr6[:,14] == pov) *
                                (arr6[:,16])
                            )
                        elif smoking_state == 6 and arr6 is None:
                            count = 0
<<<<<<< HEAD
                        elif arr2345 is None and arr1 is None:
                            count = 0
                        elif arr1 is None:
                            count = np.sum(
                                (arr2345[:,11] == black) *
                                (arr2345[:,14] == pov) *
                                (arr2345[:,4 + smoking_state] == 1) * 
                                (arr2345[:,16])
                            )
                        elif arr2345 is None:
                            count += np.sum(
                                (arr1[:,11] == black) *
                                (arr1[:,14] == pov) *
                                (arr1[:,4 + smoking_state] == 1) * 
                                (arr1[:,16])
=======
                        elif arr234 is None and arr15 is None:
                            count = 0
                        elif arr15 is None:
                            count = np.sum(
                                (arr234[:,11] == black) *
                                (arr234[:,14] == pov) *
                                (arr234[:,4 + smoking_state] == 1) * 
                                (arr234[:,16])
                            )
                        elif arr234 is None:
                            count += np.sum(
                                (arr15[:,11] == black) *
                                (arr15[:,14] == pov) *
                                (arr15[:,4 + smoking_state] == 1) * 
                                (arr15[:,16])
>>>>>>> 370f1bbab88447a1db8048fffb46c7388f62d2a8
                            )
                        else:
                            count = np.sum(
                                (arr2345[:,11] == black) *
                                (arr2345[:,14] == pov) *
                                (arr2345[:,4 + smoking_state] == 1) * 
                                (arr2345[:,16])
                            )
                            count += np.sum(
                                (arr1[:,11] == black) *
                                (arr1[:,14] == pov) *
                                (arr1[:,4 + smoking_state] == 1) * 
                                (arr1[:,16])
                            )
                        
                        # write list and numpy arr
                        out_list.append([
                            cy + self.start_year,
                            black,
                            pov,
                            smoking_state,
                            count,
                        ])

<<<<<<< HEAD
                        out_np[cy,black,pov - 1,smoking_state - 1] = count

        # Next step is to loop over years, updating the pop each year
        # and writing out the stats
        # cy means current year
        for cy in range(self.end_year - self.start_year):

            """
            Main loop and crux of the program.
            Steps:
                1. write data to appropriate structures to be saved for later analysis
                2. kill people according to life tables
                3. update people's smoking statuses
                    a. make sure to take care of hassmoked flag
                4. update people's ages
            """

            # start by writing out the appropriate data

            write_data(cy, arr2345, arr1, arr6, self.output_list_to_df, self.output_numpy)

            # time to update the population
=======
                        self.output_numpy[cy,black,pov,smoking_state - 1] = count
            
            # ok writing the output stats is done
            # time to actually update the population
>>>>>>> 370f1bbab88447a1db8048fffb46c7388f62d2a8

            # TODO: insert a new cohort of 18yearolds

            # continue by randomly determining if people
            # will die this year
            # first we will define some variables to use later
            
            # male = 0
            # female = 1

            life_table_year = min(self.start_year + cy, 2018)
            life_table_year = max(life_table_year, 2016)

            adr_male = self.life_tables[life_table_year][0].astype(np.float64)
            adr_female = self.life_tables[life_table_year][1].astype(np.float64)

            arr2345_death_chances_male = None
            arr2345_death_chances_female = None

            arr1_death_chances_male = None
            arr1_death_chances_female = None

            arr2345_ages = arr2345[:, 12].astype(np.int32)
            arr2345_ages = list(arr2345_ages.clip(min=0, max = 100))
            arr2345_sex = arr2345[:, 13].astype(np.bool_) # True = Female, False = Male

            arr1_ages = arr1[:, 12].astype(np.int32)
            arr1_ages = list(arr1_ages.clip(min=0, max = 100))
            arr1_sex = arr1[:, 13].astype(np.bool_) # True = Female, False = Male


            if self.use_adjusted_death_rates:
                """
                Need to take into account relative risk of death for smokers (state 3,4) and nonsmokers (state 1,2,5)
                RR = % mortality smokers / % mortality nonsmokers
                average death rate = (% mortality smokers * % smokers) + (% mortality nonsmokers * % nonsmokers)
                We have prevalence of smoking for each age, ages 55-90

                I found the following equations for death rates:

                smoker_deathrate = average_deathrate / (proportion_smokers + (1 - proportion_smokers) / current_smoker_RR)
                nonsmoker_deathrate = average_deathrate / (proportion_smokers * current_smoker_RR + (1 - proportion_smokers))
                former_smoker_deathrate = average_deathrate * former_smoker_RR / (proportion_smokers + (1 - proportion_smokers) / current_smoker_RR)
                """

                # first work on arr2345

                proportion_smoking = self.smoking_prevalences[life_table_year] / 100

                # here I am assuming prevalence of smoking is 0 in ages 90-100
                proportion_smoking = np.concatenate([proportion_smoking, np.zeros(10)])
                RR_indices_array = np.array([0] * 5 + [1] * 5 + [2] * 5 + [3] * 5 + [4] * 5 + [5] * 5 + [6] * 16)

                csrr_male_55plus = self.current_smoker_RR[:,0]
                csrr_female_55plus = self.current_smoker_RR[:,1]

                fsrr_male_55plus = self.former_smoker_RR[:,0]
                fsrr_female_55plus = self.former_smoker_RR[:,1]

                arr2345_death_chances_male = adr_male[arr2345_ages] 
                arr2345_death_chances_female = adr_female[arr2345_ages]

                arr2345_smoker_mask = arr2345[:,7] + arr2345[:,8] + (np.sum(arr2345[:,5:9], axis=1) == 0)
                arr2345_smoker_mask = arr2345_smoker_mask.astype(np.bool_)
                # arr2345_nonsmoker_mask = arr2345[:,5].astype(np.bool_)
                arr2345_formersmoker_mask = arr2345[:,6].astype(np.bool_)

                # smokers
                arr2345_smoker_55plus_ages = list(arr2345[np.logical_and(arr2345_smoker_mask, arr2345[:,12] >= 55).astype(np.bool_)].astype(np.int32).clip(min=0, max=100) - 55)
                arr2345_death_chances_male[arr2345_smoker_mask] = arr2345_death_chances_male[arr2345_smoker_mask] / (proportion_smoking[arr2345_smoker_55plus_ages] + (1 - proportion_smoking[arr2345_smoker_55plus_ages]) / csrr_male_55plus[list(RR_indices_array[arr2345_smoker_55plus_ages])])
                arr2345_death_chances_female[arr2345_smoker_mask] = arr2345_death_chances_female[arr2345_smoker_mask] / (proportion_smoking[arr2345_smoker_55plus_ages] + (1 - proportion_smoking[arr2345_smoker_55plus_ages]) / csrr_female_55plus[list(RR_indices_array[arr2345_smoker_55plus_ages])])

                # formersmokers
                arr2345_formersmoker_55plus_ages = list(arr2345[np.logical_and(arr2345_formersmoker_mask, arr2345[:,12] >= 55)].astype(np.bool_).astype(np.int32).clip(min=0, max=100) - 55)
                arr2345_death_chances_male[arr2345_formersmoker_mask] = fsrr_male_55plus[list(RR_indices_array[arr2345_formersmoker_55plus_ages])] * arr2345_death_chances_male[arr2345_formersmoker_mask] / (proportion_smoking[arr2345_formersmoker_55plus_ages] + (1 - proportion_smoking[arr2345_formersmoker_55plus_ages]) / csrr_male_55plus[list(RR_indices_array[arr2345_formersmoker_55plus_ages])])
                arr2345_death_chances_female[arr2345_formersmoker_mask] = fsrr_female_55plus[list(RR_indices_array[arr2345_formersmoker_55plus_ages])] * arr2345_death_chances_female[arr2345_formersmoker_mask] / (proportion_smoking[arr2345_formersmoker_55plus_ages] + (1 - proportion_smoking[arr2345_formersmoker_55plus_ages]) / csrr_female_55plus[list(RR_indices_array[arr2345_formersmoker_55plus_ages])])

                # nonsmokers
                pass

                # actually decide deaths
                chance = np.random.rand(len(arr2345)).astype(np.float64)

            else:
                print("Not using death rates adjusted for smokers, formersmokers, nonsmokers.")

                arr2345_death_chances_male = adr_male[arr2345_ages] 
                arr2345_death_chances_female = adr_female[arr2345_ages]

            chance = np.random.rand(len(arr2345)).astype(np.float64)

            deaths_male = arr2345_death_chances_male > chance # bool arr
            deaths_female = arr2345_death_chances_female > chance # bool arr

            deaths_all = np.logical_or(
                np.logical_and(deaths_male, np.logical_not(arr2345_sex)),
                np.logical_and(deaths_female, arr2345_sex)
            )

            if arr6 is None:
                arr6 = arr2345[deaths_all]
            else:
                arr6 = np.concatenate([arr6, arr2345[deaths_all]], axis=0, dtype=np.float64)
            
            arr2345 = arr2345[np.logical_not(deaths_all)]

            # people in arr1 randomly die

            chance = np.random.rand(len(arr1)).astype(np.float64)
            arr1_ages = arr1[:, 12].astype(np.int32)
            arr1_ages = list(arr1_ages.clip(min=0, max = 100))
            arr1_sex = arr1[:, 13].astype(np.bool_) # True = Female, False = Male

            deaths_male = adr_male[arr1_ages] > chance # bool arr
            deaths_female = adr_female[arr1_ages] > chance # bool arr

            deaths_all = np.logical_or(
                np.logical_and(deaths_male, np.logical_not(arr1_sex)),
                np.logical_and(deaths_female, arr1_sex)
            )

            if arr6 is None:
                arr6 = arr1[deaths_all]
            else:
                arr6 = np.concatenate([arr6, arr1[deaths_all]], axis=0, dtype=np.float64)
            
            arr1 = arr1[np.logical_not(deaths_all)]

            # TODO: take into account instantaneous menthol ban effects

            # next we update the smoking status of people
            logits_2345 = np.matmul(arr2345, beta_2345_aug).astype(np.float64)
            assert(logits_2345.shape[1] == 3)

            logits_1 = np.matmul(arr1, beta_1_aug).astype(np.float64)
            assert(logits_1.shape[1] == 4)

            # convert logits to probabilities

            exps = np.exp(logits_2345)
            p4 = 1 / (1 + np.sum(exps, axis=1))
            probs2345 = np.asarray([
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

            # take into account hassmoked flag

            hassmoked_1 = arr1[:,18]

            probs1[:,1] += probs1[:,0] * hassmoked_1
            probs1[:,0] -= probs1[:,0] * hassmoked_1

            # TODO: take into account menthol ban long-term effects

            if self.menthol_ban:
                pass

            # update current state, old state

            # proud of this
            # need to think of a better name for this function
            def random_select_arg_multinomial(probs):
                """"
                Takes in probs
                returns indicator for next state
                in a format like: [0,0,1,0,0]
                TODO: Check probs is length 4 is ok
                """
                chance = np.random.rand(probs.shape[0], 1)
                forward = np.concatenate([chance < np.sum(probs[:,:i], axis=1)[:,np.newaxis] for i in range(1, probs.shape[1] + 1)], axis=1)
                backward = np.concatenate([(1 - chance) < np.sum(probs[:,i:], axis=1)[:,np.newaxis] for i in range(probs.shape[1])], axis=1)
                arg_selection = forward * backward
                return arg_selection

            new_states2345 = random_select_arg_multinomial(probs2345)[:,:-1]
            # print(new_states2345.shape) # (9508, 3)

<<<<<<< HEAD
            # people are not going into state 1 (never smoker) from 2345
            new_states2345 = np.concatenate([ 
                np.zeros((new_states2345.shape[0], 1)),
                new_states2345,
=======
            # people are not going into state 1 (never smoker) from 234
            new_states234 = np.concatenate([ 
                np.zeros((new_states234.shape[0], 1)),
                new_states234,
>>>>>>> 370f1bbab88447a1db8048fffb46c7388f62d2a8
            ], axis=1, dtype=np.float64)

            # if all the elements in a row of new_states2345 are 0, then they are going to state 5
            # the first element of each row is 0
            staying_2345 = np.sum(new_states2345[:,1:], axis=1).astype(np.bool_) 

<<<<<<< HEAD
            new_states1 = random_select_arg_multinomial(probs1)[:,:-1].astype(np.float64)
            # leaving_1 is 1 for each row in new_states1 which has chosen to transition to 2, 3, or 4
            leaving_1 = np.sum(new_states1[:,1:], axis=1).astype(np.bool_)

            # move current states to last years states and
            # the new states into the current states
=======
            # move current states to last years states and
            # the new states into the current states

            arr234[:,1:5] = arr234[:,5:9]
            arr15[:,1:5] = arr15[:,5:9]

            arr234[:,5:9] = new_states234
            arr15[:,5:9] = new_states15
            
            #check that states are valid
            # test = np.sum(arr234[:,1:9], axis=1)
            # print(test[test != 1])

            # assert(np.all(np.logical_or(np.sum(arr234[:,1:5], axis=1) == 1, np.sum(arr234[:,1:5], axis=1) == 0)))
            # assert(np.all(np.logical_or(np.sum(arr234[:,5:9], axis=1) == 1, np.sum(arr234[:,5:9], axis=1) == 0)))
            # assert(np.all(np.logical_or(np.sum(arr15[:,1:5], axis=1) == 1, np.sum(arr15[:,1:5], axis=1) == 0)))
            # assert(np.all(np.logical_or(np.sum(arr15[:,5:9], axis=1) == 1, np.sum(arr15[:,5:9], axis=1) == 0)))

            # assert(np.all(arr234[:,1] == 0))
            # assert(np.all(arr234[:,5] == 0))

            # record the state transition numbers
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
                transition_numbers.append([
                    np.sum(arr15[:,16][np.logical_and(arr15[:,1], arr15[:,5])]),
                    np.sum(arr15[:,16][np.logical_and(arr15[:,1], arr15[:,6])]),
                    np.sum(arr15[:,16][np.logical_and(arr15[:,1], arr15[:,7])]),
                    np.sum(arr15[:,16][np.logical_and(arr15[:,1], arr15[:,8])]),
                    np.sum(arr15[:,16][np.logical_and(arr15[:,1], np.sum(arr15[:,5:9], axis=1) == 0)]), #state 5 indicator not explictly tracked
                    np.sum(arr234[:,16][np.logical_and(arr234[:,2], arr234[:,6])]),
                    np.sum(arr234[:,16][np.logical_and(arr234[:,2], arr234[:,7])]),
                    np.sum(arr234[:,16][np.logical_and(arr234[:,2], arr234[:,8])]),
                    np.sum(arr234[:,16][np.logical_and(arr234[:,2], np.sum(arr234[:,5:9], axis=1) == 0)]), #state 5 indicator not explictly tracked
                    np.sum(arr234[:,16][np.logical_and(arr234[:,3], arr234[:,6])]),
                    np.sum(arr234[:,16][np.logical_and(arr234[:,3], arr234[:,7])]),
                    np.sum(arr234[:,16][np.logical_and(arr234[:,3], arr234[:,8])]),
                    np.sum(arr234[:,16][np.logical_and(arr234[:,3], np.sum(arr234[:,5:9], axis=1) == 0)]), #state 5 indicator not explictly tracked
                    np.sum(arr234[:,16][np.logical_and(arr234[:,4], arr234[:,6])]),
                    np.sum(arr234[:,16][np.logical_and(arr234[:,4], arr234[:,7])]),
                    np.sum(arr234[:,16][np.logical_and(arr234[:,4], arr234[:,8])]),
                    np.sum(arr234[:,16][np.logical_and(arr234[:,4], np.sum(arr234[:,5:9], axis=1) == 0)]), #state 5 indicator not explictly tracked
                    np.sum(arr15[:,16][np.logical_and(np.sum(arr15[:,1:5], axis=1) == 0, arr15[:,5])]),
                    np.sum(arr15[:,16][np.logical_and(np.sum(arr15[:,1:5], axis=1) == 0, arr15[:,6])]) + 
                    np.sum(arr234[:,16][np.logical_and(np.sum(arr234[:,1:5], axis=1) == 0, arr234[:,6])]),
                    np.sum(arr15[:,16][np.logical_and(np.sum(arr15[:,1:5], axis=1) == 0, arr15[:,7])]) +
                    np.sum(arr234[:,16][np.logical_and(np.sum(arr234[:,1:5], axis=1) == 0, arr234[:,7])]),
                    np.sum(arr15[:,16][np.logical_and(np.sum(arr15[:,1:5], axis=1) == 0, arr15[:,8])]) +
                    np.sum(arr234[:,16][np.logical_and(np.sum(arr234[:,1:5], axis=1) == 0, arr234[:,8])]),
                    np.sum(arr15[:,16][np.logical_and(np.sum(arr15[:,1:5], axis=1) == 0, np.sum(arr15[:,5:9], axis=1) == 0)]) +
                    np.sum(arr234[:,16][np.logical_and(np.sum(arr234[:,1:5], axis=1) == 0, np.sum(arr234[:,5:9], axis=1) == 0)]), #state 5 indicator not explictly tracked
                ])
>>>>>>> 370f1bbab88447a1db8048fffb46c7388f62d2a8

            arr2345[:,1:5] = arr2345[:,5:9]
            arr1[:,1:5] = arr1[:,5:9]

            arr2345[:,5:9] = new_states2345
            arr1[:,5:9] = new_states1
            
            # record the state transition numbers
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
            # 1 4->4
            # 16 4->5
            # 17 5->1
            # 18 5->2
            # 19 5->3
            # 20 5->4
            # 21 5->5

            if self.save_transition_np_fname is not None:
                transition_numbers.append([
                    np.sum(arr1[:,16][np.logical_and(arr1[:,1], arr1[:,5])]),
                    np.sum(arr1[:,16][np.logical_and(arr1[:,1], arr1[:,6])]),
                    np.sum(arr1[:,16][np.logical_and(arr1[:,1], arr1[:,7])]),
                    np.sum(arr1[:,16][np.logical_and(arr1[:,1], arr1[:,8])]),
                    np.sum(arr1[:,16][np.logical_and(arr1[:,1], np.sum(arr1[:,5:9], axis=1) == 0)]), #state 5 indicator not explictly tracked
                    np.sum(arr2345[:,16][np.logical_and(arr2345[:,2], arr2345[:,6])]),
                    np.sum(arr2345[:,16][np.logical_and(arr2345[:,2], arr2345[:,7])]),
                    np.sum(arr2345[:,16][np.logical_and(arr2345[:,2], arr2345[:,8])]),
                    np.sum(arr2345[:,16][np.logical_and(arr2345[:,2], np.sum(arr2345[:,5:9], axis=1) == 0)]), #state 5 indicator not explictly tracked
                    np.sum(arr2345[:,16][np.logical_and(arr2345[:,3], arr2345[:,6])]),
                    np.sum(arr2345[:,16][np.logical_and(arr2345[:,3], arr2345[:,7])]),
                    np.sum(arr2345[:,16][np.logical_and(arr2345[:,3], arr2345[:,8])]),
                    np.sum(arr2345[:,16][np.logical_and(arr2345[:,3], np.sum(arr2345[:,5:9], axis=1) == 0)]), #state 5 indicator not explictly tracked
                    np.sum(arr2345[:,16][np.logical_and(arr2345[:,4], arr2345[:,6])]),
                    np.sum(arr2345[:,16][np.logical_and(arr2345[:,4], arr2345[:,7])]),
                    np.sum(arr2345[:,16][np.logical_and(arr2345[:,4], arr2345[:,8])]),
                    np.sum(arr2345[:,16][np.logical_and(arr2345[:,4], np.sum(arr2345[:,5:9], axis=1) == 0)]), #state 5 indicator not explictly tracked
                    np.sum(arr1[:,16][np.logical_and(np.sum(arr1[:,1:5], axis=1) == 0, arr1[:,5])]),
                    np.sum(arr1[:,16][np.logical_and(np.sum(arr1[:,1:5], axis=1) == 0, arr1[:,6])]) + 
                    np.sum(arr2345[:,16][np.logical_and(np.sum(arr2345[:,1:5], axis=1) == 0, arr2345[:,6])]),
                    np.sum(arr1[:,16][np.logical_and(np.sum(arr1[:,1:5], axis=1) == 0, arr1[:,7])]) +
                    np.sum(arr2345[:,16][np.logical_and(np.sum(arr2345[:,1:5], axis=1) == 0, arr2345[:,7])]),
                    np.sum(arr1[:,16][np.logical_and(np.sum(arr1[:,1:5], axis=1) == 0, arr1[:,8])]) +
                    np.sum(arr2345[:,16][np.logical_and(np.sum(arr2345[:,1:5], axis=1) == 0, arr2345[:,8])]),
                    np.sum(arr1[:,16][np.logical_and(np.sum(arr1[:,1:5], axis=1) == 0, np.sum(arr1[:,5:9], axis=1) == 0)]) +
                    np.sum(arr2345[:,16][np.logical_and(np.sum(arr2345[:,1:5], axis=1) == 0, np.sum(arr2345[:,5:9], axis=1) == 0)]), #state 5 indicator not explictly tracked
                ])

            # move people from arr1 to arr2345 and vice versa as needed

            tmp_to_1 = arr2345[np.logical_not(staying_2345)]
            arr2345 = arr2345[staying_2345]

            tmp_to_2345 = arr1[leaving_1]
            arr1 = arr1[np.logical_not(leaving_1)]

            arr2345 = np.concatenate([arr2345, tmp_to_2345], axis=0, dtype=np.float64)
            arr1 = np.concatenate([arr1, tmp_to_1], axis=0, dtype=np.float64)

            # update hassmoked flag
            arr2345[:,18] = np.ones(arr2345.shape[0])

            # update year_last_smoked variable
            # smokers currently in state 3,4 get their last year updated
            arr2345[np.logical_or(arr2345[:,7],arr2345[:,8]),19] = cy

            # smokers currently in state 5 get their last year updated
            arr2345[np.sum(arr2345[:,5:9], axis=1) == 0,19] = cy
            arr1[np.sum(arr1[:,5:9], axis=1) == 0,19] = cy

            # people who made the transition 1->2 get their last year updated
            # this is after switching, so anybody who made that transition will be in arr2345
            arr2345[np.logical_and(arr2345[:,1], arr2345[:,6]),19] = cy

            # update agegrp and age params as needed

            arr2345[:,12] += 1
            arr1[:,12] += 1

            # here is where agegrp should be updated but I'm not 
            # going to do it just yet since
            # we don't write it out and it doesn't matter in the simulation

            # update inital age
            # if ia=1 == 0 and hassmoked == 1 and age >= 18 then ia = 2
            # if hassmoked == 1 and age < 18 then ia = 1

<<<<<<< HEAD
            arr2345[:,10] = (arr2345[:,9] == 0) * arr2345[:,18] * (arr2345[:,12] >= 18)
            arr2345[:,9] = arr2345[:,18] * (arr2345[:,12] < 18)
=======
            arr234[:,10] = (arr234[:,9] == 0) * arr234[:,18] * (arr234[:,12] >= 18)
            arr234[:,9] = arr234[:,18] * (arr234[:,12] < 18)
>>>>>>> 370f1bbab88447a1db8048fffb46c7388f62d2a8

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
