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
        one for people in state 2,3,4 (former smoker, nonmenthol, menthol)
            in wave 2 OR wave 3
        another for people in state 1,5 (nonsmoker, ecig only)
            in wave 2 AND wave 3

    Need to transform ints into indicators (booleans)

    We have 6 states:
        1 -> never smoker
        2 -> former smoker
        3 -> menthol smoker
        4 -> nonmenthol smoker
        5 -> ecig only
        6 -> dead
    
    Here are the independent variables we need to track 
    for logistic regression (mostly indicators):
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

    The simulation population arrays will keep track of the following things
    at the following indices:

        0. one
        1. prev state = 1
        2. prev state = 2
        3. prev state = 3
        4. prev state = 4
        5. current state = 1
        6. current state = 2
        7. current state = 3
        8. current state = 4
        9. initial age = 1
        10. initial age = 2
        11. black
        12. age
        13. sex
        14. poverty
        15. start_age
        16. weight
        17. agegrp
        18. has smoked
        19. year last smoked

    """

    def __init__(self, 
                 pop_df: pd.DataFrame, 
                 beta234: np.ndarray, 
                 beta15: np.ndarray, 
                 life_tables: dict,
                 save_xl_fname: str=None, 
                 save_np_fname: str=None, 
                 save_transition_np_fname: str=None,
                 save_dir: str= '../../outputs/',
                 end_year: int=2066, 
                 start_year: int=2016,
                 menthol_ban: bool=False):
        
        self.pop_df = pop_df
        self.life_tables = life_tables # dict int (year), int(sex) -> array
        self.end_year=end_year
        self.start_year=start_year
        self.beta234 = np.asarray(beta234, dtype=np.float64) # arr
        self.beta15 = np.asarray(beta15, dtype=np.float64) # arr
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
        # one for state {2,3,4} called arr234
        # another for state {1,5} called arr15
        # another for state 6 = death called arr6

        # print("initial count:", np.sum(pop_arr[:,8])) # 188430785.536...

        arr234 = np.asarray([row for row in pop_arr 
                  if (row[4] == 2 or row[4] == 3 or row[4] == 4
                  or row[3] == 2 or row[3] == 3 or row[3] == 4)], dtype=np.float64)
        arr15 = np.asarray([row for row in pop_arr 
                  if (row[4] == 1 or row[4] == 5)
                  and (row[3] == 1 or row[3] == 5)], dtype=np.float64)
        arr6 = None

        # print(pop_arr.shape) # (19212, 11)
        # print(arr234.shape) # (9533, 11)
        # print(arr15.shape) # (9679, 11)

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
                a[:,2][:,np.newaxis],
                a[:,9][:,np.newaxis],
                a[:,1][:,np.newaxis] - 1, # change sex from {1,2} to {0,1}
                a[:,6][:,np.newaxis],  # dont change poverty from {1,2} to {0,1}
                a[:,10][:,np.newaxis],
                a[:,8][:,np.newaxis],
                a[:,0][:,np.newaxis],
                np.zeros((a.shape[0],1)), # hassmoked flag
                -1 * np.ones((a.shape[0],1)), # year last smoked
            ], axis=1, dtype=np.float64)
            return a

        arr234 = path_to_indicator_form(arr234)
        arr15 = path_to_indicator_form(arr15)


        arr234[:,18] = np.ones((arr234.shape[0])) # hassmoked flag = 1 for people in 234
        # for people whose last state is 3,4 the year last smoked is self.start_year - 1
        arr234[np.logical_or(arr234[:,3] == 1,arr234[:,4] == 1)][19] = self.start_year - 1
        # for people currently in groups 3,4 the year last smoked is self.start_year
        arr234[np.logical_or(arr234[:,7] == 1,arr234[:,8] == 1)][19] = self.start_year
        # TODO: initialization strategy for those in groups 2,5
        arr234[np.logical_and(
            np.logical_or(arr234[:,6] == 1,
                          np.sum(arr234[:,5:9], axis=1) == 0
                          ),
            np.logical_or(arr234[:,2] == 1,
                          np.sum(arr234[:,1:5], axis=1) == 0
                          ),
        )][19] = self.start_year - 5

        # now the population arrays are in the right format for matrix mult

        # test = np.sum(arr234[:,1:9], axis=1)
        # print(test[test < 1])
        # quit()

        # TODO: For experimentation, lets keep only the people that are in 
        # a specified group



        # next step is to format the betas

        # print(self.beta234.dtype) # float64
        # print(self.beta15.dtype) # float64

        beta_234_aug = np.concatenate([
            self.beta234[:,:5],
            np.zeros((len(self.beta234), 1)),
            self.beta234[:,5:],
            np.zeros((len(self.beta234), 5)),
        ], axis=1, dtype=np.float64)

        beta_15_aug = np.concatenate([
            self.beta15[:,:2],
            np.zeros((len(self.beta15), 3)),
            self.beta15[:,2][:,np.newaxis],
            np.zeros((len(self.beta15), 3)),
            self.beta15[:,3:],
            np.zeros((len(self.beta15), 5)),
        ], axis=1, dtype=np.float64)

        beta_234_aug = np.transpose(beta_234_aug)
        beta_15_aug = np.transpose(beta_15_aug)

        # print(beta_234_aug.shape) # (19,3)
        # print(beta_15_aug.shape) # (19,4)

        assert(beta_15_aug.shape[0] == arr15.shape[1])

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
            # probably a way to do this without loops but idk
            # TODO: make this a function of arr234, arr15, arr6, and out arrays
            for black in [0,1]:
                for pov in [1,2]:
                    for smoking_state in [1,2,3,4,5,6]: 
                        # determine count of people which fit the descriptors
                        # note smoking state == 6 means dead
                        count = None
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
                            count = np.sum(
                                (arr234[:,11] == black) *
                                (arr234[:,14] == pov) *
                                (arr234[:,4 + 1] == 0) * 
                                (arr234[:,4 + 2] == 0) * 
                                (arr234[:,4 + 3] == 0) * 
                                (arr234[:,4 + 4] == 0) * 
                                (arr234[:,16])
                            )
                            count += np.sum(
                                (arr15[:,11] == black) *
                                (arr15[:,14] == pov) *
                                (arr15[:,4 + 1] == 0) * 
                                (arr15[:,4 + 2] == 0) * 
                                (arr15[:,4 + 3] == 0) * 
                                (arr15[:,4 + 4] == 0) * 
                                (arr15[:,16])
                            )
                        elif smoking_state == 6 and arr6 is not None:
                            count = np.sum(
                                (arr6[:,11] == black) *
                                (arr6[:,14] == pov) *
                                (arr6[:,16])
                            )
                        elif smoking_state == 6 and arr6 is None:
                            count = 0
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
                            )
                        else:
                            count = np.sum(
                                (arr234[:,11] == black) *
                                (arr234[:,14] == pov) *
                                (arr234[:,4 + smoking_state] == 1) * 
                                (arr234[:,16])
                            )
                            count += np.sum(
                                (arr15[:,11] == black) *
                                (arr15[:,14] == pov) *
                                (arr15[:,4 + smoking_state] == 1) * 
                                (arr15[:,16])
                            )
                        
                        # write list and numpy arr
                        self.output_list_to_df.append([
                            cy + self.start_year,
                            black,
                            pov,
                            smoking_state,
                            count,
                        ])

                        self.output_numpy[cy,black,pov - 1,smoking_state - 1] = count
            
            # ok writing the output stats is done
            # time to actually update the population

            # TODO: insert a new cohort of 18yearolds

            # continue by randomly determining if people
            # will die this year
            
            # male = 0
            # female = 1

            life_table_year = min(self.start_year + cy, 2018)
            life_table_year = max(life_table_year, 2016)
            life_table_arr = np.concatenate([
                self.life_tables[life_table_year][0].astype(np.float64)[np.newaxis, :],
                self.life_tables[life_table_year][1].astype(np.float64)[np.newaxis, :],
            ], axis=0, dtype=np.float64)

            """
            Need to take into account relative risk of death for smokers (state 3,4) and nonsmokers (state 1,2,5)
            RR = % mortality smokers / % mortality nonsmokers
            average death rate = (% mortality smokers * % smokers) + (% mortality nonsmokers * % nonsmokers)
            We have prevalence of smoking for each age, ages 55-90
            """

            # people in arr235 randomly die

            chance = np.random.rand(len(arr234)).astype(np.float64)
            arr234_ages = arr234[:,12].astype(np.int32)
            arr234_ages = list(arr234_ages.clip(min=0, max = 100))
            arr234_sex = arr234[:, 13].astype(np.bool_) # True = Female, False = Male

            deaths_male = life_table_arr[0][arr234_ages] > chance # bool arr
            deaths_female = life_table_arr[1][arr234_ages] > chance # bool arr

            deaths_all = np.logical_or(
                np.logical_and(deaths_male, np.logical_not(arr234_sex)),
                np.logical_and(deaths_female, arr234_sex)
            )

            if arr6 is None:
                arr6 = arr234[deaths_all]
            else:
                arr6 = np.concatenate([arr6, arr234[deaths_all]], axis=0, dtype=np.float64)
            
            arr234 = arr234[np.logical_not(deaths_all)]

            # people in arr15 randomly die

            chance = np.random.rand(len(arr15)).astype(np.float64)
            arr15_ages = arr15[:,12].astype(np.int32)
            arr15_ages = list(arr15_ages.clip(min=0, max = 100))
            arr15_sex = arr15[:, 13].astype(np.bool_) # True = Female, False = Male

            deaths_male = life_table_arr[0][arr15_ages] > chance # bool arr
            deaths_female = life_table_arr[1][arr15_ages] > chance # bool arr

            deaths_all = np.logical_or(
                np.logical_and(deaths_male, np.logical_not(arr15_sex)),
                np.logical_and(deaths_female, arr15_sex)
            )

            if arr6 is None:
                arr6 = arr15[deaths_all]
            else:
                arr6 = np.concatenate([arr6, arr15[deaths_all]], axis=0, dtype=np.float64)
            
            arr15 = arr15[np.logical_not(deaths_all)]

            # TODO: take into account instantaneous menthol ban effects

            # next we update the smoking status of people
            logits_234 = np.matmul(arr234, beta_234_aug).astype(np.float64)
            assert(logits_234.shape[1] == 3)

            logits_15 = np.matmul(arr15, beta_15_aug).astype(np.float64)
            assert(logits_15.shape[1] == 4)

            # convert logits to probabilities

            exps = np.exp(logits_234)
            p4 = 1 / (1 + np.sum(exps, axis=1))
            probs234 = np.asarray([
                p4*exps[:,0], # p2
                p4*exps[:,1], # p3
                p4,           # p4
                p4*exps[:,2], # p5
            ], dtype=np.float64).transpose()

            exps = np.exp(logits_15)
            p4 = 1 / (1 + np.sum(exps, axis=1))
            probs15 = np.asarray([
                p4*exps[:,0], # p1
                p4*exps[:,1], # p2
                p4*exps[:,2], # p3
                p4,           # p4
                p4*exps[:,3], # p5
            ], dtype=np.float64).transpose()

            # take into account hassmoked flag

            hassmoked_15 = arr15[:,18]

            probs15[:,1] += probs15[:,0] * hassmoked_15
            probs15[:,0] -= probs15[:,0] * hassmoked_15

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

            new_states234 = random_select_arg_multinomial(probs234)[:,:-1]

            # people are not going into state 1 (never smoker) from 234
            new_states234 = np.concatenate([ 
                np.zeros((new_states234.shape[0], 1)),
                new_states234,
            ], axis=1, dtype=np.float64)

            staying_234 = np.sum(new_states234[:,1:], axis=1).astype(np.bool_)

            new_states15 = random_select_arg_multinomial(probs15)[:,:-1].astype(np.float64)
            leaving_15 = np.sum(new_states15[:,1:], axis=1).astype(np.bool_)

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

            # move people from arr15 to arr234 and vice versa as needed

            tmp_to_15 = arr234[np.logical_not(staying_234)]
            arr234 = arr234[staying_234]

            tmp_to_234 = arr15[leaving_15]
            arr15 = arr15[np.logical_not(leaving_15)]

            arr234 = np.concatenate([arr234, tmp_to_234], axis=0, dtype=np.float64)
            arr15 = np.concatenate([arr15, tmp_to_15], axis=0, dtype=np.float64)

            # update hassmoked flag

            arr234[:,18] = np.ones(arr234.shape[0])

            # update agegrp and age params as needed

            arr234[:,12] += 1
            arr15[:,12] += 1

            # here is where agegrp should be updated but I'm not 
            # going to do it just yet since
            # we don't write it out and it doesn't matter in the simulation

            # update inital age
            # if ia=1 == 0 and hassmoked == 1 and age >= 18 then ia = 2
            # if hassmoked == 1 and age < 18 then ia = 1

            arr234[:,10] = (arr234[:,9] == 0) * arr234[:,18] * (arr234[:,12] >= 18)
            arr234[:,9] = arr234[:,18] * (arr234[:,12] < 18)

            # endfor 

        # write data one last time for the final year
        # this code is copied from earlier in this file
        for black in [0,1]:
            for pov in [1,2]:
                for smoking_state in [1,2,3,4,5,6]: 
                    # determine count of people which fit the descriptors
                    # note smoking state == 6 means dead
                    count = None
                    if smoking_state == 5:
                        count = np.sum(
                            (arr234[:,11] == black) *
                            (arr234[:,14] == pov) *
                            (arr234[:,4 + 1] == 0) * 
                            (arr234[:,4 + 2] == 0) * 
                            (arr234[:,4 + 3] == 0) * 
                            (arr234[:,4 + 4] == 0) * 
                            (arr234[:,16])
                        )
                        count += np.sum(
                            (arr15[:,11] == black) *
                            (arr15[:,14] == pov) *
                            (arr15[:,4 + 1] == 0) * 
                            (arr15[:,4 + 2] == 0) * 
                            (arr15[:,4 + 3] == 0) * 
                            (arr15[:,4 + 4] == 0) * 
                            (arr15[:,16])
                        )
                    elif smoking_state == 6 and arr6 is not None:
                        count = np.sum(
                            (arr6[:,11] == black) *
                            (arr6[:,14] == pov) *
                            (arr6[:,16])
                        )
                    else:
                        count = np.sum(
                            (arr234[:,11] == black) *
                            (arr234[:,14] == pov) *
                            (arr234[:,4 + smoking_state] == 1) * 
                            (arr234[:,16])
                        )
                        count += np.sum(
                            (arr15[:,11] == black) *
                            (arr15[:,14] == pov) *
                            (arr15[:,4 + smoking_state] == 1) * 
                            (arr15[:,16])
                        )

                    # write list and numpy arr
                    self.output_list_to_df.append([
                        self.end_year,
                        black,
                        pov,
                        smoking_state,
                        count,
                    ])

                    self.output_numpy[self.end_year - self.start_year,black,pov - 1,smoking_state - 1] = count

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
