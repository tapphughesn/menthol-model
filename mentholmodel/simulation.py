import pandas as pd
import numpy as np
import os
from datetime import datetime
from typing import Tuple


class Simulation(object):

    """
    The array output of a simulation needs to have 4 dimensions:
        year
        race
        poverty
        smoking state
    where each number in the array is the count of people who belong to these categories

    This will also be written out as a pandas dataframe (excel spreadsheet) with columns
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

    We have 6 states in this simulation:
        1 -> never smoker
        2 -> former smoker
        3 -> menthol smoker
        4 -> nonmenthol smoker
        5 -> ecig
        6 -> dead

    Here are the independent variables we need to track 
    for logistic regression (mostly indicators):
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

    """

    def __init__(self,
                 pop_df: pd.DataFrame = None,
                 beta2345: np.ndarray = None,
                 beta1: np.ndarray = None,
                 life_tables: dict = None,
                 cohorts: dict = None,
                 last_year_cohort_added: int = np.inf,
                 smoking_prevalences: dict = None,
                 current_smoker_RR: np.ndarray = None,
                 former_smoker_RR: np.ndarray = None,
                 use_adjusted_death_rates: bool = True,
                 save_xl_fname: str = None,
                 save_np_fname: str = None,
                 save_transition_np_fname: str = None,
                 save_disease_np_fname: str = None,
                 save_LYL_np_fname: str = None,
                 save_dir: str = '../../outputs/',
                 end_year: int = 2116,
                 start_year: int = 2016,
                 menthol_ban: bool = False,
                 short_term_option: int = 1,
                 long_term_option: int = 1,
                 menthol_ban_year: int = 2024,
                 target_initial_smoking_proportion: float = 0.15,
                 initiation_rate_decrease: float = 0.0,
                 continuation_rate_decrease: float = 0.0,
                 print_now_str: bool = False,
                 simulate_disease: bool = False,
                 postban_18yo_cohort:np.ndarray = None,
                 ):

        self.pop_df = pop_df
        self.life_tables = life_tables  # dict int (year), int (sex) -> array
        self.cohorts = cohorts  # dict int (year) -> array
        self.last_year_cohort_added = last_year_cohort_added
        # dict int (year) -> dict int (sex) -> 2darray (age 55+ X (never, current, former))
        self.smoking_prevalences = smoking_prevalences
        # Relative Risk of all cause mortality vs nonsmokers
        self.current_smoker_RR = current_smoker_RR
        # Relative Risk of all cause mortality vs current smokers
        self.former_smoker_RR = former_smoker_RR
        self.end_year = end_year
        self.start_year = start_year
        self.beta2345 = np.asarray(beta2345)  # arr
        self.beta1 = np.asarray(beta1)  # arr
        self.save_xl_fname = save_xl_fname
        self.save_np_fname = save_np_fname
        self.save_transition_np_fname = save_transition_np_fname
        self.save_disease_np_fname = save_disease_np_fname
        self.save_LYL_np_fname = save_LYL_np_fname
        self.save_dir = save_dir
        self.output_columns = [
            "year",
            "black",
            "poverty",
            "65plus",
            "smoking state",
            "count"
        ]
        if pop_df is not None:
            self.input_columns = pop_df.columns
        else:
            self.input_columns = None
        self.output_list_to_df = []
        self.output_numpy = np.zeros((end_year - start_year + 1, 2, 2, 2, 6))
        self.output_transitions = []
        self.now_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
        if print_now_str:
            print(f"timestamp for this simulation object: {self.now_str}")
        self.menthol_ban = menthol_ban
        self.short_term_option = short_term_option
        self.long_term_option = long_term_option
        self.menthol_ban_year = menthol_ban_year
        self.initiation_rate_decrease = initiation_rate_decrease
        assert(0.0 <= initiation_rate_decrease <= 1.0)
        self.continuation_rate_decrease = continuation_rate_decrease
        assert(0.0 <= continuation_rate_decrease <= 1.0)
        self.target_initial_smoking_percentage = target_initial_smoking_proportion
        # proportion should be between 0 and 1
        assert(0.0 <= target_initial_smoking_proportion <= 1.0)

        if self.menthol_ban:
            assert(short_term_option in [1, 2, 3, 4])
            assert(long_term_option in [1, 2, 3, 4, 5])
            self.save_xl_fname += "_menthol_ban_" + \
                str(short_term_option) + '_' + str(long_term_option)
            self.save_np_fname += "_menthol_ban_" + \
                str(short_term_option) + '_' + str(long_term_option)
            self.save_transition_np_fname += "_menthol_ban_" + \
                str(short_term_option) + '_' + str(long_term_option)
        self.age_last_smoked_for_ia1 = 17

        self.use_adjusted_death_rates = use_adjusted_death_rates
        if (self.use_adjusted_death_rates):
            try:
                assert(
                    self.current_smoker_RR is not None and self.former_smoker_RR is not None and self.smoking_prevalences is not None)
            except AssertionError:
                print("use_adjusted_death_rates was set to True but not all of the following were provided: current_smoker_RR, former_smoker_RR, smoking_prevalences")
                raise

        self.arr2345 = None
        self.arr1 = None
        self.arr6 = None
        self.arr6_noncohort = None # arr6 is used for the LYL measure, so this is needed to exclude people not in the cohort of interest

        self.simulate_disease = simulate_disease
        self.num_cvd_cases = {
            "total": 0,
            "black": 0,
            "nonblack": 0,
            "pov": 0,
            "nonpov": 0,
        }
        self.num_lc_cases = {
            "total": 0,
            "black": 0,
            "nonblack": 0,
            "pov": 0,
            "nonpov": 0,
        }
        self.total_65yos = {
            "total": 0,
            "black": 0,
            "nonblack": 0,
            "pov": 0,
            "nonpov": 0,
        }
        self.output_cvd = np.zeros((end_year - start_year + 1, 5))
        self.output_lc = np.zeros((end_year - start_year + 1, 5))
        self.output_65yos = np.zeros((end_year - start_year + 1, 5))

        self.postban_18yo_cohort = postban_18yo_cohort

        self.output_LYL = np.zeros(5)
        return

    def person_to_death_rate(self, p, ever_smoker: bool, current_year: int, use_previous_smoking_state: bool = False):
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
        RRsn == sdr / ndr

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
        pn, ps, pf = self.smoking_prevalences[life_table_year][sex].astype(np.float64)[
            min(age - 55, 29), :] / 100

        # grab relative risks
        try:
            RRsn = self.current_smoker_RR[min((age - 55) // 5, 6), sex]
        except:
            print("------------------")
            print(age)
            print(sex)
            print(self.current_smoker_RR)
        # use the RR for former smokers who have not smoked in 10-19 years by default
        RRfc = self.former_smoker_RR[3, sex]

        # separate into cases depending on the smoking status of the person
        if use_previous_smoking_state:
            if p[2]:
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
                    # < 2 years since smoked
                    RRfc = self.former_smoker_RR[0, sex]
                elif years_since_smoked < 5:
                    # 2-4 years since smoked
                    RRfc = self.former_smoker_RR[1, sex]
                elif years_since_smoked < 10:
                    # 5-9 years since smoked
                    RRfc = self.former_smoker_RR[2, sex]
                elif years_since_smoked < 20:
                    # 10-19 years since smoked
                    RRfc = self.former_smoker_RR[3, sex]
                elif years_since_smoked < 30:
                    # 20-29 years since smoked
                    RRfc = self.former_smoker_RR[4, sex]
                elif years_since_smoked < 40:
                    # 30-39 years since smoked
                    RRfc = self.former_smoker_RR[5, sex]
                elif years_since_smoked < 50:
                    # 40-49 years since smoked
                    RRfc = self.former_smoker_RR[6, sex]
                else:
                    # >= 50 years since smoked
                    RRfc = self.former_smoker_RR[7, sex]

                # fdr -> (adr RRfc * RRsn)/(pn + ps * RRsn + pf * RRfc * RRsn)

                return (adr * RRfc * RRsn) / (pn + ps * RRsn + pf * RRfc * RRsn)
            elif p[3] or p[4] or ever_smoker:
                # current smoker
                # sdr == (adr * RRsn)/(pn + ps * RRsn + pf * RRfc * RRsn)

                return min((adr * RRsn) / (pn + ps * RRsn + pf * RRfc * RRsn), 1.0)
            else:
                # don't check `ever_smoker` here since we are using
                # previous smoking states, not current states
                # never smoker
                # ndr -> adr/(pn + ps RRsn + pf RRfc RRsn)

                return adr / (pn + ps * RRsn + pf * RRfc * RRsn)

        else:
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
                    # < 2 years since smoked
                    RRfc = self.former_smoker_RR[0, sex]
                elif years_since_smoked < 5:
                    # 2-4 years since smoked
                    RRfc = self.former_smoker_RR[1, sex]
                elif years_since_smoked < 10:
                    # 5-9 years since smoked
                    RRfc = self.former_smoker_RR[2, sex]
                elif years_since_smoked < 20:
                    # 10-19 years since smoked
                    RRfc = self.former_smoker_RR[3, sex]
                elif years_since_smoked < 30:
                    # 20-29 years since smoked
                    RRfc = self.former_smoker_RR[4, sex]
                elif years_since_smoked < 40:
                    # 30-39 years since smoked
                    RRfc = self.former_smoker_RR[5, sex]
                elif years_since_smoked < 50:
                    # 40-49 years since smoked
                    RRfc = self.former_smoker_RR[6, sex]
                else:
                    # >= 50 years since smoked
                    RRfc = self.former_smoker_RR[7, sex]

                # fdr -> (adr RRfc * RRsn)/(pn + ps * RRsn + pf * RRfc * RRsn)

                return (adr * RRfc * RRsn) / (pn + ps * RRsn + pf * RRfc * RRsn)
            elif p[6] or p[7] or ever_smoker:
                # current smoker
                # sdr == (adr * RRsn)/(pn + ps * RRsn + pf * RRfc * RRsn)

                res = min((adr * RRsn) / (pn + ps *
                          RRsn + pf * RRfc * RRsn), 1.0)
                return res
            elif not ever_smoker:
                # never smoker
                # ndr -> adr/(pn + ps RRsn + pf RRfc RRsn)

                return adr / (pn + ps * RRsn + pf * RRfc * RRsn)

        print("While trying to determine person's death chance, they didn't fit into any smoking category")
        raise Exception

    def simulate_deaths(self, in_arr2345, in_arr1, in_arr6, current_year: int, use_previous_smoking_state: bool = False, in_arr6_noncohort = None):
        """
        Move people (rows) from in_arr2345, in_arr1 to in_arr6 as they die.
        Calculate mortality rates and use them as death chances for individual people.
        """

        # probability of death for each person
        arr2345_death_rates = None
        arr1_death_rates = None

        if self.use_adjusted_death_rates:
            arr2345_death_rates = np.array([self.person_to_death_rate(row, ever_smoker=True, current_year=current_year,
                                           use_previous_smoking_state=use_previous_smoking_state) for row in in_arr2345]).astype(np.float64)
            arr1_death_rates = np.array([self.person_to_death_rate(row, ever_smoker=False, current_year=current_year,
                                        use_previous_smoking_state=use_previous_smoking_state) for row in in_arr1]).astype(np.float64)

            try:
                assert(np.max(arr2345_death_rates) - 1 < 0.2)
                assert(np.min(arr2345_death_rates) >= 0)
                assert(np.max(arr1_death_rates) - 1 < 0.2)
                assert(np.min(arr1_death_rates) >= 0)
            except:
                print("""
                print(np.max(arr2345_death_rates))
                print(np.min(arr2345_death_rates))
                print(np.max(arr1_death_rates))
                print(np.min(arr1_death_rates))
                """)
                print(np.max(arr2345_death_rates))
                print(np.min(arr2345_death_rates))
                print(np.max(arr1_death_rates))
                print(np.min(arr1_death_rates))

        else:
            life_table_year = min(self.start_year - 2, 2018)
            life_table_year = max(life_table_year, 2016)

            adr_male = self.life_tables[life_table_year][0].astype(np.float64)
            adr_female = self.life_tables[life_table_year][1].astype(
                np.float64)

            arr2345_ages = in_arr2345[:, 11].astype(np.int32)
            arr2345_ages = list(arr2345_ages.clip(min=0, max=100))

            arr1_ages = in_arr1[:, 11].astype(np.int32)
            arr1_ages = list(arr1_ages.clip(min=0, max=100))

            arr2345_death_chances_male = adr_male[arr2345_ages]
            arr2345_death_chances_female = adr_female[arr2345_ages]

            arr1_death_chances_male = adr_male[arr1_ages]
            arr1_death_chances_female = adr_female[arr1_ages]

            arr2345_death_rates = in_arr2345[:, 12] * arr2345_death_chances_female + (
                1 - in_arr2345[:, 12]) * arr2345_death_chances_male
            arr1_death_rates = in_arr1[:, 12] * arr1_death_chances_female + (
                1 - in_arr1[:, 12]) * arr1_death_chances_male

            assert(np.max(arr2345_death_rates) <= 1)
            assert(np.min(arr2345_death_rates) >= 0)
            assert(np.max(arr1_death_rates) <= 1)
            assert(np.min(arr1_death_rates) >= 0)

        # determine deaths in arr2345
        chance_2345 = np.random.rand(len(in_arr2345)).astype(np.float64)
        deaths_2345 = arr2345_death_rates > chance_2345

        # determine deaths in arr1
        chance_1 = np.random.rand(len(in_arr1)).astype(np.float64)
        deaths_1 = arr1_death_rates > chance_1

        # now it is time to put dead people in arr6 and arr6_noncohort

        """
        The Life Years Lived (LYL) cohort minimum age is 18 in 2035, and max age is 74 in 2035
        But, we write out data at the start of the year, then calculate deaths, then age up people.
        So, we are actually looking at people who are 18-74 in 2034, since that data will be written for 2035
        So, someone's birth year has to be 1960-2016 (inclusive).
        Birth year can be calculated as current year - age
        """
        LYL_cohort_max_birth_year = 2016
        LYL_cohort_min_birth_year = 1960

        # determine who will be in the life years lived (LYL) cohort
        arr1_ages = in_arr1[:, 11].astype(np.int32)
        arr1_inLylCohort = np.logical_and(
            (current_year - arr1_ages) <= LYL_cohort_max_birth_year,
            (current_year - arr1_ages) >= LYL_cohort_min_birth_year,
        )
        arr2345_ages = in_arr2345[:, 11].astype(np.int32)
        arr2345_inLylCohort = np.logical_and(
            (current_year - arr2345_ages) <= LYL_cohort_max_birth_year,
            (current_year - arr2345_ages) >= LYL_cohort_min_birth_year,
        )

        arr2345_dead_LYL = np.copy(in_arr2345)[deaths_2345 & arr2345_inLylCohort]
        arr2345_dead_nonLYL  = np.copy(in_arr2345)[deaths_2345 & np.logical_not(arr2345_inLylCohort)]
            
        arr1_dead_LYL = np.copy(in_arr1)[deaths_1 & arr1_inLylCohort]
        arr1_dead_nonLYL  = np.copy(in_arr1)[deaths_1 & np.logical_not(arr1_inLylCohort)]

        # put dead people in the appropriate arr6
        # start with peole in the LYL cohort

        if in_arr6 is None or (len(in_arr6) == 0):
            in_arr6 = np.concatenate([
                arr1_dead_LYL,
                arr2345_dead_LYL,
            ])
        else:
            in_arr6 = np.concatenate([
                in_arr6, 
                arr1_dead_LYL,
                arr2345_dead_LYL,
            ])
            
        if (in_arr6_noncohort is None) or (len(in_arr6_noncohort) == 0):
            in_arr6_noncohort = np.concatenate([
                arr1_dead_nonLYL,
                arr2345_dead_nonLYL,
            ])
        else:
            in_arr6_noncohort = np.concatenate([
                in_arr6_noncohort, 
                arr1_dead_nonLYL,
                arr2345_dead_nonLYL,
            ])

        # take the dead people out of arr2345, arr1
        in_arr2345 = in_arr2345[np.logical_not(deaths_2345)]
        in_arr1 = in_arr1[np.logical_not(deaths_1)]

        return in_arr2345, in_arr1, in_arr6, in_arr6_noncohort

    def sample_disease_outcomes(self, cy, arr2345, arr1):
        """
        Estimate 1-year incidence of CVD and LC for 65 year olds

        for each person, get the cvd_risk and ls_risk 
        these are a number between 0 and 1 that represents the chance
        that the person gets cvd or ls in the next year.
        These come from Duncan et al. 2019, Freeman et al. 2008

        Then, use those risks to sample whether or not they got the disease.
        Count the total number of diseases gotten.
        """
        current_year = cy+self.start_year

        # num_65yo = 0
        num_gotLC = 0
        num_gotCVD = 0

        # either current or former smokers
        for p in arr2345:
            # only estimate incidence for 65 year olds
            if p[11] != 65:
                continue

            # track the total number of 65 year olds, for a denominator
            self.total_65yos["total"] += p[15]
            if p[10]:
                # black
                self.total_65yos["black"] += p[15]
            else:
                # nonblack
                self.total_65yos["nonblack"] += p[15]
            if p[13]:
                # pov
                self.total_65yos["pov"] += p[15]
            else:
                # nonpov
                self.total_65yos["nonpov"] += p[15]

            # num_65yo += 1

            cvd_risk = None
            lc_risk = None
            if p[5]:
                # former smoker
                years_since_smoked = current_year - int(p[16])
                try:
                    assert(years_since_smoked >= 0)
                except AssertionError:
                    print(years_since_smoked)
                    print(p[16])
                    print(current_year)
                    raise
                assert(isinstance(years_since_smoked, int))

                if years_since_smoked < 5:
                    # 1-4 years since smoked
                    cvd_risk = 6.94 / 1000
                    if p[12]:
                        # female
                        lc_risk = 377.5 / 100000
                    else:
                        # male
                        lc_risk = 451.8 / 100000
                elif years_since_smoked < 10:
                    # 5-9 years since smoked
                    cvd_risk = 7.04 / 1000
                    if p[12]:
                        # female
                        lc_risk = 248.5 / 100000
                    else:
                        # male
                        lc_risk = 285.1 / 100000
                elif years_since_smoked < 15:
                    # 10-14 years since smoked
                    cvd_risk = 6.31 / 1000
                    if p[12]:
                        # female
                        lc_risk = 109.1 / 100000
                    else:
                        # male
                        lc_risk = 97.4 / 100000
                elif years_since_smoked < 25:
                    # 15-24 years since smoked
                    cvd_risk = 6.11 / 1000
                    if p[12]:
                        # female
                        lc_risk = 109.1 / 100000
                    else:
                        # male
                        lc_risk = 97.4 / 100000
                else:
                    # 25 or more years since smoked
                    cvd_risk = 5.02 / 1000
                    if p[12]:
                        # female
                        lc_risk = 109.1 / 100000
                    else:
                        # male
                        lc_risk = 97.4 / 100000

            else:
                # current smoker
                cvd_risk = 5.02 / 1000
                if p[12]:
                    # female
                    lc_risk = 612.8 / 100000
                else:
                    # male
                    lc_risk = 732.8 / 100000

            # Now we have the cvd_risk and lc_risk for former and current smokers
            # sample whether or not they get it
            gets_cvd = np.random.rand() < cvd_risk
            gets_lc = np.random.rand() < lc_risk

            num_gotCVD += gets_cvd
            num_gotLC += gets_lc

            # Track a weighted count of who gets what
            if gets_cvd:
                self.num_cvd_cases["total"] += p[15]
                if p[10]:
                    # black
                    self.num_cvd_cases["black"] += p[15]
                else:
                    # nonblack
                    self.num_cvd_cases["nonblack"] += p[15]
                if p[13]:
                    # pov
                    self.num_cvd_cases["pov"] += p[15]
                else:
                    # nonpov
                    self.num_cvd_cases["nonpov"] += p[15]

            if gets_lc:
                self.num_lc_cases["total"] += p[15]
                if p[10]:
                    # black
                    self.num_lc_cases["black"] += p[15]
                else:
                    # nonblack
                    self.num_lc_cases["nonblack"] += p[15]
                if p[13]:
                    # pov
                    self.num_lc_cases["pov"] += p[15]
                else:
                    # nonpov
                    self.num_lc_cases["nonpov"] += p[15]

        for p in arr1:
            if p[11] != 65:
                continue

            # track the total number of 65 year olds, for a denominator
            self.total_65yos["total"] += p[15]
            if p[10]:
                # black
                self.total_65yos["black"] += p[15]
            else:
                # nonblack
                self.total_65yos["nonblack"] += p[15]
            if p[13]:
                # pov
                self.total_65yos["pov"] += p[15]
            else:
                # nonpov
                self.total_65yos["nonpov"] += p[15]


            # num_65yo += 1

            cvd_risk = None
            lc_risk = None
            # never smoker
            cvd_risk = 5.09 / 1000
            if p[12]:
                # female
                lc_risk = 25.3 / 100000
            else:
                # male
                lc_risk = 20.3 / 100000

            # sample whether or not they get it
            gets_cvd = np.random.rand() < cvd_risk
            gets_lc = np.random.rand() < lc_risk

            # num_gotCVD += gets_cvd
            # num_gotLC += gets_lc

            if gets_cvd:
                self.num_cvd_cases["total"] += p[15]
                if p[10]:
                    # black
                    self.num_cvd_cases["black"] += p[15]
                else:
                    # nonblack
                    self.num_cvd_cases["nonblack"] += p[15]
                if p[13]:
                    # pov
                    self.num_cvd_cases["pov"] += p[15]
                else:
                    # nonpov
                    self.num_cvd_cases["nonpov"] += p[15]

            if gets_lc:
                self.num_lc_cases["total"] += p[15]
                if p[10]:
                    # black
                    self.num_lc_cases["black"] += p[15]
                else:
                    # nonblack
                    self.num_lc_cases["nonblack"] += p[15]
                if p[13]:
                    # pov
                    self.num_lc_cases["pov"] += p[15]
                else:
                    # nonpov
                    self.num_lc_cases["nonpov"] += p[15]

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

        s2 = a[:, 3]
        s3 = a[:, 4]
        ia = a[:, 5]
        a = np.concatenate([
            np.ones((a.shape[0], 1)),
            (s2 == 1)[:, np.newaxis],
            (s2 == 2)[:, np.newaxis],
            (s2 == 3)[:, np.newaxis],
            (s2 == 4)[:, np.newaxis],
            (s3 == 2)[:, np.newaxis],
            (s3 == 3)[:, np.newaxis],
            (s3 == 4)[:, np.newaxis],
            (ia == 1)[:, np.newaxis],
            (ia == 2)[:, np.newaxis],
            a[:, 2][:, np.newaxis],  # black
            a[:, 9][:, np.newaxis],  # age
            a[:, 1][:, np.newaxis] - 1,  # change sex from {1,2} to {0,1}
            # poverty is already {0,1} now, not {1,2} like before
            a[:, 6][:, np.newaxis],
            a[:, 10][:, np.newaxis],  # start age
            a[:, 8][:, np.newaxis],  # weight
            # year last smoked, initialize to -1 for nonsmokers
            -1 * np.ones((a.shape[0], 1)),
        ], axis=1)
        return a

    def cohort_to_indicator_form(self, c, current_year: int = 2016):
        # get it in PATH form (each row a person)
        # with columns like the path population spreadsheet
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
                0 + 17 * (round(row[4]) == 1) + 18 * \
                (round(row[4]) == 2),    # start_age
            ]), (round(row[7]), 1))
            for row in c], axis=0)

        path_form_arr = path_form_arr.astype(np.float64)

        arr2345 = np.asarray([row for row in path_form_arr
                              if (row[4] == 2 or row[4] == 3 or row[4] == 4 or row[4] == 5
                                  or row[3] == 2 or row[3] == 3 or row[3] == 4 or row[3] == 5)])
        arr1 = np.asarray([row for row in path_form_arr
                           if (row[4] == 1 and row[3] == 1)])

        arr2345 = self.path_to_indicator_form(arr2345)
        arr1 = self.path_to_indicator_form(arr1)

        # assign the last year smoked

        # for people whose last state is 3,4 the year last smoked is current_year - 1
        arr2345[np.logical_or(arr2345[:, 3], arr2345[:, 4]),
                16] = current_year - 1

        # for people currently in groups 3,4 the year last smoked is self.start_year
        arr2345[np.logical_or(arr2345[:, 6], arr2345[:, 7]), 16] = current_year

        # for people whose last state is 5, the year last smoked is self.start_year - 1
        # we are treating ecig users the same as smokers here
        arr2345[np.sum(arr2345[:, 1:5], axis=1) == 0, 16] = current_year - 1

        # for people whose current state is 5, the year last smoked is self.start_year
        arr2345[np.sum(arr2345[:, 5:8], axis=1) == 0, 16] = current_year

        # TODO: randomize year last smoked for people with ia 1 as well
        # for people in group 2 last state AND this state
        # if initialization age is 1 then year last smoked is self.year_last_smoked_for_ia1 + self.start_year - age
        ind = np.logical_and(
            arr2345[:, 2], arr2345[:, 5], arr2345[:, 8]).astype(np.bool_)
        arr2345[ind, 16] = self.age_last_smoked_for_ia1 + \
            self.start_year - arr2345[ind, 11]

        # if initialization age is 2 for former smokers then year last smoked is randomly chosen between start_age and current age
        ind = np.logical_and(
            arr2345[:, 2], arr2345[:, 5], arr2345[:, 9]).astype(np.bool_)
        # use starting age if available, otherwise use 16
        age_started = np.maximum(16, arr2345[ind, 14])
        to_multiply_rand = arr2345[ind, 11] - age_started + 1 - 1e-8
        to_add_after_multiply = current_year - arr2345[ind, 11] - 0.5 + 1e-8
        arr2345[ind, 16] = np.round(np.random.rand(
            np.sum(ind)) * to_multiply_rand + to_add_after_multiply)

        # for people whose current state is 2 and previous state is 1, the year last smoked is this year
        ind = np.logical_and(arr2345[:, 1], arr2345[:, 5]).astype(np.bool_)
        arr2345[ind, 16] = current_year

        # finally, check that no one is left with a year last smoked = -1
        assert(np.all(arr2345[:, 16] != -1))

        return arr2345, arr1

    def zero_a_prob(self, probs, idx):
        """
        probs sum to one along the 2nd (last axis)
        zero a column while maintaining the sum-to-1 property
        """
        probs[:, idx] = 0
        probs *= np.sum(probs, axis=1)

    def write_data(self, cy, arr2345, arr1, arr6, output_list_to_df, output_numpy):
        """
        Given the current year, arrays with the current state,
        and output destination arrays, write data accordingly
        """
        assert(len(output_numpy.shape) == 5)

        # probably a way to do this without loops but idk

        # TODO: account for arr6_noncohort (not LYL cohort) in total deaths. 
        # We aren't actually using those non-cohort death numbers though, so I won't bother accounting for that.
        for black in [0, 1]:
            for pov in [0, 1]:
                for plus65 in [0, 1]:
                    for smoking_state in [1, 2, 3, 4, 5, 6]:
                        # determine count of people which fit the descriptors
                        # count is weighted
                        # note smoking state == 6 means dead
                        count = None
                        if smoking_state == 5 and arr2345 is None:
                            count = 0
                        elif smoking_state == 5:
                            count = np.sum(
                                (arr2345[:, 10] == black) *
                                (arr2345[:, 13] == pov) *
                                # check for if age is 65 plus
                                ((arr2345[:, 11] >= 65) == plus65) *
                                (arr2345[:, 5] == 0) *
                                (arr2345[:, 6] == 0) *
                                (arr2345[:, 7] == 0) *
                                (arr2345[:, 15])
                            )
                        elif smoking_state == 6 and arr6 is not None:
                            count = np.sum(
                                (arr6[:, 10] == black) *
                                (arr6[:, 13] == pov) *
                                # check for if age is 65 plus
                                ((arr6[:, 11] >= 65) == plus65) *
                                (arr6[:, 15])
                            )
                        elif smoking_state == 6 and arr6 is None:
                            count = 0
                        elif smoking_state == 1 and arr1 is None:
                            count = 0
                        elif smoking_state == 1:
                            count = np.sum(
                                (arr1[:, 10] == black) *
                                (arr1[:, 13] == pov) *
                                # check for if age is 65 plus
                                ((arr1[:, 11] >= 65) == plus65) *
                                (arr1[:, 15])
                            )
                        elif arr2345 is None and arr1 is None:
                            count = 0
                        elif arr2345 is None:
                            count = 0
                        elif smoking_state in [2, 3, 4]:
                            # smoking state must be 2, 3, or 4
                            count = np.sum(
                                (arr2345[:, 10] == black) *
                                (arr2345[:, 13] == pov) *
                                # check for if age is 65 plus
                                ((arr2345[:, 11] >= 65) == plus65) *
                                arr2345[:, 4 + smoking_state - 1] *
                                (arr2345[:, 15])
                            )
                        else:
                            raise Exception

                        # write list and numpy arr
                        output_list_to_df.append([
                            cy + self.start_year,
                            black,
                            pov,
                            plus65,
                            smoking_state,
                            count,
                        ])

                        output_numpy[cy, black, pov, plus65,
                                     smoking_state - 1] = count
                        # endfor

        if (self.simulate_disease):
            # CVD
            self.output_cvd[cy, 0] = self.num_cvd_cases["total"]
            self.output_cvd[cy, 1] = self.num_cvd_cases["black"]
            self.output_cvd[cy, 2] = self.num_cvd_cases["nonblack"]
            self.output_cvd[cy, 3] = self.num_cvd_cases["pov"]
            self.output_cvd[cy, 4] = self.num_cvd_cases["nonpov"]

            # LC
            self.output_lc[cy, 0] = self.num_lc_cases["total"]
            self.output_lc[cy, 1] = self.num_lc_cases["black"]
            self.output_lc[cy, 2] = self.num_lc_cases["nonblack"]
            self.output_lc[cy, 3] = self.num_lc_cases["pov"]
            self.output_lc[cy, 4] = self.num_lc_cases["nonpov"]

            # total number of 65 year olds who could've got a disease
            self.output_65yos[cy, 0] = self.total_65yos["total"]
            self.output_65yos[cy, 1] = self.total_65yos["black"]
            self.output_65yos[cy, 2] = self.total_65yos["nonblack"]
            self.output_65yos[cy, 3] = self.total_65yos["pov"]
            self.output_65yos[cy, 4] = self.total_65yos["nonpov"]
        return output_list_to_df, output_numpy

    def set_year_last_smoked(self, in_arr2345, current_year: int = 2016):
        """
        Here we figure out the year_last_smoked variable for all cases

        This is to be used for the arr2345 array from the original PATH population
        """

        # for people whose last state is 3,4 the year last smoked is self.start_year - 1
        in_arr2345[np.logical_or(
            in_arr2345[:, 3], in_arr2345[:, 4]), 16] = current_year - 1

        # for people currently in groups 3,4 the year last smoked is self.start_year
        in_arr2345[np.logical_or(
            in_arr2345[:, 6], in_arr2345[:, 7]), 16] = current_year

        # for people whose last state is 5, the year last smoked is self.start_year - 1
        # we are treating ecig users the same as smokers here
        in_arr2345[np.sum(in_arr2345[:, 1:5], axis=1)
                   == 0, 16] = current_year - 1

        # for people whose current state is 5, the year last smoked is self.start_year
        in_arr2345[np.sum(in_arr2345[:, 5:8], axis=1) == 0, 16] = current_year

        # TODO: randomize year last smoked for people with ia 1 as well
        # for people in group 2 last state AND this state
        # if initialization age is 1 then year last smoked is self.age_last_smoked_for_ia1 + self.start_year - age
        ind = np.logical_and(
            in_arr2345[:, 2], in_arr2345[:, 5], in_arr2345[:, 8]).astype(np.bool_)
        in_arr2345[ind, 16] = self.age_last_smoked_for_ia1 + \
            current_year - in_arr2345[ind, 11]

        # if initialization age is 2 for former smokers then year last smoked is randomly chosen between start_age and current age - 2
        ind = np.logical_and(
            in_arr2345[:, 2], in_arr2345[:, 5], in_arr2345[:, 9]).astype(np.bool_)
        # use starting age if available, otherwise use 18
        age_started = np.maximum(18, in_arr2345[ind, 14])
        # doing some tricks here to select a random integer by sampling a random float
        to_multiply_rand = (in_arr2345[ind, 11] - 1) - age_started + 1 - 1e-8
        to_add_after_multiply = self.start_year - \
            in_arr2345[ind, 11] - 0.5 + 1e-8
        in_arr2345[ind, 16] = np.round(np.random.rand(
            np.sum(ind)) * to_multiply_rand + to_add_after_multiply)

        # for people whose current state is 2 and previous state is 1, the year last smoked is this year
        ind = np.logical_and(
            in_arr2345[:, 1], in_arr2345[:, 5]).astype(np.bool_)
        in_arr2345[ind, 16] = current_year

        # finally, check that no one is left with a year last smoked = -1
        assert(np.all(in_arr2345[:, 16] != -1))
        return in_arr2345

    def random_select_arg_multinomial(self, probs):
        """"
        Takes in probs which are like [0.1, 0.2, 0.3, 0.2, 0.2] -- sum to 1
        returns indicator for next state
        in a format like: [0,0,1,0,0]
        return array has same length as input array "probs"
        """
        if probs is None:
            return None

        chance = np.random.rand(probs.shape[0], 1)
        forward = np.concatenate([chance < np.sum(probs[:, :i], axis=1)[
                                 :, np.newaxis] for i in range(1, probs.shape[1] + 1)], axis=1)
        backward = np.concatenate([(1 - chance) < np.sum(probs[:, i:], axis=1)[
                                  :, np.newaxis] for i in range(probs.shape[1])], axis=1)
        arg_selection = forward * backward
        return arg_selection

    def get_augmented_betas(self):
        beta_2345_aug = np.concatenate([
            self.beta2345,
            np.zeros((len(self.beta2345), 3)),
        ], axis=1)

        beta_1_aug = np.concatenate([
            self.beta1[:, 0][:, np.newaxis],
            np.zeros((len(self.beta1), 9)),
            self.beta1[:, 1:],
            np.zeros((len(self.beta1), 3)),
        ], axis=1)

        beta_2345_aug = np.transpose(beta_2345_aug)
        beta_1_aug = np.transpose(beta_1_aug)

        return beta_2345_aug, beta_1_aug

    def get_transition_probs_from_LR(self, in_arr2345, in_arr1, in_beta_2345_aug, in_beta_1_aug):
        """
        Gets the transition probabilities for everyone

        Transition probabilities have the shape (numpeople, 5)
        Each row corresponds to a person
        Each column corresponds to the transition probability for that state. Listed by index:
            0 - never smoker
            1 - former smoker
            2 - menthol smoker
            3 - nonmenthol smoker
            4 - ecig/dual smoker
        """

        if in_arr2345 is not None:
            logits_2345 = np.matmul(
                in_arr2345, in_beta_2345_aug).astype(np.float64)
            assert(logits_2345.shape[1] == 3)

        if in_arr1 is not None:
            logits_1 = np.matmul(in_arr1, in_beta_1_aug).astype(np.float64)
            assert(logits_1.shape[1] == 4)

        # convert logits to probabilities

        if in_arr2345 is not None:
            exps = np.exp(logits_2345)
            p4 = 1 / (1 + np.sum(exps, axis=1))
            probs2345 = np.asarray([
                np.zeros(in_arr2345.shape[0]),  # p1
                p4*exps[:, 0],  # p2
                p4*exps[:, 1],  # p3
                p4,             # p4
                p4*exps[:, 2],  # p5
            ]).transpose()
        else:
            probs2345 = None

        if in_arr1 is not None:
            exps = np.exp(logits_1)
            p4 = 1 / (1 + np.sum(exps, axis=1))
            probs1 = np.asarray([
                p4*exps[:, 0],  # p1
                p4*exps[:, 1],  # p2
                p4*exps[:, 2],  # p3
                p4,             # p4
                p4*exps[:, 3],  # p5
            ]).transpose()
        else:
            probs1 = None

        return probs2345, probs1

    def adjust_transition_probs_according_to_menthol_ban(self, in_arr2345, in_arr1, in_probs2345, in_probs1, current_year: int, shortbanparams=None, longbanparams=None):
        """
        Adjust transition probabilities according to menthol ban scenario parameters.

        There will be a short-term effect and a long-term effect.
        """

        if current_year == self.menthol_ban_year - self.start_year:
            """
            Instantaneous menthol ban effects at year 1:

            Example option:
            Among those 25+ years, 
                23% of menthol cigarette smokers quit smoking, 
                44% of menthol cigarette smokers switch to non-menthol cigarettes (state 4), 
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

            probs_25minus = None
            probs_25plus = None

            if shortbanparams is None:
                raise Exception("""You need to supply shortban params for menthol ban scenario. 
                                This is what I am doing in the uncertainty analysis.""")

            assert shortbanparams.shape == (2, 5)
            probs_25minus = shortbanparams[0]
            probs_25plus = shortbanparams[1]

            try:
                assert(abs(sum(probs_25minus) - 1.) < 1e-5)
                assert(abs(sum(probs_25plus) - 1.) < 1e-5)
            except:
                print(sum(probs_25minus))
                print(sum(probs_25plus))
                raise AssertionError

            probs_25minus /= np.sum(probs_25minus)
            probs_25plus /= np.sum(probs_25plus)

            # for indexing by age less than or greater than 25
            are_25plus_2345 = in_arr2345[:, 11] >= 25
            are_25minus_2345 = np.logical_not(are_25plus_2345)
            are_25plus_1 = in_arr1[:, 11] >= 25
            are_25minus_1 = np.logical_not(are_25plus_1)

            for i in [1, 3, 4]:
                # i = 1,3,4 = former, nonmenthol, ecig
                # 2 = menthol
                in_probs2345[are_25plus_2345, i] += probs_25plus[i] * \
                    in_probs2345[are_25plus_2345, 2]
                in_probs2345[are_25minus_2345, i] += probs_25minus[i] * \
                    in_probs2345[are_25minus_2345, 2]

                in_probs1[are_25plus_1, i] += probs_25plus[i] * \
                    in_probs1[are_25plus_1, 2]
                in_probs1[are_25minus_1, i] += probs_25minus[i] * \
                    in_probs1[are_25minus_1, 2]

            in_probs2345[are_25plus_2345, 2] *= probs_25plus[2]
            in_probs2345[are_25minus_2345, 2] *= probs_25minus[2]

            in_probs1[are_25plus_1, 2] *= probs_25plus[2]
            in_probs1[are_25minus_1, 2] *= probs_25minus[2]
        elif current_year > self.menthol_ban_year - self.start_year:
            """
            Long term menthol ban effects

            Prob that menthol smoker "remains the same" is reduced
            Where those menthol smokers that would have stayed the same
            go varies from option to option. Either quit (former smoker) or nonmenthol smoker or ecig

            All non-menthol smokers have 0 chance to transition to menthol smoking
            the other probabilites are scaled so that all sum to 1.0
            """

            if longbanparams is not None:
                are_menthol_smokers = in_arr2345[:, 6] == 1
                not_menthol_smokers = in_arr2345[:, 6] == 0
                assert(len(in_probs2345) == sum(
                    are_menthol_smokers) + sum(not_menthol_smokers))
                assert(len(in_probs2345) == len(in_arr2345))

                # long term ban params are 4 numbers:
                # 1. proportion of menthol-continuers that transition to former smoker
                # 2. proportion of menthol-continuers that still continue smoking menthol
                # 3. proportion of menthol-continuers that transition to non-menthol smoker
                # 4. proportion of menthol-continuers that transition to ecig/dual

                """
                For the first 5 years of the ban, we will modify the transition probabilities of menthol smokers
                If someone is still smoking menthol after 5 years of ban, we no longer modify their transition probabilities
                (they are considered a staunch or hard-to-change person)
                """
                if current_year < self.menthol_ban_year - self.start_year + 5:
                    tmp = in_probs2345[are_menthol_smokers]
                    tmp[:, 1] += tmp[:, 2] * longbanparams[0]  # to former
                    tmp[:, 3] += tmp[:, 2] * longbanparams[2]  # to nonmenthol
                    tmp[:, 4] += tmp[:, 2] * longbanparams[3]  # to ecig
                    tmp[:, 2] *= longbanparams[1]  # still menthol
                    in_probs2345[are_menthol_smokers] = tmp

                # Make the chance that people transition from something else to menthol smoking zero
                tmp = in_probs2345[not_menthol_smokers]
                tmp[:, 2] = np.zeros_like(tmp[:, 2])
                sum_chances = np.sum(tmp, axis=1)
                tmp /= sum_chances.reshape(-1, 1)
                in_probs2345[not_menthol_smokers] = tmp

                # Don't forget about arr1 (never smokers)
                in_probs1[:, 2] = np.zeros_like(in_probs1[:, 2])
                sum_chances = np.sum(in_probs1, axis=1)
                in_probs1 /= sum_chances.reshape(-1, 1)

            else:
                # I used to code several long-term ban options explicitly, but that is deprecated now.
                raise Exception(
                    "We have a menthol ban but no long-term ban parameters.")

        return in_probs2345, in_probs1

    def adjust_transition_probs_according_to_initiation_cessation_params(self, in_probs2345, in_probs1):
        """
        Tune probabilities according to initiation_rate_decrease
        and continuation_rate_decrease parameters. 
        Note: continuation is the opposite of cessation.

        The initiation_rate_decrease param tells you by
        how much to deacrease the initiation rate, that is,
        1 - (probability of a never smoker staying a never smoker).

        The continuation_rate_decrease param tells you by how much to multiply
        the cessation probability, i.e. the probability that people 
        transition into group 2 (former smokers). "Reducing continuation"
        is semantically equivalent to "increasing cessation."

        E.g. If the probability of a person making the transition 1->1
        is .8, and we decrease the initiation rate by 30%, then the
        new probability of that person making the 1->1 is .86. 

        E.g. If the probability of a person making the transition 3->2
        has probability 0.1 and the continuation rate is decreased by 30%
        then the new probability of making the transition 3->2 is .37.
        """

        if self.initiation_rate_decrease > 0 and in_probs1 is not None:
            in_probs1[:, 0] += (1 - in_probs1[:, 0]) * \
                self.initiation_rate_decrease
            in_probs1[:, 1:] -= in_probs1[:, 1:] * \
                self.initiation_rate_decrease

        # the first column of in_probs2345 is all zeros
        if self.continuation_rate_decrease > 0 and in_probs2345 is not None:
            in_probs2345[:, 1] += (1 - in_probs2345[:, 1]) * \
                self.continuation_rate_decrease
            in_probs2345[:, 2:] -= in_probs2345[:, 2:] * \
                self.continuation_rate_decrease

        return in_probs2345, in_probs1

    def record_transitions(self, in_arr2345, in_arr1, in_leaving_1):
        """ 
        Records how many of each transision is made in a year.

        we can calculate the number who died also from these numbers
        Here's what the list means
        list index | number of people in transition
        0 1->1
        1 1->2
        2 1->3
        3 1->4
        4 1->5
        5 2->2
        6 2->3
        7 2->4
        8 2->5
        9 3->2
        10 3->3
        11 3->4
        12 3->5
        13 4->2
        14 4->3
        15 4->4
        16 4->5
        17 5->1
        18 5->2
        19 5->3
        20 5->4
        21 5->5
        """

        to_append = [
            np.sum(in_arr1[:, 15][np.logical_not(in_leaving_1)]),
            np.sum(in_arr1[:, 15][np.logical_and(
                in_arr1[:, 1], in_arr1[:, 5])]),
            np.sum(in_arr1[:, 15][np.logical_and(
                in_arr1[:, 1], in_arr1[:, 6])]),
            np.sum(in_arr1[:, 15][np.logical_and(
                in_arr1[:, 1], in_arr1[:, 7])]),
            0,  # placeholder -- fill in after this
            np.sum(in_arr2345[:, 15][np.logical_and(
                in_arr2345[:, 2], in_arr2345[:, 5])]),
            np.sum(in_arr2345[:, 15][np.logical_and(
                in_arr2345[:, 2], in_arr2345[:, 6])]),
            np.sum(in_arr2345[:, 15][np.logical_and(
                in_arr2345[:, 2], in_arr2345[:, 7])]),
            np.sum(in_arr2345[:, 15][np.logical_and(in_arr2345[:, 2], np.sum(
                in_arr2345[:, 5:8], axis=1) == 0)]),  # state 5 indicator not explictly tracked
            np.sum(in_arr2345[:, 15][np.logical_and(
                in_arr2345[:, 3], in_arr2345[:, 5])]),
            np.sum(in_arr2345[:, 15][np.logical_and(
                in_arr2345[:, 3], in_arr2345[:, 6])]),
            np.sum(in_arr2345[:, 15][np.logical_and(
                in_arr2345[:, 3], in_arr2345[:, 7])]),
            np.sum(in_arr2345[:, 15][np.logical_and(in_arr2345[:, 3], np.sum(
                in_arr2345[:, 5:8], axis=1) == 0)]),  # state 5 indicator not explictly tracked
            np.sum(in_arr2345[:, 15][np.logical_and(
                in_arr2345[:, 4], in_arr2345[:, 5])]),
            np.sum(in_arr2345[:, 15][np.logical_and(
                in_arr2345[:, 4], in_arr2345[:, 6])]),
            np.sum(in_arr2345[:, 15][np.logical_and(
                in_arr2345[:, 4], in_arr2345[:, 7])]),
            np.sum(in_arr2345[:, 15][np.logical_and(in_arr2345[:, 4], np.sum(
                in_arr2345[:, 5:8], axis=1) == 0)]),  # state 5 indicator not explictly tracked
            0,  # 5 -> 1 not allowed
            np.sum(in_arr2345[:, 15][np.logical_and(
                np.sum(in_arr2345[:, 1:5], axis=1) == 0, in_arr2345[:, 5])]),
            np.sum(in_arr2345[:, 15][np.logical_and(
                np.sum(in_arr2345[:, 1:5], axis=1) == 0, in_arr2345[:, 6])]),
            np.sum(in_arr2345[:, 15][np.logical_and(
                np.sum(in_arr2345[:, 1:5], axis=1) == 0, in_arr2345[:, 7])]),
            np.sum(in_arr2345[:, 15][np.logical_and(np.sum(in_arr2345[:, 1:5], axis=1) == 0, np.sum(
                in_arr2345[:, 5:8], axis=1) == 0)]),  # state 5 indicator not explictly tracked
        ]
        to_append[4] = np.sum(in_arr1[:, 15]) - np.sum(to_append[:4])
        return to_append

    def calibrate_smoking_percentage(self, in_arr2345: np.ndarray, in_arr1: np.ndarray, target_smoker_percentage: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calibrates the weights of a population so that the percentage of smokers
        is equal to some target percentage.

        Takes the indicator-form numpy arrays to represent the population.

        Returns a tuple of the modified indicator-form numpy arrays.

        Input arrays are not mutated!
        """

        smokers_arr = np.sum(in_arr2345[:, 6:8], axis=1, dtype=bool)
        nonsmokers_2345_arr = np.logical_not(smokers_arr)

        # sanity check
        assert(np.sum(smokers_arr) +
               np.sum(nonsmokers_2345_arr) == len(in_arr2345))

        smoker_weight = np.sum(in_arr2345[:, 15][smokers_arr])
        nonsmoker_weight = np.sum(
            in_arr1[:, 15]) + np.sum(in_arr2345[:, 15][nonsmokers_2345_arr])
        total_weight = np.sum(in_arr2345[:, 15]) + np.sum(in_arr1[:, 15])

        # sanity check
        # some tolerance, these might not be exact due to rounding error
        assert(abs(smoker_weight + nonsmoker_weight - total_weight) < 1e-2)

        smoker_weight_factor = total_weight * target_smoker_percentage / smoker_weight
        nonsmoker_weight_factor = (
            total_weight - total_weight * target_smoker_percentage) / nonsmoker_weight

        out_arr1 = np.copy(in_arr1)
        out_arr2345 = np.copy(in_arr2345)

        out_arr1[:, 15] *= nonsmoker_weight_factor
        out_arr2345[:, 15][nonsmokers_2345_arr] *= nonsmoker_weight_factor
        out_arr2345[:, 15][smokers_arr] *= smoker_weight_factor

        return out_arr2345, out_arr1

    def calibrate_initial_population(self, arr1, arr2345, arr6, beta_1_aug, beta_2345_aug, arr6_noncohort):
        """
        FURTHER CONSTRUCTING THE POPULATION

        At this stage, the PATH population represents demographics from 
        the time of wave 1 (2014), except for the previous and current
        smoking states, which come from wave 2 (2015) and wave 3 (2016).
        All the other covariates such as age, poverty, black, etc. are
        representative of 2014. We can fix this by aging the population
        twice, adding in cohorts of 18-year-olds as we go.

        We also wish to calibrate this population so that the initial
        smoking rate matches the rates given by NHIS data. 

        To accomplish these goals, we do the following:
        1. Simulate death for the population (remove some people from the population)
           using their wave 2 tobacco use states as if they were the wave 1 states
        2. Age the population by one year
        3. Incoporate a cohort of 18-year-olds from wave 2 (2015) into the simulation
        4. Simulate death for the population using wave 2 states again
           NOTE: wave 2 18-year-olds have wave 1 and wave 2 state data
        5. Simulate a tobacco use state for wave 3 for the wave 2 18-year-olds
        6. Age the population by one year
        7. Incorporate the wave 3 18-year-olds into the population
        8. Remove individuals who are 65 and 66 years old
        9. Calibrate the smoking rates to match NHIS data

        After these steps, we have a good starting population for the main simulation loop 
        """

        # Simulate death for 1 year using wave 2 rates:

        arr2345, arr1, arr6, arr6_noncohort = self.simulate_deaths(
            arr2345,
            arr1,
            arr6,
            current_year=self.start_year - 2,
            use_previous_smoking_state=True,
            in_arr6_noncohort=arr6_noncohort,
        )

        # age up the population

        arr2345[:, 11] += 1
        arr1[:, 11] += 1

        # incorporate cohort of 18-year-olds from wave 2
        # don't actually add them yet, because death rates will be calculated slightly differently
        # (using previous smoking states for PATH pop, current smoking states for cohort)

        cohort_idx = 2015  # wave 2
        cohort_arr = self.cohorts[cohort_idx]
        c2345, c1 = self.cohort_to_indicator_form(cohort_arr)

        # deaths using wave 2 smoking states

        arr2345, arr1, arr6, arr6_noncohort = self.simulate_deaths(
            arr2345,
            arr1,
            arr6,
            current_year=self.start_year - 1,
            use_previous_smoking_state=True,
            in_arr6_noncohort=arr6_noncohort,
        )

        c2345, c1, arr6, arr6_noncohort = self.simulate_deaths(
            c2345,
            c1,
            arr6,
            current_year=self.start_year - 1,
            use_previous_smoking_state=False,
            in_arr6_noncohort=arr6_noncohort,
        )

        # Simulate wave 3 smoking states for the wave 2 18-year-olds cohort

        probsc2345, probsc1 = self.get_transition_probs_from_LR(
            c2345, c1, beta_2345_aug, beta_1_aug)
        new_states2345 = self.random_select_arg_multinomial(probsc2345)[:, :]
        new_states1 = self.random_select_arg_multinomial(
            probsc1)[:, :].astype(np.float64)
        leaving_1 = np.sum(new_states1[:, 1:], axis=1).astype(np.bool_)
        c2345[:, 2:5] = c2345[:, 5:8]
        # no more previous never smokers
        c2345[:, 1] = np.zeros(c2345.shape[0])
        c2345[:, 5:8] = new_states2345[:, 1:-1]
        c1[:, 5:8] = new_states1[:, 1:-1]
        tmp_to_2345 = c1[leaving_1]
        c1 = c1[np.logical_not(leaving_1)]
        c2345 = np.concatenate([c2345, tmp_to_2345], axis=0)
        # update year_last_smoked variable (column index 16)
        # smokers currently in state 3,4 get their last year updated
        # this is for 2015 (wave 2)
        c2345[np.logical_or(c2345[:, 6], c2345[:, 7]), 16] = 2015
        # smokers currently in state 5 get their last year updated
        c2345[np.sum(c2345[:, 5:8], axis=1) == 0, 16] = 2015
        # people who made the transition 1->2 get their last year updated
        # this is after switching, so anybody who made that transition will be in arr2345
        c2345[np.logical_and(c2345[:, 1], c2345[:, 5]), 16] = 2015

        # Now actually add the cohort

        arr2345 = np.concatenate([arr2345, c2345], axis=0)
        arr1 = np.concatenate([arr1, c1], axis=0)

        # Age the population

        arr2345[:, 11] += 1
        arr1[:, 11] += 1

        # Incorporate Wave 3 (2016) 18-year-olds

        cohort_idx = 2016  # wave 2
        cohort_arr = self.cohorts[cohort_idx]
        c2345, c1 = self.cohort_to_indicator_form(cohort_arr)
        arr2345 = np.concatenate([arr2345, c2345], axis=0)
        arr1 = np.concatenate([arr1, c1], axis=0)

        # Remove individuals who are 65 or older

        arr2345 = arr2345[arr2345[:, 11] < 65]
        arr1 = arr1[arr1[:, 11] < 65]

        # rewrite year_last_smoked variable for the current year

        arr2345 = self.set_year_last_smoked(
            arr2345, current_year=self.start_year)

        # Calibrate the population to NHIS smoking rate

        arr2345, arr1 = self.calibrate_smoking_percentage(
            arr2345,
            arr1,
            target_smoker_percentage=self.target_initial_smoking_percentage,
        )

        return arr1, arr2345, arr6, arr6_noncohort

    def format_population(self):
        pop_arr = self.pop_df.to_numpy(dtype=np.float64)

        # here we will re-randomize the ages according to the agegrp variable
        # agegrp is at index 0, age is at index 9 at this stage
        for row in pop_arr:
            agegrp = row[0]
            if agegrp == 40:
                row[9] = np.random.randint(59, 65)
            else:
                row[9] = 18.0 + agegrp + np.random.randint(0, 10)

        # now we need to construct 3 arrays which get updated
        # during the course of the simulation
        # one for state {2,3,4,5} called arr2345
        # another for state 1 called arr1
        # another for state 6 = death called arr6
        self.arr2345 = np.asarray([row for row in pop_arr
                                   if (row[4] == 2 or row[4] == 3 or row[4] == 4 or row[4] == 5
                                       or row[3] == 2 or row[3] == 3 or row[3] == 4 or row[3] == 5)])
        self.arr1 = np.asarray([row for row in pop_arr
                                if (row[4] == 1 and row[3] == 1)])
        self.arr6 = None
        self.arr6_noncohort = None

        # These arrays need to be in "indicator form"
        # So that they can be mulitplied with the betas
        # during the logistic regression inference
        self.arr2345 = self.path_to_indicator_form(self.arr2345)
        self.arr1 = self.path_to_indicator_form(self.arr1)

        # Here we figure out the year_last_smoked variable for all cases
        # This variable is used for death rates calculation
        # might also be used for health checks (not yet implemented)
        self.set_year_last_smoked(self.arr2345, current_year=2014)

    def simulation_loop(self, beta_1_aug, beta_2345_aug, shortbanparams=None, longbanparams=None):
        """
        Next step is to loop over years, updating the pop each year
        this is the main loop, simulating years 2016 - 2066
        and writing out the stats
        cy means current year
        """
        for cy in range(self.end_year - self.start_year):
            """
            Main loop and crux of the program.
            Steps:
                1. add cohorts of 18-year-olds if needed
                2. write data to appropriate structures to be saved for later analysis
                3. kill people according to life tables
                4. update people's smoking statuses
                    a. make sure to take care of hassmoked flag
                5. update people's ages
                6. update initation age group
            """

            # insert new cohort(s) of 18yearolds
            # only do this until self.last_year_cohort_added, which by default is np.inf
            if self.start_year + cy <= self.last_year_cohort_added:
                # add postban cohort if we are 3 years post ban
                if self.menthol_ban and (self.menthol_ban_year <= self.start_year + cy + 3) and self.postban_18yo_cohort is not None:
                    c2345, c1 = self.cohort_to_indicator_form(self.postban_18yo_cohort)
                    self.arr2345 = np.concatenate([self.arr2345, c2345], axis=0)
                    self.arr1 = np.concatenate([self.arr1, c1], axis=0)
                elif self.cohorts is not None:
                    cohort_idx = max(self.start_year + cy, 2015)
                    cohort_idx = min(cohort_idx, 2017)
                    cohort_arr = self.cohorts[cohort_idx]
                    c2345, c1 = self.cohort_to_indicator_form(cohort_arr)
                    self.arr2345 = np.concatenate([self.arr2345, c2345], axis=0)
                    self.arr1 = np.concatenate([self.arr1, c1], axis=0)

            # start by writing out the appropriate data

            self.write_data(cy, self.arr2345, self.arr1, self.arr6,
                            self.output_list_to_df, self.output_numpy)

            # determine which 65-year olds will get lung cancer or cardiovascular disease (CVD)
            if self.simulate_disease:
                self.sample_disease_outcomes(cy, self.arr2345, self.arr1)

            # continue by randomly determining if people
            # will die this year

            self.arr2345, self.arr1, self.arr6, self.arr6_noncohort= self.simulate_deaths(
                self.arr2345, self.arr1, self.arr6, current_year=cy + self.start_year, in_arr6_noncohort=self.arr6_noncohort)

            # next we get the transition probabilities for people

            probs2345, probs1 = self.get_transition_probs_from_LR(
                self.arr2345, self.arr1, beta_2345_aug, beta_1_aug)

            # next we augment transition probabilities according to initiation and cessation parameters

            probs2345, probs1 = self.adjust_transition_probs_according_to_initiation_cessation_params(
                probs2345, probs1)

            # next we augment transition probabilities according to menthol ban effects

            if self.menthol_ban:
                probs2345, probs1 = self.adjust_transition_probs_according_to_menthol_ban(
                    self.arr2345, self.arr1, probs2345, probs1, current_year=cy, shortbanparams=shortbanparams, longbanparams=longbanparams)

            # update current state, old state

            new_states2345 = self.random_select_arg_multinomial(
                probs2345)[:, :]  # (s1, s2, s3, s4, s5)
            new_states1 = self.random_select_arg_multinomial(
                probs1)[:, :]  # (s1, s2, s3, s4, s5)

            # leaving_1 is True for each row in new_states1 which has chosen to transition to 2, 3, 4, or 5
            leaving_1 = np.sum(new_states1[:, 1:], axis=1).astype(np.bool_)

            # move current states to last years states and
            # the new states into the current states

            self.arr2345[:, 2:5] = self.arr2345[:, 5:8]
            # no more previous never smokers
            self.arr2345[:, 1] = np.zeros(self.arr2345.shape[0])
            # dont need to move arr1 stuff because arr1's previous state has not changed at all

            self.arr2345[:, 5:8] = new_states2345[:, 1:-1]
            self.arr1[:, 5:8] = new_states1[:, 1:-1]

            # record the state transition numbers :)

            if self.save_transition_np_fname is not None:
                to_append = self.record_transitions(
                    self.arr2345, self.arr1, leaving_1)
                self.output_transitions.append([to_append])

            # move people from arr1 to arr2345 if they became a smoker

            tmp_to_2345 = self.arr1[leaving_1]
            self.arr1 = self.arr1[np.logical_not(leaving_1)]
            self.arr2345 = np.concatenate([self.arr2345, tmp_to_2345], axis=0)

            # update year_last_smoked variable

            # smokers currently in state 3,4 get their last year updated
            self.arr2345[np.logical_or(
                self.arr2345[:, 6], self.arr2345[:, 7]), 16] = cy
            # smokers currently in state 5 get their last year updated
            self.arr2345[np.sum(self.arr2345[:, 5:8], axis=1) == 0, 16] = cy
            # people who made the transition 1->2 get their last year updated
            # this is after switching, so anybody who made that transition will be in arr2345
            self.arr2345[np.logical_and(
                self.arr2345[:, 1], self.arr2345[:, 5]), 16] = cy

            # increment age param

            self.arr2345[:, 11] += 1
            self.arr1[:, 11] += 1

            # update inital age for people in arr2345 (ever smokers)
            # if ia=1 == 0 and and age >= 18 then ia = 2
            # NOTE: never need to set ia=2 == 1 because everyone is 19 or older at this point

            self.arr2345[:, 9] = (self.arr2345[:, 8] == 0) * \
                (self.arr2345[:, 12] >= 18)

            # endfor

        # write data one last time for the final year

        self.output_list_to_df, self.output_numpy = self.write_data(
            self.end_year - self.start_year, self.arr2345, self.arr1, self.arr6, self.output_list_to_df, self.output_numpy)

        # write out LYL

        self.output_LYL = self.determine_LYL(self.arr6, self.arr1, self.arr2345)

        return

    def determine_LYL(self, in_arr6, in_arr1, in_arr2345):
        """
        Calculate the number of Life Years Lived (LYL)
        for this run of the simulation, stratified by our demographic groups.

        This assumes that the in_arr6 parameter only contains dead people
        that belong to the LYL cohort.

        output_LYL is a 1D array with 5 elements
        here are the indices as well as a description of each element:
        0 - number of LYL in the total LYL cohort
        1 - number of LYL in the black LYL cohort
        2 - number of LYL in the nonblack LYL cohort
        3 - number of LYL in the pov LYL cohort
        4 - number of LYL in the nonpov LYL cohort
        """

        self.output_LYL[0] = np.sum(in_arr6[:,11] * in_arr6[:,15])
        self.output_LYL[1] = np.sum((in_arr6[:,10] == 1) * in_arr6[:,11] * in_arr6[:,15])
        self.output_LYL[2] = np.sum((in_arr6[:,10] == 0) * in_arr6[:,11] * in_arr6[:,15])
        self.output_LYL[3] = np.sum((in_arr6[:,13] == 1) * in_arr6[:,11] * in_arr6[:,15])
        self.output_LYL[4] = np.sum((in_arr6[:,13] == 0) * in_arr6[:,11] * in_arr6[:,15])

        print("")
        print("")
        print("Total number of people in the LYL cohort:")
        print(np.sum(in_arr6[:,15]))
        print("")
        print("Total number black people in the LYL cohort:")
        print(np.sum((in_arr6[:,10] == 1) * in_arr6[:,15]))
        print("")
        print("Total number nonblack people in the LYL cohort:")
        print(np.sum((in_arr6[:,10] == 0) * in_arr6[:,15]))
        print("")
        print("Total number of pov in the LYL cohort:")
        print(np.sum((in_arr6[:,13] == 1) * in_arr6[:,15]))
        print("")
        print("Total number of nonpov in the LYL cohort:")
        print(np.sum((in_arr6[:,13] == 0) * in_arr6[:,15]))
        print("")
        print("")


        return self.output_LYL

    def simulate(self):
        """
        Calling this function causes 1 run of the simulation to happen.
        Results are written according to save_xl_fname and save_np_fname.
        Optionally, transition numbers are written to save_transition_np_fname.

        Note, here I am using 'self.arr1' instead of just initializing a local variable 'arr1'.
        Same for arr2345 and arr6.
        I have chosen not to reference these arrays in the helper functions with 'self.',
        instead I use a function parameter to pass them in
        This is totally fine because numpy arrays are passed by reference not by value
        I have chosen to do this because I want to be able to reference Simulation.arr1, Simulation.arr2345, etc... 
        from code outside the Simulation class.

        Args:
            None

        Output:
            self.output: the data written out from the simulation
        """

        # format the population from pandas dataframe to numpy array form to be used in the simulation
        self.format_population()

        # next step is to format the betas as a nice clean matrix
        # that can just be multiplied against the covariate matrix
        beta_2345_aug, beta_1_aug = self.get_augmented_betas()

        self.arr1, self.arr2345, self.arr6 = self.calibrate_initial_population(
            self.arr1, self.arr2345, self.arr6, beta_1_aug, beta_2345_aug, arr6_noncohort=self.arr6_noncohort)

        # do the main simulation loop over all years!
        self.simulation_loop(beta_1_aug=beta_1_aug,
                             beta_2345_aug=beta_2345_aug)

        # writeout the results of the simulation to disk
        if self.save_xl_fname is not None:
            out = pd.DataFrame(self.output_list_to_df,
                               columns=self.output_columns)
            fname = os.path.join(self.save_dir, 'excel_files/', os.path.basename(
                self.save_xl_fname) + '_' + self.now_str + '.xlsx')
            out.to_excel(fname)

        if self.save_np_fname is not None:
            fname = os.path.join(self.save_dir, 'numpy_arrays/', os.path.basename(
                self.save_np_fname) + '_' + self.now_str + '.npy')
            np.save(fname, self.output_numpy)

        if self.save_transition_np_fname is not None:
            fname = os.path.join(self.save_dir, 'transition_numbers/', os.path.basename(
                self.save_transition_np_fname) + '_' + self.now_str + '.npy')
            np.save(fname, np.asarray(self.output_transitions))

        if self.save_disease_np_fname is not None and self.simulate_disease:
            fname_cvd = os.path.join(self.save_dir, 'disease/', "CVD_" + os.path.basename(
                self.save_disease_np_fname) + '_' + self.now_str + '.npy')
            np.save(fname_cvd, self.output_cvd)

            fname_lc = os.path.join(self.save_dir, 'disease/', "LC_" + os.path.basename(
                self.save_disease_np_fname) + '_' + self.now_str + '.npy')
            np.save(fname_lc, self.output_lc)

            fname_total = os.path.join(self.save_dir, 'disease/', "TOTAL_" + os.path.basename(
                self.save_disease_np_fname) + '_' + self.now_str + '.npy')
            np.save(fname_total, self.output_65yos)
        
        if self.save_LYL_np_fname is not None:
            fname = os.path.join(self.save_dir, 'LYL/', 'LYL_' + os.path.basename(
                self.save_LYL_np_fname) + '_' + self.now_str + '.npy')
            np.save(fname, self.output_LYL)

        return self.output_list_to_df, self.output_numpy
