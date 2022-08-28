import numpy as np
from typing import Tuple

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
