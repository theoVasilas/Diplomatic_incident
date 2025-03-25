from typing import List
import numpy as np

def Fed_Average(flatten_params_list):
    sum_array = np.sum(flatten_params_list, axis=0)
    avg_array = sum_array / len(flatten_params_list)

    return avg_array