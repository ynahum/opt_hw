import numpy as np


# 1.1.5
def calc_h_value_and_derivatives(x, calc_first_derivative=True, calc_second_derivative=True):
    h_value = np.sqrt(1 + (x ** 2))
    ret = [h_value]
    if calc_first_derivative:
        ret.append(x/np.sqrt(1+(x**2)))
    if calc_second_derivative:
        ret.append(1 / np.power(1 + (x ** 2), 1.5))
    return ret

