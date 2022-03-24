import numpy as np


# 1.1.5.1
def calc_h_value_and_derivatives(x, calc_first_derivative=True, calc_second_derivative=True):
    sin_x = np.sin(x)
    v = 1 + (sin_x ** 2)
    h_value = np.sqrt(v)
    ret = [h_value]
    if calc_first_derivative:
        cos_x = np.cos(x)
        derv1 = sin_x * (cos_x / h_value)
        ret.append(derv1)
    if calc_second_derivative:
        sin_x_squared = np.sin(x) ** 2
        cos_x_squared = np.cos(x) ** 2
        derv2 = ((cos_x_squared - sin_x_squared) / h_value ) -\
                ((sin_x_squared * cos_x_squared)/ np.power(v,3/2))
        ret.append(derv2)
    return ret

