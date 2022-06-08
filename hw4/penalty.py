import numpy as np


def penalty_func(t):
    if t >= -0.5:
        f = 0.5*(t**2) + t
    else:
        f = -0.25*np.log(-2*t) - 0.375
    return f


def penalty_first_derivative(t):
    if t >= -0.5:
        f = t + 1
    else:
        f = -0.25/t
    return f


def penalty_second_derivative(t):
    if t >= -0.5:
        f = 1
    else:
        f = 0.25/(t**2)
    return f


