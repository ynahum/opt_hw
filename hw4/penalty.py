import numpy as np


def penalty_func(t, mu_multiplier=1):
    if t >= -0.5:
        f = 0.5*(t**2) + mu_multiplier * t
    else:
        f = -0.25*np.log(-2*t) - 0.375
    return f


def penalty_first_derivative(t, mu_multiplier=1):
    if t >= -0.5:
        f = t + mu_multiplier
    else:
        f = -0.25/t
    return f


def penalty_second_derivative(t):
    if t >= -0.5:
        f = 1
    else:
        f = 0.25/(t**2)
    return f


class Penalty:

    def __init__(self, p_init, multiplier_init=1):
        self.p = p_init
        self.multiplier = multiplier_init

    def func(self,x):
        return (1.0/self.p) * penalty_func(self.p * x, self.multiplier)

    def first_derivative(self,x):
        return penalty_first_derivative(self.p * x, self.multiplier)

    def second_derivative(self,x):
        return self.p * penalty_second_derivative(self.p * x)
