import numpy as np


def obj_func(x):
    return 2 * ((x[0] - 5) ** 2) + (x[1] - 1) ** 2


def obj_grad(x):
    grad = np.zeros_like(x)
    grad[0] = 4 * (x[0] - 5)
    grad[1] = 2 * (x[1] - 1)
    return grad


def obj_hessian(x):
    hessian = np.zeros((len(x),len(x)))
    hessian[0][0] = 2
    hessian[1][1] = 1
    return hessian


def g1_func(x):
    return 0.5 * x[0] + x[1] - 1


def g1_grad(x):
    grad = np.zeros_like(x)
    grad[0] = 0.5
    grad[1] = 1
    return grad


def g1_hessian(x):
    return np.zeros((len(x),len(x)))


def g2_func(x):
    return x[0] - x[1]


def g2_grad(x):
    grad = np.zeros_like(x)
    grad[0] = 1
    grad[1] = -1
    return grad


def g2_hessian(x):
    return np.zeros((len(x),len(x)))


def g3_func(x):
    return -x[0] - x[1]


def g3_grad(x):
    grad = np.zeros_like(x)
    grad[0] = -1
    grad[1] = -1
    return grad


def g3_hessian(x):
    return np.zeros((len(x),len(x)))
