import numpy as np


def rosenbrock_func(x):
    f = 0
    for i in range(len(x)-1):
        f += (100 * (x[i+1] - (x[i] ** 2)) ** 2 + ((1 - x[i]) ** 2))
    return f


def rosenbrock_grad(x):
    n = len(x)
    assert(n>=2)
    grad = np.zeros_like(x)
    grad[0] = 400 * x[0] * (x[0] ** 2 - x[1]) + 2 * x[0] - 2
    for i in range(1,(n-1)):
        grad[i] = 400 * x[i] * (x[i] ** 2 - x[i+1]) + 202 * x[i] - 200 * (x[i-1] ** 2) - 2
    grad[n-1] = 200 * (x[n-1] - x[n - 2] ** 2)
    return grad


def rosenbrock_hessian(x):
    n = len(x)
    assert(n>=2)
    hessian = np.zeros((n,n))

    hessian[0][0] = 1200 * (x[0] ** 2) - 400 * x[1] + 2
    hessian[0][1] = -400 * x[0]
    for i in range(1,(n-1)):
        hessian[i][i-1] = -400 * x[i-1]
        hessian[i][i] = 1200 * (x[i] ** 2) - 400 * x[i+1] + 202
        hessian[i][i+1] =  -400 * x[i]
    hessian[n-1][n-2] = -400 * x[n-2]
    hessian[n-1][n-1] = 200
    return hessian


