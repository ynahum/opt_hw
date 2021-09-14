import numpy as np
import matplotlib.pyplot as plt


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


def rosenbrock_plot(trajectory_points, title=''):

    f = [ rosenbrock_func(x) for x in trajectory_points]
    #f = [ np.log10(rosenbrock_func(x)) for x in trajectory_points]
    fig = plt.figure()
    fig.suptitle(title)
    plt.yscale("log")
    plt.xlabel(r'Number of iterations (tolerance on gradient norm = $10^{-5}$)')
    plt.ylabel(r'$log{(f(x_k)-f(x^*))}$')
    plt.plot(f, 'k')
    plt.show()
