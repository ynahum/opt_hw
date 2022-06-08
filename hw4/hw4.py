from newton_method import newton_method
from rosenbrock import *
import matplotlib.pyplot as plt
import numpy as np


def traj_plot(trajectory_points, f_func, title=''):

    f = [ f_func(x) for x in trajectory_points]
    #f = [ np.log10(rosenbrock_func(x)) for x in trajectory_points]
    fig = plt.figure()
    fig.suptitle(title)
    plt.yscale("log")
    plt.xlabel(r'Number of iterations (tolerance on gradient norm = $10^{-5}$)')
    plt.ylabel(r'$log{(f(x_k)-f(x^*))}$')
    plt.plot(f, 'k')
    plt.show()


if __name__ == '__main__':

    test_rosen = True
    if test_rosen:
        x_0 = np.zeros((10, 1))

        trajectory_points = newton_method(x_0, f_func=rosenbrock_func, f_grad=rosenbrock_grad, f_hessian=rosenbrock_hessian)
        title = f'Newton Method over rosenbrock trajectory plot'
        traj_plot(trajectory_points, rosenbrock_func, title=title)


