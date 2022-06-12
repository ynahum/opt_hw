import matplotlib.pyplot as plt
import numpy as np


def augmented_grads_plot(grads, y_scale_to_log=True, title=''):

    grads_norm = [ np.linalg.norm(x) for x in grads]
    #print(grads_norm)
    fig = plt.figure()
    fig.suptitle(title)
    if y_scale_to_log:
        plt.yscale("log")
    plt.xlabel(r'Number of iterations')
    plt.ylabel(r'$|\nabla ||F_{p,\mu}(x_k)||_2$')
    plt.plot(grads_norm, 'k')
    plt.show()


def maximal_cont_violation_plot(violations, title=''):
    fig = plt.figure()
    fig.suptitle(title)
    plt.xlabel(r'Number of iterations')
    plt.ylabel(r'max violation')
    plt.plot(violations, 'k')
    plt.show()


def f_abs_diff_plot(trajectory_points, f_func, x_optimal, y_scale_to_log=True, title=''):

    f_optimal = f_func(x_optimal)
    abs_diff = [np.abs(f_func(x) -f_optimal) for x in trajectory_points]
    #f = [ np.log10(rosenbrock_func(x)) for x in trajectory_points]
    fig = plt.figure()
    fig.suptitle(title)
    if y_scale_to_log:
        plt.yscale("log")
    plt.xlabel(r'Number of iterations')
    plt.ylabel(r'$|(f(x_k)-f(x^*))|$')
    plt.plot(abs_diff, 'k')
    plt.show()


def x_diff_plot(trajectory_points, x_optimal, y_scale_to_log=True, title=''):
    diff = [np.linalg.norm(x-x_optimal) for x in trajectory_points]
    fig = plt.figure()
    fig.suptitle(title)
    if y_scale_to_log:
        plt.yscale("log")
    plt.xlabel(r'Number of iterations')
    plt.ylabel(r'$||x-x^*||_2$')
    plt.plot(diff, 'k')
    plt.show()

def multipliers_diff_plot(multipliers, optimal_multiplier, y_scale_to_log=True, title=''):
    diff = [np.linalg.norm(mult-optimal_multiplier) for mult in multipliers]
    fig = plt.figure()
    fig.suptitle(title)
    if y_scale_to_log:
        plt.yscale("log")
    plt.xlabel(r'Number of iterations')
    plt.ylabel(r'$||\lambda-\lambda^*||_2$')
    plt.plot(diff, 'k')
    plt.show()
