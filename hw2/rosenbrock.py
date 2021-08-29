import numpy as np
import matplotlib.pyplot as plt
from newton_method_utils import *

# 2.7
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

def rosenbrock_inexact_line_search_armijo_rule_backtrack(d, x, alpha_0=1, beta=0.5, sigma=0.25):
    alpha = alpha_0
    while True:
        f_x_k_1 = rosenbrock_func(x + alpha * d)
        f_x_k = rosenbrock_func(x)
        grad_f_x_k = rosenbrock_grad(x)
        if f_x_k_1 <= f_x_k +  sigma * alpha * d.T @ grad_f_x_k:
             break
        alpha = beta * alpha
    return alpha


def rosenbrock_inexact_line_search_grad_descent(x_0, grad_norm_thresh=(10**(-5))):
    x_k_list = []

    print('Started Rosenbrock inexact line search gradient descent')

    x = x_0
    num_of_iteration = 0

    while True:
        #print(f"Running iteration {num_of_iteration}")
        # direction is the opposite to gradient

        d = -rosenbrock_grad(x)
        x_k_list.append(x)
        if np.linalg.norm(d, ord=2) <= grad_norm_thresh:
            print(f'Gradient Descent has converged after {num_of_iteration} iterations')
            break

        num_of_iteration += 1

        alpha = rosenbrock_inexact_line_search_armijo_rule_backtrack(d, x)
        x = x + alpha * d

    return np.array(x_k_list)

def rosenbrock_inexact_line_search_newton_method(x_0, grad_norm_thresh=(10**(-5))):
    x_k_list = []

    print('Started Rosenbrock inexact line search newton method')

    x = x_0
    num_of_iteration = 0

    while True:
        #print(f"Running iteration {num_of_iteration}")
        # direction is the opposite to gradient

        d = newton_direction(rosenbrock_hessian(x), rosenbrock_grad(x))
        x_k_list.append(x)
        if np.linalg.norm(d, ord=2) <= grad_norm_thresh:
            print(f'Gradient Descent has converged after {num_of_iteration} iterations')
            break

        num_of_iteration += 1

        alpha = rosenbrock_inexact_line_search_armijo_rule_backtrack(d, x)
        x = x + alpha * d

    return np.array(x_k_list)

def rosenbrock_plot(trajectory_points, title=''):

    f = [ rosenbrock_func(x) for x in trajectory_points]
    fig = plt.figure()
    fig.suptitle(title)
    plt.yscale("log")
    plt.xlabel(r'Number of iterations (tolerance on gradient norm = $10^{-5}$)')
    plt.ylabel(r'$log{(f(x_k)-f(x^*))}$')
    plt.plot(f, 'k')
    plt.show()
