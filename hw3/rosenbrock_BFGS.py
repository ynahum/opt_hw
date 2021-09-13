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


def inexact_line_search_armijo_rule_backtrack(d, x, alpha_0=1, beta=0.5, sigma=0.25, c_2=0.9):
    alpha = alpha_0
    while True:
        x_k_1 = x + alpha * d
        f_x_k = rosenbrock_func(x)
        f_x_k_1 = rosenbrock_func(x_k_1)
        grad_x_k = rosenbrock_grad(x)
        grad_x_k_1 = rosenbrock_grad(x_k_1)
        decrease_cond =(f_x_k_1 <= f_x_k + sigma * alpha * d.T @ grad_x_k)
        curvature_cond = (grad_x_k_1.T @ d >= c_2 * grad_x_k.T @ d)
        if decrease_cond and curvature_cond:
             break
        alpha = beta * alpha
    return alpha


def BFGS(x_0, grad_norm_thresh=(10**(-5))):
    x_k_list = []

    print('Started BFGS on rosenbrock')
    n = len(x_0)
    x_k = x_0
    g_x_k = rosenbrock_grad(x_k)
    num_of_iteration = 0
    I = np.identity(n)
    B_k = I

    while True:
        #print(f"Running iteration {num_of_iteration}")

        d = -B_k @ g_x_k
        x_k_list.append(x_k)
        if np.linalg.norm(d, ord=2) <= grad_norm_thresh:
            print(f'BFGS has converged after {num_of_iteration} iterations')
            break

        num_of_iteration += 1

        alpha = inexact_line_search_armijo_rule_backtrack(d, x_k)

        x_k_next = x_k + alpha * d
        g_x_k_next = rosenbrock_grad(x_k_next)
        s_k = x_k_next - x_k
        y_k = g_x_k_next - g_x_k
        curve_factor = (y_k.T @ s_k)
        # the below condition is true as we check the curvature condition in
        # the inexact line search with armijo_rule_backtrack
        #if curve_factor > 0:
        B_k = (I - ((s_k @ y_k.T) / curve_factor)) @ B_k @ (I - ((y_k@s_k.T)/curve_factor)) + ((s_k@s_k.T)/curve_factor)

        x_k = x_k_next
        g_x_k = g_x_k_next

    return np.array(x_k_list)

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
