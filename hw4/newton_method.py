import numpy as np
from newton_method_utils import *


def inexact_line_search_armijo_rule_backtrack(f_func, f_grad, d, x, alpha_0=1, beta=0.5, sigma=0.25):
    alpha = alpha_0
    while True:
        f_x_k_1 = f_func(x + d * alpha)
        f_x_k = f_func(x)
        grad_x_k = f_grad(x)
        if f_x_k_1 <= f_x_k +  sigma * alpha * d.T @ grad_x_k:
            break
        alpha = beta * alpha
    return alpha


def newton_method(x_0, f_func, f_grad, f_hessian, grad_norm_thresh=(10**(-5))):
    x_k_list = []

    print('Started Newton Method with:')

    x = x_0
    num_of_iteration = 0

    while True:
        #print(f"Running iteration {num_of_iteration}")

        d = newton_direction(f_hessian(x), f_grad(x))

        x_k_list.append(x)
        if np.linalg.norm(d, ord=2) <= grad_norm_thresh:
            print(f'Newton Method has converged after {num_of_iteration} iterations')
            break

        num_of_iteration += 1
        alpha = inexact_line_search_armijo_rule_backtrack(f_func, f_grad, d, x)
        x = x + alpha * d

    return np.array(x_k_list)
