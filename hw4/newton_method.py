from optim_problem import *
from newton_method_utils import *
import numpy as np


def inexact_line_search_armijo_rule_backtrack(
        optim_problem: OptimizationProblem, d, x, alpha_0=1, beta=0.5, sigma=0.25):

    alpha = alpha_0
    while True:
        f_x_k_1 = optim_problem.func(x + d * alpha)
        f_x_k = optim_problem.func(x)
        grad_x_k = optim_problem.grad(x)
        if f_x_k_1 <= f_x_k + sigma * alpha * d.T @ grad_x_k:
            break
        alpha = beta * alpha
    return alpha


def newton_method(x_0, optim_problem: OptimizationProblem, grad_norm_thresh=(10**(-5))):
    x_k_list = []
    grad_k_list = []

    x = x_0
    num_of_iteration = 0

    while True:
        #print(f"Running iteration {num_of_iteration}")
        grad_k_list.append(optim_problem.grad(x))
        d = newton_direction(optim_problem.hessian(x), grad_k_list[-1])

        x_k_list.append(x)
        if np.linalg.norm(grad_k_list[-1], ord=2) <= grad_norm_thresh:
            print(f'Newton Method has converged after {num_of_iteration} iterations')
            break

        num_of_iteration += 1
        alpha = inexact_line_search_armijo_rule_backtrack(optim_problem, d, x)
        x = x + alpha * d

    return_hash = {}
    return_hash['x_list'] = np.array(x_k_list)
    return_hash['grad_list'] = np.array(grad_k_list)
    return return_hash
