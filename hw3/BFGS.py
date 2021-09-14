import numpy as np


def inexact_line_search_armijo_rule_backtrack(d, x, f_func_ptr, g_func_ptr, alpha_0=1, beta=0.5, sigma=0.25, c_2=0.9):
    alpha = alpha_0
    while True:
        x_k_1 = x + alpha * d
        f_x_k = f_func_ptr(x)
        f_x_k_1 = f_func_ptr(x_k_1)
        grad_x_k = g_func_ptr(x)
        grad_x_k_1 = g_func_ptr(x_k_1)
        decrease_cond =(f_x_k_1 <= f_x_k + sigma * alpha * d.T @ grad_x_k)
        curvature_cond = (grad_x_k_1.T @ d >= c_2 * grad_x_k.T @ d)
        if decrease_cond and curvature_cond:
             break
        alpha = beta * alpha
    return alpha


def BFGS(x_0, f_func_ptr, g_func_ptr, grad_norm_thresh=(10**(-5))):
    x_k_list = []

    print('Started BFGS on rosenbrock')
    n = len(x_0)
    x_k = x_0
    g_x_k = g_func_ptr(x_k)
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

        alpha = inexact_line_search_armijo_rule_backtrack(d, x_k, f_func_ptr, g_func_ptr)

        x_k_next = x_k + alpha * d
        g_x_k_next = g_func_ptr(x_k_next)
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
