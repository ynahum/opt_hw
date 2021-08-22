import numpy as np

def inexact_armijo_rule_backtrack_line_search(Q, d, x, alpha_0=1, beta=0.5, sigma=0.25 ):
    alpha = alpha_0
    while True:
        f_x_k_1 = 0.5 * (x + d * alpha).T @ Q @ (x + d * alpha)
        f_x_k = 0.5 * x.T @ Q @ x
        grad_f_x_k = 0.5 * (Q + Q.T) @ x
        if f_x_k_1 <= f_x_k +  sigma * alpha * d.T @ grad_f_x_k:
             break
        alpha = beta * alpha
    return alpha


# 2.3
def grad_descent(x_0, Q, exact_line_search=True, grad_norm_thresh=(10**(-5))):
    x_k_list = []
    symQ = (Q + Q.T)

    print('Started Gradient Descent with:')
    print(f"Q={repr(Q)}")
    print(f"x_0={repr(x_0)}")

    x = x_0
    num_of_iteration = 0

    while True:
        #print(f"Running iteration {num_of_iteration}")
        # direction is the opposite to gradient
        d = -0.5 * symQ @ x
        x_k_list.append(x)
        if np.linalg.norm(d, ord=2) <= grad_norm_thresh:
            print(f'Gradient Descent has converged after {num_of_iteration-1} iterations')
            break

        num_of_iteration += 1

        if exact_line_search:
            denom = (d.T @ symQ @ d)
            if denom == 0:
                print(f"At iteration {num_of_iteration} we got "
                      f"illegal denominator 0: (d.T (Q + Q.T) d == 0)")
            else:
                alpha = -(x.T @ symQ @ d) / denom
        else:
            alpha = inexact_armijo_rule_backtrack_line_search(Q, d, x)
        x = x + alpha * d

    return np.array(x_k_list)
