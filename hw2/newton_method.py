import numpy as np
from inexact_line_search_backtrack import inexact_line_search_armijo_rule_backtrack
from mcholmz.mcholmz import modifiedChol


# 2.4.1
def forward(L, x):
    y = np.zeros_like(x)
    for idx, y_elem in enumerate(y):
        y[idx] = (x[idx] - L[idx] @ y) / L[idx, idx]
    return y


def inverse_diagonal(d, y):
    inv_d = 1/d
    n = len(d)
    inv_D = np.zeros((n,n))
    np.fill_diagonal(inv_D, inv_d)
    return inv_D @ y


def newton_direction(hessian, gradient):
    L, d, e = modifiedChol(hessian)
    L = np.array(L)
    d = np.array(d)
    y = forward(L, -gradient)
    z = inverse_diagonal(d, y)
    x = forward(L.T, z)
    return x


# 2.4
def newton_method(x_0, Q, exact_line_search=True, grad_norm_thresh=(10**(-5))):
    x_k_list = []
    symQ = (Q + Q.T)

    print('Started Newton Method with:')
    print(f"Q={repr(Q)}")
    print(f"x_0={repr(x_0)}")

    x = x_0
    num_of_iteration = 0

    while True:
        #print(f"Running iteration {num_of_iteration}")
        gradient = 0.5 * symQ @ x
        hessian = 0.5 * symQ
        d = newton_direction(hessian, gradient)

        x_k_list.append(x)
        if np.linalg.norm(d, ord=2) <= grad_norm_thresh:
            print(f'Gradient Descent has converged after {num_of_iteration} iterations')
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
            alpha = inexact_line_search_armijo_rule_backtrack(Q, d, x)
        x = x + alpha * d

    return np.array(x_k_list)
