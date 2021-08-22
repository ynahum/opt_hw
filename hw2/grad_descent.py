import numpy as np

def inexact_armijo_rule_backtrack_line_search(Q, d, x, alpha_0=1, beta=0.5, sigma=0.25 ):
    alpha = alpha_0
    while(x - d * alpha).T @ Q @ (x - d * alpha) > ( x.T @ Q @ x + sigma * alpha * ((d.T) @ Q @ (x)) * -sigma * 2):
        alpha = beta * alpha
    return alpha



# 2.3
def grad_descent(x_0, Q, exact_line_search=True, grad_norm_thresh=(10**(-5))):
    x_k_list = []
    #print(f"Q={repr(Q)}")
    #print(f"x_0={repr(x_0)}")
    symQ = (Q + Q.T)
    x = x_0
    num_of_iteration = 0
    while True:
        num_of_iteration += 1
        #print(f"Running iteration {num_of_iteration}")
        d = 0.5 * symQ @ x
        x_k_list.append(x)
        if np.linalg.norm(d, ord=2) <= grad_norm_thresh:
            print('Gradient Descent has converged')
            break
        if exact_line_search:
            denom = (d.T @ symQ @ d)
            if denom == 0:
                print(f"At iteration {num_of_iteration} we got "
                      f"illegal denominator 0: (d.T (Q + Q.T) d == 0)")
            else:
                alpha = -(x.T @ symQ @ d) / denom
        else:
            alpha = -inexact_armijo_rule_backtrack_line_search(Q, d, x)
        x = x + alpha * d

    return np.array(x_k_list)
