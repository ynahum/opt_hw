import numpy as np

# 2.3
def grad_descent(x_0, Q, exact_line_search=True, grad_norm_thresh=(10**(-5))):
    x_k_list = []
    x = x_0
    x_k_list.append(x)
    if exact_line_search:
        while True:
            d = 0.5 * (Q + Q.T) @ x
            denom = (d.T @ Q @ d)
            if denom == 0:
                print("d.T Q d == 0")
                alpha = 0.1
            else:
                alpha = (x.T @ Q @ d) / denom
            if np.linalg.norm(d,ord=2) <= grad_norm_thresh:
                print('Gradient Descent has converged')
                break
            x = x - alpha * d
            x_k_list.append(x)
    return np.array(x_k_list)
