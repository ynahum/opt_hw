
def inexact_line_search_armijo_rule_backtrack(Q, d, x, alpha_0=1, beta=0.5, sigma=0.25 ):
    alpha = alpha_0
    while True:
        f_x_k_1 = 0.5 * (x + d * alpha).T @ Q @ (x + d * alpha)
        f_x_k = 0.5 * x.T @ Q @ x
        grad_f_x_k = 0.5 * (Q + Q.T) @ x
        if f_x_k_1 <= f_x_k +  sigma * alpha * d.T @ grad_f_x_k:
             break
        alpha = beta * alpha
    return alpha

