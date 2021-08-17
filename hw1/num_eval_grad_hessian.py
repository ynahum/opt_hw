import numpy as np


# 1.2.3
def num_eval_grad_hessian(x, eps, func, func_pos_args):
    # TODO: consider to calculate with only function values (analytical gradient is not known)
    func_key_args = {'calc_grad': True, 'calc_hessian': False}
    n = x.size
    grad = np.zeros((n, 1))
    hessian = np.zeros((n, n))
    for i in range(n):
        e_i = np.zeros((n, 1))
        e_i[i] = eps
        func_value_x_plus_eps, func_grad_x_plus_eps = func(x + e_i, *func_pos_args, **func_key_args)
        func_value_x_minus_eps, func_grad_x_minus_eps = func(x - e_i, *func_pos_args, **func_key_args)
        grad[i] = (func_value_x_plus_eps - func_value_x_minus_eps) / (2 * eps)
        #TODO: remove this squeeze somehow
        hessian[:, i] = ((func_grad_x_plus_eps - func_grad_x_minus_eps) / (2 * eps)).squeeze(1)
    return grad, hessian


