import numpy as np
from num_eval_grad_hessian import num_eval_grad_hessian


def func_compare_analytical_vs_numerical(x, eps_array, func, func_pos_args):
    func_value, func_grad, func_hessian = func(x, *func_pos_args)
    func_grad_max_norm = []
    func_hessian_max_norm = []
    for eps in eps_array:
        func_num_grad, func_num_hessian =\
            num_eval_grad_hessian(
                x,
                eps,
                func,
                func_pos_args)
        func_grad_diff = np.linalg.norm(func_grad-func_num_grad, ord=np.inf)
        func_grad_max_norm.append(func_grad_diff)
        func_hessian_diff = np.linalg.norm(func_hessian-func_num_hessian, ord=np.inf)
        func_hessian_max_norm.append(func_hessian_diff)
    return func_grad_max_norm, func_hessian_max_norm

