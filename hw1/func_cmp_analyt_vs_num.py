import numpy as np
from num_eval_grad_hessian import num_eval_grad_hessian


# 1.3
def func_compare_analytical_vs_numerical(x, eps_array, func, func_pos_args):
    func_value, func_grad, func_hessian = func(x, *func_pos_args)
    func_grad_diff_max_norm = []
    func_hessian_diff_max_norm = []
    for eps in eps_array:
        func_num_grad, func_num_hessian =\
            num_eval_grad_hessian(
                x,
                eps,
                func,
                func_pos_args)
        func_grad_diff_max_norm.append(np.linalg.norm(func_grad-func_num_grad, ord=np.inf))
        func_hessian_diff_max_norm.append(np.linalg.norm(func_hessian-func_num_hessian, ord=np.inf))
    return func_grad_diff_max_norm, func_hessian_diff_max_norm

