import numpy as np
import matplotlib.pyplot as plt
from f1 import calc_f1_value_grad_hessian
from f2 import calc_f2_value_grad_hessian
from func_cmp_analyt_vs_num import func_compare_analytical_vs_numerical


def compare_analytical_vs_numerical():

    np.random.seed(42)

    x_range = np.arange(61)
    eps_array = np.power(2.0, -x_range)

    x = np.random.rand(3, 1)
    A = np.random.rand(3, 3)
    f1_grad_norm, f1_hessian_norm = func_compare_analytical_vs_numerical(x, eps_array, calc_f1_value_grad_hessian, [A])
    f2_grad_norm, f2_hessian_norm = func_compare_analytical_vs_numerical(x, eps_array, calc_f2_value_grad_hessian, [])

    f1_grad_min_err = np.min(f1_grad_norm)
    f1_grad_min_err_ind = np.argmin(f1_grad_norm)
    f1_hessian_min_err = np.min(f1_hessian_norm)
    f1_hessian_min_err_ind = np.argmin(f1_hessian_norm)

    f2_grad_min_err = np.min(f2_grad_norm)
    f2_grad_min_err_ind = np.argmin(f2_grad_norm)
    f2_hessian_min_err = np.min(f2_hessian_norm)
    f2_hessian_min_err_ind = np.argmin(f2_hessian_norm)

    # Plot norm difference

    plt.subplot()
    plt.plot(x_range, f1_grad_norm)
    plt.yscale('log')
    plt.xlabel('Exponent absolute value')
    # plt.ylabel('Infinity norm of difference(log scale)')
    plt.suptitle('f1 gradient difference')
    plt.show()

    plt.plot(x_range, f1_hessian_norm)
    plt.yscale('log')
    plt.xlabel('Exponent absolute value')
    # plt.ylabel('Infinity norm of difference(log scale)')
    plt.suptitle('f1 hessian difference')
    plt.show()

    plt.subplot()
    plt.plot(x_range, f2_grad_norm)
    plt.yscale('log')
    plt.xlabel('Exponent absolute value')
    # plt.ylabel('Infinity norm of difference(log scale)')
    plt.suptitle('f2 gradient difference')
    plt.show()

    plt.plot(x_range, f2_hessian_norm)
    plt.yscale('log')
    plt.xlabel('Exponent absolute value')
    # plt.ylabel('Infinity norm of difference(log scale)')
    plt.suptitle('f2 hessian difference')
    plt.show()

    print(f'f1 gradient min error is: {f1_grad_min_err} at exponent: {f1_grad_min_err_ind}')
    print(f'f1 hessian min error is: {f1_hessian_min_err} at exponent: {f1_hessian_min_err_ind}')

    print(f'f2 gradient min error is: {f2_grad_min_err} at exponent: {f2_grad_min_err_ind}')
    print(f'f2 hessian min error is: {f2_hessian_min_err} at exponent: {f2_hessian_min_err_ind}')


if __name__ == '__main__':
    compare_analytical_vs_numerical()
