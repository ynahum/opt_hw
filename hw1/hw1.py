import numpy as np
import matplotlib.pyplot as plt
from f1 import calc_f1_value_grad_hessian
from f2 import calc_f2_value_grad_hessian
from num_eval_grad_hessian import num_eval_grad_hessian


def compare_analytical_vs_numerical():
    np.random.seed(42)
    x_range = np.arange(61)
    x = np.random.rand(3, 1)
    A = np.random.rand(3, 3)
    f1_value, f1_grad, f1_hessian = calc_f1_value_grad_hessian(x, A)
    f2_value, f2_grad, f2_hessian = calc_f2_value_grad_hessian(x)

    f1_grad_norm = []
    f1_hessian_norm = []
    f2_grad_norm = []
    f2_hessian_norm = []

    for i in x_range:
        f1_num_grad, f1_num_hessian =\
            num_eval_grad_hessian(
                x,
                np.power(2.0, -i),
                calc_f1_value_grad_hessian,
                [A])

        f1_grad_diff = np.linalg.norm(f1_grad-f1_num_grad, ord=np.inf)
        f1_grad_norm.append(f1_grad_diff)
        f1_hessian_diff = np.linalg.norm(f1_hessian-f1_num_hessian, ord=np.inf)
        f1_hessian_norm.append(f1_hessian_diff)

        f2_num_grad, f2_num_hessian =\
            num_eval_grad_hessian(
                x,
                np.power(2.0, -i),
                calc_f2_value_grad_hessian,
                [])

        f2_grad_diff = np.linalg.norm(f2_grad-f2_num_grad, ord=np.inf)
        f2_grad_norm.append(f2_grad_diff)
        f2_hessian_diff = np.linalg.norm(f2_hessian-f2_num_hessian, ord=np.inf)
        f2_hessian_norm.append(f2_hessian_diff)

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
