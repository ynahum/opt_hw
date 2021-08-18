import numpy as np
import matplotlib.pyplot as plt
from f1 import calc_f1_value_grad_hessian
from f2 import calc_f2_value_grad_hessian
from func_cmp_analyt_vs_num import func_compare_analytical_vs_numerical

run_on_f1 = False
run_on_f2 = True

def compare_analytical_vs_numerical():

    np.random.seed(42)

    exp_range = np.arange(61)
    eps_array = np.power(2.0, -exp_range)

    x = np.random.rand(3,)

    if run_on_f1:
        A = np.random.rand(3, 3)
        f1_grad_diff_max_norm, f1_hessian_diff_max_norm = func_compare_analytical_vs_numerical(x, eps_array, calc_f1_value_grad_hessian, [A])

        plt.subplot()
        plt.plot(exp_range, f1_grad_diff_max_norm)
        plt.suptitle(r'$\log{||\nabla_{x}^{analytical}f_1 - \nabla_{x}^{numerical}f_1||_{\infty}}$ vs  $\epsilon$ accuracy')
        plt.xlabel(r'Absolute value of $\epsilon$ exponent')
        plt.yscale('log')
        plt.show()

        plt.plot(exp_range, f1_hessian_diff_max_norm)
        plt.suptitle(r'$\log{||\nabla_{x}^{2\;analytical}f_1 - \nabla_{x}^{2\;numerical}f_1||_{\infty}}$ vs  $\epsilon$ accuracy')
        plt.xlabel(r'Absolute value of $\epsilon$ exponent')
        plt.yscale('log')
        plt.show()

        f1_grad_diff_max_norm_min = np.min(f1_grad_diff_max_norm)
        f1_grad_diff_max_norm_argmin = np.argmin(f1_grad_diff_max_norm)

        print(f'f1 gradient min infinity norm error : {f1_grad_diff_max_norm_min} ')
        print(f'epsilon exponent: {-f1_grad_diff_max_norm_argmin}')
        print(f'epsilon value: {np.power(2.0, -f1_grad_diff_max_norm_argmin)}')
        print()

        f1_hessian_diff_max_norm_min = np.min(f1_hessian_diff_max_norm)
        f1_hessian_diff_max_norm_argmin = np.argmin(f1_hessian_diff_max_norm)

        print(f'f1 hessian min infinity norm error : {f1_hessian_diff_max_norm_min} ')
        print(f'epsilon exponent: {-f1_hessian_diff_max_norm_argmin}')
        print(f'epsilon value: {np.power(2.0, -f1_hessian_diff_max_norm_argmin)}')
        print()

    if run_on_f2:
        f2_grad_diff_max_norm, f2_hessian_diff_max_norm = func_compare_analytical_vs_numerical(x, eps_array, calc_f2_value_grad_hessian, [])

        plt.subplot()
        plt.plot(exp_range, f2_grad_diff_max_norm)
        plt.suptitle(r'$\log{||\nabla_{x}^{analytical}f_2 - \nabla_{x}^{numerical}f_2||_{\infty}}$ vs  $\epsilon$ accuracy')
        plt.xlabel(r'Absolute value of $\epsilon$ exponent')
        plt.yscale('log')
        plt.show()

        plt.plot(exp_range, f2_hessian_diff_max_norm)
        plt.suptitle(r'$\log{||\nabla_{x}^{2\;analytical}f_2 - \nabla_{x}^{2\;numerical}f_2||_{\infty}}$ vs  $\epsilon$ accuracy')
        plt.xlabel(r'Absolute value of $\epsilon$ exponent')
        plt.yscale('log')
        plt.show()

        f2_grad_diff_max_norm_min = np.min(f2_grad_diff_max_norm)
        f2_grad_diff_max_norm_argmin = np.argmin(f2_grad_diff_max_norm)

        print(f'f2 gradient min infinity norm error : {f2_grad_diff_max_norm_min} ')
        print(f'epsilon exponent: {-f2_grad_diff_max_norm_argmin}')
        print(f'epsilon value: {np.power(2.0, -f2_grad_diff_max_norm_argmin)}')
        print()

        f2_hessian_diff_max_norm_min = np.min(f2_hessian_diff_max_norm)
        f2_hessian_diff_max_norm_argmin = np.argmin(f2_hessian_diff_max_norm)

        print(f'f2 hessian min infinity norm error : {f2_hessian_diff_max_norm_min} ')
        print(f'epsilon exponent: {-f2_hessian_diff_max_norm_argmin}')
        print(f'epsilon value: {np.power(2.0, -f2_hessian_diff_max_norm_argmin)}')
        print()


if __name__ == '__main__':
    compare_analytical_vs_numerical()
