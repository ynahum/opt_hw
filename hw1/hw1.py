import numpy as np
import matplotlib.pyplot as plt

# 1.1.5
def calc_phi_value_grad_hessian(x, calc_grad=True, calc_hessian=True):
    x_vec = x.squeeze()
    u_value = np.prod(x, axis=0)
    sin_value = np.sin(u_value)
    cos_value = np.cos(u_value)
    value = sin_value
    ret = [value]

    if calc_grad:
        u_grad = np.array([x[1] * x[2],
                           x[0] * x[2],
                           x[0] * x[1]])
        phi_grad = u_grad * cos_value
        ret.append(phi_grad)
    if calc_hessian:
        H1 = np.array([[0, x_vec[2], x_vec[1]],
                       [x_vec[2], 0, x_vec[0]],
                       [x_vec[1], x_vec[0], 0]])
        H2_11 = (x_vec[1]*x_vec[2])**2
        H2_12 = x_vec[0] * x_vec[1] * (x_vec[2]**2)
        H2_13 = x_vec[0] * (x_vec[1]**2) * x_vec[2]
        H2_22 = (x_vec[0]*x_vec[2])**2
        H2_23 = (x_vec[0]**2) * x_vec[1] * x_vec[2]
        H2_33 = (x_vec[0] * x_vec[1])**2
        H2 = np.array([[H2_11, H2_12, H2_13],
                       [H2_12, H2_22, H2_23],
                       [H2_13, H2_23, H2_33]])
        phi_hessian = cos_value * H1 - sin_value * H2
        ret.append(phi_hessian)
    return ret


def calc_f1_value_grad_hessian(x, A, calc_grad=True, calc_hessian=True):
    phi_value, phi_grad, phi_hessian = calc_phi_value_grad_hessian(A @ x)
    ret = [phi_value]
    if calc_grad:
        ret.append(A.T @ phi_grad)
    if calc_hessian:
        ret.append(A.T @ phi_hessian @ A)
    return ret


def calc_h_value_and_derivatives(x, calc_first_derivative=True, calc_second_derivative=True):
    h_value = np.sqrt(1 + (x ** 2))
    ret = [h_value]
    if calc_first_derivative:
        ret.append(x/np.sqrt(1+(x**2)))
    if calc_second_derivative:
        ret.append(1 / np.power(1 + (x ** 2), 1.5))
    return ret


def calc_f2_value_grad_hessian(x, calc_grad=True, calc_hessian=True):
    phi_value, phi_grad, phi_hessian = calc_phi_value_grad_hessian(x)
    h_value, h_first_derivative, h_second_derivative = calc_h_value_and_derivatives(phi_value)
    ret = [h_value]
    if calc_grad:
        ret.append(h_first_derivative * phi_grad)
    if calc_hessian:
        ret.append(h_first_derivative * phi_hessian + h_second_derivative * (phi_grad @ phi_grad.T))
    return ret


# 1.2.3
def numerical_grad_hessian(x, eps, func, func_pos_args):
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
            numerical_grad_hessian(
                x,
                np.power(2.0, -i),
                calc_f1_value_grad_hessian,
                [A])

        f1_grad_diff = np.linalg.norm(f1_grad-f1_num_grad, ord=np.inf)
        f1_grad_norm.append(f1_grad_diff)
        f1_hessian_diff = np.linalg.norm(f1_hessian-f1_num_hessian, ord=np.inf)
        f1_hessian_norm.append(f1_hessian_diff)

        f2_num_grad, f2_num_hessian =\
            numerical_grad_hessian(
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
