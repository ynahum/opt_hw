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


def h_val_der_sec_der(x, calc_g=True, calc_h=True):
    h_val = np.sqrt(1 + (x ** 2))
    to_ret = [h_val]
    if calc_g:
        h_der = x/np.sqrt(1+(x**2))
        to_ret.append(h_der)
    if calc_h:
        h_sec_der = 1 / np.power(1 + (x ** 2), 1.5)
        to_ret.append(h_sec_der)
    return to_ret


def f2_val_grad_hessian(x, calc_grad=True, calc_hessian=True):
    phi_val, phi_grad, phi_hessian = calc_phi_value_grad_hessian(x)
    h_val, h_der, h_sec_der = h_val_der_sec_der(phi_val)
    f2_val = h_val
    to_ret = [f2_val]
    if calc_grad:
        f2_grad = h_der * phi_grad
        to_ret.append(f2_grad)
    if calc_hessian:
        f2_hessian = h_der * phi_hessian + h_sec_der * np.matmul(phi_grad, np.transpose(phi_grad))
        to_ret.append(f2_hessian)
    return to_ret

# 1.2
def numerical_evaluation(f, x, eps, A=None):
    n = x.shape[0]
    I = np.identity(n) * eps
    grad = np.zeros((n, 1))
    hessian = np.zeros((n, n))
    for i in range(n):
        if A is None:
            f_val_add, f_g_add = f(x + np.expand_dims(I[:, i], 1), calc_grad=True, calc_hessian=False)
            f_val_sub, f_g_sub = f(x - np.expand_dims(I[:, i], 1), calc_grad=True, calc_hessian=False)
        else:
            f_val_add, f_g_add = f(x + np.expand_dims(I[:, i], 1), A, calc_grad=True, calc_hessian=False)
            f_val_sub, f_g_sub = f(x - np.expand_dims(I[:, i], 1), A, calc_grad=True, calc_hessian=False)

        grad[i] = (f_val_add - f_val_sub) / (2 * eps)
        hessian[:, i] = ((f_g_add - f_g_sub) / (2 * eps)).squeeze(1)

    return grad, hessian


def comparison():
    # This function computes the numerical and analytical expression for f1 and f2 gradient and hessian for epsilon
    # values between 2^0 until 2^-60 and outputs comparison graphs.
    np.random.seed(42)
    x = np.random.rand(3, 1)
    A = np.random.rand(3, 3)
    _, f1_g, f1_h = calc_f1_value_grad_hessian(x, A)
    _, f2_g, f2_h = f2_val_grad_hessian(x)

    f1_g_norm = []
    f1_h_norm = []

    f2_g_norm = []
    f2_h_norm = []

    x_axis = np.arange(61)

    for i in x_axis:
        f1_num_g, f1_num_h = numerical_evaluation(calc_f1_value_grad_hessian, x,
                                                  np.power(2.0, -i), A)

        f1_g_diff = np.linalg.norm(f1_g-f1_num_g, ord=np.inf)
        f1_g_norm.append(f1_g_diff)
        f1_h_diff = np.linalg.norm(f1_h-f1_num_h, ord=np.inf)
        f1_h_norm.append(f1_h_diff)

        f2_num_g, f2_num_h = numerical_evaluation(f2_val_grad_hessian, x,
                                                  np.power(2.0, -i))
        f2_g_diff = np.linalg.norm(f2_g-f2_num_g, ord=np.inf)
        f2_g_norm.append(f2_g_diff)
        f2_h_diff = np.linalg.norm(f2_h-f2_num_h, ord=np.inf)
        f2_h_norm.append(f2_h_diff)

    f1_g_min_err = np.min(f1_g_norm)
    f1_g_min_err_ind = np.argmin(f1_g_norm)
    f1_h_min_err = np.min(f1_h_norm)
    f1_h_min_err_ind = np.argmin(f1_h_norm)

    f2_g_min_err = np.min(f2_g_norm)
    f2_g_min_err_ind = np.argmin(f2_g_norm)
    f2_h_min_err = np.min(f2_h_norm)
    f2_h_min_err_ind = np.argmin(f2_h_norm)

    # Plot norm difference

    plt.subplot()
    plt.plot(x_axis, f1_g_norm)
    plt.yscale('log')
    plt.xlabel('Exponent absolute value')
    # plt.ylabel('Infinity norm of difference(log scale)')
    plt.suptitle('f1 gradient difference')
    plt.show()

    plt.plot(x_axis, f1_h_norm)
    plt.yscale('log')
    plt.xlabel('Exponent absolute value')
    # plt.ylabel('Infinity norm of difference(log scale)')
    plt.suptitle('f1 hessian difference')
    plt.show()

    plt.subplot()
    plt.plot(x_axis, f2_g_norm)
    plt.yscale('log')
    plt.xlabel('Exponent absolute value')
    # plt.ylabel('Infinity norm of difference(log scale)')
    plt.suptitle('f2 gradient difference')
    plt.show()

    plt.plot(x_axis, f2_h_norm)
    plt.yscale('log')
    plt.xlabel('Exponent absolute value')
    # plt.ylabel('Infinity norm of difference(log scale)')
    plt.suptitle('f2 hessian difference')
    plt.show()

    print(f'f1 gradient min error is: {f1_g_min_err} at exponent: {f1_g_min_err_ind}')
    print(f'f1 hessian min error is: {f1_h_min_err} at exponent: {f1_h_min_err_ind}')

    print(f'f2 gradient min error is: {f2_g_min_err} at exponent: {f2_g_min_err_ind}')
    print(f'f2 hessian min error is: {f2_h_min_err} at exponent: {f2_h_min_err_ind}')


if __name__ == '__main__':
    comparison()
