import numpy as np
import matplotlib.pyplot as plt

# 1.1.5
def calc_phi_grad(x):
    u_value = np.prod(x, axis=0)
    cos_value = np.cos(u_value)
    u_grad = np.array([x[1] * x[2], x[0] * x[2], x[0] * x[1]])
    return  cos_value * u_grad


def phi_val_grad_hessian(x, calc_g=True, calc_h=True):
    # Input : Vector x of shape 3x1
    # Input : calc_g(bool), calc_h(bool) - whether to calculate gradient or hessian(respectively)
    # Value, gradient and hessian calculation for function: phi(x) = sin(x[0]*x[1]*x[2])
    cos = np.cos(np.prod(x, 0))
    sin = np.sin(np.prod(x, 0))
    phi_val = sin
    to_ret = [phi_val]
    if calc_g:
        phi_grad = np.array([x[1] * x[2],
                             x[0] * x[2],
                             x[0] * x[1]]) * cos
        # phi_grad=np.array([x[1] * x[2],
        #           x[0] * x[2],
        #           x[0] * x[1]]) * np.cos(np.prod(x, axis=0))
        to_ret.append(phi_grad)
    if calc_h:
        h11 = -(x[1] ** 2) * (x[2] ** 2) * sin
        h12 = x[2] * cos - x[0] * x[1] * (x[2] ** 2) * sin
        h13 = x[1] * cos - x[0] * (x[1] ** 2) * x[2] * sin
        h22 = -(x[0] ** 2) * (x[2] ** 2) * sin
        h23 = x[0] * cos - (x[0] ** 2) * x[1] * x[2] * sin
        h33 = -(x[0] ** 2) * (x[1] ** 2) * sin

        phi_hessian = np.array([[h11, h12, h13],
                                [h12, h22, h23],
                                [h13, h23, h33]]).squeeze(2)
        to_ret.append(phi_hessian)
    return to_ret


def f1_val_grad_hessian(x, A, calc_g=True, calc_h=True):
    # Input : Vector x of shape 3x1, Matrix A of shape 3x3
    # Input : calc_g(bool), calc_h(bool) - whether to calculate gradient or hessian(respectively)
    # Value, gradient and hessian calculation for function: f1(x) = phi(A@x)
    phi_val, phi_grad, phi_hessian = phi_val_grad_hessian(np.matmul(A,x))
    to_ret = [phi_val]
    if calc_g:
        f1_grad = np.matmul(np.transpose(A), phi_grad)
        to_ret.append(f1_grad)
    if calc_h:
        f1_hessian = np.matmul(np.matmul(np.transpose(A), phi_hessian), A)
        to_ret.append(f1_hessian)
    return to_ret


def h_val_der_sec_der(x, calc_g=True, calc_h=True):
    # Input : Scalar x
    # Input : calc_g(bool), calc_h(bool) - whether to calculate gradient or hessian(respectively)
    # Value, gradient and hessian calculation for function: h(x) = sqrt(1+x^2)
    h_val = np.sqrt(1 + (x ** 2))
    to_ret = [h_val]
    if calc_g:
        h_der = x/np.sqrt(1+(x**2))
        to_ret.append(h_der)
    if calc_h:
        h_sec_der = 1 / np.power(1 + (x ** 2), 1.5)
        to_ret.append(h_sec_der)
    return to_ret


def f2_val_grad_hessian(x, calc_g=True, calc_h=True):
    # Input : Vector x of shape 3x1
    # Input : calc_g(bool), calc_h(bool) - whether to calculate gradient or hessian(respectively)
    # Value, gradient and hessian calculation for function: f2(x) = h(phi(x))
    phi_val, phi_grad, phi_hessian = phi_val_grad_hessian(x)
    h_val, h_der, h_sec_der = h_val_der_sec_der(phi_val)
    f2_val = h_val
    to_ret = [f2_val]
    if calc_g:
        f2_grad = h_der * phi_grad
        to_ret.append(f2_grad)
    if calc_h:
        f2_hessian = h_der * phi_hessian + h_sec_der * np.matmul(phi_grad, np.transpose(phi_grad))
        to_ret.append(f2_hessian)
    return to_ret


def numerical_evaluation(f, x, eps, A=None):
    # This function gets as input a function reference f, vector x and matrix A(optional depends on f)
    # as inputs for f.
    # Output: grad - gradient numerical estimation for f at point x
    # Output: hessian - hessian numerical estimation for f at point x
    n = x.shape[0]
    I = np.identity(n) * eps
    grad = np.zeros((n, 1))
    hessian = np.zeros((n, n))
    for i in range(n):
        if A is None:
            f_val_add, f_g_add = f(x + np.expand_dims(I[:, i], 1), calc_g=True, calc_h=False)
            f_val_sub, f_g_sub = f(x - np.expand_dims(I[:, i], 1), calc_g=True, calc_h=False)
        else:
            f_val_add, f_g_add = f(x + np.expand_dims(I[:, i], 1), A, calc_g=True, calc_h=False)
            f_val_sub, f_g_sub = f(x - np.expand_dims(I[:, i], 1), A, calc_g=True, calc_h=False)

        grad[i] = (f_val_add - f_val_sub) / (2 * eps)
        hessian[:, i] = ((f_g_add - f_g_sub) / (2 * eps)).squeeze(1)

    return grad, hessian

def comparison():
    # This function computes the numerical and analytical expression for f1 and f2 gradient and hessian for epsilon
    # values between 2^0 until 2^-60 and outputs comparison graphs.
    np.random.seed(42)
    x = np.random.rand(3, 1)
    A = np.random.rand(3, 3)
    _, f1_g, f1_h = f1_val_grad_hessian(x, A)
    _, f2_g, f2_h = f2_val_grad_hessian(x)

    f1_g_norm = []
    f1_h_norm = []

    f2_g_norm = []
    f2_h_norm = []

    x_axis = np.arange(61)

    for i in x_axis:
        f1_num_g, f1_num_h = numerical_evaluation(f1_val_grad_hessian, x,
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
