import numpy as np

# 1.1.5.2
def calc_phi_value_grad_hessian(x, calc_grad=True, calc_hessian=True):
    x_1_squared = x[1]**2
    u_value = x[0] * x_1_squared * x[2]
    cos_value = np.cos(u_value)
    sin_value = np.sin(u_value)
    value = cos_value
    ret = [value]

    if calc_grad:
        u_grad = np.array([x_1_squared * x[2],
                           2 * x[0] * x[1] * x[2],
                           x[0] * x_1_squared])
        phi_grad = -sin_value * u_grad
        ret.append(phi_grad)
    if calc_hessian:
        u_hessian = np.array([[0,                 2 * x[1] * x[2],   x_1_squared],
                             [2 * x[1] * x[2],   2 * x[0] * x[2],   2 * x[0] * x[1]],
                             [x_1_squared,       2 * x[0] * x[1],   0]])
        H2_11 = np.power(x[1],4) * np.power(x[2],2)
        H2_12 = 2 * x[0] * np.power(x[1],3) * np.power(x[2],2)
        H2_13 = x[0] * np.power(x[1],4) * x[2]
        H2_22 = 4 * np.power(x[0],2) * np.power(x[1],2) * np.power(x[2],2)
        H2_23 = 2 * np.power(x[0],2) * np.power(x[1],3) * x[2]
        H2_33 = np.power(x[0],2) * np.power(x[1],2)
        H2 = np.array([[H2_11, H2_12, H2_13],
                       [H2_12, H2_22, H2_23],
                       [H2_13, H2_23, H2_33]])
        phi_hessian = - cos_value * H2 - sin_value * u_hessian
        ret.append(phi_hessian)
    return ret

