import numpy as np

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

