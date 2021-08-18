import numpy as np
from phi import calc_phi_value_grad_hessian
from h import calc_h_value_and_derivatives


# 1.1.5
def calc_f2_value_grad_hessian(x, calc_grad=True, calc_hessian=True):
    phi_value, phi_grad, phi_hessian = calc_phi_value_grad_hessian(x)
    h_value, h_first_derivative, h_second_derivative = calc_h_value_and_derivatives(phi_value)
    ret = [h_value]
    if calc_grad:
        ret.append(h_first_derivative * phi_grad)
    if calc_hessian:
        first_comp = h_first_derivative * phi_hessian
        phi_grad_expanded = np.expand_dims(phi_grad, axis=1)
        outer_prod = phi_grad_expanded @ phi_grad_expanded.T
        second_comp = h_second_derivative * outer_prod
        ret.append(first_comp + second_comp)
    return ret

