from phi import calc_phi_value_grad_hessian


# 1.1.5
def calc_f1_value_grad_hessian(x, A, calc_grad=True, calc_hessian=True):
    phi_value, phi_grad, phi_hessian = calc_phi_value_grad_hessian(A @ x)
    ret = [phi_value]
    if calc_grad:
        ret.append(A.T @ phi_grad)
    if calc_hessian:
        ret.append(A.T @ phi_hessian @ A)
    return ret

