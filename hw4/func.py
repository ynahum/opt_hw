

class Func:
    def __init__(self, func, grad=None, hessian=None):
        self.func = func
        self.grad = grad
        self.hessian = hessian

