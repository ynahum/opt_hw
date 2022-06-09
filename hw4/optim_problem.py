from func import Func


class OptimizationProblem:

    def __init__(self, objective_func: Func):
        self.objective_func_operator = objective_func

    def func(self, x):
        return self.objective_func_operator.func(x)

    def grad(self, x):
        return self.objective_func_operator.grad(x)

    def hessian(self, x):
        return self.objective_func_operator.hessian(x)
