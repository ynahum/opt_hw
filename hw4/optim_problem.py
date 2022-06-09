from func import Func


class OptimizationProblem:

    def __init__(self, objective_func: Func, ineq_constraints: [] = []):
        self.objective_func_operator = objective_func
        self.ineq_constraints = ineq_constraints.copy()

    def func(self, x):
        return self.objective_func_operator.func(x)

    def grad(self, x):
        return self.objective_func_operator.grad(x)

    def hessian(self, x):
        return self.objective_func_operator.hessian(x)

    # return g(x) <= 0 constrained function value
    def ineq_func(self, x, i):
        assert(0 <= i < len(self.ineq_constraints))
        return self.ineq_constraints[i].func(x)

    def ineq_grad(self, x, i):
        assert(0 <= i < len(self.ineq_constraints))
        return self.ineq_constraints[i].grad(x)

    def ineq_hessian(self, x, i):
        assert(0 <= i < len(self.ineq_constraints))
        return self.ineq_constraints[i].hessian(x)
