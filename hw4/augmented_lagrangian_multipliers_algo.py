from optim_problem import *
from newton_method import newton_method


class AugmentedLagrangianSolver:
    def __init__(self, optim_problem: OptimizationProblem, penalty_func: Func, p_init=100, alpha=2):
        self.optim_problem = optim_problem
        self.penalty_func = penalty_func
        self.p = p_init
        self.alpha = alpha

    def func(self, x):
        f_value = self.optim_problem.func(x)
        for i, _ in enumerate(self.optim_problem.ineq_constraints):
            f_value += (1 / self.p) * self.penalty_func.func(self.p * self.optim_problem.ineq_func(x, i))
        return f_value

    def grad(self, x):
        g_value = self.optim_problem.grad(x)
        for i, _ in enumerate(self.optim_problem.ineq_constraints):
            g_i_x = self.optim_problem.ineq_func(x, i)
            nabla_g_i_x = self.optim_problem.ineq_grad(x, i)
            g_value += self.penalty_func.grad(self.p * g_i_x) * nabla_g_i_x
        return g_value

    def hessian(self, x):
        h_value = self.optim_problem.hessian(x)
        for i, _ in enumerate(self.optim_problem.ineq_constraints):
            g_i_x = self.optim_problem.ineq_func(x, i)
            nabla_g_i_x = self.optim_problem.ineq_grad(x, i)
            first_comp = (self.p * self.penalty_func.hessian(self.p * g_i_x) *
                          nabla_g_i_x @ nabla_g_i_x.T)
            nabla_square_g_i_x = self.optim_problem.ineq_hessian(x, i)
            second_comp = (self.penalty_func.grad(self.p * g_i_x) * nabla_square_g_i_x)
            h_value += first_comp + second_comp
        return h_value

    def solve(self, x_0):
        x_traj = []
        # loop of the algo
        # create unconstrained optimization problem using
        # the original optimization probem with penalty function and multiplies
        func_aggregate = Func(self.func, self.grad, self.hessian)

        unconstrained_optim_problem = OptimizationProblem(func_aggregate)

        # run newton method on the new unconstrained problem and get optimal x
        x_traj = newton_method(x_0, unconstrained_optim_problem)
        # calculate new multiplier phi'(x_optimal)

        return x_traj


