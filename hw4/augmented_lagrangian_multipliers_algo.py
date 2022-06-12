from optim_problem import *
from penalty import *
from newton_method import newton_method
import numpy as np


class AugmentedLagrangianSolver:

    def __init__(self, optim_problem: OptimizationProblem, p_init=100, alpha=2):
        self.optim_problem = optim_problem
        num_of_constraints = len(self.optim_problem.ineq_constraints)
        self.penalty = [Penalty(p_init) for _ in range(num_of_constraints)]
        self.alpha = alpha
        self.multipliers = np.ones((num_of_constraints, num_of_constraints))

    def func(self, x):
        f_value = self.optim_problem.func(x)
        for i, _ in enumerate(self.optim_problem.ineq_constraints):
            g_i_x = self.optim_problem.ineq_func(x, i)
            f_value += self.penalty[i].func(g_i_x)
        return f_value

    def grad(self, x):
        g_value = self.optim_problem.grad(x)
        for i, _ in enumerate(self.optim_problem.ineq_constraints):
            g_i_x = self.optim_problem.ineq_func(x, i)
            nabla_g_i_x = self.optim_problem.ineq_grad(x, i)
            g_value += self.penalty[i].first_derivative(g_i_x) * nabla_g_i_x
        return g_value

    def hessian(self, x):
        h_value = self.optim_problem.hessian(x)
        for i, _ in enumerate(self.optim_problem.ineq_constraints):
            g_i_x = self.optim_problem.ineq_func(x, i)
            nabla_g_i_x = self.optim_problem.ineq_grad(x, i)
            first_comp = (self.penalty[i].second_derivative(g_i_x) * nabla_g_i_x @ nabla_g_i_x.T)
            nabla_square_g_i_x = self.optim_problem.ineq_hessian(x, i)
            second_comp = (self.penalty[i].first_derivative(g_i_x) * nabla_square_g_i_x)
            h_value += first_comp + second_comp
        return h_value

    def solve(self, x_0):
        num_of_constraints = len(self.optim_problem.ineq_constraints)
        self.multipliers = np.ones((num_of_constraints, num_of_constraints))

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


