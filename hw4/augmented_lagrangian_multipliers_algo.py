from audioop import mul

from optim_problem import *
from penalty import *
from newton_method import newton_method
import numpy as np


class AugmentedLagrangianSolver:

    def __init__(self, optim_problem: OptimizationProblem, p_init=100, alpha=2):
        self.optim_problem = optim_problem
        self.num_of_constraints = len(self.optim_problem.ineq_constraints)
        self.penalty = [Penalty(p_init) for _ in range(self.num_of_constraints)]
        self.p = p_init
        self.alpha = alpha
        self.p_max = 1000

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

    def update_multipliers(self, x):
        for i, _ in enumerate(self.optim_problem.ineq_constraints):
            g_i_x = self.optim_problem.ineq_func(x, i)
            current_multiplier = self.penalty[i].multiplier
            next_multiplier = self.penalty[i].first_derivative(g_i_x)
            next_multiplier = min(current_multiplier*3, next_multiplier)
            next_multiplier = max(current_multiplier/3.0, next_multiplier)
            self.penalty[i].multiplier = next_multiplier

    def update_penalty_param(self):
        for i, _ in enumerate(self.optim_problem.ineq_constraints):
            self.penalty[i].p = self.p

    def solve(self, x_0):
        x_trajectory = []
        aggregate_grads = []
        mult_trajectory = []

        current_multipliers = np.zeros((self.num_of_constraints, 1))

        func_aggregate = Func(self.func, self.grad, self.hessian)

        # loop of the algo
        while self.p < self.p_max:

            # update penalty parameters (to every constraint penalty function)
            self.update_penalty_param()

            # create unconstrained optimization problem using
            # the original optimization probem with penalty function and multiplies
            unconstrained_optim_problem = OptimizationProblem(func_aggregate)

            # run newton method on the new unconstrained problem and get optimal x

            returned_hash = newton_method(x_0, unconstrained_optim_problem)
            x_trajectory.extend(returned_hash['x_list'])
            x_optimal = x_trajectory[-1]

            aggregate_grads.extend(returned_hash['grad_list'])

            for i, _ in enumerate(self.optim_problem.ineq_constraints):
                current_multipliers[i] = self.penalty[i].multiplier
            mult_trajectory.append(current_multipliers.copy())

            # calculate new multiplier phi'(x_optimal)
            self.update_multipliers(x_optimal)

            # increase p by alpha
            self.p *= self.alpha

        return {'x_list': x_trajectory, 'grad_list': aggregate_grads, 'mult_list': mult_trajectory}


