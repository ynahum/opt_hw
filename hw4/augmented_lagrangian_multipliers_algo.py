from optim_problem import *
from newton_method import newton_method



class PenaltyAggregateFunction:
    def __init__(self, optim_problem: OptimizationProblem, penalty_func: Func, p_init=100, alpha=2):
        self.optim_problem = optim_problem
        self.penalty_func = penalty_func
        self.p = p_init
        self.alpha = alpha

    def func(self, x):
        f_value = self.optim_problem.func(x)
        for i, _ in enumerate(self.optim_problem.ineq_constraints):
            f_value += (1 / self.p) * self.penalty_func(self.p * self.optim_problem.ineq_func(x, i))
        return f_value

    def grad(self, x):
        # TODO: add ineq
        return self.optim_problem.grad(x)

    def hessian(self, x):
        # TODO: add ineq
        return self.optim_problem.hessian(x)


def augmented_lagrangian_mult_algo(optim_problem: OptimizationProblem, penalty_func: Func, x_0, init_p=100, alpha=2):

    x_traj = []
    aggregate = PenaltyAggregateFunction(optim_problem, penalty_func, p_init=init_p, alpha=alpha)
    p = init_p
    # loop of the algo
        # create unconstrained optimization problem using
        # the original optimization probem with penalty function and multiplies
    func_aggregate = Func(aggregate.func, aggregate.grad, aggregate.hessian)

    unconstrained_optim_problem = OptimizationProblem(func_aggregate)

        # run newton method on the new unconstrained problem and get optimal x
    x_traj = newton_method(x_0, unconstrained_optim_problem)
        # calculate new multiplier phi'(x_optimal)

    return x_traj
