from newton_method import newton_method
from rosenbrock import *
from optim_problem import *
from plot_utils import *
import numpy as np


if __name__ == '__main__':

    test_rosen = True
    if test_rosen:
        x_0 = np.zeros((10, 1))

        f = Func(func=rosenbrock_func, grad=rosenbrock_grad, hessian=rosenbrock_hessian)
        rosen_op_problem = OptimizationProblem(objective_func=f)

        trajectory_points = newton_method(x_0, rosen_op_problem)
        title = f'Newton Method over rosenbrock trajectory plot'
        traj_plot(trajectory_points, rosenbrock_func, title=title)


