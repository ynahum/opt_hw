from rosenbrock import *
from quadratic_func_and_constraints import *
from plot_utils import *
from augmented_lagrangian_multipliers_algo import *
from newton_method import newton_method
from optim_problem import *
from penalty import *
import numpy as np


if __name__ == '__main__':


    test_rosen = False
    if test_rosen:
        x_0 = np.zeros((10, 1))

        f = Func(func=rosenbrock_func, grad=rosenbrock_grad, hessian=rosenbrock_hessian)
        rosen_op_problem = OptimizationProblem(objective_func=f)

        trajectory_points = newton_method(x_0, rosen_op_problem)
        title = f'Newton Method over rosenbrock trajectory plot'
        traj_plot(trajectory_points, rosenbrock_func, title=title)

    test_augmented_rosen = False
    if test_augmented_rosen:
        x_0 = np.zeros((10, 1))
        f = Func(func=rosenbrock_func, grad=rosenbrock_grad, hessian=rosenbrock_hessian)
        penalty_f = Func(func=penalty_func, grad=penalty_first_derivative, hessian=penalty_second_derivative)
        rosen_op_problem = OptimizationProblem(objective_func=f)
        x_traj = augmented_lagrangian_mult_algo(rosen_op_problem, penalty_f, x_0)
        title = f'Augmented Lagrangian over rosenbrock trajectory plot'
        traj_plot(x_traj, rosenbrock_func, title=title)

    test_augmented = True
    if test_augmented:
        x_0 = np.zeros((2, 1))
        f = Func(func=obj_func, grad=obj_grad, hessian=obj_hessian)
        penalty_f = Func(func=penalty_func, grad=penalty_first_derivative, hessian=penalty_second_derivative)
        op_problem = OptimizationProblem(objective_func=f)
        x_traj = augmented_lagrangian_mult_algo(op_problem, penalty_f, x_0)
        title = f'Augmented Lagrangian over report problem trajectory plot'
        print(x_traj)
        traj_plot(x_traj, obj_func, title=title)
