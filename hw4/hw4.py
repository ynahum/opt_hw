from rosenbrock import *
from quadratic_func_and_constraints import *
from plot_utils import *
from augmented_lagrangian_multipliers_algo import *
from newton_method import newton_method
from optim_problem import *
from penalty import *
import numpy as np


if __name__ == '__main__':

    test_unconst_rosen = False
    if test_unconst_rosen:
        x_0 = np.zeros((10, 1))

        f = Func(func=rosenbrock_func, grad=rosenbrock_grad, hessian=rosenbrock_hessian)
        rosen_op_problem = OptimizationProblem(objective_func=f)

        trajectory_points = newton_method(x_0, rosen_op_problem)
        title = f'Newton Method over rosenbrock trajectory plot'
        traj_plot(trajectory_points, rosenbrock_func, title=title)

    test_unconst_rosen_2 = False
    if test_unconst_rosen_2:
        x_0 = np.zeros((10, 1))
        f = Func(func=rosenbrock_func, grad=rosenbrock_grad, hessian=rosenbrock_hessian)
        rosen_op_problem = OptimizationProblem(objective_func=f)
        augmented_lagrangian_solver = AugmentedLagrangianSolver(rosen_op_problem)
        x_traj = augmented_lagrangian_solver.solve(x_0)
        title = f'Augmented Lagrangian over rosenbrock trajectory plot'
        traj_plot(x_traj, rosenbrock_func, title=title)

    test_augmented = True
    if test_augmented:
        x_0 = np.zeros((2, 1))

        f = Func(func=obj_func, grad=obj_grad, hessian=obj_hessian)

        g1 = Func(func=g1_func, grad=g1_grad, hessian=g1_hessian)
        g2 = Func(func=g2_func, grad=g2_grad, hessian=g2_hessian)
        g3 = Func(func=g3_func, grad=g3_grad, hessian=g3_hessian)
        ineq_const = [g1, g2, g3]

        op_problem = OptimizationProblem(objective_func=f, ineq_constraints=ineq_const)

        augmented_lagrangian_solver = AugmentedLagrangianSolver(op_problem, p_init=100)

        x_traj = augmented_lagrangian_solver.solve(x_0)

        title = f'Augmented Lagrangian over report problem trajectory plot'
        print(x_traj)
        traj_plot(x_traj, augmented_lagrangian_solver.func, y_scale_to_log=False, title=title)
