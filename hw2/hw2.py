import numpy as np
from grad_descent import grad_descent
from newton_method import newton_method
from plot_utils import plot_search_method_graphs
from rosenbrock import *

if __name__ == '__main__':

    Qs = np.array([
            [[3, 0.],
             [0, 3]],
            [[10, 0.],
             [0, 1]],
            [[10, 0.],
             [0, 1]],
        ])
    x_0s = np.array([
            [[1.5], [2.]],
            [[1.5], [2.]],
            [[1.5], [2.]],
        ])

    exact_line_search_config = [ True, True, False]

    run_grad_descent = True

    if run_grad_descent:
        for idx, Q in enumerate(Qs):
            trajectory_points = grad_descent(
                x_0s[idx], Q, exact_line_search=exact_line_search_config[idx])
            line_search = " exact " if exact_line_search_config[idx] else " inexact "
            title = f'Gradient Descent with {line_search} line search diag(Q)=({Q[0][0]},{Q[1][1]}) x0=({x_0s[idx][0][0]},{x_0s[idx][1][0]})'
            plot_search_method_graphs(Q, trajectory_points, title=title)

    run_newton_method = True

    if run_newton_method:
        for idx, Q in enumerate(Qs):
            trajectory_points = newton_method(
                x_0s[idx], Q, exact_line_search=exact_line_search_config[idx])
            line_search = " exact " if exact_line_search_config[idx] else " inexact "
            title = f'Newton Method with {line_search} line search diag(Q)=({Q[0][0]},{Q[1][1]}) x0=({x_0s[idx][0][0]},{x_0s[idx][1][0]})'
            plot_search_method_graphs(Q, trajectory_points, title=title)

    run_rosenbrock_inexact = True
    if run_rosenbrock_inexact:
        x_0 = np.ones((10,1))
        print(f"rosenbrock func minimum at all ones vector: {rosenbrock_func(x_0)}")

        x_0 = np.zeros((10,1))

        trajectory_points = rosenbrock_inexact_line_search_grad_descent(x_0)
        title = f'Rosenbrock optimization with Gradient Descent inexact line search:\n'
        title += r'$log{(f(x_k)-f(x^*))}$ vs number of iterations'
        rosenbrock_plot(trajectory_points, title)

        trajectory_points = rosenbrock_inexact_line_search_newton_method(x_0)
        title = f'Rosenbrock optimization with Newton Method inexact line search\n'
        title += r'$log{(f(x_k)-f(x^*))}$ vs number of iterations'
        rosenbrock_plot(trajectory_points, title)

