import numpy as np
from grad_descent import grad_descent
from newton_method import newton_method
from plot_utils import plot_search_method_graphs

if __name__ == '__main__':

    Qs = np.array([
            [[10, 0.],
            [0, 1]],
            [[3, 0.],
             [0, 3]],
            [[10, 0.],
             [0, 1]],
            [[10, 0.],
             [0, 1]],
            [[3, 0.],
             [0, 3]],
            [[10, 0.],
             [0, 1]],
        ])
    x_0s = np.array([
            [[-0.2], [-2.]],
            [[-0.2], [-2.]],
            [[-2], [0]],
            [[-0.2], [-2.]],
            [[-0.2], [-2.]],
            [[-2], [0]],
        ])

    exact_line_search_config = [ True, True, True, False, False, False]

    run_grad_descent = True

    if run_grad_descent:
        for idx, Q in enumerate(Qs):
            trajectory_points = grad_descent(
                x_0s[idx], Q, exact_line_search=exact_line_search_config[idx])
            plot_search_method_graphs(Q, trajectory_points)

    run_newton_method = True

    if run_newton_method:
        for idx, Q in enumerate(Qs):
            trajectory_points = newton_method(
                x_0s[idx], Q, exact_line_search=exact_line_search_config[idx])
            plot_search_method_graphs(Q, trajectory_points)
