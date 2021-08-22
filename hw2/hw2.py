import numpy as np
from grad_descent import grad_descent
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

    grad_descent_exact_line_search_config = [ True, True, True, False, False, False]

    for idx, Q in enumerate(Qs):
        trajectory_points = grad_descent(
            x_0s[idx], Q, exact_line_search=grad_descent_exact_line_search_config[idx])
        plot_search_method_graphs(Q, trajectory_points)
