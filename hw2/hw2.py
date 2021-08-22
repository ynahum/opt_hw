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
    ])
    x_0s = np.array([
            [[-0.2], [-2.]],
            [[-0.2], [-2.]],
            [[-2], [0]],
    ])

    for idx, Q in enumerate(Qs):
        x_0 = x_0s[idx]
        trajectory_points = grad_descent(x_0, Q)
        plot_search_method_graphs(Q, trajectory_points)
