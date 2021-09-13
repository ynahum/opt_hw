import numpy as np
from rosenbrock_BFGS import *

if __name__ == '__main__':
    run_rosenbrock_BFGS = True
    if run_rosenbrock_BFGS:
        x_0 = np.ones((10, 1))
        print(f"rosenbrock func minimum at all ones vector: {rosenbrock_func(x_0)}")

        x_0 = np.zeros((10, 1))

        trajectory_points = BFGS(x_0)
        title = f'Rosenbrock optimization with BFGS\n'
        title += r'$log{(f(x_k)-f(x^*))}$ vs number of iterations'
        rosenbrock_plot(trajectory_points, title)
