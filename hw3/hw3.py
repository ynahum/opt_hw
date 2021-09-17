import numpy as np
from rosenbrock import *
from BFGS import *


# 1.3.5
def f(x):
    return x[0] * np.exp(-(x[0] ** 2 + x[1] ** 2))


# 1.3.6
def phi(x):
    return np.tanh(x)


def g_phi(x):
    return 1 - (np.tanh(x) ** 2)


# 1.3.7
def g_loss(F, y):
    return 2 * (F - y)


def vec_to_dict(w_vec, sizes_dict):
    w3_size = sizes_dict['w3']
    start = 0
    end = w3_size[0] * w3_size[1]
    w3 = w_vec[start:end].reshape(w3_size)

    b3_size = sizes_dict['b3']
    start = end
    end = end + b3_size[0] * b3_size[1]
    b3 = np.squeeze(w_vec[start:end].reshape(b3_size),axis=1)

    w2_size = sizes_dict['w2']
    start = end
    end = end + w2_size[0] * w2_size[1]
    w2 = w_vec[start:end].reshape(w2_size)

    b2_size = sizes_dict['b2']
    start = end
    end = end + b2_size[0] * b2_size[1]
    b2 = np.squeeze(w_vec[start:end].reshape(b2_size),axis=1)

    w1_size = sizes_dict['w1']
    start = end
    end = end + w1_size[0] * w1_size[1]
    w1 = w_vec[start:end].reshape(w1_size)

    b1_size = sizes_dict['b1']
    start = end
    end = end + b1_size[0] * b1_size[1]
    b1 = np.squeeze(w_vec[start:end].reshape(b1_size),axis=1)

    w_dict = {'w3' : w3, 'b3' : b3, 'w2' : w2, 'b2' : b2, 'w1' : w1, 'b1' : b1}

    return w_dict


if __name__ == '__main__':
    run_rosenbrock_BFGS = False
    if run_rosenbrock_BFGS:
        x_0 = np.ones((10, 1))
        print(f"rosenbrock func minimum at all ones vector: {rosenbrock_func(x_0)}")

        x_0 = np.zeros((10, 1))

        trajectory_points = BFGS(x_0, rosenbrock_func, rosenbrock_grad)
        title = f'Rosenbrock optimization with BFGS\n'
        title += r'$log{(f(x_k)-f(x^*))}$ vs number of iterations'
        rosenbrock_plot(trajectory_points, title)

    w_vec = np.arange(0,31)
    print(w_vec)
    W_sizes_dict = {'w3': (3,1), 'b3': (1,1), 'w2': (4,3), 'b2': (3,1), 'w1': (2,4), 'b1' : (4,1)}
    w_dict = vec_to_dict(w_vec, W_sizes_dict)
    print(f"{w_dict['w3']}")
    print(f"{w_dict['b3']}")
    print(f"{w_dict['w2']}")
    print(f"{w_dict['b2']}")
    print(f"{w_dict['w1']}")
    print(f"{w_dict['b1']}")