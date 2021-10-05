import numpy as np
import matplotlib.pyplot as plt

# =============== Q1.1

def inexact_line_search(x, f, g, d):
    beta = 0.5
    sigma = 0.25
    alpha = 1
    sigma2 = 0.9
    while True:
        if f(x + alpha * d) <= f(x) + sigma * alpha * np.matmul(g(x).T, d):
            if np.matmul(g(x + alpha * d).T, d) >= sigma2 * np.matmul(g(x).T, d):
                break
        alpha = alpha * beta

    return alpha

def BFGS(f, g, x, epsilon):
    H = np.identity(x.shape[0])
    values_list = []
    points_list = []
    iteration = 0
    while np.linalg.norm(g(x)) > epsilon:
        iteration += 1
        values_list.append(f(x))
        points_list.append(x)
        d = np.matmul(-H, g(x))
        alpha = inexact_line_search(x, f, g, d)
        prev_x = x
        prev_grad = g(x)
        x = x + alpha * d
        s_k = x - prev_x
        y_k = g(x) - prev_grad

        H = (np.identity(x.shape[0]) - ((s_k @ y_k.T) / (y_k.T @ s_k))) @ H @ (np.identity(x.shape[0]) - ((y_k@s_k.T)/(y_k.T@s_k))) +\
            ((s_k@s_k.T)/(y_k.T@s_k))

    return x, values_list, points_list

# ====================

# =============== Q1.2
def frosenbrock_f(x):
    return sum(100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0)

def frosenbrock_g(x):
    der = np.zeros_like(x)
    der[1:-1] = -2 * (1 - x[1:-1]) -400 * x[1:-1] * (x[2:] - x[1:-1]**2.0) + 200 * (x[1:-1] - x[:-2]**2.0)
    der[0] = -2 * (1-x[0]) -400 * x[0] * (x[1] - x[0]**2.0)
    der[-1] = 200 * (x[-1] - x[-2]**2.0)
    return der

def plot_curve(values_list):
    plt.plot(np.arange(len(values_list)), values_list)
    plt.yscale('log')
    plt.xlabel('iteration #')
    plt.suptitle('Covergence curve of Ross function using BFGS method')
    plt.show()

# ====================

if __name__ == '__main__':
    x, values_list, points_list = BFGS(frosenbrock_f, frosenbrock_g, np.zeros((10,1)), epsilon=1e-5)
    plot_curve(values_list)
    pass
