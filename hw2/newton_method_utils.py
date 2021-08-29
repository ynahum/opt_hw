import numpy as np
from mcholmz.mcholmz import modifiedChol


# 2.4.1
def forward(L, x):
    y = np.zeros_like(x)
    for idx, y_elem in enumerate(y):
        y[idx] = (x[idx] - L[idx] @ y) / L[idx, idx]
    return y


def backward(U, y):
    n = len(y)
    x = np.zeros_like(y)
    for i in np.arange(n-1,-1,-1):
        if U[i,i] == 0:
            continue
        x[i] = y[i]/U[i][i]
        temp = U[0:i,i] * x[i]
        y[0:i] = y[0:i] - temp.reshape((i,1))
    return x


def newton_direction(hessian, gradient):
    L, d, e = modifiedChol(hessian)
    L = np.array(L)
    d = np.array(d)
    y = forward(L, -gradient)
    n = len(d)
    D = np.zeros((n,n))
    np.fill_diagonal(D, d)
    x = backward(D@L.T, y)
    return x
