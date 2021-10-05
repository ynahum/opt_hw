import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# Q1.3.5
def f(x1, x2=None):
    if x2 is None:
        x1, x2 = x1
    return x1 * np.exp(-(x1 ** 2 + x2 ** 2))


# Q1.3.6
def tanh(x):
    return np.tanh(x)


def tanh_backward(x):
    return 1.0 - (np.tanh(x) ** 2)


# Q1.3.7
def loss_derivative(y, y_hat):
    return 2 * (y_hat - y)


# Q1.3.8
def forward(w : dict, x):
    if type(w) == np.ndarray:
        w = vector_to_param(w)
    outputs = {}
    A = x.reshape(2, 1)
    for i, (w_m, b_v) in enumerate(zip([w['w1'], w['w2'], w['w3']], [w['b1'], w['b2'], w['b3']])):
        z = np.dot(w_m.T, A) + b_v
        A = tanh(z)
        outputs[f'layer{i+1}_tanh'] = A
        outputs[f'layer{i+1}'] = z

    return z, outputs

def forward_backward(w, x, y):
    w = vector_to_param(w)
    y_hat, outputs = forward(w, x)
    dr = loss_derivative(y, y_hat)
    derivatives = {}
    tanh_backward_1 = np.diagflat(tanh_backward(outputs['layer1']))
    tanh_backward_2 = np.diagflat(tanh_backward(outputs['layer2']))

    derivatives['db3'] = dr
    derivatives['dw3'] = np.dot(outputs['layer2_tanh'], dr).reshape(-1, 1)

    derivatives['db2'] = np.dot(np.dot(tanh_backward_2, w['w3']), dr)
    derivatives['dw2'] = (outputs['layer1_tanh'] @ derivatives['db2'].T).reshape(-1, 1)

    derivatives['db1'] = tanh_backward_1 @ (w['w2'] @ derivatives['db2'])
    derivatives['dw1'] = (x.reshape(2,1) @ derivatives['db1'].T).reshape(-1, 1)

    vector = [derivatives['dw3'], derivatives['db3'], derivatives['dw2'], derivatives['db2'], derivatives['dw1'], derivatives['db1']]

    return np.concatenate(vector), y_hat


# Q1.3.9
def batch_forward_backward(w, train_x, train_y):
    derivatives = None
    for x, y in zip(train_x, train_y):
        point_der, _ = forward_backward(w, x, y)
        if derivatives is None:
            derivatives = point_der
        else:
            derivatives += point_der

    derivatives /= len(train_y)

    return derivatives

# Q1.3.10 + Q1.3.11
def generate_data(n):
    x = 4 * np.random.rand(n, 2) - 2
    y = []
    for p_x in x:
        y.append(f(p_x))

    return x, np.array(y)

# Q1.3.12
def get_model():

    w3 = np.random.randn(3, 1) / np.sqrt(1)
    b3 = np.zeros((1, 1))

    w2 = np.random.randn(4, 3) / np.sqrt(3)
    b2 = np.zeros((3, 1))

    w1 = np.random.randn(2, 4) / np.sqrt(4)
    b1 = np.zeros((4, 1))

    return np.concatenate([w3.reshape(-1, 1), b3, w2.reshape(-1, 1), b2, w1.reshape(-1, 1), b1])

# Q1.3.13
def plot(f, x, y, b, msg, w=None):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    # Make data.
    X1 = np.arange(-2, 2, 0.2)
    X2 = np.arange(-2, 2, 0.2)
    if w is None:
        X1, X2 = np.meshgrid(X1, X2)
        Z = f(X1, X2)
    else:
        Z = []
        for x2 in X2:
            Z_row = []
            for x1 in X1:
                Z_row.append(f(w, np.array([x1, x2]))[0].squeeze(1))
            Z.append(Z_row)
        X1, X2 = np.meshgrid(X1, X2)
        Z = np.array(Z).squeeze(2)

    # Plot the surface.
    surf = ax.plot_surface(X1, X2, Z, alpha=0.5, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)

    if b:
        for p_x, p_y in zip(x, y):
            x1, x2 = p_x
            ax.scatter(x1, x2, p_y, marker='o', color='red', s=0.5)

    ax.set_xlabel('X1')
    ax.set_xticks(np.arange(-2, 2, 0.5))
    ax.set_ylabel('X2')
    ax.set_yticks(np.arange(-2, 2, 0.5))
    ax.set_zlabel('Y')
    ax.set_zticks(np.arange(-2, 2, 0.5))
    ax.view_init(elev=15, azim=150)
    plt.suptitle(msg)
    plt.show()

# Q1.3.14 - training and validation
def full_pipeline():
    x_train, y_train = generate_data(500)
    x_test, _ = generate_data(200)
    w = get_model()
    #print(w)
    #epsilons = [1e-1, 1e-2, 1e-3, 1e-4]
    epsilons = [1e-2]

    for eps in epsilons:
        print(f'Running with epsilon = {eps}')
        w_opt, values_list, points_list = BFGS(loss, batch_forward_backward, w, eps, x_train, y_train)
        plot(forward, x_train, y_train, False, f'F(x,W*) using epsilon={eps}', w_opt)
        print(f'Finished train set loss = {values_list[-1]}, starting test set')
        y_test = []
        for x in x_test:
            out, _ = forward(w_opt, x)
            y_test.append(out)
        plot(f, x_test, y_test, True, f'Test set recovered points epsilon={eps} on f(x1, x2)')
        print('Finished test set')


def loss(w, train_x, train_y):
    w = vector_to_param(w)
    loss = 0
    for x, y in zip(train_x, train_y):
        y_hat, _ = forward(w, x)
        loss += (y_hat - y) ** 2

    loss /= len(train_y)

    return loss


def vector_to_param(w):
    w3 = w[:3].reshape(3, 1)
    b3 = w[3:4]

    w2 = w[4:16].reshape(4, 3)
    b2 = w[16:19]

    w1 = w[19:27].reshape(2, 4)
    b1 = w[27:31]

    dict = {'w3' : w3, 'b3' : b3, 'w2' : w2, 'b2' : b2, 'w1' : w1, 'b1' : b1}

    return dict


def inexact_line_search(x, f, g, d, train_x, train_y):
    beta = 0.5
    sigma = 0.25
    alpha = 1
    sigma2 = 0.9
    while True:
        if f(x + alpha * d, train_x, train_y) <= f(x, train_x, train_y) + sigma * alpha * np.matmul(g(x, train_x, train_y).T, d):
            # if np.matmul(g(x + alpha * d, train_x, train_y).T, d) >= sigma2 * np.matmul(g(x, train_x, train_y).T, d):
            if np.abs(np.matmul(g(x + alpha * d, train_x, train_y).T, d)) <= sigma2 * np.abs(np.matmul(g(x, train_x, train_y).T, d)):
                break
        alpha = alpha * beta

    return alpha

def BFGS(f, g, x, epsilon, train_x, train_y):
    H = np.identity(x.shape[0])
    I = np.identity(x.shape[0])
    values_list = []
    points_list = []
    iteration = 0
    gradient = g(x, train_x, train_y)
    while np.linalg.norm(gradient) > epsilon:
        iteration += 1
        loss = f(x, train_x, train_y)
        print(f'it={iteration} loss={loss}')
        values_list.append(loss)
        points_list.append(x)
        d = -np.matmul(H, gradient)
        alpha = inexact_line_search(x, f, g, d, train_x, train_y)
        print(f"alpha {alpha}")
        prev_x = x
        prev_grad = gradient
        x = x + alpha * d
        s_k = x - prev_x
        gradient = g(x, train_x, train_y)
        y_k = gradient - prev_grad
        rho = (y_k.T @ s_k)
        A = (s_k @ y_k.T) / rho
        B = (y_k @ s_k.T) / rho
        C = (s_k@s_k.T) / rho
        H = ((I - A ) @ H @ (I - B)) +  C


    print(f'Optimization took: {iteration} iterations')
    return x, values_list, points_list


if __name__ == '__main__':
    np.random.seed(10)

    full_pipeline()
