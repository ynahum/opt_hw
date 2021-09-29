import numpy as np
from rosenbrock import *
from BFGS import *
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator

# 1.3.5
def f(x_1, x_2):
    return x_1 * np.exp(-(x_1 ** 2 + x_2 ** 2))


# 1.3.6
def phi(x):
    return np.tanh(x)


def g_phi(x):
    return 1 - (np.tanh(x) ** 2)


def loss(F, y):
    return (F - y)**2


# 1.3.7
def g_loss(F, y):
    return 2 * (F - y)


# 1.3.10-11
def get_data(num_of_samples, sample_vec_size):
    X = 4 * np.random.rand(num_of_samples, sample_vec_size) - 2
    labels = []
    for x in X:
        labels.append(f(x))
    return X, np.array(labels)


# 1.3.8,9,12
class My_FC_NN_Model(object):
    def __init__(self, loss_type='mse'):
        self.layers = {}
        self.layer_index = 0
        self.loss_type = loss_type

    def add_layer(self, in_size, layer_size, W=None, b=None):

        # 1.3.12
        # generating the parameters randomly as requested
        if W is None:
            W = np.random.randn(in_size, layer_size) / np.sqrt(layer_size)
        if b is None:
            b = np.zeros((layer_size,1))
        assert (len(b) == layer_size)
        assert (W.shape[1] == layer_size)
        assert (W.shape[0] == in_size)
        self.layer_index += 1
        self.layers[f'l{self.layer_index}'] = {}
        self.layers[f'l{self.layer_index}']['W'] = W
        self.layers[f'l{self.layer_index}']['b'] = b

    def print(self):
        for i in np.arange(1,self.layer_index+1):
            print(f"layer {i}:")
            W = self.layers[f'l{i}']['W']
            print(f"W:\n{W}")
            b = self.layers[f'l{i}']['b']
            print(f"b:\n{b}")

    def fwd(self, input_vec):
        layers_outputs = {}
        next_layer_input = input_vec
        for i in np.arange(1, self.layer_index+1):
            W = self.layers[f'l{i}']['W']
            b = self.layers[f'l{i}']['b']
            y = np.dot(W.T, next_layer_input) + b
            layers_outputs[f'l{i}_lin'] = y
            next_layer_input = phi(y)
            layers_outputs[f'l{i}'] = next_layer_input
        return layers_outputs, y

    # 1.3.8
    def fwd_and_backprop(self, input_vec, label):
        outputs, y_hat = self.fwd(input_vec)
        grad_z = self.calc_loss_grad(label, y_hat)
        grads = {}
        for li in np.arange(start=self.layer_index, stop=0, step=-1):
            if li > 1:
                x = outputs[f'l{li-1}']
            else:
                x = input_vec
            W = self.layers[f'l{li}']['W']
            y = outputs[f'l{li}_lin']
            grads[f'l{li}'] = {}
            # if we're not in the last layer then we have an activation function
            # and thus it's gradient needs to be considered
            if li < self.layer_index:
                grad_act = g_phi(y)
                phi_grad_diag = np.diagflat(grad_act)
            else:
                phi_grad_diag = np.expand_dims(1, axis=(0, 1))
            grads[f'l{li}']['x'] = W @ phi_grad_diag @ grad_z
            grads[f'l{li}']['W'] = x @ grad_z.T @ phi_grad_diag
            grads[f'l{li}']['b'] = phi_grad_diag @ grad_z
            grad_z = grads[f'l{li}']['x']

        params_list = []
        for li in np.arange(start=self.layer_index, stop=0, step=-1):
            params_list.append(grads[f'l{li}']['W'])
            params_list.append(grads[f'l{li}']['b'])

        return outputs, np.concatenate(params_list, axis=None), y_hat

    # 1.3.9
    def fwd_and_backprop_batch(self, inputs, labels):
        n = len(labels)
        _, params_vec, _ = self.fwd_and_backprop(inputs[0], labels[0])
        grads_sum = np.zeros_like(params_vec)
        for input, label in zip(inputs, labels):
            _, params_vec, _ = self.fwd_and_backprop(input, label)
            grads_sum += params_vec
        grads_delta = grads_sum / n
        return grads_delta

    def set_loss_type(self, loss_type):
        self.loss_type = loss_type

    def calc_loss(self, y, y_hat):
        if self.loss_type == 'mse':
            L = loss(y_hat,y)
        else:
            assert(0, f'the loss type {self.loss_type} is not supported')
        return L

    def calc_loss_grad(self, y, y_hat):
        if self.loss_type == 'mse':
            L_der = g_loss(y_hat,y)
        else:
            assert(0, f'the loss type {self.loss_type} is not supported')
        return L_der


# 1.3.10,11 - generating data samples for training and testing
def gen_samples(func_ptr, n):
    labels = []
    samples = 4*np.random.rand(n, 2)-2
    for sample in samples:
        labels.append(func_ptr(sample[0], sample[1]))
    return samples, np.array(labels)

# 1.3.13
def plot_func(func_ptr, title="", model=None, plot_samples=False, samples=None, labels=None):

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    # Make data.
    X_1 = np.arange(-2, 2, 0.2)
    X_2 = np.arange(-2, 2, 0.2)
    X_1, X_2 = np.meshgrid(X_1, X_2)
    Z = func_ptr(X_1, X_2)

    # Plot the surface.
    surf = ax.plot_surface(X_1, X_2, Z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False, alpha=0.5)

    # Customize the z axis.
    ax.zaxis.set_major_locator(LinearLocator(10))
    # A StrMethodFormatter is used automatically
    ax.zaxis.set_major_formatter('{x:.02f}')

    # Add a color bar which maps values to colors.
    #fig.colorbar(surf, shrink=0.5, aspect=5, location='left')

    if plot_samples:
        for sample, label in zip(samples, labels):
            x_1, x_2 = sample
            ax.scatter(x_1, x_2, label, marker='o', color='green', s=5)

    ax.set_xlabel(r'$x_1$')
    ax.set_xticks(np.arange(-2, 2, 1))
    ax.set_ylabel(r'$x_2$')
    ax.set_yticks(np.arange(-2, 2, 1))
    ax.set_zlabel(r'$f(x_1,x_2)$')
    ax.set_zticks(np.arange(-2, 2, 0.2))
    ax.set_zlim(-0.42, 0.42)
    ax.view_init(azim=135,elev=30)

    plt.suptitle(title)

    plt.show()

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
    layers_sizes_dict = {'l1': (2,4), 'l2': (4,3), 'l3': (3,1)}

    nn_model = My_FC_NN_Model()
    layer_index = 1
    start_offset = 0
    for i in np.arange(1,len(layers_sizes_dict)+1):
        layer_size = layers_sizes_dict[f'l{layer_index}']
        print(f"layer size: {layer_size}")
        start_w = start_offset
        print(f"start_w: {start_w}")
        end_w = start_w + layer_size[0] * layer_size[1]
        print(f"end_w: {end_w}")
        start_b = end_w
        print(f"start_b: {start_b}")
        end_b = start_b + layer_size[1]
        print(f"end_b: {end_b}")
        nn_model.add_layer(
            in_size=layer_size[0],
            layer_size=layer_size[1],
            W=w_vec[start_w:end_w].reshape(layer_size),
            b=w_vec[start_b:end_b].reshape((layer_size[1],1))
        )
        start_offset = end_b
        layer_index += 1

    nn_model.print()

    fwd_outputs, y_estimate = nn_model.fwd([[1], [2]])
    print(f"fwd_outputs: {fwd_outputs}")
    print(f"y_estimate: {y_estimate}")
    fwd_bwd_outputs, gradients_vec, y_estimate = nn_model.fwd_and_backprop([[1], [2]], label=1)
    print(f"fwd_bwd_outputs: {fwd_bwd_outputs}")
    print(f"gradients_vec {gradients_vec}")
    print(f"y_estimate {y_estimate}")


    train_samples, train_labels = gen_samples(f, 200)
    plot_func(f, plot_samples=True, samples=train_samples, labels=train_labels)
