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


def loss(F, y):
    return (F - y)**2

# 1.3.7
def g_loss(F, y):
    return 2 * (F - y)


# Q1.3.8
class MyNNModel(object):
    def __init__(self, loss_type='mse'):
        self.layers = {}
        self.layer_index = 0
        self.loss_type = loss_type

    def add_layer(self, in_size, layer_size, weights, bias):
        assert (len(bias) == layer_size)
        assert (weights.shape[1] == layer_size)
        assert (weights.shape[0] == in_size)
        self.layers[f'l{self.layer_index}'] = {}
        self.layers[f'l{self.layer_index}']['W'] = weights
        self.layers[f'l{self.layer_index}']['b'] = bias
        self.layer_index += 1

    def print(self):
        for i in np.arange(0,self.layer_index):
            print(f"layer {i}:")
            W = self.layers[f'l{i}']['W']
            print(f"W:\n{W}")
            b = self.layers[f'l{i}']['b']
            print(f"b:\n{b}")

    def forward(self, input_vec):
        layers_outputs = {}
        next_layer_input = input_vec
        for i in np.arange(0, self.layer_index):
            W = self.layers[f'l{i}']['W']
            b = self.layers[f'l{i}']['b']
            y = np.dot(W.T, next_layer_input) + b
            layers_outputs[f'l{i}_lin'] = y
            next_layer_input = phi(y)
            layers_outputs[f'l{i}'] = next_layer_input
        return layers_outputs, y

    def forward_and_backprop(self, input_vec, label):
        outputs, y_hat = self.forward(input_vec)
        grad_z = self.calc_example_loss_grad(label, y_hat)
        grads = {}
        for li in np.arange(start=self.layer_index-1, stop=0, step=-1):
            x = outputs[f'l{li-1}']
            W = self.layers[f'l{li}']['W']
            y = outputs[f'l{li}_lin']
            grads[f'l{li}'] = {}
            # if we're not in the last layer then we have an activation function
            # and thus it's gradient needs to be considered
            if li < self.layer_index-1:
                phi_grad_diag = np.diagflat(g_phi(y))
            else:
                phi_grad_diag = np.expand_dims(1, axis=(0, 1))
            grads[f'l{li}']['x'] = W @ phi_grad_diag @ grad_z
            grads[f'l{li}']['W'] = x @ grad_z.T @ phi_grad_diag
            grads[f'l{li}']['b'] = phi_grad_diag @ grad_z
            grad_z = grads[f'l{li}']['x']

        return outputs, grads, y_hat

    def set_loss_type(self, loss_type):
        self.loss_type = loss_type

    def calc_example_loss(self, y, y_hat):
        if self.loss_type == 'mse':
            L = loss(y_hat,y)
        else:
            assert(0, f'the loss type {self.loss_type} is not supported')
        return L

    def calc_example_loss_grad(self, y, y_hat):
        if self.loss_type == 'mse':
            L_der = g_loss(y_hat,y)
        else:
            assert(0, f'the loss type {self.loss_type} is not supported')
        return L_der

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
    layers_sizes_dict = {'l0': (2,4), 'l1': (4,3), 'l2': (3,1)}

    nn_model = MyNNModel()
    layer_index = 0
    start_offset = 0
    for i in np.arange(0,len(layers_sizes_dict)):
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
            weights=w_vec[start_w:end_w].reshape(layer_size),
            bias=w_vec[start_b:end_b].reshape((layer_size[1],1))
        )
        start_offset = end_b
        layer_index += 1

    nn_model.print()

    fwd_outputs, y_estimate = nn_model.forward([[1], [2]])
    print(f"fwd_outputs: {fwd_outputs}")
    print(f"y_estimate: {y_estimate}")
    fwd_bwd_outputs, gradients, y_estimate = nn_model.forward_and_backprop([[1], [2]], label=1)
    print(f"fwd_bwd_outputs: {fwd_bwd_outputs}")
    print(f"gradients {gradients}")
    print(f"y_estimate {y_estimate}")

