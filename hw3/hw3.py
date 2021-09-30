import numpy as np
from rosenbrock import *
from BFGS import *
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator


def f(x_1, x_2):
    return x_1 * np.exp(-(x_1 ** 2 + x_2 ** 2))


# 1.3.5-9,12
class My_FC_NN_Model(object):
    def __init__(self, in_size, loss_type='mse', activation_type='tanh'):
        self.layers = {}
        self.layer_index = 0
        self.loss_type = loss_type
        self.activation_type = activation_type
        self.input_size = in_size
        self.last_layer_size = self.input_size

    # API functions

    # API function for adding a hidden layer with specified size
    def add_layer(self, layer_size, W=None, b=None):
        prev_layer_size = self.last_layer_size

        # 1.3.12
        # generating the parameters randomly if not given
        if W is None:
            W = np.random.randn(prev_layer_size, layer_size) / np.sqrt(layer_size)
        if b is None:
            b = np.zeros((layer_size,1))
        assert (len(b) == layer_size)
        assert (W.shape[1] == layer_size)
        assert (W.shape[0] == prev_layer_size)
        self.layer_index += 1
        self.layers[f'l{self.layer_index}'] = {}
        self.layers[f'l{self.layer_index}']['W'] = W
        self.layers[f'l{self.layer_index}']['b'] = b
        self.last_layer_size = layer_size

    # API function for setting loss type:
    # 'mse' is the default
    def set_loss_type(self, loss_type):
        self.loss_type = loss_type

    # API function for setting activation type:
    # 'tanh' is the default
    def set_activation_type(self, activation_type):
        self.activation_type = activation_type

    # API function for model prediction (wrapping above fwd)
    def predict(self, input_vec):
        return self.fwd(input_vec)

    # API function for printing the model layers' params
    def print(self):
        for i in np.arange(1,self.layer_index+1):
            print(f"layer {i}:")
            W = self.layers[f'l{i}']['W']
            print(f"W:\n{W}")
            b = self.layers[f'l{i}']['b']
            print(f"b:\n{b}")

    # Internal functions

    # 1.3.5
    # Forwarding an input sample through the model layers
    def fwd(self, input_vec):
        layers_outputs = {}
        next_layer_input = input_vec
        for i in np.arange(1, self.layer_index+1):
            W = self.layers[f'l{i}']['W']
            b = self.layers[f'l{i}']['b']
            y = np.dot(W.T, next_layer_input) + b
            layers_outputs[f'l{i}_lin'] = y
            next_layer_input = self.phi(y)
            layers_outputs[f'l{i}'] = next_layer_input
        return layers_outputs, y

    # 1.3.6
    def phi(self, x):
        act_out = None
        if self.activation_type == 'tanh':
            act_out = np.tanh(x)
        else:
            assert 0, f'the activation type {self.activation_type} is not supported'
        return act_out

    def g_phi(self, x):
        g_act_out = None
        if self.activation_type == 'tanh':
            g_act_out = 1 - (np.tanh(x) ** 2)
        else:
            assert 0, f'the activation type {self.activation_type} is not supported'
        return g_act_out

    # 1.3.7
    def loss(self, y, F):
        if self.loss_type == 'mse':
            L = (F - y)**2
        else:
            assert 0, f'the loss type {self.loss_type} is not supported'
        return L

    def batch_loss(self, labels, predictions):
        if self.loss_type == 'mse':
            L = 0
            for y, F in zip(labels, predictions):
                L += self.loss(y, F)
            L /= len(labels)
        else:
            assert 0, f'the loss type {self.loss_type} is not supported'
        return L

    def loss_grad(self, y, F):
        if self.loss_type == 'mse':
            L_der = 2 * (F - y)
        else:
            assert 0, f'the loss type {self.loss_type} is not supported'
        return L_der

    # 1.3.8
    def fwd_and_backprop(self, input_vec, label):
        outputs, y_hat = self.fwd(input_vec)
        grad_z = self.loss_grad(label, y_hat)
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
                grad_act = self.g_phi(y)
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
    def batch_fwd_and_backprop(self, inputs, labels):
        n = len(labels)
        _, params_vec, _ = self.fwd_and_backprop(inputs[0], labels[0])
        grads_sum = np.zeros_like(params_vec)
        for input, label in zip(inputs, labels):
            _, params_vec, _ = self.fwd_and_backprop(input, label)
            grads_sum += params_vec
        grads_delta = grads_sum / n
        return grads_delta


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


# helper function to build the model layers
def build_model(in_size, hidden_layers_sizes):
    nn_model = My_FC_NN_Model(in_size=in_size)
    for li in np.arange(1,len(hidden_layers_sizes)+1):
        nn_model.add_layer(layer_size=hidden_layers_sizes[li-1])
    return nn_model


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

    in_size = 2
    hidden_layers_sizes = (4,3,1)
    nn_model = build_model(in_size=in_size, hidden_layers_sizes=hidden_layers_sizes)
    nn_model.print()

    train_samples, train_labels = gen_samples(f, 500)
    #plot_func(f, plot_samples=True, samples=train_samples, labels=train_labels)

    test_samples, test_labels = gen_samples(f, 200)
    plot_func(f, plot_samples=True, samples=test_samples, labels=test_labels)

    run_func_approximation = False
    if run_func_approximation:
        epsilons = [1e-1, 1e-2, 1e-3, 1e-4]

        for epsilon in epsilons:
            print(f'Optimizing model with epsilon = {epsilon}')
            print(f'Start training')
            #w_opt, values_list, points_list = BFGS(loss, batch_forward_backward, w, eps, x_train, y_train)
            #plot(forward, x_train, y_train, False, f'F(x,W*) using epsilon={eps}', w_opt)
            print(f'Ended training')
            #print(f'Trained examples Loss = {values_list[-1]}')
            print(f'Start testing')
            predictions = []
            for x in test_samples:
                _, prediction = nn_model.predict(x)
                predictions.append(prediction)
            plot_func(f,
                      title=f'Predicted test samples with epsilon {epsilon}',
                      plot_samples=True,
                      samples=test_samples,
                      labels=predictions)
            print('Ended testing')


    '''
    w_vec = np.arange(0,31)
    print(w_vec)
    in_size = 2
    layers_sizes_dict = {'l1': 4, 'l2': 3, 'l3': 1}

    nn_model = My_FC_NN_Model(in_size=in_size)
    layer_index = 1
    start_offset = 0
    prev_layer_size = in_size
    for i in np.arange(1,len(layers_sizes_dict)+1):
        layer_size = layers_sizes_dict[f'l{layer_index}']
        #print(f"layer size: {layer_size}")
        start_w = start_offset
        #print(f"start_w: {start_w}")
        end_w = start_w + prev_layer_size * layer_size
        #print(f"end_w: {end_w}")
        start_b = end_w
        #print(f"start_b: {start_b}")
        end_b = start_b + layer_size
        #print(f"end_b: {end_b}")
        nn_model.add_layer(
            layer_size=layer_size,
            W=w_vec[start_w:end_w].reshape((prev_layer_size, layer_size)),
            b=w_vec[start_b:end_b].reshape((layer_size,1))
        )
        start_offset = end_b
        prev_layer_size = layer_size
        layer_index += 1
    fwd_outputs, y_estimate = nn_model.fwd([[1], [2]])
    print(f"fwd_outputs: {fwd_outputs}")
    print(f"y_estimate: {y_estimate}")
    fwd_bwd_outputs, gradients_vec, y_estimate = nn_model.fwd_and_backprop([[1], [2]], label=1)
    print(f"fwd_bwd_outputs: {fwd_bwd_outputs}")
    print(f"gradients_vec {gradients_vec}")
    print(f"y_estimate {y_estimate}")
    '''
