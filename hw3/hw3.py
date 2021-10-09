import numpy as np
from rosenbrock import *
from BFGS import *
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator

save_plot_to_file = False

def f(x_1, x_2):
    return x_1 * np.exp(-(x_1 ** 2 + x_2 ** 2))

# calss that contains the architechture properties of the model
class My_FC_NN_Model(object):

    # Constructor
    def __init__(self, in_size, hidden_layers_sizes):
        self.layers_sizes = hidden_layers_sizes[:]
        self.input_size = in_size

    def get_random_params_dict(self):
        params_dict = {}
        for li in np.arange(start=num_of_layers - 1, stop=-1, step=-1):
            layer_size = self.layers_sizes[li]
            if li > 0:
                prev_layer_size = self.layers_sizes[li-1]
            else:
                prev_layer_size = self.input_size
            params_dict[f'l{li}'] = {}
            params_dict[f'l{li}']['W'] = np.random.randn(prev_layer_size, layer_size) / np.sqrt(layer_size)
            params_dict[f'l{li}']['b'] = np.zeros((layer_size, 1))
        return params_dict

# Utils
def params_vec_to_dict(model, params_vec):
    params_dict = {}
    start_offset = 0
    for li in np.arange(start=num_of_layers - 1, stop=-1, step=-1):
        layer_size = model.layers_sizes[li]
        if li > 0:
            prev_layer_size = model.layers_sizes[li - 1]
        else:
            prev_layer_size = model.input_size
        start_w = start_offset
        end_w = start_w + prev_layer_size * layer_size
        start_b = end_w
        end_b = start_b + layer_size
        params_dict[f'l{li}'] = {}
        params_dict[f'l{li}']['W'] = params_vec[start_w:end_w].reshape((prev_layer_size, layer_size))
        params_dict[f'l{li}']['b'] = params_vec[start_b:end_b].reshape((layer_size, 1))
        start_offset = end_b
    return params_dict

    params_dict = {}
    start_offset = 0
    prev_layer_size = model.input_size
    for li, layer_size in enumerate(model.layers_sizes):
        start_w = start_offset
        #print(f"start_w: {start_w}")
        end_w = start_w + prev_layer_size * layer_size
        #print(f"end_w: {end_w}")
        start_b = end_w
        #print(f"start_b: {start_b}")
        end_b = start_b + layer_size
        #print(f"end_b: {end_b}")
        params_dict[f'l{li}'] = {}
        if params_vec is None:
            W = np.random.randn(prev_layer_size, layer_size) / np.sqrt(layer_size)
            b = np.zeros((layer_size, 1))
        else:
            W = params_vec[start_w:end_w].reshape((prev_layer_size, layer_size))
            b = params_vec[start_b:end_b].reshape((layer_size, 1))
        params_dict[f'l{li}']['W'] = W
        params_dict[f'l{li}']['b'] = b
        start_offset = end_b
        prev_layer_size = layer_size
    return params_dict


def params_dict_to_vec(params_dict, num_of_layers):
    params_list = []
    for li in np.arange(start=num_of_layers - 1, stop=-1, step=-1):
        params_list.append(params_dict[f'l{li}']['W'])
        params_list.append(params_dict[f'l{li}']['b'])
    params_vec = np.concatenate(params_list, axis=None)
    return params_vec

def print_params_dict(params_dict, num_of_layers):
    for li in np.arange(start=num_of_layers - 1, stop=-1, step=-1):
        print(f"layer {li}:")
        W = params_dict[f'l{li}']['W']
        print(f"W:\n{W}")
        b = params_dict[f'l{li}']['b']
        print(f"b:\n{b}")

# 1.3.5
def fwd(input_vec, params_dict, num_of_layers):
    layers_outputs = {}
    next_layer_input = input_vec
    for li in np.arange(0,num_of_layers):
        W = params_dict[f'l{li}']['W']
        b = params_dict[f'l{li}']['b']
        y = W.T @ next_layer_input + b
        layers_outputs[f'l{li}_lin'] = y
        next_layer_input = phi(y)
        layers_outputs[f'l{li}'] = next_layer_input
    return y, layers_outputs

# 1.3.6
def phi(x):
    return np.tanh(x)

def g_phi(x):
    return 1 - (np.tanh(x) ** 2)

# 1.3.7
def loss(label, prediction):
    return (prediction - label)**2

def g_loss(label, prediction):
    return 2 * (prediction - label)

def batch_fwd(inputs, params_dict, num_of_layers):
    predictions = []
    for input in inputs:
        input_vec = np.expand_dims(input, axis=1)
        prediction, _ = fwd(input_vec, params_dict, num_of_layers)
        predictions.append(prediction)
    return predictions

def batch_loss(inputs, labels, params_dict, num_of_layers):
    sum_of_losses = 0
    for input, label in zip(inputs, labels):
        input_vec = np.expand_dims(input, axis=1)
        prediction, _ = fwd(input_vec, params_dict, num_of_layers)
        sum_of_losses += loss(label, prediction)
    average_loss = sum_of_losses / len(labels)
    return average_loss

# 1.3.8
def fwd_and_backprop(input_vec, label, params_dict, num_of_layers):
    prediction, layers_outputs = fwd(input_vec, params_dict, num_of_layers)
    grad_z = g_loss(label, prediction)
    grads_dict = {}
    for li in np.arange(start=num_of_layers-1, stop=-1, step=-1):
        if li > 0:
            x = layers_outputs[f'l{li-1}']
        else:
            x = input_vec
        W = params_dict[f'l{li}']['W']
        y = layers_outputs[f'l{li}_lin']
        grads_dict[f'l{li}'] = {}
        # if we're not in the last layer then we have an activation function
        # and thus it's gradient needs to be considered
        if li < num_of_layers-1:
            grad_act = g_phi(y)
            phi_grad_diag = np.diagflat(grad_act)
        else:
            phi_grad_diag = np.expand_dims(1, axis=(0, 1))
        grads_dict[f'l{li}']['x'] = W @ phi_grad_diag @ grad_z
        grads_dict[f'l{li}']['W'] = x @ grad_z.T @ phi_grad_diag
        grads_dict[f'l{li}']['b'] = phi_grad_diag @ grad_z
        grad_z = grads_dict[f'l{li}']['x']
    return prediction, layers_outputs, grads_dict

# 1.3.9
def batch_fwd_and_backprop(inputs, labels, params_dict, num_of_layers):
    n = len(labels)
    params_vec = params_dict_to_vec(params_dict, num_of_layers)
    grads_sum = np.zeros_like(params_vec)
    for input, label in zip(inputs, labels):
        input = np.expand_dims(input, axis=1)
        _, _, grads_dict = fwd_and_backprop(input, label, params_dict, num_of_layers)
        grads_vec = params_dict_to_vec(grads_dict, num_of_layers)
        grads_sum += grads_vec
    return grads_sum / n

# 1.3.10-11 - generating data samples for training and testing
def gen_samples(func_ptr, n):
    labels = []
    samples = 4*np.random.rand(n, 2)-2
    for sample in samples:
        labels.append(func_ptr(sample[0], sample[1]))
    return samples, np.array(labels)

def inexact_line_search(w_k, d, model, inputs, labels):
    alpha_0 = 1
    beta = 0.5
    sigma = 0.25
    c_2 = 0.9
    alpha = alpha_0
    num_of_layers = len(model.layers_sizes)
    w_k_dict = params_vec_to_dict(model, w_k)
    f_w_k = batch_loss(inputs, labels, w_k_dict, num_of_layers)
    grad_w_k = batch_fwd_and_backprop(inputs, labels, w_k_dict, num_of_layers)

    while True:
        w_k_1 = w_k + alpha * d
        w_k_1_dict = params_vec_to_dict(model, w_k_1)
        f_w_k_1 = batch_loss(inputs, labels, w_k_1_dict, num_of_layers)
        grad_w_k_1 = batch_fwd_and_backprop(inputs, labels, w_k_1_dict, num_of_layers)
        decrease_cond =(f_w_k_1 <= f_w_k + sigma * alpha * d.T @ grad_w_k)
        #curvature_cond = (grad_w_k_1.T @ d >= c_2 * grad_w_k.T @ d)
        curvature_cond = (np.abs(grad_w_k_1.T @ d) <= c_2 * np.abs(grad_w_k.T @ d))
        if decrease_cond and curvature_cond:
             break
        alpha = beta * alpha
    return alpha

def NN_BFGS(w_0, model, epsilon, inputs, labels):

    w_k = w_0
    w_k_dict = params_vec_to_dict(model, w_k)
    num_of_layers = len(model.layers_sizes)
    g_w_k = batch_fwd_and_backprop(inputs, labels, w_k_dict, num_of_layers)
    g_w_k_norm = np.linalg.norm(g_w_k)

    n = len(w_k)
    I = np.identity(n)
    B_k = np.identity(n)

    w_k_list = []
    losses_list = []
    iteration_idx = 0

    while g_w_k_norm > epsilon:
        iteration_idx += 1
        w_k_dict = params_vec_to_dict(model, w_k)
        loss = batch_loss(inputs, labels, w_k_dict, num_of_layers)
        loss = np.squeeze(loss)
        print(f'it={iteration_idx} grad_norm={g_w_k_norm} loss={loss}')
        losses_list.append(loss)
        w_k_list.append(w_k)
        d = - B_k @ g_w_k
        alpha = inexact_line_search(w_k, d, model, inputs, labels)
        w_k_next = w_k + alpha * d
        w_k_next_dict = params_vec_to_dict(model, w_k_next)
        g_w_k_next = batch_fwd_and_backprop(inputs, labels, w_k_next_dict, num_of_layers)
        s_k = w_k_next - w_k
        y_k = g_w_k_next - g_w_k

        s_k = s_k.reshape(n,1)
        y_k = y_k.reshape(n,1)

        curve_factor = (y_k.T @ s_k)
        A = (s_k @ y_k.T) / curve_factor
        B = (y_k @ s_k.T) / curve_factor
        C = (s_k @ s_k.T) / curve_factor
        B_k = ((I - A ) @ B_k @ (I - B)) +  C

        w_k = w_k_next
        g_w_k = g_w_k_next
        g_w_k_norm = np.linalg.norm(g_w_k)

    print(f'Optimization took: {iteration_idx} iterations')
    return w_k, losses_list, w_k_list

# 1.3.13
def plot_func(func_ptr, title="", plot_samples=False, samples=None, predictions=None, plot_filename=""):

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    # Make data.
    X_1 = np.arange(-2, 2.02, 0.2)
    X_2 = np.arange(-2, 2.02, 0.2)
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
        for sample, prediction in zip(samples, predictions):
            x_1, x_2 = sample
            ax.scatter(x_1, x_2, prediction, marker='o', color='green', s=5)

    ax.set_xlabel(r'$x_1$')
    ax.set_xticks(np.arange(-2, 2, 1))
    ax.set_ylabel(r'$x_2$')
    ax.set_yticks(np.arange(-2, 2, 1))
    ax.set_zlabel(r'$f(x_1,x_2)$')
    ax.set_zticks(np.arange(-2, 2, 0.2))
    ax.set_zlim(-0.42, 0.42)
    ax.view_init(azim=210,elev=15)

    plt.suptitle(title)

    plt.show()

    if save_plot_to_file and plot_filename != "":
        print(f"saving plot to file {plot_filename}")
        fig.savefig(plot_filename)

def plot_model_func_approximation(fwd_func_ptr, model, params_vec, title="", plot_filename=""):

    params_dict = params_vec_to_dict(model, params_vec)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    # Make data.
    X_1 = np.arange(-2, 2.02, 0.2)
    X_2 = np.arange(-2, 2.02, 0.2)
    X_1, X_2 = np.meshgrid(X_1, X_2)
    Z_shape = np.shape(X_1)
    Z = np.zeros((Z_shape[0] * Z_shape[1],1))
    for i, t in enumerate(zip(X_1.ravel(), X_2.ravel())):
        x = np.array([[t[0]],[t[1]]])
        approx_value, _ = fwd_func_ptr(x, params_dict, len(model.layers_sizes))
        Z[i] = approx_value
    Z = Z.reshape(Z_shape)

    # Plot the surface.
    surf = ax.plot_surface(X_1, X_2, Z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False, alpha=0.5)

    # Customize the z axis.
    ax.zaxis.set_major_locator(LinearLocator(10))
    # A StrMethodFormatter is used automatically
    ax.zaxis.set_major_formatter('{x:.02f}')

    # Add a color bar which maps values to colors.
    #fig.colorbar(surf, shrink=0.5, aspect=5, location='left')

    ax.set_xlabel(r'$x_1$')
    ax.set_xticks(np.arange(-2, 2, 1))
    ax.set_ylabel(r'$x_2$')
    ax.set_yticks(np.arange(-2, 2, 1))
    ax.set_zlabel(r'$f(x_1,x_2)$')
    ax.set_zticks(np.arange(-2, 2, 0.2))
    ax.set_zlim(-0.42, 0.42)
    ax.view_init(azim=210,elev=15)

    plt.suptitle(title)

    plt.show()

    if save_plot_to_file and plot_filename != "":
        print(f"saving plot to file {plot_filename}")
        fig.savefig(plot_filename)

if __name__ == '__main__':

    # for getting the reports' plots (need also to disable the rosenbrock BFGS run)
    #np.random.seed(10)

    run_rosenbrock_BFGS = True
    if run_rosenbrock_BFGS:
        x_0 = np.ones((10, 1))
        print(f"rosenbrock func minimum at all ones vector: {rosenbrock_func(x_0)}")

        x_0 = np.zeros((10, 1))

        trajectory_points = BFGS(x_0, rosenbrock_func, rosenbrock_grad)
        title = f'Rosenbrock optimization with BFGS\n'
        title += r'$log{(f(x_k)-f(x^*))}$ vs number of iterations'
        rosenbrock_plot(trajectory_points, title)

    train_samples, train_labels = gen_samples(f, 500)
    test_samples, test_labels = gen_samples(f, 200)
    in_size = 2
    hidden_layers_sizes = (4, 3, 1)
    nn_model = My_FC_NN_Model(in_size, hidden_layers_sizes)
    num_of_layers = len(nn_model.layers_sizes)

    run_func_approximation = True
    if run_func_approximation:

        epsilons = [1e-1, 1e-2, 1e-3, 1e-4]

        for i, epsilon in enumerate(epsilons):

            print(f'Optimizing model with epsilon = {epsilon}')
            w_0_dict = nn_model.get_random_params_dict()
            w_0_vec = params_dict_to_vec(w_0_dict, num_of_layers)
            print(f'Start training')
            opt_params_vec, values_list, points_list =\
                NN_BFGS(w_0_vec, nn_model, epsilon, train_samples, train_labels)
            print(f'Ended training')
            print(f'Training loss = {values_list[-1]}')
            opt_params_dict = params_vec_to_dict(nn_model, opt_params_vec)
            predictions = batch_fwd(test_samples, opt_params_dict, num_of_layers)

            print(f'Plot function approximation')
            plot_model_func_approximation(
                fwd_func_ptr=fwd,
                model=nn_model,
                params_vec=opt_params_vec,
                title=f'F(x,W*) when epsilon={epsilon}',
                plot_filename=f'./plots/model_approx_epsilon_{i}.png')

            print(f'Plot predictions')
            plot_func(
                func_ptr=f,
                title=f'Predicted test samples over reference function when epsilon {epsilon}',
                plot_samples=True,
                samples=test_samples,
                predictions=predictions,
                plot_filename=f'./plots/model_test_over_ref_func_{i}.png')

