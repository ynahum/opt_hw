import numpy as np
import matplotlib.pyplot as plt

def plot_search_method_graphs(Q,trajectory_points,title='', num_of_contours=30):

    traj_x = trajectory_points[:,0]
    traj_y = trajectory_points[:,1]
    x_vec_start = traj_x[:-1]
    y_vec_start = traj_y[:-1]
    x_vec_offset = traj_x[1:] - x_vec_start
    y_vec_offset = traj_y[1:] - y_vec_start

    x = np.linspace(-2, 2, 100)
    y = np.linspace(-2, 2, 100)
    X, Y = np.meshgrid(x, y)
    F = (X ** 2) * (Q[0, 0]) + 2 * X * Y * (Q[0, 1]) + (Y ** 2) * (Q[1, 1])

    fig = plt.figure(figsize=plt.figaspect(0.5))
    fig.suptitle(title)

    ax = fig.add_subplot(1, 2, 1)
    ax.contour(X, Y, F, num_of_contours, cmap="coolwarm")
    #ax.scatter(traj_x, traj_y, s=7)
    ax.quiver(x_vec_start, y_vec_start, x_vec_offset, y_vec_offset,
              color='red', angles='xy', scale_units='xy', scale=1)

    ax = fig.add_subplot(1, 2, 2, projection='3d')
    ax.contour(X, Y, F, num_of_contours, cmap="coolwarm", offset=0)
    #ax.scatter(traj_x, traj_y, s=7)
    ax.plot_surface(X, Y, F, cmap="coolwarm", rstride=1, cstride=1)
    z_vec_start = np.zeros(np.shape(x_vec_start))
    z_vec_offset = np.zeros(np.shape(x_vec_offset))
    ax.quiver(x_vec_start, y_vec_start, z_vec_start, x_vec_offset, y_vec_offset, z_vec_offset, color='red')

    plt.show()
