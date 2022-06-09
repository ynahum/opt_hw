import matplotlib.pyplot as plt

def traj_plot(trajectory_points, f_func, title=''):

    f = [ f_func(x) for x in trajectory_points]
    #f = [ np.log10(rosenbrock_func(x)) for x in trajectory_points]
    fig = plt.figure()
    fig.suptitle(title)
    plt.yscale("log")
    plt.xlabel(r'Number of iterations (tolerance on gradient norm = $10^{-5}$)')
    plt.ylabel(r'$log{(f(x_k)-f(x^*))}$')
    plt.plot(f, 'k')
    plt.show()
