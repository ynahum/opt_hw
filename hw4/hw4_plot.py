import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from matplotlib import patheffects



if __name__ == '__main__':

    fig, ax = plt.subplots(figsize=(6, 6))

    nx_1 = 201
    nx_2 = 201

    # Set up survey vectors
    x1_vec = np.linspace(-3, 8, nx_1)
    x2_vec = np.linspace(-3, 5, nx_2)

    # Set up survey matrices.  Design disk loading and gear ratio.
    x1, x2 = np.meshgrid(x1_vec, x2_vec)

    # Evaluate some stuff to plot
    obj = 2*((x1-5) ** 2) + (x2-1) ** 2
    g1 = 0.5*x1 + x2 - 1
    g2 = x1 - x2
    g3 = -x1 - x2

    cntr = ax.contour(x1, x2, obj, [0.5, 1, 2, 4, 8, 16, 32, 64],
                      colors='blue')
    ax.clabel(cntr, fmt="%2.1f", use_clabeltext=True)

    cg1 = ax.contour(x1, x2, g1, [0], colors='sandybrown')
    plt.setp(cg1.collections,
             path_effects=[patheffects.withTickedStroke(angle=135)])

    cg2 = ax.contour(x1, x2, g2, [0], colors='orangered')
    plt.setp(cg2.collections,
             path_effects=[patheffects.withTickedStroke(angle=60, length=2)])

    cg3 = ax.contour(x1, x2, g3, [0], colors='mediumblue')
    plt.setp(cg3.collections,
             path_effects=[patheffects.withTickedStroke(spacing=7)])

    ax.set_xlim(-3, 8)
    ax.set_ylim(-3, 5)

    plt.show()
