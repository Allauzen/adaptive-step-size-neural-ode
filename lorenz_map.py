# Author: Alexandre Allauzen
# License: MIT
# Starting code thanks to Alessandro Bucci

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


class LorenzMap:
    """
    LorenzMap in 3D (x,y,z)
    Default values:
    sigma=10, rho=28, and beta=8/3
    delta_t = 1e-3

    The main method is full_traj which generates the trajectory
    """

    def __init__(self, sigma=10, rho=28, beta=8 / 3, delta_t=1e-3):
        self.sigma, self.rho, self.beta = sigma, rho, beta
        self.delta_t = delta_t

    def v_eq(self, t=None, v=None):
        x, y, z = v[0], v[1], v[2]
        dot_x = self.sigma * (-x + y)
        dot_y = x * (self.rho - z) - y
        dot_z = x * y - self.beta * z
        return np.array([dot_x, dot_y, dot_z])

    def euler_step(self, v):
        return v + self.v_eq(v=v) * self.delta_t

    def jacobian(self, v):
        x, y, z = v[0], v[1], v[2]
        res = np.array([[-self.sigma, self.sigma, 0], [self.rho - z, -1, -x],
                        [y, x, -self.beta]])
        return res

    def full_traj(self, nb_steps, init_pos):
        """
        Generate the trajectory.
        - nb_steps : the number of steps of generation
        - init pos : the starting point, it's a numpy array in 3D

        Returns: a numpy array of size (nb_steps,3)
        Generate "nb_steps" points.
        The associated timeline starts at 0 and ends at nb_steps*delta_t

        For example if nb_steps=1000 and delta_t=1e-3, the trajetory ends at
        time = 1. If you want to generate 1000 points from t=0 to 20, you
        should use delta_t = 20/1000 =  2e-2
        """
        t = np.linspace(0, nb_steps * self.delta_t, nb_steps)
        f = solve_ivp(self.v_eq, [0, nb_steps * self.delta_t],
                      init_pos,
                      method='RK45',
                      t_eval=t)
        # f.y contains a tensor with the time as the last dim. Switch the axis:
        return np.moveaxis(f.y, -1, 0)


def plot_traj(data, filename=None, marker=None, start=False):
    """Plot the result of the simulation.
    -If the filename is set to None
    (default), the figure is not saved and you can see it.
    -  data is a numpy array of size (nb_steps,3).
    If you want to plot only the 100 first points:
    plot_traj(res[:100])
    """
    x, y, z = data[:, 0], data[:, 1], data[:, 2]
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot(x, y, z, lw=0.5, marker=marker)
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")
    if start:
        ax.plot(x[0], y[0], z[0], 'ro')
    if filename is not None:
        plt.savefig(filename, format="pdf")
    else:
        plt.show()


def scatterGraph(x, size=(12, 10), mark_downsampling=1):
    """
    Plot the result of the Lorenz simulation in 3D
     - x is a numpy array of size (nb_steps,3).
     - size allows you to set the size of the figure
     - mark_downsampling: every "mark_downsampling" points, we add a red cross
    """
    _ = plt.figure(figsize=size)
    ax = plt.axes(projection='3d')
    ax.plot3D(x[:, 0], x[:, 1], x[:, 2], 'blue')
    ax.scatter3D(x[::mark_downsampling, 0],
                 x[::mark_downsampling, 1],
                 x[::mark_downsampling, 2],
                 color='red')


def simpleTest():
    gen = LorenzMap(delta_t=1e-2)
    res = gen.full_traj(init_pos=np.ones(3) * 0.01, nb_steps=10000)
    plot_traj(res[:100], marker=".", start=True)
    scatterGraph(res[:200], mark_downsampling=10)
    plt.show()


def main():
    simpleTest()


if __name__ == "__main__":
    main()
