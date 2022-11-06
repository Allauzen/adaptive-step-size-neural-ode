# Author: Alexandre Allauzen
# Licence: MIT

import torch as th
import torch.nn as nn
import numpy as np


######################################################################
class ODENet(nn.Module):
    """ A simple Neural ODE implementation:
    We define the Neural ODE class (ODENet) and
    its methods needed for learning:
    - This is a simple wrapper that relies on a external solver
    (see examples of solvers above).
    - Note that here the model is a simple feed forward model with
    2 hidden layers, and $h$ is the hidden size.
    """
    def __init__(self, dim, h=10):
        super(ODENet, self).__init__()
        # The NNet
        self.nn = nn.Sequential(
            nn.Linear(dim, h),
            nn.ReLU(),
            nn.Linear(h, h),
            nn.ReLU(),
            nn.Linear(h, dim),
        )

    def forward(self, z):
        """
        Compute the function f which is the time derivative of the quantity
        under study.
        - It does not depends on time
        - just on the state z: compute the time derivative at z.
        """
        return self.nn(z)


######################################################################
# Solvers
######################################################################
def euler_solve(x0, t0, t1, f, n_steps=1):
    """
    Simplest Euler ODE initial value solver
    """
    h = (t1 - t0)/n_steps
    tn = t0
    yn = x0
    for i_step in range(n_steps):
        a1 = yn + h*f(yn)
        tn = tn + h
    return a1


def euler_step(x0, stepsize, f):
    return x0 + stepsize*f(x0)


def ode_solve_2steps(x0, t0, t1, f, n_steps=1):
    """
      Euler-2step ODE solver
    """
    global hp
    hp = (t1 - t0)/n_steps
    tn = t0
    yn = x0
    for i_step in range(n_steps):
        a2 = yn + (hp/2)*f(yn) + (hp/2)*f(yn + hp/2)
        tn = tn + hp/2
    return a2


def improved_euler_solve(x0, t0, t1, f, n_steps=1):
    """
      Improved Euler forward solver
    """
    hp = (t1 - t0)/n_steps
    yn = x0
    for i_step in range(n_steps):
        f1 = f(yn)
        f2 = f(yn + hp*f1)
        f3 = f(yn + (hp/4)*(f1+f2))
        A2 = yn + (hp/6)*(f1+f2+4*f3)
        yn = A2
    return yn


def fehlberg_batch_inference(f, batch, h=1, S=0.9, eps=0.1, numsteps=1):
    """
    Fehlberg's method: for each input value, compute the useful quantities
    for integration.
    - The model f of dx/dt
    - The dimensions of the return objects A1 and A2: N timesteps * d
    just like the input
    - r and hprime are  of dimension N
    """
    inputs = batch
    for _ in range(numsteps):
        f1 = f(inputs)
        f2 = f(inputs + h*f1)
        f3 = f(inputs + (h/4)*(f1+f2))
        A1 = inputs + (h/2)*(f1+f2)
        A2 = inputs + (h/6)*(f1+f2+4*f3)
        inputs = A2
    r = th.abs(A1-A2).sum(dim=-1)/h
    hprime = h*S*(eps/r)**(0.5)
    return A1, A2, r, hprime


def rk3(f, batch, h=1, numsteps=1):
    """
     Basic RK3 integration scheme.
    """
    inputs = batch
    for _ in range(numsteps):
        f1 = f(inputs)
        f2 = f(inputs + h*f1)
        f3 = f(inputs + (h/4)*(f1+f2))
        outputs = inputs + (h/6)*(f1+f2+4*f3)
        inputs = outputs
    return outputs


def fehlberg_solver(x0, f, S=0.9, eps=0.1, max_step=10):
    """
    Wrap the Fehlberg strategy using the useful quantities:
    For each input points: compute the next one with f as a model of dx/dt
    - eps: is the tolerance on the estimated of truncation error
    - S  : the safety factor
    """
    newstep = 1
    a1, a2, r, hp = fehlberg_batch_inference(f, x0, h=1,
                                             S=S, eps=eps,
                                             numsteps=1)
    r = th.abs(a1-a2).sum(dim=-1)
    if r > eps:
        newstep = int(np.ceil(1/max(hp.item(), 1/max_step)))
        a1, a2, r, hp = fehlberg_batch_inference(f, x0, h=1/newstep,
                                                 S=S, eps=eps,
                                                 numsteps=newstep)
    return a2, newstep


def generate_fehlberg(x0, f, Ntimes=1, S=0.9, eps=0.1, max_step=10):
    """
    Generate Ntimes-1 points from x0 given f
    The step size is adapted with the Fehlberg strategy
    """
    assert(Ntimes > 1)
    d = x0.shape[-1]
    x0 = x0.unsqueeze(0)
    xs = th.zeros(Ntimes, d).to(x0)
    ns = th.ones(Ntimes, 1).to(x0)
    xs[0] = x0
    for i_t in range(Ntimes - 1):
        xs[i_t+1], ns[i_t+1] = fehlberg_solver(xs[i_t], f, S, eps, max_step)
    return xs, ns


def generate_rk3(x0, f, Ntimes=2, nsteps=1):
    """
    Generate Ntimes-1 points from x0 given f
    Unfold the rk3 formula
    """
    assert(Ntimes > 1)
    d = x0.shape[-1]
    x0 = x0.unsqueeze(0)
    xs = th.zeros(Ntimes, d).to(x0)
    xs[0] = x0
    for i_t in range(Ntimes - 1):
        xs[i_t+1] = rk3(f, xs[i_t], h=1/nsteps, numsteps=nsteps)
    return xs
