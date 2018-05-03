# parameter file

import numpy as np

class params():
    '''This is the parameter class where you specify all the necessary parameters'''
    # time step
    dt = 0.001
    #grid points
    npoints = 1000
    #grid is computed
    grid = np.linspace(0, 1, npoints)
    # spatial discretization
    dx = 1 / npoints
    # laminar diffusion
    D = 0.001
    # turbulent diffusion
    Dt = D * 3
    # number of stochastic fields
    fields = 8
    # time steps to compute
    tsteps = 2000