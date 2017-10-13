'''
This is a framework to check the impact of Ito and Stratonovich integration on the 
pure 1D-diffusive process of an inert scalar by using the stochastic differential equation.
 
@author: M. Hansinger 

'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve

#set parameters

npoints = 1000
grid = np.linspace(0,1,npoints)


dx = 1/npoints
D = 0.0005
Dt = 0.00001

# time step
dt = 0.1
tsteps = 10
fields = 8

Phi_0 = gaussianDist(grid, grid[int(npoints/2)], 0.05)

# initialize the fields
Phi_fields_old = np.zeros((npoints,fields))
Phi_fields_new = np.zeros((npoints,fields))
for i in range(fields):
    Phi_fields_old[:,i] = Phi_0
    Phi_fields_new[:,i] = Phi_0


def setDiffMatrix(D,Dt,npoints,dt,dx):
    # set up coefficient matrix for pure diffusion
    A = np.zeros([npoints, npoints])

    east = (D+Dt) / dx ** 2 * dt
    middle = -2 * (D+Dt) / dx ** 2 * dt
    west = (D+Dt) / dx ** 2 * dt

    for i in range(1, npoints - 1):
        A[i][i - 1] = west
        A[i][i] = middle
        A[i][i + 1] = east

    #INLET:
    A[0][0] = middle
    A[0][1] = west

    #OUTLET
    A[npoints-1][npoints-1] = middle
    A[npoints-1][npoints-2] = east

    return A


def setStochMatrix(D,Dt,npoints,dt,dx,dW,fields):
    # set up coefficient matrix for the diffusion with stochastic fields
    # the matrix has the size fields x npoints x npoints

    # span for each field a different coefficient Matrix, as dW always differs
    A = np.zeros([fields, npoints, npoints])

    east = (D+Dt) / dx ** 2 * dt + 1/(2*dx)*np.sqrt(2*Dt)* dW
    middle = -2 * (D+Dt) / dx ** 2 * dt
    west = (D+Dt) / dx ** 2 * dt -1/(2*dx)*np.sqrt(2*Dt) *dW

    # loop over the fields
    for j in range(fields):
        #print(j)
        for i in range(1, npoints - 1):
            A[j][i][i - 1] = west[j]
            A[j][i][i] = middle
            A[j][i][i + 1] = east[j]

        # INLET:
        A[j][0][0] = middle
        A[j][0][1] = west[j]

        # OUTLET
        A[j][npoints - 1][npoints - 1] = middle
        A[j][npoints - 1][npoints - 2] = east[j]

    return A


# 1D diffusion equation
def pureDiffusion(Phi_0,A):
    Phi = np.matmul(Phi_0,A) + Phi_0
    return Phi

# Advance 1 time step of stochastic diffusion with Euler-Maruyama
def stochasticTimestep(Phi_fields_old, A_stoch, fields, T_eddy, npoints, tsteps=1000):
    #dW = dWiener(dt,fields)
    Phi_new = np.zeros((npoints, fields))
    Phi_mean = Phi_fields_old.sum(axis=1)*(1/fields)
    IEM_term = IEM(Phi_fields_old, T_eddy, Phi_mean)

    for i in range(fields):
        Phi_new[:,i] = np.matmul(Phi_fields_old[:,i],A_stoch[i,:,:])  # check different lin Algebra methods!
        Phi_new[:,i] += Phi_fields_old[:,i] + IEM_term[:,i]

    return Phi_new

# main routine
def computeStochDiffusion(Phi_fields_old,D,Dt,dt,fields,T_eddy,npoints,tsteps=1000):

    for i in range(tsteps):
        # compute wiener vector for number of fields
        dW = dWiener(dt, fields)
        # set up the coefficient matrix; size = npoints x fields
        A_stoch = setStochMatrix(D, Dt, npoints, dt, dx, dW, fields)

        Phi_fields_new = stochasticTimestep(Phi_fields_old, A_stoch, fields, T_eddy, npoints)

        Phi_mean = Phi_fields_old.sum(axis=1)*(1/fields)

        Phi_fields_old = np.copy(Phi_fields_new)

        plt.figure(1)
        plt.plot(np.linspace(0,1,npoints),Phi_mean)
        plt.figure(2)
        plt.plot(np.linspace(0,1,npoints),Phi_fields_old[:,2])
        plt.plot(np.linspace(0,1, npoints), Phi_fields_old[:, 6])

    return Phi_fields_new, Phi_mean


def advanceDiffusion(Phi_0,A,tsteps=1000):
    for i in range(1,tsteps):
        if i ==1:
            Phi_old = pureDiffusion(Phi_0,A)
        else:
            Phi_new = pureDiffusion(Phi_old,A)
            Phi_old = Phi_new

       # plt.plot(Phi_old)

    return Phi_old


#def analyticalSolution(Phi_0,A,D,grid)


# Gaussian function for initialization
def gaussianDist(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))


def dWiener(dt,fields):
    # compute the Wiener Term
    #initialize gamma vector
    gamma = np.ones(fields)
    gamma[0:int(fields/2)] = -1
    #shuffle it
    np.random.shuffle(gamma)
    dW = gamma * np.sqrt(dt)

    return dW


def IEM(Phi_fields_old,T_eddy, Phi_mean):
    fields = Phi_fields_old.shape[1]
    IEM = np.zeros((len(Phi_mean),fields))

    for i in range(fields):
        IEM[:,i] = 1/T_eddy * (Phi_fields_old[:,i] -Phi_mean[:])

    return IEM

def comparePlots():
    plt.figure(1)
    plt.plot(grid,Phi_diff)
    plt.plot(grid,Phi_mean)
    plt.grid()
    plt.legend(['pure Diffusion','Stochastic Fields'])
    plt.title('Comparison')
    #plt.show()
    plt.figure(2)
    plt.plot(grid,Phi_fields)
    plt.title('Phi der 8 Felder')
    plt.grid()
    plt.show()


def stratHeun(f, G, y0, tspan, dW=None):
    """Use the Stratonovich Heun algorithm to integrate Stratonovich equation
    dy = f(y,t)dt + G(y,t) \circ dW(t)

    where y is the d-dimensional state vector, f is a vector-valued function,
    G is an d x m matrix-valued function giving the noise coefficients and
    dW(t) = (dW_1, dW_2, ... dW_m) is a vector of independent Wiener increments

    Args:
      f: callable(y, t) returning (d,) array
         Vector-valued function to define the deterministic part of the system
      G: callable(y, t) returning (d,m) array
         Matrix-valued function to define the noise coefficients of the system
      y0: array of shape (d,) giving the initial state vector y(t==0)
      tspan (array): The sequence of time points for which to solve for y.
        These must be equally spaced, e.g. np.arange(0,10,0.005)
        tspan[0] is the intial time corresponding to the initial state y0.
      dW: optional array of shape (len(tspan)-1, d). This is for advanced use,
        if you want to use a specific realization of the d independent Wiener
        processes. If not provided Wiener increments will be generated randomly

    Returns:
      y: array, with shape (len(tspan), len(y0))
         With the initial value y0 in the first row

    Raises:
      SDEValueError

    See also:
      W. Rumelin (1982) Numerical Treatment of Stochastic Differential
         Equations
      R. Mannella (2002) Integration of Stochastic Differential Equations
         on a Computer
      K. Burrage, P. M. Burrage and T. Tian (2004) Numerical methods for strong
         solutions of stochastic differential equations: an overview
    """
    (d, m, f, G, y0, tspan, dW, __) = _check_args(f, G, y0, tspan, dW, None)
    N = len(tspan)
    h = (tspan[N-1] - tspan[0])/(N - 1)
    # allocate space for result
    y = np.zeros((N, d), dtype=type(y0[0]))
    if dW is None:
        # pre-generate Wiener increments (for m independent Wiener processes):
        dW = deltaW(N - 1, m, h)
    y[0] = y0;
    for n in range(0, N-1):
        tn = tspan[n]
        tnp1 = tspan[n+1]
        yn = y[n]
        dWn = dW[n,:]
        fn = f(yn, tn)
        Gn = G(yn, tn)
        ybar = yn + fn*h + Gn.dot(dWn)
        fnbar = f(ybar, tnp1)
        Gnbar = G(ybar, tnp1)
        y[n+1] = yn + 0.5*(fn + fnbar)*h + 0.5*(Gn + Gnbar).dot(dWn)
    return y









