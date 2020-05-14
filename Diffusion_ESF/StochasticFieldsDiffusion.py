'''
This is a framework to check the impact of Ito and Stratonovich integration on the 
pure 1D-diffusive process of an inert scalar by using the stochastic differential equation.

@author: M. Hansinger

#######################################
THIS is an old version, which is incorrect!
#######################################

'''

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as sc


def setDiffMatrix(D, Dt, npoints, dt, dx):
    # set up coefficient matrix for pure diffusion
    A = np.zeros([npoints, npoints])

    east = (D + Dt) / dx ** 2 * dt
    middle = -2 * (D + Dt) / dx ** 2 * dt
    west = (D + Dt) / dx ** 2 * dt

    for i in range(1, npoints - 1):
        A[i][i - 1] = west
        A[i][i] = middle
        A[i][i + 1] = east

    # INLET:
    A[0][0] = middle
    A[0][1] = west

    # OUTLET
    A[npoints - 1][npoints - 1] = middle
    A[npoints - 1][npoints - 2] = east

    return A


def setStochMatrix(D, Dt, npoints, dt, dx, dW, fields):
    # set up coefficient matrix for the diffusion with stochastic fields
    # the matrix has the size fields x npoints x npoints

    # span for each field a different coefficient Matrix, as dW always differs
    A = np.zeros([fields, npoints, npoints])

    east = (D + Dt) / dx ** 2 * dt + 1 / (2 * dx) * np.sqrt(2 * Dt) * dW
    middle = -2 * (D + Dt) / dx ** 2 * dt
    west = (D + Dt) / dx ** 2 * dt - 1 / (2 * dx) * np.sqrt(2 * Dt) * dW

    # loop over the fields
    for j in range(fields):
        # print(j)
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
def pureDiffusion(Phi_0, A):
    Phi = np.matmul(Phi_0, A) + Phi_0
    return Phi


# Advance 1 time step of stochastic diffusion with Euler-Maruyama
def stochasticTimestep(Phi_fields_old, A_stoch, fields, T_eddy, npoints):
    # dW = dWiener(dt,fields)
    Phi_new = np.zeros((npoints, fields))
    Phi_mean = Phi_fields_old.sum(axis=1) * (1 / fields)
    IEM_term = IEM(Phi_fields_old, T_eddy, Phi_mean)

    for i in range(fields):
        Phi_new[:, i] = np.matmul(Phi_fields_old[:, i], A_stoch[i, :, :])  # check different lin Algebra methods!
        Phi_new[:, i] += Phi_fields_old[:, i] + IEM_term[:,i]
        # hier noch die Stochastic Geschwindigkeit rein und dafür rausnehmen bei der A Matrix

    return Phi_new


# main routine
def computeStochDiffusion(Phi_fields_old, D, Dt, dt, fields, T_eddy, npoints, tsteps=1000):
    for i in range(tsteps):
        # compute wiener vector for number of fields
        dW = dWiener(dt, fields)
        # set up the coefficient matrix; size = npoints x fields
        A_stoch = setStochMatrix(D, Dt, npoints, dt, dx, dW, fields)

        Phi_fields_new = stochasticTimestep(Phi_fields_old, A_stoch, fields, T_eddy, npoints)

        Phi_mean = Phi_fields_old.sum(axis=1) * (1 / fields)

        Phi_fields_old = np.copy(Phi_fields_new)

        #plt.figure(1)
        #plt.plot(np.linspace(0, 1, npoints), Phi_mean)
        #plt.figure(2)
        #plt.plot(np.linspace(0, 1, npoints), Phi_fields_old[:, 2])
        #plt.plot(np.linspace(0, 1, npoints), Phi_fields_old[:, 6])

    return Phi_fields_new, Phi_mean


def advanceDiffusion(Phi_0, A, tsteps=1000):
    # this is the pure diffusion process in implicit formulation
    for i in range(1, tsteps):
        if i == 1:
            Phi_old = pureDiffusion(Phi_0, A)
        else:
            Phi_new = pureDiffusion(Phi_old, A)
            Phi_old = Phi_new

            # plt.plot(Phi_old)

    return Phi_old


# def analyticalSolution(Phi_0,A,D,grid)

# Gaussian function for initialization
def gaussianDist(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))


def dWiener(dt, fields):
    # compute the Wiener Term
    # initialize gamma vector
    gamma = np.ones(fields)
    gamma[0:int(fields / 2)] = -1
    # shuffle it
    np.random.shuffle(gamma)
    dW = gamma * np.sqrt(dt)

    return dW


def IEM(Phi_fields_old, T_eddy, Phi_mean):
    fields = Phi_fields_old.shape[1]
    IEM = np.zeros((len(Phi_mean), fields))

    for i in range(fields):
        IEM[:, i] = 1 / T_eddy * (Phi_fields_old[:, i] - Phi_mean[:])

    return IEM


def comparePlots():
    plt.figure(1)
    #plt.plot(grid, Phi_diff)
    plt.plot(grid, Phi_mean)
    plt.grid()
    plt.legend(['pure Diffusion', 'Stochastic Fields'])
    plt.title('Comparison')
    # plt.show()
    plt.figure(2)
    plt.plot(grid, Phi_fields)
    plt.title('Phi der 8 Felder')
    plt.grid()
    plt.show(block=False)

########################
# MAIN PART
########################


npoints = 100
grid = np.linspace(0, 1, npoints)

dx = 1 / npoints
D = 0.000001
Dt = D*2

# time step
dt = 0.00001
tsteps = 100
fields = 8
Phi_0 = gaussianDist(grid, grid[int(npoints/2)], 0.05)

# initialize the fields
Phi_fields_old = np.zeros((npoints, fields))
Phi_fields_new = np.zeros((npoints, fields))
for i in range(fields):
    Phi_fields_old[:, i] = Phi_0
    Phi_fields_new[:, i] = Phi_0

Phi_fields, Phi_mean = computeStochDiffusion(Phi_fields_old=Phi_fields_old, D=D, Dt=Dt, dt=dt, fields=fields, T_eddy=50, npoints=npoints, tsteps=tsteps)

def mainRun():
    print('Stochastic fields computation started ...')
    Phi_fields, Phi_mean = computeStochDiffusion(Phi_fields_old=Phi_fields_old, D=D, Dt=Dt, dt=dt, fields=fields, T_eddy=10, npoints=npoints, tsteps=100)
    plt.figure(1)
    #plt.plot(grid, Phi_diff)
    plt.plot(grid, Phi_mean)
    plt.grid()
    plt.legend(['pure Diffusion', 'Stochastic Fields'])
    plt.title('Comparison')
    # plt.show()
    plt.figure(2)
    plt.plot(grid, Phi_fields)
    plt.title('Phi der 8 Felder')
    plt.grid()
    plt.show(block=False)

if __name__ == '__main__':
    mainRun()



