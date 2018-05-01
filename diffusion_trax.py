'''
This is a framework to check the impact of Ito and Stratonovich integration on the 
pure 1D-diffusive process of an inert scalar by using the stochastic differential equation.
 
@author: M. Hansinger 

'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve


def setDiffMatrix(D,npoints,dt,dx):
    # set up coefficient matrix for pure diffusion
    A = np.zeros([npoints, npoints])

    east = (D) / dx ** 2 * dt
    middle = -2 * (D) / dx ** 2 * dt
    west = (D) / dx ** 2 * dt

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


# 1D diffusion equation
def pureDiffusion(Phi_0,A):
    Phi = np.matmul(Phi_0,A) + Phi_0
    return Phi

def advanceDiffusion(Phi_0,A,tsteps=1000):
    points = len(Phi_0)
    for i in range(1,tsteps):
        if i ==1:
            Phi_old = pureDiffusion(Phi_0,A)
        else:
            Phi_new = pureDiffusion(Phi_old,A)
            Phi_old = Phi_new

        Phi_old[points-1] = Phi_old[points-2]
        Phi_old[0] = Phi_old[1]

       # plt.plot(Phi_old)

    return Phi_old


#def analyticalSolution(Phi_0,A,D,grid)

# Gaussian function for initialization
def gaussianDist(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))




################################
# main part
################################


#set parameters

npoints = 1000
grid = np.linspace(0,1,npoints)


dx = 1/npoints
D = 0.0005

# time step
dt = 0.001
tsteps = 50000

# Initial conditions
Phi_0 = np.zeros((npoints,))
Phi_0[int(npoints/2 +1):,] =1

# set up diffusion matrix
#A = setDiffMatrix(D=D,npoints=np,dt=dt,dx=dx)

# compute diffusion
Phi_end = advanceDiffusion(Phi_0=Phi_0,A=A,tsteps=tsteps)

# plot Diffusion
plt.figure(1)

plt.plot(Phi_0)
plt.plot(Phi_end)
plt.legend(['Phi_0','Phi_new'])

plt.title('Phi after '+str(tsteps)+' time steps')

plt.show(block=False)




