'''
This is a framework to check the impact of Ito and Stratonovich integration on the
pure 1D-diffusive process of an inert scalar by using the stochastic differential equation.

@author: M. Hansinger

date: April 2018
'''

import numpy as np
import matplotlib.pyplot as plt

class params():
    '''This is the parameter class where you specify all the necessary parameters'''
    # time step
    dt = 0.0001

    #grid points
    npoints = 300

    #grid is computed
    grid = np.linspace(0, 1, npoints)

    # spatial discretization
    dx = 1 / npoints

    # laminar diffusion
    D = 0.0001
    # turbulent diffusion
    Dt = D * 2

    # number of stochastic fields
    fields = 8

    # time steps to compute
    tsteps = 1000


class Diffusion(object):
    def __init__(self,params, BC='Neumann'):
        '''
        :param params:
        :param BC: Either 'Dirichlet' or 'Neumann'
        '''

        #here only fixed parameters are handed over, dt, D, etc. which are adjustable will come later
        self.npoints = params.npoints
        self.fields = params.fields
        self.dx = params.dx
        self.grid = params.grid
        self.BC = BC

        #initialize the Gaussian Distribution
        self.gaussianDist(self.grid, self.grid[int(self.npoints/2)], 0.05)

    def gaussianDist(self,x, mu, sig):
        '''Initialize the gaussian distribution'''
        self.Phi_org = np.zeros((len(x),1))
        self.Phi_org[:,0] = np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))


    def setDiffMatrix(self, D=params.D, Dt=params.Dt, dt=params.dt):
        '''
        This function will fill the diffusion Matrix A
        '''
        self.A = np.zeros([self.npoints, self.npoints])
        self.dt = dt

        east = (D + Dt) * self.dt / self.dx ** 2
        middle = -2 * (D + Dt) * self.dt / self.dx ** 2 + 1
        west = (D + Dt) * self.dt / self.dx ** 2

        for i in range(1, self.npoints - 1):
            self.A[i][i - 1] = west
            self.A[i][i] = middle
            self.A[i][i + 1] = east

        if self.BC == 'Neumann':
            # INLET:
            self.A[0][0] = middle
            self.A[0][1] = west

            # OUTLET
            self.A[-1][-1] = middle
            self.A[-1][-2] = east

        elif self.BC == 'Dirichlet':
            # INLET:
            self.A[0][0] = 0
            self.A[0][1] = 0

            # OUTLET
            self.A[-1][-1] = 0
            self.A[-1][-2] = 0

        else:
            raise Exception('Check your boundary conditions!')


    def pureDiffusion(self):
        # 1D diffusion equation
        try:
            self.Phi = np.matmul(self.A,self.Phi)

            if self.BC == 'Dirichlet':
                # update the boundary values -> so that gradient is zero at boundary!
                self.Phi[0,0] = self.Phi[1,0]
                self.Phi[-1,0] = self.Phi[-2,0]

        except AttributeError:
            print('Set first the Diffusion Matrix!\n This is now done for you...')
            self.setDiffMatrix()


    def advanceDiffusion(self, tsteps = params.tsteps):
        # this is the pure diffusion process in implicit formulation
        for i in range(0, tsteps):
            if i == 0:
                self.Phi = self.Phi_org.copy()
                # updateing diffusion equation
                self.pureDiffusion()
            else:
                self.pureDiffusion()


    # These are the functions for Stochastic Fields then
    def dWiener(self):
        # compute the Wiener Term
        # initialize gamma vector
        gamma = np.ones(self.fields)
        gamma[0:int(self.fields / 2)] = -1
        # shuffle it
        np.random.shuffle(gamma)
        self.dW = gamma * np.sqrt(self.dt)


#############################################################################

class StochasticDiffusion(object):
    def __init__(self,params, BC='Neumann', IEM_on = False):
        '''
        :param params:
        :param BC: Either 'Dirichlet' or 'Neumann'
        '''

        #here only fixed parameters are handed over, dt, D, etc. which are adjustable will come later
        self.npoints = params.npoints
        self.fields = params.fields
        self.dx = params.dx
        self.grid = params.grid
        self.BC = BC
        self.D = params.D
        self.Dt = params.Dt
        self.dt = params.dt

        self.gradPhi = np.zeros((self.npoints,self.fields))
        self.Phi_star = np.zeros((self.npoints,self.fields))

        self.Phi_fields = np.zeros((self.npoints,self.fields))

        self.Phi = np.zeros(self.npoints)

        #initialize the Gaussian Distribution
        self.gaussianDist(self.grid, self.grid[int(self.npoints/2)], 0.05)

    def gaussianDist(self,x, mu, sig):
        '''Initialize the gaussian distribution'''
        self.Phi_org = np.zeros((len(x),1))
        self.Phi_org[:,0] = np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))


    def setDiffMatrix(self):
        '''
        This function will fill the diffusion Matrix A
        '''
        self.A = np.zeros([self.npoints, self.npoints])


        east = (self.D + self.Dt) * self.dt / self.dx ** 2
        middle = -2 * (self.D + self.Dt) * self.dt / self.dx ** 2 + 1
        west = (self.D + self.Dt) * self.dt / self.dx ** 2

        for i in range(1, self.npoints - 1):
            self.A[i][i - 1] = west
            self.A[i][i] = middle
            self.A[i][i + 1] = east

        if self.BC == 'Neumann':
            # INLET:
            self.A[0][0] = middle
            self.A[0][1] = west

            # OUTLET
            self.A[-1][-1] = middle
            self.A[-1][-2] = east

        elif self.BC == 'Dirichlet':
            # INLET:
            self.A[0][0] = 0
            self.A[0][1] = 0

            # OUTLET
            self.A[-1][-1] = 0
            self.A[-1][-2] = 0

        else:
            raise Exception('Check your boundary conditions!')


    def pureDiffusion(self):
        # 1D diffusion equation
        # this is equivalent to the 1st fractional step (eq. 2.11)
        try:
            # again do this for each field separately
            for f in range(self.fields):
                self.Phi_star[:,f] = np.matmul(self.A, self.Phi_fields[:,f])

                if self.BC == 'Dirichlet':
                    # update the boundary values -> so that gradient is zero at boundary!
                    self.Phi_star[0, f] = self.Phi_fields[1, f].copy()
                    self.Phi_star[-1, f] = self.Phi_fields[-2, f].copy()

        except AttributeError:
            print('Set first the Diffusion Matrix!\nIt has been done for you...')
            self.setDiffMatrix()


    def addStochastic(self):
        # the Wiener element is between 1,..,nFields of Wiener term
        # this is done in an explicit manner!

        # compute Wiener term
        self.dWiener()

        # compute the gradient of Phi_star for each field separately
        self.getGradients()

        # now compute Phi for each field
        for f in range(self.fields):
            self.Phi_fields[:,f] = self.Phi_star[:,f] + np.sqrt(2*self.Dt)*self.gradPhi[:,f]*self.dW[f]

        # finally get the average over all fields -> new Phi, though it is never used for further computation
        for n in range(self.npoints):
            self.Phi[n] = 1/self.fields * sum(self.Phi_fields[n,:])


    def advanceStochasticDiffusion(self, tsteps = params.tsteps):
        # this is the stochastic version

        #first update the diffusion matrix (implicit), in case dt, D, Dt have changed
        self.setDiffMatrix()

        for i in range(0, tsteps):
            if i == 0:

                # this is the first time step, assuming all fields are the same,
                # so fill them:
                self.initFields()

                # updateing diffusion equation
                # 1. part of fractional step
                self.pureDiffusion()

                # 2. part of fractional step, add the stochastic velocity
                self.addStochastic()

            else:
                # 1 part of fractional step
                self.pureDiffusion()

                # 2. part of fractional step, add the stochastic velocity
                self.addStochastic()


    # These are the functions for Stochastic Fields then
    def dWiener(self):
        # compute the Wiener Term
        # initialize gamma vector
        gamma = np.ones(self.fields)
        gamma[0:int(self.fields / 2)] = -1
        # shuffle it
        np.random.shuffle(gamma)
        self.dW = gamma * np.sqrt(self.dt)

    def getGradients(self):
        # it computes the scalar gradient dPhi/dx

        for f in range(self.fields):
            # now compute the gradients with CDS, except the boundaries, they are up and downwind
            for i in range(self.npoints):
                # iter over 0,...,npoints - 1
                if i == 0:
                    self.gradPhi[0,f] = (self.Phi_star[1,f] - self.Phi_star[0,f]) / self.dx
                elif i == self.npoints - 1:
                    self.gradPhi[i,f] = (self.Phi_star[i,f] - self.Phi_star[i-1,f]) / self.dx
                else:
                    self.gradPhi[i,f] = (self.Phi_star[i+1,f] - self.Phi_star[i-1,f]) / (2 * self.dx)


    def upDateDiffusion(self,D,Dt,dt):
        self.D = D
        self.Dt = Dt
        self.dt = dt

    def initFields(self):
        # helper to initialize the fields
        for f in range(self.fields):
            self.Phi_fields[:, f] = self.Phi_org[:,0].copy()


####################################################################

myParams = params()

ESF = StochasticDiffusion(myParams)

Diff = Diffusion(myParams)

ESF.advanceStochasticDiffusion(tsteps=5000)
Diff.advanceDiffusion(tsteps=5000)

# plot to compare the results
plt.plot(ESF.Phi)
plt.plot(Diff.Phi)

plt.show()

plt.plot(ESF.Phi_fields)
plt.show()