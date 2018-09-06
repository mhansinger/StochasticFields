'''
This is a framework to check the impact of Ito and Stratonovich integration on the
pure 1D-diffusive process of an inert scalar by using the stochastic differential equation.

@author: M. Hansinger

date: April 2018
'''

import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
from abc import ABC

class params():
    '''This is the parameter class where you specify all the necessary parameters'''
    # time step
    dt = 0.001
    #grid points
    npoints = 100
    #grid is computed
    grid = np.linspace(0, 1, npoints)
    # spatial discretization
    dx = 1 / npoints
    # laminar diffusion
    D = 0.001
    # turbulent diffusion
    Dt = D * 6
    # number of stochastic fields
    fields = 8
    # time steps to compute
    tsteps = 2000


#############################################################################

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

    def stepFunction(self,x):
        self.Phi_org = np.zeros((len(x),1))
        self.Phi_org[int(self.npoints/2):, 0] = 1

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
            #self.Phi = sc.linalg.solve(self.A,self.Phi)

            if self.BC == 'Dirichlet':
                # update the boundary values -> so that gradient is zero at boundary!
                self.Phi[0,0] = self.Phi[1,0]
                self.Phi[-1,0] = self.Phi[-2,0]

        except AttributeError:
            print('Set first the Diffusion Matrix!\nThis is now done for you...')
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
        self.D = params.D
        self.Dt = params.Dt
        self.dt = params.dt
        self.IEM_on = False

        self.gradPhi = np.zeros((self.npoints,self.fields))
        self.Phi_star = np.zeros((self.npoints,self.fields))

        self.Phi_fields = np.zeros((self.npoints,self.fields))
        self.Phi_RMS = np.zeros((self.npoints, self.fields))
        self.IEM = np.zeros((self.npoints, self.fields))

        #initialize the Gaussian Distribution
        self.gaussianDist(self.grid, self.grid[int(self.npoints/2)], 0.05)
        self.Phi = self.Phi_org.copy()

    def gaussianDist(self,x, mu, sig):
        '''Initialize the gaussian distribution'''
        self.Phi_org = np.zeros((len(x),1))
        self.Phi_org[:,0] = np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

    def stepFunction(self,x):
        self.Phi_org = np.zeros((len(x),1))
        self.Phi_org[int(self.npoints/2):, 0] = 1

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
        self.Phi = self.Phi_fields.mean(axis=1)

        #**************************************
        # additional step if IEM_on = True
        # **************************************
        if self.IEM_on:
            # compute IEM terms
            self.getIEM()
            # at this stage self.Phi is already the averaged field, so use it for further computation
            for f in range(self.fields):
                self.Phi_fields[:, f] = self.Phi_fields[:,f] + self.IEM[:,f]

            # finally get the average over all fields -> new Phi, though it is never used for further computation
            self.Phi = self.Phi_fields.mean(axis=1)
            # get the RMS of the fields
            self.Phi_RMS = self.Phi_fields.std(axis=1)

        # get the RMS of the fields
        self.Phi_RMS = self.Phi_fields.std(axis=1)


    def startStochasticDiffusion(self, tsteps = params.tsteps, IEM_on = False):
        '''
            Start from 0 to advance the stochastic diffusion process
        '''

        # choose for IEM
        self.IEM_on = IEM_on

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


    def continueStochasticDiffusion(self, tsteps = params.tsteps):
        # to advance the diffusion process

        if self.Phi == self.Phi_org:
            print('Use ".startStochasticDiffusion" to start')
        else:
            for i in range(0, tsteps):
                # 1. part of fractional step
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

    def getIEM(self):
        # compute at first the Eddy turn over time: Teddy
        T_eddy = self.dx**2 /(2*self.Dt)

        # compute IEM for each field:
        for f in range(self.fields):
            self.IEM[:,f] = - (self.Phi_fields[:,f] - self.Phi) * (self.dt / T_eddy)


#############################################################################

class StochasticDiffusion_oldPhi(object):
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
        self.D = params.D
        self.Dt = params.Dt
        self.dt = params.dt
        self.IEM_on = False

        self.gradPhi = np.zeros((self.npoints,self.fields))
        self.Phi_star = np.zeros((self.npoints,self.fields))

        self.Phi_fields = np.zeros((self.npoints,self.fields))
        self.Phi_RMS = np.zeros((self.npoints, self.fields))

        self.IEM = np.zeros((self.npoints, self.fields))

        #initialize the Gaussian Distribution
        self.gaussianDist(self.grid, self.grid[int(self.npoints/2)], 0.05)
        self.Phi = self.Phi_org[:,0].copy()

    def gaussianDist(self,x, mu, sig):
        ''' Initialize the gaussian distribution '''
        self.Phi_org = np.zeros((len(x),1))
        self.Phi_org[:,0] = np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

    def stepFunction(self,x):
        self.Phi_org = np.zeros((len(x),1))
        self.Phi_org[int(self.npoints/2):, 0] = 1

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

        #**************************************
        # additional step if IEM_on = True
        # **************************************
        if self.IEM_on:
            # compute IEM terms
            self.getIEM()
            # at this stage self.Phi is already the averaged field, so use it for further computation
            for f in range(self.fields):
                self.Phi_fields[:, f] = self.Phi_fields[:,f] + self.IEM[:,f]

        # finally get the average over all fields -> new Phi, though it is never used for further computation
        self.Phi = self.Phi_fields.mean(axis=1)

        # get the RMS of the fields
        self.Phi_RMS = self.Phi_fields.std(axis=1)

    def startStochasticDiffusion(self, tsteps = params.tsteps, IEM_on = False):
        '''
            Start from 0 to advance the stochastic diffusion process
        '''

        # choose for IEM
        self.IEM_on = IEM_on

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


    def continueStochasticDiffusion(self, tsteps = params.tsteps):
        # to advance the diffusion process

        if self.Phi == self.Phi_org:
            print('Use ".startStochasticDiffusion" to start')
        else:
            for i in range(0, tsteps):
                # 1. part of fractional step
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


    def getIEM(self):
        # compute at first the Eddy turn over time: Teddy
        T_eddy = self.dx**2 /(2*self.Dt)

        # compute IEM for each field:
        for f in range(self.fields):
            self.IEM[:,f] = - (self.Phi_fields[:,f] - self.Phi) * (self.dt / T_eddy)


#############################################################################

# check that with runge-Kutta
class StochasticDiffusion_Stratonovich(object):
    def __init__(self, params, BC='Neumann'):
        '''
        :param params:
        :param BC: Either 'Dirichlet' or 'Neumann'
        This the Version where the stochastic velocity is already included into the diffusion matrix, hence, no fractional
        step is applied.
        '''

        # here only fixed parameters are handed over, dt, D, etc. which are adjustable will come later
        self.npoints = params.npoints
        self.fields = params.fields
        self.dx = params.dx
        self.grid = params.grid
        self.BC = BC
        self.D = params.D
        self.Dt = params.Dt
        self.dt = params.dt
        self.IEM_on = False

        self.gradPhi = np.zeros((self.npoints, self.fields))
        self.Phi_star = np.zeros((self.npoints, self.fields))

        self.Phi_fields = np.zeros((self.npoints, self.fields))
        self.Phi_RMS = np.zeros((self.npoints, self.fields))
        self.IEM = np.zeros((self.npoints, self.fields))

        # initialize the Gaussian Distribution
        self.gaussianDist(self.grid, self.grid[int(self.npoints / 2)], 0.05)
        self.Phi = self.Phi_org.copy()

    def gaussianDist(self, x, mu, sig):
        '''Initialize the gaussian distribution'''
        self.Phi_org = np.zeros((len(x), 1))
        self.Phi_org[:, 0] = np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

    def stepFunction(self,x):
        self.Phi_org = np.zeros((len(x),1))
        self.Phi_org[int(self.npoints/2):, 0] = 1

    def setStochMatrix(self):
        # this is the diffusion matrix where the stochastic term is included!
        # -> each field has its own Matrix A and for each time step a new one

        # span for each field a different coefficient Matrix, as dW always differs
        self.dWiener()
        self.A_stoch = np.zeros([self.fields, self.npoints, self.npoints])

        # computes east, west as vectors, middle as scalar
        east = (self.D + self.Dt) / self.dx ** 2 * self.dt + 1 / (2 * self.dx) * np.sqrt(2 * self.Dt) * self.dW
        middle = -2 * (self.D + self.Dt) / self.dx ** 2 * self.dt
        west = (self.D + self.Dt) / self.dx ** 2 * self.dt - 1 / (2 * self.dx) * np.sqrt(2 * self.Dt) * self.dW

        # loop over the fields
        for j in range(self.fields):
            # print(j)
            for i in range(1, self.npoints - 1):
                self.A_stoch[j][i][i - 1] = west[j]
                self.A_stoch[j][i][i] = middle
                self.A_stoch[j][i][i + 1] = east[j]

            if self.BC == 'Neumann':
                # INLET:
                self.A_stoch[j][0][0] = middle
                self.A_stoch[j][0][1] = west[j]

                # OUTLET
                self.A_stoch[j][- 1][- 1] = middle
                self.A_stoch[j][- 1][- 2] = east[j]

            elif self.BC == 'Dirichlet':
                # INLET:
                self.A_stoch[j][0][0] = 0
                self.A_stoch[j][0][1] = 0

                # OUTLET
                self.A_stoch[j][-1][-1] = 0
                self.A_stoch[j][-1][-2] = 0


    def mixedDiffusion(self):
        # this is the diffusion where the stochastic velocity is already implemented into the Matrix

        # upadte the stochastic matrix!
        self.setStochMatrix()

        try:
            # again do this for each field separately
            for f in range(self.fields):
                self.Phi_fields[:, f] = np.matmul(self.A_stoch[f], self.Phi_fields[:, f])

                if self.BC == 'Dirichlet':
                    # update the boundary values -> so that gradient is zero at boundary!
                    self.Phi_fields[0, f] = self.Phi_fields[1, f].copy()
                    self.Phi_fields[-1, f] = self.Phi_fields[-2, f].copy()

        except AttributeError:
            print('Set first the Diffusion Matrix!\nIt has been done for you...')
            self.setStochMatrix()


    def startStochasticDiffusion(self, tsteps=params.tsteps, IEM_on=False):
        '''
            Start from 0 to advance the stochastic diffusion process
        '''

        # choose for IEM
        self.IEM_on = IEM_on

        for i in range(0, tsteps):
            if i == 0:

                # this is the first time step, assuming all fields are the same,
                # so fill them:
                self.initFields()

                # updateing diffusion equation
                self.mixedDiffusion()
                # get the mean field
                self.Phi = self.Phi_fields.mean(axis=1)

                # additional step if IEM_on = True
                if self.IEM_on:
                    # compute IEM terms
                    self.getIEM()
                    # at this stage self.Phi is already the averaged field, so use it for further computation
                    for f in range(self.fields):
                        self.Phi_fields[:, f] = self.Phi_fields[:, f] + self.IEM[:, f]

                    # finally get the average over all fields -> new Phi, though it is never used for further computation
                    self.Phi = self.Phi_fields.mean(axis=1)

            else:
                # update the mean field
                self.Phi = self.Phi_fields.mean(axis=1)

                # additional step if IEM_on = True
                if self.IEM_on:
                    # compute IEM terms
                    self.getIEM()
                    # at this stage self.Phi is already the averaged field, so use it for further computation
                    for f in range(self.fields):
                        self.Phi_fields[:, f] = self.Phi_fields[:, f] + self.IEM[:, f]

                    # finally get the average over all fields -> new Phi, though it is never used for
                    # further computation
                    self.Phi = self.Phi_fields.mean(axis=1)

    def continueStochasticDiffusion(self, tsteps=params.tsteps):
        # to advance the diffusion process

        if self.Phi == self.Phi_org:
            print('Use ".startStochasticDiffusion" to start')
        else:
            for i in range(0, tsteps):
                # update the mean field
                self.Phi = self.Phi_fields.mean(axis=1)

                # 2. part of fractional step, add the stochastic velocity
                # **************************************
                # additional step if IEM_on = True
                # **************************************
                if self.IEM_on:
                    # compute IEM terms
                    self.getIEM()
                    # at this stage self.Phi is already the averaged field, so use it for further computation
                    for f in range(self.fields):
                        self.Phi_fields[:, f] = self.Phi_fields[:, f] + self.IEM[:, f]

                    # finally get the average over all fields -> new Phi, though it is never used for
                    # further computation
                    self.Phi = self.Phi_fields.mean(axis=1)

    # These are the functions for Stochastic Fields then
    def dWiener(self):
        # compute the Wiener Term
        # initialize gamma vector
        gamma = np.ones(self.fields)
        gamma[0:int(self.fields / 2)] = -1
        # shuffle it
        np.random.shuffle(gamma)
        self.dW = gamma * np.sqrt(self.dt)


    def upDateDiffusion(self, D, Dt, dt):
        self.D = D
        self.Dt = Dt
        self.dt = dt

    def initFields(self):
        # helper to initialize the fields
        for f in range(self.fields):
            self.Phi_fields[:, f] = self.Phi_org[:, 0].copy()

    def getIEM(self):
        # compute at first the Eddy turn over time: Teddy
        T_eddy = self.dx ** 2 / (2 * self.Dt)

        # compute IEM for each field:
        for f in range(self.fields):
            self.IEM[:, f] = - (self.Phi_fields[:, f] - self.Phi) * (self.dt / T_eddy)

#############################################################################

class StochasticDiffusion_noIEM(object):
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
        self.D = params.D
        self.Dt = params.Dt
        self.dt = params.dt
        self.IEM_on = False

        self.gradPhi = np.zeros((self.npoints,self.fields))
        self.Phi_star = np.zeros((self.npoints,self.fields))

        self.Phi_fields = np.zeros((self.npoints,self.fields))
        self.Phi_RMS = np.zeros((self.npoints, self.fields))

        self.IEM = np.zeros((self.npoints, self.fields))

        #initialize the Gaussian Distribution
        self.gaussianDist(self.grid, self.grid[int(self.npoints/2)], 0.05)
        self.Phi = self.Phi_org[:,0].copy()

    def gaussianDist(self,x, mu, sig):
        ''' Initialize the gaussian distribution '''
        self.Phi_org = np.zeros((len(x),1))
        self.Phi_org[:,0] = np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

    def stepFunction(self,x):
        self.Phi_org = np.zeros((len(x),1))
        self.Phi_org[int(self.npoints/2):, 0] = 1

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


        # compute Wiener term
        self.dWiener()

        # compute the gradient of Phi_star for each field separately
        self.getGradients()

        # now compute Phi for each field
        for f in range(self.fields):
            self.Phi_fields[:,f] = self.Phi_star[:,f] + np.sqrt(2*self.Dt)*self.gradPhi[:,f]*self.dW[f] * 2


        # finally get the average over all fields -> new Phi, though it is never used for further computation
        self.Phi = self.Phi_fields.mean(axis=1)

        # get the RMS of the fields
        self.Phi_RMS = self.Phi_fields.std(axis=1)


    def startStochasticDiffusion(self, tsteps = params.tsteps):
        '''
            Start from 0 to advance the stochastic diffusion process
        '''

        # choose for IEM
        self.IEM_on = False

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

                # **************************************
                # overwrite all fields with current Phi
                # **************************************
                for f in range(self.fields):
                    self.Phi_fields[:, f] = self.Phi

                # 1 part of fractional step
                self.pureDiffusion()

                # 2. part of fractional step, add the stochastic velocity
                self.addStochastic()


    def continueStochasticDiffusion(self, tsteps = params.tsteps):
        # to advance the diffusion process

        if self.Phi == self.Phi_org:
            print('Use ".startStochasticDiffusion" to start')
        else:
            for i in range(0, tsteps):
                # 1. part of fractional step
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

# from params import params

myParams = params()

Pe = (myParams.D+myParams.Dt)*myParams.dt / (myParams.dx**2)
print("Pe is: ", round(Pe,4))

thisBC = 'Dirichlet'

ESF_off = StochasticDiffusion(myParams,BC=thisBC)
ESF_off.stepFunction(ESF_off.grid)
ESF_on = StochasticDiffusion(myParams,BC=thisBC)
ESF_on.stepFunction(ESF_on.grid)
ESF_new = StochasticDiffusion_noIEM(myParams,BC=thisBC)
ESF_new.stepFunction(ESF_new.grid)
Diff = Diffusion(myParams,BC=thisBC)
Diff.stepFunction(Diff.grid)

tsteps = 10000

ESF_off.startStochasticDiffusion(tsteps=tsteps,IEM_on=False)
ESF_on.startStochasticDiffusion(tsteps=tsteps, IEM_on=True)
ESF_new.startStochasticDiffusion(tsteps=tsteps)
Diff.advanceDiffusion(tsteps=tsteps)



# plot to compare the results
plt.figure(1)
plt.plot(Diff.Phi_org,':m')
plt.plot(Diff.Phi,'-k')
plt.plot(ESF_off.Phi,'--b')
plt.plot(ESF_on.Phi,'-.r')
plt.legend(['init. Distribution','pure Diffusion','ESF Diffusion w/o IEM','ESF Diffusion w. IEM'])
plt.title('Scalar distribution after %s steps' % str(tsteps))
plt.ylabel('Scalar concentration [-]')
try:
    plt.savefig('Figures/Phi_%s_dt%s_D%s_%sfields.png' % (str(tsteps), str(Diff.dt), str(ESF_on.D), str(ESF_on.fields)))
    plt.savefig('Figures/Phi_%s_dt%s_D%s_%sfields.eps' % (str(tsteps), str(Diff.dt), str(ESF_on.D), str(ESF_on.fields)))
except:
    pass
plt.show(block=False)

plt.figure(2)
on, =plt.plot(ESF_on.Phi_fields[:,0],'-.r', label='IEM on')
off, = plt.plot(ESF_off.Phi_fields[:,0],'--b', label='IEM off')
#init, = plt.plot(ESF_off.Phi_org,'-m', label='init. Distribution')
plt.plot(ESF_on.Phi_fields[:,1:],'-.r')
plt.plot(ESF_off.Phi_fields[:,1:],'--b')
#plt.legend(handles=[on,off,init])
plt.legend(handles=[on,off])
plt.title('Evolution of single fields after %s steps' % str(tsteps))
plt.ylabel('Scalar concentration [-]')
try:
    plt.savefig('Figures/Fields_%s_dt%s_D%s%sfields.png' % (str(tsteps), str(Diff.dt), str(ESF_on.D), str(ESF_on.fields)))
    plt.savefig('Figures/Fields_%s_dt%s_D%s%sfields.eps' % (str(tsteps), str(Diff.dt), str(ESF_on.D), str(ESF_on.fields)))
except:
    pass

plt.show(block=False)

