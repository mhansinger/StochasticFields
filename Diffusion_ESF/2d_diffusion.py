'''
This is a framework to check the impact of Ito and Stratonovich integration on the
pure 1D-diffusive process of an inert scalar by using the stochastic differential equation.

@author: M. Hansinger

date: April 2018
'''

import numpy as np
import matplotlib.pyplot as plt
from abc import ABC
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from numba import jit

from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

# this is for latex in the plot labels and titles
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('font', size=15)


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
    Dt = D * 3
    # number of stochastic fields
    fields = 8
    # time steps to compute
    tsteps = 200


###################################################################
class Diffusion_2D_ABC(ABC):
    def __init__(self,params, BC='Neumann'):
        super().__init__()
        '''
        :param params:
        :param BC: Either 'Dirichlet' or 'Neumann'
        '''

        #here only fixed parameters are handed over, dt, D, etc. which are adjustable will come later
        self.npoints = params.npoints
        self.fields = params.fields
        self.dx = params.dx
        self.dy = params.dx
        self.grid = params.grid
        self.Phi_2d = np.zeros((params.npoints,params.npoints))
        self.BC = BC
        self.Dt = params.Dt
        self.D = params.D
        self.tsteps = params.tsteps

        #initialize the Gaussian Distribution
        self.gaussianDist(self.grid, self.grid[int(self.npoints/2)], 0.05)

        # initialize the 2d Array with Gaussian
        for i in range(0,self.npoints):
            for j in range(0,self.npoints):
                self.Phi_2d[i,j] = self.Phi_org[i] * self.Phi_org[j]

        self.Phi_2d_org = self.Phi_2d.copy()

        self.Phi_2d_vec = self.Phi_2d_org.reshape(self.npoints*self.npoints)


    def gaussianDist(self,x, mu, sig):
        '''Initialize the gaussian distribution'''
        self.Phi_org = np.zeros((len(x),1))
        self.Phi_org[:,0] = np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))


    def setDiffMatrix(self, dt=params.dt):
        '''
        This function will fill the diffusion Matrix A
        This is a bit more tricky than in the 1D case

        1. reshape your array to a vector -> length(npoints**2)
        2. set up the Diffusion matrix with fluxes in x and y direction
        '''
        #1. Vector


        self.A = np.zeros([self.npoints**2, self.npoints**2])
        self.dt = dt

        east = (self.D + self.Dt) * self.dt / (self.dx ** 2)
        middle = -2 * (self.D + self.Dt) * self.dt / (self.dx ** 2) - 2 * (self.D + self.Dt) * self.dt / (self.dy ** 2) + 1
        west = (self.D + self.Dt) * self.dt / (self.dx ** 2)

        # for 2d additional
        north = (self.D + self.Dt) * self.dt / (self.dy ** 2)
        south = (self.D + self.Dt) * self.dt / (self.dy ** 2)


        for i in range(self.npoints, self.npoints**2 - self.npoints):
            self.A[i][i - 1] = west
            self.A[i][i] = middle
            self.A[i][i + 1] = east

            # 2d additional
            self.A[i - self.npoints][i] = south
            self.A[i + self.npoints][i] = north

        if self.BC == 'Neumann':
            # LOWER:
            for i in range(1, self.npoints - 1):
                self.A[i][i] = middle
                self.A[i][i+self.npoints] = north

            # UPPER:
            for i in range(self.npoints - self.npoints ** 2, self.npoints ** 2):
                self.A[i][i] = middle
                self.A[i][i-self.npoints] = south

            # RIGHT:
            for i in range(self.npoints, self.npoints ** 2, self.npoints):
                self.A[i][i] = middle
                self.A[i][i-self.npoints] = west

            # LEFT:
            for i in range(0 , self.npoints ** 2 - self.npoints, self.npoints):
                self.A[i][i] = middle
                self.A[i][i+self.npoints] = east

        elif self.BC == 'Dirichlet':
            # LOWER:
            for i in range(1, self.npoints - 1):
                self.A[i][i] = 0

            # UPPER:
            for i in range(self.npoints - self.npoints ** 2, self.npoints ** 2):
                self.A[i][i] = 0

            # RIGHT:
            for i in range(self.npoints , self.npoints ** 2, self.npoints):
                self.A[i][i] = 0

            # LEFT:
            for i in range(0 , self.npoints ** 2 - self.npoints, self.npoints):
                self.A[i][i] = 0

        else:
            raise Exception('Check your boundary conditions!')

    def pureDiffusion(self):
        # 1D diffusion equation
        try:
            self.Phi_2d_vec = np.matmul(self.A,self.Phi_2d_vec)

            if self.BC == 'Dirichlet':
                print('Dont use Dirichlet')
                # update the boundary values -> so that gradient is zero at boundary!
                #self.Phi_2d_vec[0,0] = self.Phi[1,0]
                #self.Phi_2d[-1,0] = self.Phi[-2,0]

        except AttributeError:
            print('Set first the Diffusion Matrix!\nThis is now done for you...')
            self.setDiffMatrix()

    @jit
    def advanceDiffusion(self, tsteps = params.tsteps):
        # this is the pure diffusion process in implicit formulation
        self.tsteps = tsteps

        self.Phi_2d_vec = self.Phi_2d_org.reshape(self.npoints * self.npoints)

        self.setDiffMatrix()

        for i in range(0, self.tsteps):
            if i == 0:
                self.Phi_2d_vec = self.Phi_2d_org.reshape(self.npoints*self.npoints)
                # updateing diffusion equation
                self.pureDiffusion()
            else:
                self.pureDiffusion()

    def continueDiffusion(self, tsteps = params.tsteps):
        # to advance the diffusion process
        self.tsteps = tsteps
        self.setDiffMatrix()

        for i in range(0, tsteps):
            # 1. part of fractional step
            self.pureDiffusion()

    def upDateDiffusion(self,D,Dt,dt):
        self.D = D
        self.Dt = Dt
        self.dt = dt


###########################################
# Plotting
    def plot_3D(self,org = False):

        # Plots nebeneinander!

        self.Phi_2d = self.Phi_2d_vec.reshape(self.npoints,self.npoints)

        fig = plt.figure()
        ax = fig.gca(projection='3d')

        X, Y = np.meshgrid(self.grid,self.grid)

        if org:
            # Plot the surface.
            surf = ax.plot_surface(X, Y, self.Phi_2d_org, cmap='inferno',
                                   linewidth=0, antialiased=False)
        else:
            # Plot the surface.
            surf = ax.plot_surface(X, Y, self.Phi_2d, cmap='inferno',
                                   linewidth=0, antialiased=False)

        # Customize the z axis.
        ax.set_zlim(0, 1)
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

        # Add a color bar which maps values to colors.
        cbar = plt.colorbar(surf, shrink=0.5, aspect=5)#, ticks=[0,0.25,0.5,0.75,0.99])
        #cbar.ax.set_xticklabels(['0', '0.25', '0.5','0.75','1.0'])

        cbar.set_label('Concentration')
        cbar.set_clim(0,1)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Scalar distribution')
        plt.tight_layout()
        plt.show(block=False)

    def imshow_Phi(self,org=False):

        # Plots nebeneinander

        self.Phi_2d = self.Phi_2d_vec.reshape(self.npoints,self.npoints)

        if org:
            plt.imshow(self.Phi_2d_org, cmap='inferno',extent=[0,1,0,1])

        else:
            plt.imshow(self.Phi_2d, cmap='inferno',extent=[0,1,0,1])


        # Plot the surface with colorbar .
        cbar = plt.colorbar(ticks=[0,0.25,0.5,0.75,1.0])
        cbar.set_label('Concentration')
        cbar.set_clim(0,1.001)

        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Scalar distribution')
        plt.tight_layout()
        #cbar.set_ticklabels(ticklabels=['0', '0.25', '0.5','0.75','1.0'])
        plt.show(block=False)


###################################################################
class Diffusion_2d(Diffusion_2D_ABC):
    def __init__(self,params,BC='Neumann'):
        super().__init__(params,BC='Neumann')
        '''
        Is identical to abstract base class (ABC) diffusion, so nothing added... 
        '''

###################################################################
class StochasticDiffusion_2d_ABC(Diffusion_2D_ABC):
    def __init__(self,params,BC='Neumann'):
        super().__init__(params,BC='Neumann')
        '''
        This is the stochastic diffusion base class for 2D simulation 
        '''

        self.Phi_fields_2d = np.zeros((self.npoints**2, self.fields))

        # gradients in x and y direction
        # this is 3 dimensional: points, direction (x&y) and field
        self.gradPhi = np.zeros((self.npoints**2, self.fields, 2))

        self.IEM = self.Phi_fields_2d.copy()
        self.dW = np.zeros((self.fields,2))
        self.IEM_on = True

        # needed for operator splitting
        self.Phi_fields_2d_star = self.Phi_fields_2d.copy()

    def initFields(self):
        # helper to initialize the fields
        for f in range(self.fields):
            self.Phi_fields_2d[:, f] = self.Phi_2d_vec.copy()

    # has to overwritten as the function needs to do matrix multiplication for each field separately!
    def pureDiffusion(self):
        # 1D diffusion equation
        try:
            for f in range(self.fields):
                self.Phi_fields_2d_star[:,f] = np.matmul(self.A,self.Phi_fields_2d[:,f])

                if self.BC == 'Dirichlet':
                    print('Dont use Dirichlet')
                    # update the boundary values -> so that gradient is zero at boundary!
                    # self.Phi_2d_vec[0,0] = self.Phi[1,0]
                    # self.Phi_2d[-1,0] = self.Phi[-2,0]

        except AttributeError:
            print('Set first the Diffusion Matrix!\nThis is now done for you...')
            self.setDiffMatrix()

    @jit(parallel=True)
    def addStochastic(self):
        # the Wiener element is between 1,..,nFields of Wiener term
        # this is done in an explicit manner!

        # compute Wiener term
        self.dWiener()

        # compute the gradient of Phi_star for each field separately
        self.getGradients()

        # now compute Phi for each field and loop over points
        for f in range(self.fields):
            for p in range(self.npoints**2):
                self.Phi_fields_2d[p, f] = \
                    self.Phi_fields_2d_star[p, f] + np.sqrt(2*self.Dt) * np.dot(self.gradPhi[p, f, :],self.dW[f, :])

        # finally get the average over all fields -> new Phi, though it is never used for further computation
        self.Phi_2d_vec = self.Phi_fields_2d.mean(axis=1)

        # *************************************
        # additional step if IEM_on = True
        # *************************************
        if self.IEM_on:
            # compute IEM terms
            self.getIEM()
            # at this stage self.Phi is already the averaged field, so use it for further computation
            for f in range(self.fields):
                self.Phi_fields_2d[:, f] = self.Phi_fields_2d[:, f] + self.IEM[:, f]

            # finally get the average over all fields -> new Phi, though it is never used for further computation
            self.Phi_2d_vec = self.Phi_fields_2d.mean(axis=1)

        # get the RMS of the fields
        self.Phi_STD = self.Phi_fields_2d.std(axis=1)

    def startStochasticDiffusion(self, tsteps = params.tsteps, IEM_on = False):
        '''
            Start from 0 to advance the stochastic diffusion process
        '''
        self.tsteps = tsteps

        self.Phi_2d_vec = self.Phi_2d_org.reshape(self.npoints * self.npoints)

        # choose for IEM
        self.IEM_on = IEM_on

        #first update the diffusion matrix (implicit), in case dt, D, Dt have changed
        self.setDiffMatrix()

        for i in range(0, self.tsteps):

            if i%100 == 0:
                print(i)

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

    def continueStochasticDiffusion(self, tsteps=params.tsteps):
        # to advance the diffusion process

        self.tsteps = tsteps

        #first update the diffusion matrix (implicit), in case dt, D, Dt have changed
        self.setDiffMatrix()

        if self.Phi_2d_vec == self.Phi_2d_org:
            print('Use ".startStochasticDiffusion" to start')
        else:
            for i in range(0, tsteps):
                # update the mean field
                #self.Phi_2d_vec = self.Phi_fields_2d.mean(axis=1)

                # updateing diffusion equation
                # 1. part of fractional step
                self.pureDiffusion()

                # 2. part of fractional step, add the stochastic velocity
                self.addStochastic()

    # These are the functions for Stochastic Fields then
    def dWiener(self):
        # compute the Wiener Term, it has to be 2 dimensional now
        # initialize gamma vector
        gamma = np.ones((self.fields,2))

        for d in range(gamma.shape[1]):
            gamma[0:int(self.fields / 2),d] = -1
            # shuffle it
            np.random.shuffle(gamma[:,d])
            self.dW[:,d] = gamma[:,d] * np.sqrt(self.dt)

    @jit(parallel=True)
    def getGradients(self):
        # it computes the scalar gradient dPhi/dx and dPhi/dy
        #this WORKS!!!

        # compute for every field in x and y direction the gradient
        for f in range(self.fields):
            # get the current field value and reshape for convenience
            thisField = self.Phi_fields_2d[:,f].reshape(self.npoints,self.npoints)

            x_grad = thisField.copy()
            y_grad = thisField.copy()

            # now compute the gradients with CDS, except the boundaries, they are up and downwind
            # x-direction
            for g in range(self.npoints):   # y-direction, line
                for i in range(self.npoints):   # x-direction, column
                    # iter over 0,...,npoints - 1
                    if i == 0:
                        x_grad[g, i] = (thisField[g, 1] - thisField[g, 0]) / self.dx
                    elif i == self.npoints - 1:
                        x_grad[g, i] = (thisField[g, i] - thisField[g,i - 1]) / self.dx
                    else:
                        x_grad[g, i] = (thisField[g, i + 1] - thisField[g, i - 1]) / (2 * self.dx)

            # y-direction
            for g in range(self.npoints):   # y-direction, line
                for i in range(self.npoints):   # x-direction, column
                    # iter over 0,...,npoints - 1
                    if g == 0:
                        y_grad[g, i] = (thisField[1,i] - thisField[0,i]) / self.dx
                    elif g == self.npoints - 1:
                        y_grad[g, i] = (thisField[g, i] - thisField[g - 1,i]) / self.dx
                    else:
                        y_grad[g, i] = (thisField[g+1, i] - thisField[g - 1, i]) / (2 * self.dx)

            # now reshape and store
            self.gradPhi[:,f,0] = x_grad.reshape(self.npoints * self.npoints)
            self.gradPhi[:,f,1] = y_grad.reshape(self.npoints * self.npoints)

            # plot for testing and evaluation, remove...
            # plt.imshow(self.gradPhi[:,f,0].reshape(self.npoints,self.npoints))
            # plt.show(block=False)
            # plt.imshow(y_grad)
            # plt.show(block=False)

    @jit(parallel=True)
    def getIEM(self):
        # compute at first the Eddy turn over time: Teddy
        # -> check if sqrt is needed!!
        T_eddy = (self.dx**2 + self.dy**2) /(2*(self.Dt + self.D))

        # compute IEM for each field:
        for f in range(self.fields):
            self.IEM[:,f] = - (self.Phi_fields_2d[:,f] - self.Phi_2d_vec) * (self.dt / (2*T_eddy))

    # plotting
    def plot_3D_STD(self,org = False):

        # Plots nebeneinander!

        Phi_plot = self.Phi_STD.reshape(self.npoints,self.npoints)

        fig = plt.figure()
        ax = fig.gca(projection='3d')

        X, Y = np.meshgrid(self.grid,self.grid)

        if org:
            # Plot the surface.
            surf = ax.plot_surface(X, Y, Phi_plot, cmap='inferno',
                                   linewidth=0, antialiased=False)
        else:
            # Plot the surface.
            surf = ax.plot_surface(X, Y, Phi_plot, cmap='inferno',
                                   linewidth=0, antialiased=False)

        # Customize the z axis.
        ax.set_zlim(0, 1)
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

        # Add a color bar which maps values to colors.
        cbar = plt.colorbar(surf, shrink=0.5, aspect=5)
        #cbar.ax.set_xticklabels(['0', '0.25', '0.5','0.75','1.0'])

        cbar.set_label('STD')
        cbar.set_clim(0,1)
        plt.title('STD from the fields')
        #plt.show(block=False)

    # plotting
    def plot_3D_Field(self,field = 0):

        # Plots nebeneinander!

        fig = plt.figure()
        ax = fig.gca(projection='3d')

        X, Y = np.meshgrid(self.grid,self.grid)

        phi_plot = self.Phi_fields_2d[:, field].reshape(self.npoints, self.npoints)

        # Plot the surface.
        surf = ax.plot_surface(X, Y, phi_plot, cmap='inferno', linewidth=0, antialiased=False)
        cset = ax.contour(X, Y, phi_plot, zdir='z',  cmap=cm.coolwarm)
        cset = ax.contour(X, Y, phi_plot, zdir='x',  cmap=cm.coolwarm)
        cset = ax.contour(X, Y, phi_plot, zdir='y',  cmap=cm.coolwarm)

        # Customize the z axis.
        ax.set_zlim(0, 1)
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

        # Add a color bar which maps values to colors.
        cbar = plt.colorbar(surf, shrink=0.5, aspect=5)
        # cbar.ax.set_xticklabels(['0', '0.25', '0.5','0.75','1.0'])

        cbar.set_label('Concentration')
        cbar.set_clim(0, 1)
        #plt.show(block=False)

    # plotting of the conditional distributions
    def plot_conditional_fields(self, x=0.5, Diffusion=None, legend=False):

        plt.figure()
        x_cond = int(self.npoints*x)

        for f in range(self.fields):
            phi_field_cond = self.Phi_fields_2d[:, f].reshape(self.npoints, self.npoints)[x_cond, :]
            field, = plt.plot(self.grid, phi_field_cond, 'k--', label='stochastic field', lw=0.7, alpha=0.75)

        phi_cond = self.Phi_2d_vec.reshape(self.npoints, self.npoints)[x_cond, :]
        phi_std = self.Phi_fields_2d.std(axis=1).reshape(self.npoints,self.npoints)[x_cond,:]

        try:
            phi_pure_diff = Diffusion.Phi_2d_vec.reshape(self.npoints,self.npoints)[x_cond,:]

            phi_diff, = plt.plot(self.grid, phi_pure_diff, 'b-', label='pure diffusion', lw=1.5)
            mean, = plt.plot(self.grid,phi_cond,'r-', label='filtered value',lw=2)
            std, = plt.plot(self.grid,phi_std,'r--',label='STD fields',lw=1)

            if legend:
                plt.legend(handles=[field,mean,phi_diff,std],prop={'size':9})
            plt.title('Conditional distribution $p(x|y=0.5)$: tsteps=%s nFields=%s' % (str(self.tsteps),str(self.fields)))
            plt.ylabel('Scalar concentration [-]')
            plt.xlabel('x')

            #plt.show(block=False)

        except:
            mean, = plt.plot(self.grid,phi_cond,'r-', label='filtered value',lw=2)
            std, = plt.plot(self.grid, phi_std, 'r--', label='STD fields',lw=1)

            if legend:
                plt.legend(handles=[field,mean,std],prop={'size':9})
            plt.title('Conditional distribution $p(x|y=0.5)$: tsteps=%s, nFields=%s' % (str(self.tsteps),str(self.fields)))
            plt.ylabel('Scalar concentration [-]')
            plt.xlabel('x')

            #plt.show(block=False)


###################################################################
class StochasticDiffusion_2d(StochasticDiffusion_2d_ABC):
    def __init__(self,params,BC='Neumann'):
        super().__init__(params,BC='Neumann')
        '''
        This is the stochastic diffusion class for 2D simulation 
        inherited from StochasticDiffusion_2d
        '''

###################################################################
class StochasticDiffusion_2d_normal(StochasticDiffusion_2d_ABC):
    def __init__(self,params,BC='Neumann'):
        super().__init__(params,BC='Neumann')
        '''
        This is with normal wiener term and not dichtomic
        '''

    # This is with normal distributed gamma ~ N(0,1)
    def dWiener(self):
        # compute the Wiener Term, it has to be 2 dimensional now
        # initialize gamma vector
        gamma = np.random.normal(0,1,[self.fields,2])

        for d in range(gamma.shape[1]):
            #gamma[0:int(self.fields / 2),d] = -1
            # shuffle it
            np.random.shuffle(gamma[:,d])
            self.dW[:,d] = gamma[:,d] * np.sqrt(self.dt)


###################################################################
class StochasticDiffusion_2d_noD(StochasticDiffusion_2d_ABC):
    def __init__(self, params, BC='Neumann'):
        super().__init__(params, BC='Neumann')
        '''
        This is with normal wiener term and not dichtomic
        '''

    @jit(parallel=True)
    def getIEM(self):
        # compute at first the Eddy turn over time: Teddy
        # -> check if sqrt is needed!!
        T_eddy = (self.dx**2+ self.dy**2) /(2*(self.Dt))

        # compute IEM for each field:
        for f in range(self.fields):
            self.IEM[:,f] = - (self.Phi_fields_2d[:,f] - self.Phi_2d_vec) * (self.dt / (2*T_eddy))

###################################################################
class StochasticDiffusion_2d_oldPhi(StochasticDiffusion_2d_ABC):
    def __init__(self,params,BC='Neumann'):
        super().__init__(params,BC='Neumann')
        '''
        This is the stochastic diffusion test class for 2D simulation 
        some functions are overwritten
        '''

        self.peak =None

    def startStochasticDiffusion(self, tsteps = params.tsteps, IEM_on = False):
        '''
            Start from 0 to advance the stochastic diffusion process
        '''
        self.tsteps = tsteps

        # this is to store the location
        self.peak = np.zeros(tsteps)

        # choose for IEM
        self.IEM_on = IEM_on

        #first update the diffusion matrix (implicit), in case dt, D, Dt have changed
        self.setDiffMatrix()

        for i in range(0, self.tsteps):

            if i%100 == 0:
                print(i)

            if i == 0:

                # this is the first time step, assuming all fields are the same,
                # so fill them:
                self.initFields()

                # updateing diffusion equation
                # 1. part of fractional step
                self.pureDiffusion()

                # 2. part of fractional step, add the stochastic velocity
                self.addStochastic1(i)

            else:
                # 1 part of fractional step
                self.pureDiffusion()

                # 2. part of fractional step, add the stochastic velocity
                self.addStochastic1(i)

    # has to overwritten as the function needs to do matrix multiplication for each field separately!
    def pureDiffusion(self):
        # 1D diffusion equation
        try:
            self.Phi_2d_vec = np.matmul(self.A,self.Phi_2d_vec)
            if self.BC == 'Dirichlet':
                print('Dont use Dirichlet')
                # update the boundary values -> so that gradient is zero at boundary!
                # self.Phi_2d_vec[0,0] = self.Phi[1,0]
                # self.Phi_2d[-1,0] = self.Phi[-2,0]

        except AttributeError:
            print('Set first the Diffusion Matrix!\nThis is now done for you...')
            self.setDiffMatrix()

    # overwrite
    @jit
    def addStochastic1(self,i):
        # the Wiener element is between 1,..,nFields of Wiener term
        # this is done in an explicit manner!

        # compute Wiener term
        self.dWienerCum(i)

        # compute the gradient of Phi_star for each field separately
        self.getGradients()

        # now compute Phi for each field and loop over points
        for f in range(self.fields):
            # gaussian amplification
            #amp = np.random.normal(1,1) * np.sqrt(self.dt)

            self.Phi_fields_2d[:, f] = \
                self.Phi_2d_vec + np.sqrt( 2*self.Dt) * np.dot(self.gradPhi[:, f, :], (self.dW[f, :]))

            #print((np.sqrt(2 * self.Dt) *np.dot(self.gradPhi[:, f, :], (self.dW[f, :]))).max())

        self.getIEM()
        #self.Phi_fields_2d = self.Phi_fields_2d + self.IEM
        # get location of peak in field 1
        #self.peak[i] = self.Phi_fields_2d[:, 1].argmax()

        # finally get the average over all fields -> new Phi, though it is never used for further computation
        self.Phi_2d_vec = self.Phi_fields_2d.mean(axis=1)

        #**************************************
        # additional step if IEM_on = True
        # **************************************
        if self.IEM_on:
            print('No IEM possible here')

        # get the RMS of the fields
        self.Phi_STD = self.Phi_fields_2d.std(axis=1)

    # def getIEM(self):
    #     print('Print not implemented here!')

    @jit
    def getIEM_negative(self):
        # compute at first the Eddy turn over time: Teddy
        # -> check if sqrt is needed!!
        T_eddy = (self.dx**2 + self.dy**2) /(2*(self.Dt+self.D))

        # compute IEM for each field:
        for f in range(self.fields):
            self.IEM[:,f] = (self.Phi_fields_2d[:,f] - self.Phi_2d_vec) * (self.dt / T_eddy)

    # These are the functions for Stochastic Fields then
    def dWienerCum(self,i):
        # compute the Wiener Term, it has to be 2 dimensional now
        # initialize gamma vector
        gamma = np.ones((self.fields,2))

        for d in range(gamma.shape[1]):
            gamma[0:int(self.fields / 2),d] = -1
            # shuffle it
            np.random.shuffle(gamma[:,d])
            self.dW[:,d] = (gamma[:,d] * np.sqrt(self.dt))
    #
    # def plotPeak(self):
    #
    #     empty_vec = np.zeros(self.npoints**2)
    #
    #     for t in range(self.tsteps):
    #         pos = self.peak[t]
    #         empty_vec[pos] = 1
    #
    #     plt.imshow(empty_vec.reshape(self.npoints,self.npoints))
    #     plt.show()



###################################################################
class StochasticDiffusion_2d_Langevine(StochasticDiffusion_2d_ABC):
    def __init__(self, params, BC='Neumann', d_0=1.0):
        super().__init__(params, BC='Neumann')
        '''
        This is with normal wiener term and not dichtomic
        '''

        # these are needed for the langevine model
        self.d_0=d_0
        self.a=self.Phi_fields_2d.copy() * 1e-15
        self.b=self.Phi_fields_2d.copy() * 1e-15
        self.sig_m = self.Phi_fields_2d.copy() * 1e-15
        self.sig = self.Phi_fields_2d.copy() * 1e-15

    @jit(parallel=True)
    def addStochastic(self):
        # the Wiener element is between 1,..,nFields of Wiener term
        # this is done in an explicit manner!

        # compute Wiener term
        self.dWiener()

        # compute the gradient of Phi_star for each field separately
        self.getGradients()

        # now compute Phi for each field and loop over points
        for f in range(self.fields):
            for p in range(self.npoints**2):
                self.Phi_fields_2d[p, f] = \
                    self.Phi_fields_2d_star[p, f] + np.sqrt(2*self.Dt) * np.dot(self.gradPhi[p, f, :],self.dW[f, :])

        # finally get the average over all fields -> new Phi, though it is never used for further computation
        self.Phi_2d_vec = self.Phi_fields_2d.mean(axis=1)

        # *************************************
        # additional step if IEM_on = True
        # *************************************
        if self.IEM_on:
            # compute IEM terms
            self.getIEM()
            # at this stage self.Phi is already the averaged field, so use it for further computation
            for f in range(self.fields):
                self.Phi_fields_2d[:, f] = self.Phi_fields_2d[:, f] + self.IEM[:, f]

            # finally get the average over all fields -> new Phi, though it is never used for further computation
            self.Phi_2d_vec = self.Phi_fields_2d.mean(axis=1)

        # get the RMS of the fields
        self.Phi_STD = self.Phi_fields_2d.std(axis=1)

    def startStochasticDiffusion(self, tsteps = params.tsteps, IEM_on = False):
        '''
            Start from 0 to advance the stochastic diffusion process
        '''
        self.tsteps = tsteps

        self.Phi_2d_vec = self.Phi_2d_org.reshape(self.npoints * self.npoints)

        # choose for IEM
        # note used here
        self.IEM_on = False

        #first update the diffusion matrix (implicit), in case dt, D, Dt have changed
        self.setDiffMatrix()

        for i in range(0, self.tsteps):

            if i%100 == 0:
                print(i)

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

    @jit(parallel=True)
    def get_a(self):
        pass

    @jit(parallel=True)
    def get_b(self):
        pass

    @jit(parallel=True)
    def getIEM(self):
        print('not implemented here!')

    @jit(parallel=True)
    def getLangevine(self):
        print('to do ...')

##########################################
# to run the Simulation

myParams = params()
myParams.fields = 8

#myParams.Dt = 1e-15

time_steps=200

Diff = Diffusion_2d(myParams)
Diff.advanceDiffusion(time_steps)
#Diff.plot_3D()
#Diff.imshow_Phi()
#Diff.plot_3D(org=True)
#Diff.imshow_Phi(org=True)

# # IEM Off
# Stoch2 = StochasticDiffusion_2d(myParams)
# Stoch2.startStochasticDiffusion(time_steps,IEM_on=False)
# Stoch2.plot_3D()
# Stoch2.plot_3D_STD()
# Stoch2.plot_conditional_fields(Diffusion=Diff,legend=True)
#
 # IEM On
Stoch_on = StochasticDiffusion_2d(myParams)
Stoch_on.startStochasticDiffusion(time_steps,IEM_on=True)
# Stoch_on.plot_3D()
# Stoch_on.plot_3D_STD()
Stoch_on.plot_conditional_fields(Diffusion=Diff,legend=True)
# plt.savefig('Figures/Phi_IEM_%s_steps%s_Dt%s_%sfields.png' % (str(Stoch_on.tsteps), str(Stoch_on.tsteps), str(Stoch_on.Dt), str(Stoch_on.fields)))
# plt.savefig('Figures/Phi_IEM_%s_steps%s_Dt%s_%sfields.eps' % (str(Stoch_on.tsteps), str(Stoch_on.tsteps), str(Stoch_on.Dt), str(Stoch_on.fields)))
plt.show()

#
# # # IEM On
# Stoch_noD = StochasticDiffusion_2d_noD(myParams)
# Stoch_noD.startStochasticDiffusion(time_steps,IEM_on=True)
# # Stoch_on.plot_3D()
# # Stoch_on.plot_3D_STD()
# Stoch_noD.plot_conditional_fields(Diffusion=Diff)
# plt.savefig('Figures/Phi_noD_%s_steps%s_Dt%s_%sfields.png' % (str(Stoch_noD.tsteps), str(Stoch_noD.tsteps), str(Stoch_noD.Dt), str(Stoch_noD.fields)))
# plt.savefig('Figures/Phi_noD_%s_steps%s_Dt%s_%sfields.eps' % (str(Stoch_noD.tsteps), str(Stoch_noD.tsteps), str(Stoch_noD.Dt), str(Stoch_noD.fields)))
# plt.show()
#
# Normal distribution
# Stoch_norm = StochasticDiffusion_2d_normal(myParams)
# Stoch_norm.startStochasticDiffusion(time_steps,IEM_on=True)
# # Stoch_on.plot_3D()
# # Stoch_on.plot_3D_STD()
# Stoch_norm.plot_conditional_fields(Diffusion=Diff)
#plt.savefig('Figures/Phi_norm_%s_steps%s_Dt%s_%sfields.png' % (str(Stoch_norm.tsteps), str(Stoch_norm.tsteps), str(Stoch_norm.Dt), str(Stoch_norm.fields)))
#plt.savefig('Figures/Phi_norm_%s_steps%s_Dt%s_%sfields.eps' % (str(Stoch_norm.tsteps), str(Stoch_norm.tsteps), str(Stoch_norm.Dt), str(Stoch_norm.fields)))

# plt.show()
# #
# Diff.Dt = 1e-15
# Diff.advanceDiffusion(400)
#
# Stoch_on.Dt = 1e-15
# Stoch_on.continueStochasticDiffusion(200)
# Stoch_on.plot_conditional_fields(Diffusion=Diff)
# plt.show()
#
# Stoch_noD.Dt = 1e-15
# Stoch_noD.continueStochasticDiffusion(200)
# Stoch_noD.plot_conditional_fields(Diffusion=Diff)
# plt.show()

# Stoch_oldPhi = StochasticDiffusion_2d_oldPhi(myParams)
# Stoch_oldPhi.startStochasticDiffusion(200)
# Stoch_oldPhi.plot_conditional_fields(Diffusion=Diff)
# plt.show()

import copy

Diff_org = copy.copy(Diff)
Stoch_org = copy.copy(Stoch_on)

#Diff = Diffusion_2d(myParams)
Diff.Dt = 1e-15
Diff.continueDiffusion(200)

Stoch_on.Dt = 1e-15
Stoch_on.continueStochasticDiffusion(200)

Stoch_on.plot_conditional_fields(Diffusion=Diff,legend=True)

plt.show()