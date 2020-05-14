from diffusion_2d import *


import time
from multiprocessing import Process

import matplotlib.pyplot as plt
plt.close('all')

myParams = params
myParams.dt = 0.0001

myParams.D = 0.001
myParams.Dt = 0.005

myParams.fields = 16
myParams.npoints = 100
myParams.computeGrid()

#myParams.Dt = 1e-15

time_steps=200
myParams.tsteps = time_steps

nBlocks = 5

BC = 'Dirichlet'

Pe = (myParams.D+myParams.Dt)*myParams.dt / (myParams.dx**2)
print("Pe is: ", round(Pe,4))
#%%
# GAUSSIAN WITHOUT STOCHASTIC

Diff2 = Diffusion_2d(myParams,BC=BC)
#Diff2.gaussianDist(Diff2.grid,0.5,0.05)
# Diff2.imshow_Phi()
Diff2.advanceDiffusion(time_steps)
# Diff2.imshow_Phi()
#Diff2.plot_1d(org=True)

#%%
ESF_IEM = StochasticDiffusion_2d(myParams,BC=BC)
ESF_IEM.startStochasticDiffusion(time_steps,IEM_on=True,ESF_old=False)

# ESF_IEM_off = StochasticDiffusion_2d(myParams,BC=BC)
# ESF_IEM_off.startStochasticDiffusion(time_steps,IEM_on=False,ESF_old=False)

ESF_old_IEM = StochasticDiffusion_2d(myParams,BC=BC)
ESF_old_IEM.startStochasticDiffusion(time_steps,IEM_on=True,ESF_old=True)
#
# ESF_old_IEM_off = StochasticDiffusion_2d(myParams,BC=BC)
# ESF_old_IEM_off.startStochasticDiffusion(time_steps,IEM_on=False,ESF_old=True)

#%%
ESF_IEM.plot_1d_fields(Diff2.Phi_2d,org=True)
ESF_old_IEM.plot_1d_fields(Diff2.Phi_2d,org=True)

# ESF_IEM_off.plot_1d_fields(Diff2.Phi_2d,org=True)
# ESF_old_IEM_off.plot_1d_fields(Diff2.Phi_2d,org=True)