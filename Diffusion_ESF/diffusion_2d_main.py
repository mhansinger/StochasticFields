from diffusion_2d import *

import time
from multiprocessing import Process

import matplotlib.pyplot as plt

myParams = params
myParams.dt = 0.005

myParams.fields = 16
myParams.npoints = 100
myParams.computeGrid()

#myParams.Dt = 1e-15

time_steps=300

# Diff = Diffusion_2d(myParams,BC='Neumann')
# Diff.stepFunction(Diff.grid)
# Diff.imshow_Phi()
# Diff.advanceDiffusion(time_steps)
# Diff.imshow_Phi()

Diff2 = Diffusion_2d(myParams,BC='Dirichlet')
#Diff2.setDictionary()
Diff2.manyBlockFunction(Diff2.grid,nBlocks=9)
Diff2.imshow_Phi()
Diff2.advanceDiffusion(time_steps)
Diff2.imshow_Phi()

# start1 = time.time()
print('Now computing Stochastic Diffusion with Langevine model')

#Langev = StochasticDiffusion_2d_Langevine(myParams,d_0=1)
#Langev.startStochasticDiffusion(time_steps,IEM_on=True)

# IEM On
# start2 = time.time()
#Stoch_on = StochasticDiffusion_2d(myParams)
#Stoch_on.startStochasticDiffusion(time_steps,IEM_on=True)

#Langev.plot_conditional_fields(Diffusion=Diff,legend=True)
#Stoch_on.plot_conditional_fields(Diffusion=Diff,legend=True)
# plt.savefig('Figures/Phi_IEM_%s_steps%s_Dt%s_%sfields.png' % (str(Stoch_on.tsteps), str(Stoch_on.tsteps), str(Stoch_on.Dt), str(Stoch_on.fields)))
# plt.savefig('Figures/Phi_IEM_%s_steps%s_Dt%s_%sfields.eps' % (str(Stoch_on.tsteps), str(Stoch_on.tsteps), str(Stoch_on.Dt), str(Stoch_on.fields)))

Diff2.plot_3D()

plt.show(block=False)