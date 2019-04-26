from diffusion_2d import *


import time
from multiprocessing import Process

import matplotlib.pyplot as plt
plt.close('all')

myParams = params
myParams.dt = 0.05

myParams.fields = 8
myParams.npoints = 100
myParams.computeGrid()

#myParams.Dt = 1e-15

time_steps=500
myParams.tsteps = time_steps

nBlocks = 5

BC = 'Dirichlet'

Diff2 = Diffusion_2d(myParams,BC=BC)

#Diff2.setDictionary()
#Diff2.manyBlockFunction(Diff2.grid,nBlocks=nBlocks)

Diff2.advanceDiffusion(time_steps)


# IEM On
# start2 = time.time()
IEM = StochasticDiffusion_2d(myParams,BC=BC)
#IEM.manyBlockFunction(Diff2.grid,nBlocks=nBlocks)
IEM.startStochasticDiffusion(time_steps,IEM_on=True)

IEM.plot_conditional_fields(Diffusion=Diff2,legend=True)
#IEM.plot_conditional_error(Diffusion=Diff2)
plt.show(block=False)

print(' ')
print('Now computing Stochastic Diffusion with SPMM model')

SPMM = StochasticDiffusion_2d_SPMM_simple(myParams,BC=BC)
#SPMM.manyBlockFunction(Diff2.grid,nBlocks=nBlocks)
SPMM.startStochasticDiffusion(time_steps,SPMM_on=True)
SPMM.plot_conditional_fields(Diffusion=Diff2, legend=True)
plt.show(block=False)

# SPMM_off = StochasticDiffusion_2d_SPMM_simple(myParams,BC=BC)
# #SPMM.manyBlockFunction(Diff2.grid,nBlocks=nBlocks)
# SPMM_off.startStochasticDiffusion(time_steps,SPMM_on=False)
# SPMM_off.plot_conditional_fields(Diffusion=Diff2, legend=True)

#SPMM.plot_conditional_error(Diffusion=Diff2)
#SPMM.imshow_error(Diffusion=Diff2)

# SPMM2 = StochasticDiffusion_2d_SPMM_full(myParams,BC=BC)
# #SPMM.manyBlockFunction(Diff2.grid,nBlocks=nBlocks)
# SPMM2.startStochasticDiffusion(time_steps,SPMM_on=True)
# SPMM2.plot_conditional_fields(Diffusion=Diff2, legend=True)

plt.show(block=False)

#plt.show(block=False)
#
#
# # plot SPMMine and IEM components
# SPMM = SPMM.SPMM[:,1].reshape(myParams.npoints,myParams.npoints)
# iem = SPMM.IEM[:,1].reshape(myParams.npoints,myParams.npoints)
#
# plt.figure()
# plt.plot(SPMM[:,int(myParams.npoints/2)])
# plt.plot(iem[:,int(myParams.npoints/2)])
# plt.legend(['BLM','IEM'])
# plt.ylabel('Scalar concentration [-]')
# plt.xlabel('x')
# plt.title('Compare SPMMine and IEM at $p(x|y=0.5)$')
#
# # plot conditional error
# plt.figure()
# plt.plot(SPMM.Phi_error[:,int(myParams.npoints/2)])
# plt.plot(IEM.Phi_error[:,int(myParams.npoints/2)])
# plt.legend(['BLM','IEM'])
# plt.title('Conditional error $p(x|y=0.5)$: tsteps=%s, nFields=%s' % (str(myParams.tsteps), str(myParams.fields)))
# plt.ylabel('Scalar concentration [-]')
# plt.xlabel('x')
#
# # plot different STD
# plt.figure()
# plt.plot(SPMM.Phi_std)
# plt.plot(IEM.Phi_std)
# plt.legend(['BLM','IEM'])
# plt.title('Conditional STD $p(x|y=0.5)$: tsteps=%s, nFields=%s' % (str(myParams.tsteps), str(myParams.fields)))
# plt.ylabel('Scalar concentration [-]')
# plt.xlabel('x')

plt.show(block=False)