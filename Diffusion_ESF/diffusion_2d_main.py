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

time_steps=20
myParams.tsteps = time_steps

nBlocks = 5

BC = 'Dirichlet'

# Diff = Diffusion_2d(myParams,BC='Neumann')
# Diff.stepFunction(Diff.grid)
# Diff.imshow_Phi()
# Diff.advanceDiffusion(time_steps)
# Diff.imshow_Phi()

Diff2 = Diffusion_2d(myParams,BC=BC)

#Diff2.setDictionary()
Diff2.manyBlockFunction(Diff2.grid,nBlocks=nBlocks)
# Diff2.imshow_Phi()
# Diff2.plot_3D()
Diff2.advanceDiffusion(time_steps)
#Diff2.imshow_Phi()
#plt.show(block=False)

# start1 = time.time()

# IEM On
# start2 = time.time()
Stoch_on = StochasticDiffusion_2d(myParams,BC=BC)
Stoch_on.manyBlockFunction(Diff2.grid,nBlocks=nBlocks)
#Stoch_on.imshow_Phi()
Stoch_on.startStochasticDiffusion(time_steps,IEM_on=True)

Stoch_on.plot_conditional_fields(Diffusion=Diff2,legend=True)
Stoch_on.plot_conditional_error(Diffusion=Diff2)

print(' ')
print('Now computing Stochastic Diffusion with Langevine model')

Langev = StochasticDiffusion_2d_Langevine(myParams,BC=BC,d_0=1)
Langev.manyBlockFunction(Diff2.grid,nBlocks=nBlocks)
Langev.startStochasticDiffusion(time_steps,IEM_on=True)
Langev.plot_conditional_fields(Diffusion=Diff2, legend=True)
Langev.plot_conditional_error(Diffusion=Diff2)
#Langev.imshow_error(Diffusion=Diff2)
plt.show(block=False)

# Stoch_on.plot_conditional_fields(Diffusion=Diff,legend=True)
# plt.savefig('Figures/Phi_IEM_%s_steps%s_Dt%s_%sfields.png' % (str(Stoch_on.tsteps), str(Stoch_on.tsteps), str(Stoch_on.Dt), str(Stoch_on.fields)))
# plt.savefig('Figures/Phi_IEM_%s_steps%s_Dt%s_%sfields.eps' % (str(Stoch_on.tsteps), str(Stoch_on.tsteps), str(Stoch_on.Dt), str(Stoch_on.fields)))

#Diff2.plot_3D()
#Stoch_on.imshow_Phi()

#plt.show(block=False)


# plot Langevine and IEM components
langev = Langev.LANGEV[:,1].reshape(myParams.npoints,myParams.npoints)
iem = Langev.IEM[:,1].reshape(myParams.npoints,myParams.npoints)

plt.figure()
plt.plot(langev[:,int(myParams.npoints/2)])
plt.plot(iem[:,int(myParams.npoints/2)])
plt.legend(['BLM','IEM'])
plt.ylabel('Scalar concentration [-]')
plt.xlabel('x')
plt.title('Compare Langevine and IEM at $p(x|y=0.5)$')

# plot conditional error
plt.figure()
plt.plot(Langev.Phi_error[:,int(myParams.npoints/2)])
plt.plot(Stoch_on.Phi_error[:,int(myParams.npoints/2)])
plt.legend(['BLM','IEM'])
plt.title('Conditional error $p(x|y=0.5)$: tsteps=%s, nFields=%s' % (str(myParams.tsteps), str(myParams.fields)))
plt.ylabel('Scalar concentration [-]')
plt.xlabel('x')

# plot different STD
plt.figure()
plt.plot(Langev.Phi_std)
plt.plot(Stoch_on.Phi_std)
plt.legend(['BLM','IEM'])
plt.title('Conditional STD $p(x|y=0.5)$: tsteps=%s, nFields=%s' % (str(myParams.tsteps), str(myParams.fields)))
plt.ylabel('Scalar concentration [-]')
plt.xlabel('x')

plt.show(block=False)