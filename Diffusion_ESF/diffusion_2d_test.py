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

time_steps=500

Diff = Diffusion_2d(myParams)
Diff.advanceDiffusion(time_steps)

# start1 = time.time()
print('Now computing Stochastic Diffusion with Langevine model')

Langev = StochasticDiffusion_2d_Langevine(myParams,d_0=1)
#Langev = StochasticDiffusion_2d(myParams)

p1 = Process(target=Langev.startStochasticDiffusion(time_steps,IEM_on=False))


# Langev.startStochasticDiffusion(time_steps)
# Langev.plot_conditional_fields(Diffusion=Diff,legend=True)
# end1 = time.time()
# print('Time to compute LANGEV: ', end1-start1)


# IEM On
# start2 = time.time()
Stoch_on = StochasticDiffusion_2d(myParams)
# Stoch_on.startStochasticDiffusion(time_steps,IEM_on=True)

p2 = Process(target = Stoch_on.startStochasticDiffusion(time_steps,IEM_on=False))

p1.start()
p2.start()

p1.join()
p2.join()

Langev.plot_conditional_fields(Diffusion=Diff,legend=True)
Stoch_on.plot_conditional_fields(Diffusion=Diff,legend=True)
# plt.savefig('Figures/Phi_IEM_%s_steps%s_Dt%s_%sfields.png' % (str(Stoch_on.tsteps), str(Stoch_on.tsteps), str(Stoch_on.Dt), str(Stoch_on.fields)))
# plt.savefig('Figures/Phi_IEM_%s_steps%s_Dt%s_%sfields.eps' % (str(Stoch_on.tsteps), str(Stoch_on.tsteps), str(Stoch_on.Dt), str(Stoch_on.fields)))

# end2 = time.time()


plt.show(block=False)