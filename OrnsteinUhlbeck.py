#!/bin/python3

import numpy as np
import matplotlib.pyplot as plt
t_0 = 0
# define model parameters
t_end = 2
length = 1000
theta  = 1.1
mu     = 0.5
sigma  = 0.3

t = np.linspace(t_0,t_end,length)

# define time axis
dt = np.mean(np.diff(t))
y = np.zeros(length)
y0 = np.random.normal(loc=0.0,scale=1.0)

# initial condition
drift = lambda y,t: theta*(mu-y)

# define drift term
diffusion = lambda y,t: sigma

# define diffusion term
noise = np.random.normal(loc=0.0,scale=1.0,size=length)*np.sqrt(dt)

#define noise process
# solve SDE
for j in range(1,20):
    # define diffusion term
    noise = np.random.normal(loc=0.0, scale=1.0, size=length) * np.sqrt(dt)
    for i in range(1,length):
        y[i] = y[i-1] + drift(y[i-1],i*dt)*dt + diffusion(y[i-1],i*dt)*noise[i]
    plt.plot(t,y)

plt.show()
