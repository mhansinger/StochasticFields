# A very simple numerical implementation of the Black-Scholes Model

# dS = \mu*s*dt + \sigma*S*dW

import numpy as np
import matplotlib.pyplot as plt

# time
T = 2

# start price
S0 = 20
dt = 0.01
N = round(T/dt)
t = np.linspace(0, T, N)

def run_samples(runs=20,mu=0.09,sigma=0.05):

    for i in range(runs):
        W = np.random.standard_normal(size=N)
        W = np.cumsum(W) * np.sqrt(dt)  # standard brownian motion
        X = (mu - 0.5 * sigma ** 2) * t + sigma * W
        S = S0 * np.exp(X)              # geometric brownian motion
        plt.plot(t, S)
        plt.title('Price development with mu = %s and sigma = %s' % (str(mu),str(sigma)))

    plt.show()