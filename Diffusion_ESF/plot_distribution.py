import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import os
from scipy.stats import gaussian_kde


# this is for latex in the plot labels and titles
plt.rc('text', usetex=True)
plt.rc('font', family='serif')


n_points =200

x = np.linspace(0,1,n_points)
sigma = 0.15
mu = 0.5

gaussian_hist = np.array([np.random.normal(mu,sigma)*n_points for f in range(n_points)])

#gaussian = np.zeros((len(x),1))
gaussian = np.exp(-np.power(x - mu, 2.) / (2 * np.power(sigma, 2.)))

# normalization
gaussian = gaussian/sum(gaussian)

kde = gaussian_kde(gaussian_hist)
gaussian_fit = kde.evaluate(x)

plt.figure()
plt.hist(gaussian_hist,normed=True,bins=20)
#plt.plot(gaussian,'r',lw = 2)
plt.xticks([0,n_points/4,n_points/2,n_points*3/4,n_points])
# plt.xti(['0','0.25','0.5','0.75','1.0'])

plt.show()
