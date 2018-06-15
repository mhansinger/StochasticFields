# Author: Jake Vanderplas <jakevdp@cs.washington.edu>
#
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import norm
from sklearn.neighbors import KernelDensity


#this is for latex in the plot labels and titles
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('font', size=17)

# mpl.rcParams['text.usetex']=True
# mpl.rcParams['text.latex.unicode']=True

# Plot a 1D density example
N = 2000
mu1 = 0.7
sig1 = 0.1

np.random.seed(42)
X = (np.random.normal(mu1, sig1, int(1 * N)))

X_plot = np.linspace(0, 1, 1000)[:, np.newaxis]

true_dens = (norm(mu1, sig1).pdf(X_plot[:, 0]))

# transformation of pdfs
X_square = X**3*np.sin(X)
np.nan_to_num(X_square,copy=True)
true_dens_square = X_plot**2 * true_dens

############################
# plots
fig, ax = plt.subplots()
#ax.plot(X_plot[:, 0], true_dens, 'k--', lw=2, label='fitted gaussian')

ax.hist(X,normed=True,bins=30,alpha=0.8,label='histogram',edgecolor='black')
# ax.vlines(0.7, 4.11,0, lw=7, color='r',label=r'$\overline{f}$')
#
ax.set_xlim(0.35,1.05)
ax.set_ylim(0,4.5)
ax.legend(loc='upper left')
plt.ylabel('Frequency of f')
plt.xlabel('f')
plt.title(r'Distribution of scalar f with $\overline{f} = %s $' % str(mu1))
plt.savefig('dist_nobars.png')
plt.show(block=False)


# plots
fig2, ax2 = plt.subplots()

mu_2 = round(X_square.mean(),3)

ax2.hist(X_square,normed=True,bins=30,alpha=0.5,label='histogram',edgecolor='black')
#
ax2.set_xlim(0,1)
ax2.set_ylim(0,3.7)

ax2.vlines(mu1**3, 1.6,0, lw=7, color='r',label=r'$(\overline{f})^3$')
#ax2.vlines(mu_2, 3.3,0, lw=7, color='lime',label=r'$\overline{f^3}$')

plt.ylabel(r'Frequency of $f^3$')
plt.xlabel(r'f^3')
plt.title(r'Distribution of scalar $f^3$')
ax2.legend(loc='upper right')
plt.savefig('dist_3_onebars.png')
plt.show(block=False)