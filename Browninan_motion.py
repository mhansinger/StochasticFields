
import numpy as np
import matplotlib.pyplot as plt
import tikzplotlib

t = 2
n = 1000

t_vec = np.linspace(0,t,n+1)
delta_t = np.sqrt(t / n)

paths=10

dichotomic = np.ones((n,paths))
dichotomic[:,int(paths/2)-1:-1] = -1

#shuffle independently
for ni in range(n):
    np.random.shuffle(dichotomic[ni,:])


# Create each dW step
dW = delta_t * np.random.normal(0, 1, size=(n,paths))
dW_dichotomic = delta_t * dichotomic

W = np.zeros([n + 1,paths])
W_dichotomic = np.zeros([n + 1,paths])

# Add W_{j-1} + dW_{j-1}
W[1:,:] = np.cumsum(dW,axis=0)
W_dichotomic[1:,:] = np.cumsum(dW_dichotomic,axis=0)

W_mean = np.mean(W,axis=1)
W_d_mean = np.mean(W_dichotomic,axis=1)

#%%
#DICHOTOMIC
plt.plot(t_vec,W_dichotomic,'grey',lw=0.8)
plt.plot(t_vec,W_d_mean,'k',lw=3)
plt.xlim(0,t)
plt.xlabel('$t$')
plt.ylabel('$\textbf{S}_n$')
ax = plt.gca()
#ax.set_aspect('equal')
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
ax.axis('off')
ax.margins(0, 0)
ax.xaxis.set_major_locator(plt.NullLocator())
ax.yaxis.set_major_locator(plt.NullLocator())
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.savefig('plots/brownian_paths_%i_dichotomic.pdf' % paths)
# #tikzplotlib.save('plots/brownian_paths_dichotomic%i.tex' % n)
plt.show()


#%%
#NORMAL
plt.plot(t_vec,W,'grey',lw=0.8)
plt.plot(t_vec,W_mean,'k',lw=3)
plt.xlim(0,t)
plt.xlabel('$t$')
plt.ylabel('$\textbf{S}_n$')
ax = plt.gca()
#ax.set_aspect('equal')
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
ax.axis('off')
ax.margins(0, 0)
ax.xaxis.set_major_locator(plt.NullLocator())
ax.yaxis.set_major_locator(plt.NullLocator())
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.savefig('plots/brownian_paths_%i.pdf' % paths)
# #tikzplotlib.save('plots/brownian_paths_dichotomic%i.tex' % n)
plt.show()