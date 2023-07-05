import numpy as np
import matplotlib.pyplot as plt
import cmocean
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
plt.rc('text', usetex=True) 

k = np.load('kappa.npz')['kappa']
y = np.load('y.npz')['y']


for i in range(0,k.shape[0]):
    fig = plt.figure(figsize=(10,4))
        
    gs  = gridspec.GridSpec(1, 2,top=0.92,bottom=0.1,hspace=0.4,wspace=0.05,right=0.98,left=0.08)

    ax0 = plt.subplot(gs[0])
    a=ax0.imshow(k[i,:,:],cmap='bone',origin='lower',vmin=-0.05,vmax=0.05)
    divider = make_axes_locatable(ax0)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(a, cax=cax,label=r'${\rm Convergence}$')

    ax1 = plt.subplot(gs[1])
    b=ax1.imshow(y[i,:,:],cmap=cmocean.cm.tempo_r,origin='lower',vmin=1e-26,vmax=3.5e-26)
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(b, cax=cax,label=r'${\rm Compton}$-$y$')
    plt.savefig('fig/fig_%d.png'%i,dpi=200)
