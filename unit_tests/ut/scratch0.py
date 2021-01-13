import numpy as np
import scipy.stats as sps
import sys
import os

current = os.path.abspath(os.getcwd())
sys.path.append(os.path.normpath(os.path.join(os.path.dirname(current), '..')))
from base.emulation import emulator


#2-d x (30 x 2), 2-d theta (50 x 2), f (30 x 50)
x = np.vstack(( np.array(list(np.arange(0, 15))*2), np.repeat([1, 2], 15))).T
theta = np.vstack((sps.norm.rvs(0, 5, size=50), sps.gamma.rvs(2, 0, 10, size=50))).T
f = np.zeros((theta.shape[0], x.shape[0]))
for k in range(0, theta.shape[0]):
    f[k, :] = x[:, 0]*theta[k, 0] + x[:, 1]*theta[k, 1] 
f = f.T
#2-d x (30 x 2), 2-d theta (50 x 2), f1 (15 x 50)
f1 = f[0:15,:]
#2-d x (30 x 2), 2-d theta (50 x 2), f2 (30 x 25)
f2 = f[:,0:25]
#2-d x (30 x 2), 2-d theta1 (25 x 2), f (30 x 50)
theta1 = theta[0:25,:]
#2-d x1 (15 x 2), 2-d theta (50 x 2), f (30 x 50)
x1 = x[0:15,:]
x3 = np.vstack(( np.array(list(np.arange(0, 10))*2), np.repeat([1, 2], 10), np.repeat([2, 3], 10))).T
# 
f0d = np.array(1)
theta0d = np.array(1)
x0d = np.array(1)
import pdb
pdb.set_trace() 
#emu = emulator(x = x, theta = theta, f = f, method = 'PCGPwM', args = {'epsilon': 1.5, 'hypregmean': -10, 'hypregLB': -20})
emu = emulator(x = x, theta = theta.T, f = f, method = 'PCGPwM', options = {'xrmnan': 'XXX'})
#pred = emu.predict(x = x, theta = theta) 
#supp = emu.supplement(size = 5, theta = theta, thetachoices = theta1)
#updated_emu = emu.update(x=x1, theta = None, f=f1, options = {'xreps' : True})
#removed_emu = emu.remove(theta = theta1)