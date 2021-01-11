import numpy as np
import scipy.stats as sps
import sys
import os
import pytest
from contextlib import contextmanager

current = os.path.abspath(os.getcwd())
sys.path.append(os.path.normpath(os.path.join(os.path.dirname(current), '..')))
from base.emulation import emulator
from base.calibration import calibrator
##############################################
#            Simple scenarios                #
##############################################

#2-d x (30 x 2), 2-d theta (50 x 2), f (30 x 50)
x = np.vstack(( np.array(list(np.arange(0, 15))*2), np.repeat([1, 2], 15))).T
theta = np.vstack((sps.norm.rvs(0, 5, size=50), sps.gamma.rvs(2, 0, 10, size=50))).T
f = np.zeros((theta.shape[0], x.shape[0]))
for k in range(0, theta.shape[0]):
    f[k, :] = x[:, 0]*theta[k, 0] + x[:, 1]*theta[k, 1] 
f = f.T
#
y = np.array(sps.norm.rvs(0, 5, size=30)).reshape(30,1)
#
obsvar = 0.001*sps.uniform.rvs(10, 20, size=30)
#
class fake_prior:
    def lpdf(theta):
        return np.array([1,2,3])
    def rnd(n):
        return np.array([1,2,3])
    
class fake_empty_prior:  
    def nothing():
        return None
        
class prior_example:
    def lpdf(theta):
        if theta.ndim > 1.5:
            return np.squeeze(sps.norm.logpdf(theta[:, 0], 0, 5) + 
                              sps.gamma.logpdf(theta[:, 1], 2, 0, 10)) 
        else:
            return np.squeeze(sps.norm.logpdf(theta[0], 0, 5) + 
                              sps.gamma.logpdf(theta[1], 2, 0, 10)) 

    def rnd(n):
        return np.vstack((sps.norm.rvs(0, 5, size=n), 
                          sps.gamma.rvs(2, 0, 10, size=n))).T 
        
#2-d x (30 x 2), 2-d theta (50 x 2), f1 (15 x 50)
f1 = f[0:15,:]
#2-d x (30 x 2), 2-d theta (50 x 2), f2 (30 x 25)
f2 = f[:,0:25]
#2-d x (30 x 2), 2-d theta1 (25 x 2), f (30 x 50)
theta1 = theta[0:25,:]
#2-d x1 (15 x 2), 2-d theta (50 x 2), f (30 x 50)
x1 = x[0:15,:]
# 
f0d = np.array(1)
theta0d = np.array(1)
x0d = np.array(1)
##############################################
# Unit tests to initialize an emulator class #
##############################################
emulator_test = emulator(x = x, theta = theta, f = f, method = 'PCGPwM')
import pdb
pdb.set_trace() 
cal_test = calibrator(emu = emulator_test, y = y, x = x, thetaprior = prior_example, 
                   method = 'MLcal', yvar = obsvar, 
                   args = {'theta0': np.array([1, 1]), 
                           'numsamp' : 50, 
                           'stepType' : 'normal', 
                           'stepParam' : [0.3, 0.3]})
cal_pred = cal_test.predict(x=x)