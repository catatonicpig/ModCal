# -*- coding: utf-8 -*-
"""Header here."""
import numpy as np
import scipy.stats as sps
import sys
import os
import copy

from boreholetestfunctions import borehole_model, borehole_true
SCRIPT_DIR = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, '..')))
from base.emulation import emulator
from base.calibration import calibrator

class thetaprior:
    """ This defines the class instance of priors provided to the methods. """
    def lpdf(theta):
        if theta.ndim > 1.5:
            return np.squeeze(np.sum(sps.norm.logpdf(theta, 1, 0.5),1))
        else:
            return np.squeeze(np.sum(sps.norm.logpdf(theta, 1, 0.5)))
    def lpdf_grad(theta):
        f_base = thetaprior.lpdf(theta)
        if theta.ndim > 1.5:
            grad = np.zeros((theta.shape[0], theta.shape[1]))
            for k in range(0,theta.shape[1]):
                thetaprop = copy.copy(theta)
                thetaprop[:,k] += 10 ** (-5)
                f_base2 = thetaprior.lpdf(thetaprop)
                grad[:,k] = 10 ** (5) * (f_base2 - f_base)
        else:
            grad = np.zeros(theta.shape[0])
            for k in range(0,theta.shape[0]):
                thetaprop = copy.copy(theta)
                thetaprop[k] += 10 ** (-5)
                f_base2 = thetaprior.lpdf(thetaprop)
                grad[k] = 10 ** (5) * (f_base2 - f_base)
        return grad
    def rnd(n):
        return np.vstack((sps.norm.rvs(1, 0.5, size=(n,4))))

x = sps.uniform.rvs(0,1,[50,3])
yt = np.squeeze(borehole_true(x))
yvar = (10 ** (-2)) * np.ones(yt.shape)
x[:,2] = np.round(x[:,2])
thetacompexp = (thetaprior.rnd(30))
f = (borehole_model(x, thetacompexp).T ).T

y = np.squeeze(borehole_true(x)) + sps.norm.rvs(0,np.sqrt(yvar))
emu = emulator(x, thetacompexp, f, method = 'PCGPwM')  # this builds an emulator 

cal = calibrator( emu, y, x, thetaprior, yvar, method = 'directbayes_wgrad')
adas

thetatrial = thetaprior.rnd(10)
xtrial = x[:3]
emu2 = emulator(passthroughfunc = borehole_model)
g = emu2.predict(x,thetacompexp).mean()
thetatrial = thetaprior.rnd(1000)


cal2 = calibrator( emu2, y, x, thetaprior, yvar, method = 'directbayes')

print(np.round(np.quantile(cal2.theta.rnd(10000), (0.01, 0.99), axis = 0),3))
cal = calibrator( emu, y, x, thetaprior, yvar, method = 'directbayes')
print(np.round(np.quantile(cal.theta.rnd(10000), (0.01, 0.99), axis = 0),3))
for k in range(0,5):
    thetanew, info = emu.supplement(size = 10, cal = cal)
    fadd = (borehole_model(x, thetanew).T).T
    emu.update(f = fadd)
    cal.fit()
    print(np.round(np.quantile(cal.theta.rnd(10000), (0.01, 0.99), axis = 0),3))
