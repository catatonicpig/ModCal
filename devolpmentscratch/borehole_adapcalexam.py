# -*- coding: utf-8 -*-
"""Header here."""
import numpy as np
import scipy.stats as sps
import sys
import os
import copy
import time
from line_profiler import LineProfiler
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
    def rnd(n):
        return np.vstack((sps.norm.rvs(1, 0.5, size=(n,4))))

x = sps.uniform.rvs(0,1,[50,3])
x[:,2] = x[:,2] > 0.5
yt = np.squeeze(borehole_true(x))
yvar = (10 ** (-2)) * np.ones(yt.shape)
thetatot = (thetaprior.rnd(30))
f = (borehole_model(x, thetatot).T ).T
y = yt + sps.norm.rvs(0,np.sqrt(yvar))
emu = emulator(x, thetatot, f, method = 'PCGPwM')

emu2 = emulator(passthroughfunc = borehole_model)
cal2 = calibrator( emu2, y, x, thetaprior, yvar, method = 'directbayes')
print(np.round(np.quantile(cal2.theta.rnd(10000), (0.01, 0.99), axis = 0),3))
cal = calibrator( emu, y, x, thetaprior, yvar, method = 'directbayes_wgrad')
print(np.round(np.quantile(cal.theta.rnd(10000), (0.01, 0.99), axis = 0),3))
for k in range(0,5):
    thetanew, info = emu.supplement(size = 10, cal = cal, overwrite=True)
    thetatot = np.vstack((thetatot,thetanew))
    ftot = (borehole_model(x, thetatot).T).T
    emu.update(theta=thetatot, f = ftot)
    cal.fit()
    print(np.round(np.quantile(cal.theta.rnd(10000), (0.01, 0.99), axis = 0),3))
