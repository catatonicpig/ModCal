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
            return np.squeeze(np.sum(sps.norm.logpdf(theta, 0.6, 0.25),1))
        else:
            return np.squeeze(np.sum(sps.norm.logpdf(theta, 0.6, 0.25)))
    def rnd(n):
        return np.vstack((sps.norm.rvs(0.6, 0.25, size=(n,6))))

x = sps.uniform.rvs(0,1,[50,3])
x[:,2] = np.round(x[:,2])
thetacompexp = thetaprior.rnd(30)
f = borehole_model(x, thetacompexp)

y = borehole_true(x)
emu = emulator(x, thetacompexp, f, method = 'PCGPwM')  # this builds an emulator 

thetatrial = thetaprior.rnd(1000)

ftest = borehole_model(x, thetatrial)
predobj = emu.predict(x, theta = thetatrial)
yvar = 10 ** (-1) *np.ones(y.shape)
cal = calibrator( emu, y, x, thetaprior, yvar, method = 'directbayes')
print(cal.theta.mean() + np.sqrt(cal.theta.var()) * 3)
print(cal.theta.mean() - np.sqrt(cal.theta.var()) * 3)
for k in range(0,5):
    thetanew, info = emu.supplement(size = 10, cal = cal)
    fadd = borehole_model(x, thetanew)
    emu.update(f = fadd)
    cal.fit()
    print(cal.theta.mean() + np.sqrt(cal.theta.var()) * 3)
    print(cal.theta.mean() - np.sqrt(cal.theta.var()) * 3)
