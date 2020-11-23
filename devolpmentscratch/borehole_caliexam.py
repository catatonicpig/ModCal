# -*- coding: utf-8 -*-
"""Header here."""
import numpy as np
import scipy.stats as sps
import sys
import os
import copy
import time
from line_profiler import LineProfiler
from boreholetestfunction import borehole_model, borehole_true
SCRIPT_DIR = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, '..')))
from base.emulation import emulator
from base.calibration import calibrator


from base.calibrationmethods.directbayes_wgrad import fit as fit1
from base.calibrationmethods.directbayes import fit as fit2
from base.utilitiesmethods.plumleeMCMC_wgrad import plumleepostsampler_wgrad as sampler1

class thetaprior:
    """ This defines the class instance of priors provided to the methods. """
    def lpdf(theta):
        if theta.ndim > 1.5:
            return np.squeeze(np.sum(sps.norm.logpdf(theta, 1, 0.5),1))
        else:
            return np.squeeze(np.sum(sps.norm.logpdf(theta, 1, 0.5)))
    def rnd(n):
        return np.vstack((sps.norm.rvs(1, 0.5, size=(n,6))))

x = sps.uniform.rvs(0,1,[50,3])
x[:,2] = x[:,2] > 0.5
yt = np.squeeze(borehole_true(x))
yvar = (10 ** (-2)) * np.ones(yt.shape)
thetacompexp = (thetaprior.rnd(30))
f = (borehole_model(x, thetacompexp).T ).T
def emulation_test_borehole():
    y = np.squeeze(borehole_true(x)) + sps.norm.rvs(0,np.sqrt(yvar))
    emu = emulator(x, thetacompexp, f, method = 'PCGPwM')  # this builds an emulator 
    t = time.time()
    cal = calibrator( emu, y, x, thetaprior, yvar, method = 'directbayes_wgrad')
    print(np.round(np.quantile(cal.theta.rnd(1000), (0.01, 0.99), axis = 0),3))
    print(time.time()-t)
    t = time.time()
    cal = calibrator( emu, y, x, thetaprior, yvar, method = 'directbayes')
    print(np.round(np.quantile(cal.theta.rnd(1000), (0.01, 0.99), axis = 0),3))
    print(time.time()-t)
# asdada
# thetatrial = thetaprior.rnd(10)
# xtrial = x[:3]
# emu2 = emulator(passthroughfunc = borehole_model)
# g = emu2.predict(x,thetacompexp).mean()
# thetatrial = thetaprior.rnd(1000)
# cal2 = calibrator( emu2, y, x, thetaprior, yvar, method = 'directbayes')

# print(np.round(np.quantile(cal2.theta.rnd(10000), (0.01, 0.99), axis = 0),3))
# cal = calibrator( emu, y, x, thetaprior, yvar, method = 'directbayes_wgrad')
# print(np.round(np.quantile(cal.theta.rnd(10000), (0.01, 0.99), axis = 0),3))
# for k in range(0,5):
#     thetanew, info = emu.supplement(size = 10, cal = cal)
#     fadd = (borehole_model(x, thetanew).T).T
#     emu.update(f = fadd)
#     cal.fit()
#     print(np.round(np.quantile(cal.theta.rnd(10000), (0.01, 0.99), axis = 0),3))


lp = LineProfiler()
lp_wrapper = lp(emulation_test_borehole)    
lp.add_function(fit1)
lp.add_function(sampler1)
lp_wrapper()
#lp.print_stats()
