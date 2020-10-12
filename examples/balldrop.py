# -*- coding: utf-8 -*-
"""Header here."""
import numpy as np
import scipy.stats as sps
import sys
import os

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
from base.utilities import postsampler
from base.emulation import emulator
from base.calibration import calibrator
from testing.balldrop import balldropmodel_linear,\
    balldropmodel_quad, balldropmodel_drag, balldroptrue

class priorphys:
    def logpdf(theta):
        return np.squeeze(sps.gamma.logpdf(theta[:,0], 10, 0, 1) +
            sps.gamma.logpdf(theta[:,1], 1, 0, 40))
    def rvs(n):
        return np.vstack((sps.gamma.rvs(10, 0, 1, size=n),
                     sps.gamma.rvs(1, 0, 40, size=n))).T

tvec = np.concatenate((np.arange(0.1,1.2,0.1),
                  np.arange(0.1,2.2,0.1),
                  np.arange(0.1,2.6,0.1),
                  np.arange(0.1,3.2,0.1)))
hvec = np.concatenate((10*np.ones(11),
                  20*np.ones(21),
                  30*np.ones(25),
                  40*np.ones(31)))
xtot = np.vstack((tvec,hvec)).T

#NEED TO EMULATE EACH MODEL
thetacompexp = priorphys.rvs(50)
emu_lin = emulator(thetacompexp, balldropmodel_linear(thetacompexp, xtot), xtot)
emu_quad = emulator(thetacompexp, balldropmodel_quad(thetacompexp, xtot), xtot)
emu_drag = emulator(thetacompexp, balldropmodel_drag(thetacompexp, xtot), xtot)


# #NEED TO CALIBRATE EACH MODEL
import matplotlib.pyplot as plt
sigma2 = 2

x = np.array([[ 0.1, 10. ],
        [ 0.2, 10. ],
        [ 0.3, 10. ],
        [ 0.4, 10. ],
        [ 0.5, 10. ],
        [ 0.6, 10. ],
        [ 0.7, 10. ],
        [ 0.8, 10. ],
        [ 0.9, 10. ],
        [ 1. , 10. ],
        [ 1.1, 10. ],
        [ 0.1, 20. ],
        [ 0.4, 20. ],
        [ 0.7, 20. ],
        [ 1. , 20. ],
        [ 1.3, 20. ],
        [ 1.6, 20. ],
        [ 1.9, 20. ],
        [ 0.1, 40. ],
        [ 0.5, 40. ],
        [ 0.9, 40. ],
        [ 1.3, 40. ],
        [ 1.7, 40. ],
        [ 2.1, 40. ],
        [ 2.5, 40. ],
        [ 2.9, 40. ]])
y = np.array([[11.5],
        [11.2],
        [ 9.6],
        [ 7.9],
        [ 9.6],
        [ 6.8],
        [ 7.3],
        [ 7.9],
        [ 6.5],
        [ 2.2],
        [ 2.8],
        [21.6],
        [20.3],
        [18.8],
        [12.1],
        [11.8],
        [10.6],
        [ 3.9],
        [40.4],
        [36.2],
        [36. ],
        [32.6],
        [24.8],
        [19.7],
        [13.1],
        [ 6.8]])

emu_lin.predict(thetacompexp[0,:])
obsvar = sigma2*np.ones(y.shape[0])

Spred = np.zeros((xtot.shape[0],x.shape[0]))
cal_lin = calibrator(emu_lin, y, x,
                       thetaprior = priorphys,
                       phiprior = None,
                       passoptions = {'obsvar': obsvar})
cal_quad = calibrator(emu_quad, y, x,
                       thetaprior = priorphys,
                       phiprior = None,
                       passoptions = {'obsvar': obsvar})
cal_drag = calibrator(emu_drag, y, x,
                       thetaprior = priorphys,
                       phiprior = None,
                       passoptions = {'obsvar': obsvar})

A = cal_lin.predict(xtot)
plt.plot(xtot[:,0],A['mean'],'bo')
plt.plot(xtot[:,0],A['mean']+3*np.sqrt(A['var']),'b-')
plt.plot(xtot[:,0],A['mean']-3*np.sqrt(A['var']),'b-')

def lpostclosed(theta):
    term1 = cal_lin.logpost(theta, None)
    term2 = cal_quad.logpost(theta, None)
    term3 = cal_drag.logpost(theta, None)
    terms = np.vstack((term1,term2,term3))
    tm = np.max(terms,0)
    termsadj = terms - tm
    logpost = tm + np.log(np.sum(np.exp(termsadj),0))
    indsing = np.where(np.isnan(logpost))[0]
    logpost[indsing] = -np.inf
    return logpost
thetapostclosed = postsampler(priorphys.rvs(1000), lpostclosed)

mean1 = cal_lin.predict(xtot,thetapostclosed, None)
plt.plot(xtot[:,0],mean1['mean'],'bo')
plt.plot(xtot[:,0],mean1['mean']+3*np.sqrt(A['var']),'b-')
plt.plot(xtot[:,0],mean1['mean']-3*np.sqrt(A['var']),'b-')


class priorstat:
    def logpdf(phi):
        return np.squeeze(sps.gamma.logpdf(phi, 4, 0, 1))
    def rvs(n):
        return sps.gamma.rvs(4, 0, 1, size = n).reshape(-1,1)

#speed up computation
CorrMatDeltaT = np.exp(-np.abs(np.subtract.outer(xtot[:,0],xtot[:,0])))* (1 + 1/40*np.abs(np.subtract.outer(xtot[:,0],xtot[:,0])))
CorrMatDeltaH = np.exp(-1/40*np.abs(np.subtract.outer(xtot[:,1],xtot[:,1]))) * (1 + 1/40*np.abs(np.subtract.outer(xtot[:,1],xtot[:,1])))
CorrMatDelta = CorrMatDeltaT * CorrMatDeltaH
W,V = np.linalg.eigh(CorrMatDelta)
CorrMatHalf = (V[:,-20:] @ np.diag(np.sqrt(W[-20:]))).T
CorrMatHalf.T @ CorrMatHalf

cal_lin_plus = calibrator(emu_lin, y, x,
                       thetaprior = priorphys,
                       phiprior = priorstat,
                       passoptions = {'obsvar': obsvar, 'covhalf': CorrMatHalf})
cal_quad_plus = calibrator(emu_quad, y, x,
                       thetaprior = priorphys,
                       phiprior = priorstat,
                       passoptions = {'obsvar': obsvar, 'covhalf': CorrMatHalf})
cal_drag_plus = calibrator(emu_drag, y, x,
                       thetaprior = priorphys,
                       phiprior = priorstat,
                       passoptions = {'obsvar': obsvar, 'covhalf': CorrMatHalf})

def lpostopen(thetaphi):
    theta = thetaphi[:,:2]
    phi = np.abs(thetaphi[:,2:])
    term1 = cal_lin_plus.logpost(theta, phi[:,0])
    term2 = cal_quad_plus.logpost(theta, phi[:,1])
    term3 = cal_drag_plus.logpost(theta, phi[:,2])
    terms = np.vstack((term1,term2,term3))
    tm = np.max(terms,0)
    termsadj = terms - tm
    logpost = tm + np.log(np.sum(np.exp(termsadj),0))
    indsing = np.where(np.isnan(logpost))[0]
    logpost[indsing] = -np.inf
    return logpost
thetaphipostopen = postsampler(np.hstack((priorphys.rvs(1000),
                                          priorstat.rvs(1000),
                                          priorstat.rvs(1000),
                                          priorstat.rvs(1000))), lpostopen)


fig, axes = plt.subplots(ncols=3, nrows=1, figsize=(21, 5))

from scipy.stats import kde
def two2d(axis, theta):
    nbins = 50
    k = kde.gaussian_kde(theta.T)
    xi, yi = np.mgrid[6:13:nbins*1j, 0:60:nbins*1j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))
    axis.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='gouraud', cmap=plt.cm.BuGn_r)
    axis.contour(xi, yi, zi.reshape(xi.shape))

two2d(axes[0], priorphys.rvs(4000))
two2d(axes[1], thetapostclosed)
two2d(axes[2], thetaphipostopen[:,:2])



# # def dpriorstat(n):
# #     return np.vstack((sps.gamma.rvs(4, 0, 1, size=n),
# #                       sps.gamma.rvs(4, 0, 1, size=n),
# #                       sps.gamma.rvs(4, 0, 1, size=n))).T


# # #NEED TO DECIDE A NEW HEIGHT TO DROP IT AT