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
from base.calibration import loglik
from testing.balldrop import balldropmodel_linear,\
    balldropmodel_quad, balldropmodel_drag, balldroptrue

def lpriorphys(theta):
    return (sps.gamma.logpdf(theta[:,0], 6, 0, 2) +
            sps.gamma.logpdf(theta[:,1], 1, 0, 40))
def dpriorphys(n):
    return np.vstack((sps.gamma.rvs(6, 0, 2, size=n),
                     sps.gamma.rvs(1, 0, 40, size=n))).T

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

#NEED TO EMULATE EACH MODEL
thetacompexp = dpriorphys(50)
fcompexp = balldropmodel_linear(thetacompexp, x) 
emu_lin = emulator(thetacompexp, fcompexp)

fcompexp = balldropmodel_quad(thetacompexp, x) 
emu_quad = emulator(thetacompexp, fcompexp)

fcompexp = balldropmodel_drag(thetacompexp, x) 
emu_drag = emulator(thetacompexp, fcompexp)

# #NEED TO CALIBRATE EACH MODEL
import matplotlib.pyplot as plt
sigma2 = 4
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


Sinv = 1/sigma2*np.eye(y.shape[0])
ldetS = y.shape[0] * np.log(1/sigma2)

def llik1(theta):\
    return loglik(emu_quad, theta, y, Sinv, ldetS)
thetapost1 = postsampler(dpriorphys(1000), lpriorphys, llik1)

def llikclosed(theta):
    term1 = loglik(emu_lin, theta, y, Sinv,ldetS)
    term2 = loglik(emu_quad, theta, y, Sinv,ldetS)
    term3 = loglik(emu_drag, theta, y, Sinv,ldetS)
    terms = np.vstack((term1,term2,term3))
    tm = np.max(terms,0)
    termsadj = terms - tm
    logpost = tm + np.log(np.sum(np.exp(termsadj),0))
    return logpost
thetapostclosed = postsampler(dpriorphys(1000), lpriorphys, llikclosed)

def lpriorstat(phi):
    return (sps.gamma.logpdf(phi[:,0], 4, 0, 1) +
            sps.gamma.logpdf(phi[:,1], 4, 0, 1) +
            sps.gamma.logpdf(phi[:,2], 4, 0, 1))
def dpriorstat(n):
    return np.vstack((sps.gamma.rvs(4, 0, 1, size=n),
                      sps.gamma.rvs(4, 0, 1, size=n),
                      sps.gamma.rvs(4, 0, 1, size=n))).T

def lprioropen(thetaphi):
    return (lpriorphys(thetaphi[:,:2]) + lpriorstat(thetaphi[:,2:]))
def dprioropen(n):
    return np.vstack((dpriorphys(n).T, dpriorstat(n).T)).T

#speed up computation
CorrMatDeltaT = np.exp(-4*np.abs(np.subtract.outer(x[:,0],x[:,0])))
CorrMatDeltaH = np.exp(-1/40*np.abs(np.subtract.outer(x[:,1],x[:,1])))
CorrMatDelta = CorrMatDeltaT * CorrMatDeltaH
W, V = np.linalg.eigh(CorrMatDelta)

def loglikopen(thetaphi):
    theta = thetaphi[:,:2]
    phi = np.abs(thetaphi[:,2:])
    term1 = np.zeros(thetaphi.shape[0])
    term2 = np.zeros(thetaphi.shape[0])
    term3 = np.zeros(thetaphi.shape[0])
    for k in range(0,thetaphi.shape[0]):
        term1[k] = loglik(emu_lin, theta[k,:], y, 
                          V @ (V * (1/(phi[k,0]*W+sigma2))).T,
                          np.sum(np.log(phi[k,0]*W+sigma2)))
        term2[k] = loglik(emu_quad, theta[k,:], y,
                          V @ (V * (1/(phi[k,1]*W+sigma2))).T,
                          np.sum(np.log(phi[k,1]*W+sigma2)))
        term3[k] = loglik(emu_drag, theta[k,:], y,
                          V @ (V * (1/(phi[k,2]*W+sigma2))).T,
                          np.sum(np.log(phi[k,2]*W+sigma2)))
    terms = np.vstack((term1,term2,term3))
    tm = np.max(terms,0)
    termsadj = terms - tm
    logpost = tm + np.log(np.sum(np.exp(termsadj),0))
    return logpost

thetapostopen = postsampler(dprioropen(1000), lprioropen, loglikopen)

# print(np.mean(thetapostopen,0))

fig, axes = plt.subplots(ncols=3, nrows=1, figsize=(21, 5))

from scipy.stats import kde
def two2d(axis, theta):
    nbins = 50
    k = kde.gaussian_kde(theta.T)
    xi, yi = np.mgrid[6:13:nbins*1j, 0:60:nbins*1j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))
    axis.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='gouraud', cmap=plt.cm.BuGn_r)
    axis.contour(xi, yi, zi.reshape(xi.shape))

two2d(axes[0], dpriorphys(4000))
two2d(axes[1], thetapostclosed)
two2d(axes[2], thetapostopen[:,:2])


#plt.plot(x[:,0],y)
#NEED TO DECIDE A NEW HEIGHT TO DROP IT AT