# -*- coding: utf-8 -*-
"""Header here."""
import numpy as np
import scipy.stats as sps
import sys
import os

from balldroptestfuncs import balldropmodel_linear,\
    balldropmodel_quad, balldropmodel_drag, balldroptrue
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
from base.utilities import postsampler
from base.emulation import emulator
from base.calibration import calibrator

class priorphys:
    def logpdf(theta):
        return np.squeeze(sps.gamma.logpdf(theta[:,0], 5, 0, 2) +
            sps.gamma.logpdf(theta[:,1], 4, 0, 3))
    def rvs(n):
        return np.vstack((sps.gamma.rvs(5, 0, 2, size=n),
                     sps.gamma.rvs(4, 0, 3, size=n))).T

tvec = np.concatenate((np.arange(0.2,5.2,0.2),
                  np.arange(0.2,5.2,0.2),
                  np.arange(0.2,5.2,0.2),
                  np.arange(0.2,5.2,0.2)))
hvec = np.concatenate((10*np.ones(25),
                  20*np.ones(25),
                  30*np.ones(25),
                  40*np.ones(25)))
xtot = np.vstack((tvec,hvec)).T

thetacompexp = priorphys.rvs(100)
emu_lin = emulator(thetacompexp, balldropmodel_linear(thetacompexp, xtot), xtot)
emu_quad = emulator(thetacompexp, balldropmodel_quad(thetacompexp, xtot), xtot)

import matplotlib.pyplot as plt
sigma2 = 1

x = np.array([[ 0.2, 10. ],
        [ 0.4, 10. ],
        [ 0.6, 10. ],
        [ 0.8, 10. ],
        [ 1. , 10. ],
        [ 1.2, 10. ],
        [ 1.4, 10. ],
        [ 0.2, 20. ],
        [ 0.4, 20. ],
        [ 0.8, 20. ],
        [ 1. , 20. ],
        [ 1.2, 20. ],
        [ 1.6, 20. ],
        [ 2.0, 20. ],
        [ 2.4, 20. ],
        [ 0.2, 40. ],
        [ 0.6, 40. ],
        [ 0.8, 40. ],
        [ 1.0, 40. ],
        [ 1.4, 40. ],
        [ 1.8, 40. ],
        [ 2.8, 40. ]])
y = balldroptrue(x) + sps.norm.rvs(0, np.sqrt(sigma2),size=x.shape[0])

obsvar = sigma2*np.ones(y.shape[0])


class priorstat:
    def logpdf(phi):
        return np.squeeze(sps.gamma.logpdf(phi[:,0], 1, 0, 1)
                          +sps.gamma.logpdf(phi[:,1], 1, 0, 1))                          
    def rvs(n):
        return np.vstack((sps.gamma.rvs(1, 0, 1, size = n),
                         sps.gamma.rvs(1, 0, 1, size = n))).T

def corr_f(x,k):
    corrdict = {}
    C0 = np.exp(-1/3*np.abs(np.subtract.outer(x[:,0],x[:,0])))*(1+1/3*np.abs(np.subtract.outer(x[:,0],x[:,0])))
    C0 = 0.999*C0 + 0.001 * np.diag(np.diag(C0))
    C1 = 0.25*(np.abs(np.subtract.outer(x[:,1],x[:,1]))<10**(-4))
    if k == 0:
        adj = np.abs(20*(x[:,0] - 2*(1-np.exp(-x[:,0]/2))) - 20 * x[:,0])
        corrdict['C'] = C1 + np.diag(adj) @ C0 @ np.diag(adj)
    if k == 1:
        adj = np.abs(20*(x[:,0] - 2*(1-np.exp(-x[:,0]/2))) - 5 * x[:,0] ** 2)
        corrdict['C'] = C1 + np.diag(adj) @ C0 @ np.diag(adj)
    return corrdict

cal_quad= calibrator(emu_quad, y, x,
                       thetaprior = priorphys,
                       phiprior = None,
                       passoptions = {'obsvar': obsvar})
 

cal_BMM = calibrator((emu_lin,emu_quad), y, x,
                       thetaprior = priorphys,
                       phiprior = priorstat,
                       passoptions = {'obsvar': obsvar, 'corrf': corr_f})


cal_lin = calibrator(emu_lin, y, x,
                       thetaprior = priorphys,
                       phiprior = None,
                       passoptions = {'obsvar': obsvar})

    
cal_BMA = calibrator((emu_lin,emu_quad), y, x,
                       thetaprior = priorphys,
                       phiprior = None,
                       passoptions = {'obsvar': obsvar})



from scipy.stats import kde
def two2d(axis, theta):
    nbins = 50
    k = kde.gaussian_kde(theta.T)
    xi, yi = np.mgrid[0:13:nbins*1j, 0:20:nbins*1j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))
    axis.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='gouraud', cmap=plt.cm.BuGn_r)
    axis.contour(xi, yi, zi.reshape(xi.shape))

fig, axes = plt.subplots(ncols=5, nrows=1, figsize=(21, 5))
two2d(axes[0], priorphys.rvs(4000))
two2d(axes[1], cal_lin.thetadraw)
two2d(axes[2], cal_quad.thetadraw)
two2d(axes[3], cal_BMA.thetadraw)
two2d(axes[4], cal_BMM.thetadraw)




def plotpreds(axis, preddict):
    for k in (10,20,30,40):
        inds = np.where(xtot[:,1] == k)[0]
        uppercurve = preddict['mean'][inds] + 3*np.sqrt(preddict['var'][inds])
        lowercurve = preddict['mean'][inds] - 3*np.sqrt(preddict['var'][inds])
        axis.fill_between(xtot[inds,0], lowercurve, uppercurve, color='k', alpha=0.2)
        axis.plot(xtot[inds,0],preddict['mean'][inds],'k-')
        axis.plot(xtot[inds,0],uppercurve, 'k-', alpha=0.6,linewidth=0.5)
        axis.plot(xtot[inds,0],lowercurve, 'k-', alpha=0.6,linewidth=0.5)
    axis.plot(x,y, 'ko')
    axis.set_xlim([0,3.0])
    axis.set_ylim([0,41])

fig, axes = plt.subplots(ncols=4, nrows=1, figsize=(21, 5))    
plotpreds(axes[0], cal_lin.predict(xtot))
plotpreds(axes[1], cal_quad.predict(xtot))
plotpreds(axes[2], cal_BMA.predict(xtot))
plotpreds(axes[3], cal_BMM.predict(xtot))