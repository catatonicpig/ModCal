# -*- coding: utf-8 -*-
"""Header here."""
import numpy as np
import scipy.stats as sps
import sys
import os

from balldroptestfuncs import balldropmodel_linear,\
    balldropmodel_quad, balldropmodel_drag, balldroptrue, balldroptruealt
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
from base.utilities import postsampler
from base.emulation import emulator
from base.calibration import calibrator

class priorphys:
    def logpdf(theta):
        return np.squeeze(sps.gamma.logpdf(theta[:,0], 2, 0, 5) +
            sps.gamma.logpdf(theta[:,1], 1, 0, 40) +
            sps.norm.logpdf(theta[:,2], 0, 10))
    def rvs(n):
        return np.vstack((sps.gamma.rvs(2, 0, 5, size=n),
                     sps.gamma.rvs(1, 0, 40, size=n),
                     sps.norm.rvs(0, 10, size=n))).T

tvec = np.concatenate((np.arange(0.1,5.6,0.1),
                  np.arange(0.1,5.6,0.1),
                  np.arange(0.1,5.6,0.1),
                  np.arange(0.1,5.6,0.1)))
hvec = np.concatenate((20*np.ones(55),
                  40*np.ones(55),
                  60*np.ones(55),
                  80*np.ones(55)))
xtot = np.vstack((tvec,hvec)).T

thetacompexp = priorphys.rvs(100)
emu_lin = emulator(thetacompexp, balldropmodel_linear(thetacompexp, xtot), xtot)
emu_quad = emulator(thetacompexp, balldropmodel_quad(thetacompexp, xtot), xtot)

import matplotlib.pyplot as plt
sigma2 = 1

x = np.array([[ 0.1, 20. ],
              [ 0.2, 20. ],
              [ 0.3, 20. ],
        [ 0.4, 20. ],
        [ 0.5, 20. ],
        [ 0.6, 20. ],
        [ 0.8, 20. ],
        [ 1. , 20. ],
        [ 1.4, 20. ],
        [ 1.8, 20. ],
        [ 0.1, 40. ],
        [ 0.2, 40. ],
        [ 0.3, 40. ],
        [ 0.4, 40. ],
        [ 0.5, 40. ],
        [ 0.6, 40. ],
        [ 0.8, 40. ],
        [ 1.2, 40. ],
        [ 1.6, 40. ],
        [ 2.0, 40. ],
        [ 2.4, 40. ],
        [ 3.0, 40. ],
        [ 0.1, 80. ],
        [ 0.2, 80. ],
        [ 0.3, 80. ],
        [ 0.4, 80. ],
        [ 0.5, 80. ],
        [ 0.6, 80. ],
        [ 0.8, 80. ],
        [ 1.0, 80. ],
        [ 1.4, 80. ],
        [ 1.8, 80. ],
        [ 2.2, 80. ],
        [ 2.7, 80. ],
        [ 3.2, 80. ],
        [ 3.5, 80. ],
        [ 3.7, 80. ],
        [ 4.2, 80. ],
        [ 4.7, 80. ],
        [ 4.9, 80. ],
        [ 5.0, 80. ],
        [ 5.1, 80. ],
        [ 5.2, 80. ]])
y = balldroptruealt(x) + sps.norm.rvs(0, np.sqrt(sigma2),size=x.shape[0])
# y = np.array([[ 9.01],
#        [ 7.64],
#        [ 8.59],
#        [ 7.35],
#        [ 4.96],
#        [ 2.15],
#        [19.98],
#        [19.84],
#        [17.87],
#        [18.23],
#        [15.55],
#        [11.65],
#        [ 8.5 ],
#        [ 6.16],
#        [41.22],
#        [41.3 ],
#        [39.71],
#        [39.73],
#        [37.99],
#        [35.93],
#        [31.33],
#        [21.35],
#        [12.42]])
obsvar = sigma2*np.ones(y.shape[0])
#plt.plot(x[:,0],y, 'ko')

class priorstat1d:
    def logpdf(phi):
        return np.squeeze(sps.gamma.logpdf(phi, 1, 0, 4))
    def rvs(n):
        return sps.gamma.rvs(1, 0, 4, size = n).reshape((-1,1))

class priorstat2d:
    def logpdf(phi):
        return np.squeeze(priorstat1d.logpdf(phi[:,0])+priorstat1d.logpdf(phi[:,1]))
    def rvs(n):
        return np.hstack((priorstat1d.rvs(n),priorstat1d.rvs(n)))



def corr_f(x,k):
    corrdict = {}
    C0 = np.exp(-np.abs(np.subtract.outer(x[:,0],x[:,0])))*(1+np.abs(np.subtract.outer(x[:,0],x[:,0])))
    C1 = 0.0001*(np.abs(np.subtract.outer(x[:,1],x[:,1]))<10**(-4))
    if k == 0:
        adj = np.exp(6-x[:,0])/4
        corrdict['C'] = C1 + np.diag(adj) @ C0 @ np.diag(adj)
    if k == 1:
        adj = np.exp(x[:,0])
        corrdict['C'] = C1 + np.diag(adj) @ C0 @ np.diag(adj)
    return corrdict

cal_BMM = calibrator((emu_lin,emu_quad), y, x,
                       thetaprior = priorphys,
                       phiprior = priorstat2d,
                       passoptions = {'obsvar': obsvar, 'corrf': corr_f})

print(cal_BMM.thetadraw)
print(cal_BMM.phidraw)
asdasd
cal_lin = calibrator(emu_lin, y, x,
                       thetaprior = priorphys,
                       phiprior = priorstat1d,
                       passoptions = {'obsvar': obsvar})

cal_quad= calibrator(emu_quad, y, x,
                       thetaprior = priorphys,
                       phiprior = priorstat1d,
                       passoptions = {'obsvar': obsvar})

 
cal_BMA = calibrator((emu_lin,emu_quad), y, x,
                       thetaprior = priorphys,
                       phiprior = priorstat2d,
                       passoptions = {'obsvar': obsvar})

from scipy.stats import kde
def two2d(axis, theta):
    nbins = 50
    k = kde.gaussian_kde(theta.T)
    xi, yi = np.mgrid[0:15:nbins*1j, 0:60:nbins*1j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))
    axis.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='gouraud', cmap=plt.cm.BuGn_r)
    axis.contour(xi, yi, zi.reshape(xi.shape))

fig, axes = plt.subplots(ncols=5, nrows=1, figsize=(21, 5))
two2d(axes[0], priorphys.rvs(4000)[:,:2])
two2d(axes[1], cal_lin.thetadraw[:,:2])
two2d(axes[2], cal_quad.thetadraw[:,:2])
two2d(axes[3], cal_BMA.thetadraw[:,:2])
two2d(axes[4], cal_BMM.thetadraw[:,:2])




def plotpreds(axis, preddict):
    for k in (20,40,60,80):
        inds = np.where(xtot[:,1] == k)[0]
        uppercurve = preddict['mean'][inds] + 3*np.sqrt(preddict['var'][inds])
        lowercurve = preddict['mean'][inds] - 3*np.sqrt(preddict['var'][inds])
        axis.fill_between(xtot[inds,0], lowercurve, uppercurve, color='k', alpha=0.2)
        axis.plot(xtot[inds,0],preddict['mean'][inds],'k-')
        axis.plot(xtot[inds,0],uppercurve, 'k-', alpha=0.6,linewidth=0.5)
        axis.plot(xtot[inds,0],lowercurve, 'k-', alpha=0.6,linewidth=0.5)
    axis.plot(x,y, 'ko')
    axis.set_xlim([0,5.6])
    axis.set_ylim([0,85])

fig, axes = plt.subplots(ncols=4, nrows=1, figsize=(21, 5))    
plotpreds(axes[0], cal_lin.predict(xtot))
plotpreds(axes[1], cal_quad.predict(xtot))
plotpreds(axes[2], cal_BMA.predict(xtot))
plotpreds(axes[3], cal_BMM.predict(xtot))