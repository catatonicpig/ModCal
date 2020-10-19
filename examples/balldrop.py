# -*- coding: utf-8 -*-
"""Header here."""
import numpy as np
import scipy.stats as sps
import sys
import os

from balldroptestfuncs import balldropmodel_linear,\
    balldropmodel_grav, balldroptrue
SCRIPT_DIR = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, '..')))
from base.utilities import postsampler
from base.emulation import emulator
from base.calibration import calibrator


class priorphys:
    """ This defines the class instance of priors provided to the software. """
    def logpdf(theta):
        return np.squeeze(sps.norm.logpdf(theta[:, 0], 0, 20) +  # initial height deviation
                          sps.gamma.logpdf(theta[:, 1], 1, 0, 40) +  # terminal velocity
                          sps.gamma.logpdf(theta[:, 2], 2, 0, 5))  # gravity

    def rvs(n):
        return np.vstack((sps.norm.rvs(0, 20, size=n),  # initial height deviation
                          sps.gamma.rvs(1, 0, 40, size=n),  # terminal velocity
                          sps.gamma.rvs(2, 0, 5, size=n))).T  # gravity


tvec = np.concatenate((np.arange(0.1, 2.0, 0.1),
                       np.arange(0.1, 3.1, 0.1),
                       np.arange(0.1, 4.0, 0.1),
                       np.arange(0.1, 5.2, 0.1)))  # the time vector of interest
hvec = np.concatenate((20 * np.ones(19),
                       40 * np.ones(30),
                       60 * np.ones(39),
                       80 * np.ones(51)))  # the drop heights vector of interest
xtot = np.vstack((tvec, hvec)).T  # the input of interest
# each row is an individual vector of interest
# this should include those important to the study AND the data

# we now create a computer experiment to build an emulator
thetacompexp = priorphys.rvs(50)  # drawing 50 random parameters from the prior

linear_results = balldropmodel_linear(thetacompexp, xtot)  # the value of the linear simulation
# This is for all vectors in the input of interest
emu_lin = emulator(thetacompexp, linear_results, xtot)  # this builds an emulator 
# for the linear simulation. this is a specialized class specific to ModCal.
grav_results = balldropmodel_grav(thetacompexp, xtot)  # the value of the gravity simulation
emu_grav = emulator(thetacompexp, grav_results, xtot)  # this builds an
# emulator for the gravity simulation. this is a specialized class specific to ModCal.

x = np.array([[ 0.1, 20. ],  # input data with corresponding responses
              [ 0.2, 20. ],  # this MUST be a subset of 'xtot'
              [ 0.3, 20. ],
        [ 0.5, 20. ],
        [ 0.6, 20. ],
        [ 0.8, 20. ],
        [ 1. , 20. ],
        [ 1.4, 20. ],
        [ 1.8, 20. ],
        [ 0.1, 40. ],
        [ 0.2, 40. ],
        [ 0.3, 40. ],
        [ 0.5, 40. ],
        [ 0.6, 40. ],
        [ 0.8, 40. ],
        [ 1.2, 40. ],
        [ 1.6, 40. ],
        [ 2.0, 40. ],
        [ 2.4, 40. ],
        [ 3.0, 40. ],
        [ 0.1, 80. ],
        [ 0.3, 80. ],
        [ 0.5, 80. ],
        [ 0.8, 80. ],
        [ 1.0, 80. ],
        [ 1.4, 80. ],
        [ 1.8, 80. ],
        [ 2.2, 80. ],
        [ 2.7, 80. ],
        [ 3.2, 80. ],
        [ 3.7, 80. ],
        [ 4.2, 80. ],
        [ 4.5, 80. ],
        [ 4.7, 80. ],
        [ 4.9, 80. ],
        [ 5.1, 80. ]])
obsvar = 1*np.ones(x.shape[0])  # variance for the observations in 'y' below
y = np.array([[18.3],  # response data that aligns with 'x'
       [18.1],
       [18.7],
       [18.5],
       [18.5],
       [17. ],
       [14.4],
       [10. ],
       [ 3.9],
       [39.1],
       [40.3],
       [39.5],
       [37.9],
       [39.1],
       [36. ],
       [33.1],
       [26.2],
       [23.6],
       [16.5],
       [ 6.6],
       [79.4],
       [82. ],
       [78.1],
       [76. ],
       [75.2],
       [70.2],
       [65. ],
       [60.2],
       [53. ],
       [43.7],
       [32.5],
       [24.2],
       [18.1],
       [14.4],
       [10.2],
       [ 5.9]]) 
# to regenerate use the line
# y = balldroptrue(x) + sps.norm.rvs(0, np.sqrt(obsvar))
# alignment is provided by np.round(y.reshape((-1,1)),1)

class priorstat_1model:
    def logpdf(phi):
        return np.squeeze(sps.norm.logpdf(phi[:,0], 0, 2))
    def rvs(n):
        return np.vstack((sps.norm.rvs(1,1, size = n))).T

class priorstat_2model:
    def logpdf(phi):
        return np.squeeze(priorstat1model.logpdf(phi[:,:2]) +
                          priorstat1model.logpdf(phi[:,2:]))
    def rvs(n):
        return np.hstack((priorstat1model.rvs(n),
                          priorstat1model.rvs(n)))
    
cal_lin = calibrator(emu_lin, y, x,
                    thetaprior = priorphys,
                    phiprior = priorstatdisc_1model,
                    passoptions = {'obsvar': obsvar, 'cov_disc': cov_delta})
pred_lin = cal_lin.predict(xtot)
cal_grav = calibrator(emu_grav, y, x,
                       thetaprior = priorphys,
                       phiprior = priorstatdisc_1model,
                       passoptions = {'obsvar': obsvar, 'cov_disc': cov_delta})
pred_grav = cal_grav.predict(xtot)


class priorstatdisc_1model:
    def logpdf(phi):
        return np.squeeze(sps.norm.logpdf(phi[:,0], 0, 1) +
                          sps.norm.logpdf(phi[:,0], 0, 1))
    def rvs(n):
        return np.vstack((sps.norm.rvs(1,1, size = n),
                         sps.norm.rvs(0,1, size = n))).T

class priorstatdisc_2model:
    def logpdf(phi):
        return np.squeeze(priorstat1model.logpdf(phi[:,:2]) +
                          priorstat1model.logpdf(phi[:,2:]))
    def rvs(n):
        return np.hstack((priorstat1model.rvs(n),
                          priorstat1model.rvs(n)))

def cov_delta(x,phi):
    C0 = np.exp(-1/2*np.abs(np.subtract.outer(x[:,0],x[:,0]))) *\
        (1+1/2*np.abs(np.subtract.outer(x[:,0],x[:,0])))
    adj = np.minimum(np.exp(phi[0] + phi[1] * x[:,0]),20)
    return (np.diag(adj) @ C0 @ np.diag(adj))

def cov_disc(x,k,phi):
    return cov_delta(x, phi[2*k:(2*k+2)])



cal_BMM = calibrator((emu_grav,emu_lin), y, x,
                    thetaprior = priorphys,
                    phiprior = priorstatdisc_2model,
                    passoptions = {'obsvar': obsvar, 'cov_disc': cov_disc})
pred_BMM = cal_BMM.predict(xtot)


import matplotlib.pyplot as plt 
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
from scipy.stats import kde
def two2d(axis, theta):
    nbins = 50
    k = kde.gaussian_kde(theta.T)
    xi, yi = np.mgrid[0:15:nbins*1j, 0:60:nbins*1j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))
    axis.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='gouraud', cmap=plt.cm.BuGn_r)
    axis.contour(xi, yi, zi.reshape(xi.shape))



fig, axes = plt.subplots(ncols=5, nrows=1, figsize=(21, 5))
plotpreds(axes[0], pred_lin)
plotpreds(axes[1], pred_grav)
plotpreds(axes[2], pred_BMM)



fig, axes = plt.subplots(ncols=5, nrows=1, figsize=(21, 5))
two2d(axes[0], priorphys.rvs(1000)[:,(2,1)])
two2d(axes[1], cal_lin.thetadraw[:,(2,1)])
two2d(axes[2], cal_grav.thetadraw[:,(2,1)])
two2d(axes[3], cal_BMM.thetadraw[:,(2,1)])
