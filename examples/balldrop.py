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

import matplotlib.pyplot as plt 
def plotpreds(axis, preddict):
    for k in (20,40,60):
        inds = np.where(xtot[:,1] == k)[0]
        uppercurve = preddict['mean'][inds] + 3*np.sqrt(preddict['var'][inds])
        lowercurve = preddict['mean'][inds] - 3*np.sqrt(preddict['var'][inds])
        axis.fill_between(xtot[inds,0], lowercurve, uppercurve, color='k', alpha=0.2)
        axis.plot(xtot[inds,0],preddict['mean'][inds],'k-')
        axis.plot(xtot[inds,0],uppercurve, 'k-', alpha=0.6,linewidth=0.5)
        axis.plot(xtot[inds,0],lowercurve, 'k-', alpha=0.6,linewidth=0.5)
    axis.plot(x,y, 'ko')
    axis.set_xlim([0,5.6])
    axis.set_ylim([-5,70])
from scipy.stats import kde
def two2d(axis, theta):
    nbins = 50
    k = kde.gaussian_kde(theta.T)
    xi, yi = np.mgrid[0:15:nbins*1j, 0:60:nbins*1j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))
    axis.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='gouraud', cmap=plt.cm.BuGn_r)
    axis.contour(xi, yi, zi.reshape(xi.shape))



class priorphys:
    """ This defines the class instance of priors provided to the software. """
    def logpdf(theta):
        return np.squeeze(sps.norm.logpdf(theta[:, 0], 0, 20) +  # initial height deviation
                          sps.gamma.logpdf(theta[:, 1], 1, 0, 40) +  # terminal velocity
                          sps.gamma.logpdf(theta[:, 2], 5, 0, 2))  # gravity

    def rvs(n):
        return np.vstack((sps.norm.rvs(0, 20, size=n),  # initial height deviation
                          sps.gamma.rvs(1, 0, 40, size=n),  # terminal velocity
                          sps.gamma.rvs(5, 0, 2, size=n))).T  # gravity


tvec = np.concatenate((np.arange(0.1, 3.0, 0.1),
                       np.arange(0.1, 4.2, 0.1),
                       np.arange(0.1, 5.6, 0.1)))  # the time vector of interest
hvec = np.concatenate((20 * np.ones(29),
                       40 * np.ones(41),
                       60 * np.ones(55)))  # the drop heights vector of interest
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

x = np.array([[ 0.3, 20. ],
              [ 0.4, 20. ],
              [ 0.5, 20. ],
        [ 0.6, 20. ],
        [ 0.8, 20. ],
        [ 1.0, 20. ],
        [ 1.2, 20. ],
        [ 1.8, 20. ],
        [ 0.4, 40. ],
        [ 0.8, 40. ],
        [ 1.2, 40. ],
        [ 1.6, 40. ],
        [ 2.0, 40. ],
        [ 2.4, 40. ],
        [ 2.8, 40. ],
        [ 3.2, 40. ],
        [ 2.4, 60. ],
        [ 2.8, 60. ],
        [ 3.2, 60. ],
        [ 3.6, 60. ],
        [ 4.0, 60. ],
        [ 4.2, 60. ],
        [ 4.4, 60. ],
        [ 4.6, 60. ]])
obsvar = 4*np.ones(x.shape[0])  # variance for the observations in 'y' below
y = balldroptrue(x) + sps.norm.rvs(0, np.sqrt(obsvar)) #observations at each row of 'x'


class priorstat_1model:
    def logpdf(phi):
        return np.squeeze(sps.norm.logpdf(phi, 1, 1))
    def rvs(n):
        return sps.norm.rvs(1,1, size = n).reshape((-1,1))

class priorstat_2model:
    def logpdf(phi):
        return np.squeeze(priorstat_1model.logpdf(phi[:,0]) +
                          priorstat_1model.logpdf(phi[:,1]))
    def rvs(n):
        return np.hstack((priorstat_1model.rvs(n),
                          priorstat_1model.rvs(n)))
    
    
cal_lin = calibrator(emu_lin, y, x,
                    thetaprior = priorphys,
                    phiprior = priorstat_1model,
                    passoptions = {'obsvar': obsvar}, software = 'BMA')
pred_lin = cal_lin.predict(xtot)

cal_grav = calibrator(emu_grav, y, x,
                       thetaprior = priorphys,
                       phiprior = priorstat_1model,
                       passoptions = {'obsvar': obsvar}, software = 'BMA')
pred_grav = cal_grav.predict(xtot)

cal_BMA = calibrator((emu_lin,emu_grav), y, x,
                       thetaprior = priorphys,
                       phiprior = priorstat_2model,
                       passoptions = {'obsvar': obsvar}, software = 'BMA')
pred_BMA = cal_BMA.predict(xtot)

fig, axes = plt.subplots(ncols=4, nrows=1, figsize=(21, 5))
two2d(axes[0], priorphys.rvs(1000)[:,(2,1)])
two2d(axes[1], cal_lin.thetadraw[:,(2,1)])
two2d(axes[2], cal_grav.thetadraw[:,(2,1)])
two2d(axes[3], cal_BMA.thetadraw[:,(2,1)])

fig, axes = plt.subplots(ncols=4, nrows=1, figsize=(21, 5))
plotpreds(axes[1], pred_lin)
plotpreds(axes[2], pred_grav)
plotpreds(axes[3], pred_BMA)


class priorstatdisc_1model:
    def logpdf(phi):
        return np.squeeze(sps.norm.logpdf(phi[:,0], 1, 1) +
                          sps.norm.logpdf(phi[:,0], 0, 1))
    def rvs(n):
        return np.vstack((sps.norm.rvs(1,1, size = n),
                         sps.norm.rvs(0,1, size = n))).T

class priorstatdisc_2model:
    def logpdf(phi):
        return np.squeeze(priorstatdisc_1model.logpdf(phi[:,:2]) +
                          priorstatdisc_1model.logpdf(phi[:,2:]))
    def rvs(n):
        return np.hstack((priorstatdisc_1model.rvs(n),
                          priorstatdisc_1model.rvs(n)))

def cov_delta(x,phi):
    C0 = np.exp(-1/3*np.abs(np.subtract.outer(x[:,0],x[:,0]))) #*\
        #(1+np.abs(np.subtract.outer(x[:,0],x[:,0])))
    adj = np.minimum(np.exp(phi[0] + phi[1]*x[:,0] - np.maximum(0,phi[1]*5)),
                     50)
    return (np.diag(adj) @ C0 @ np.diag(adj))

def cov_disc(x,k,phi):
    return cov_delta(x, phi[2*k:(2*k+2)])

cal_lin = calibrator(emu_lin, y, x,
                    thetaprior = priorphys,
                    phiprior = priorstatdisc_1model,
                    passoptions = {'obsvar': obsvar, 'cov_disc': cov_delta}, software = 'BDM')
pred_lin = cal_lin.predict(xtot)

cal_grav = calibrator(emu_grav, y, x,
                       thetaprior = priorphys,
                       phiprior = priorstatdisc_1model,
                       passoptions = {'obsvar': obsvar, 'cov_disc': cov_delta}, software = 'BDM')
pred_grav = cal_grav.predict(xtot)

cal_BMM = calibrator((emu_grav,emu_lin), y, x,
                    thetaprior = priorphys,
                    phiprior = priorstatdisc_2model,
                    passoptions = {'obsvar': obsvar, 'cov_disc': cov_disc}, software = 'BDM')
pred_BMM = cal_BMM.predict(xtot)

fig, axes = plt.subplots(ncols=4, nrows=1, figsize=(21, 5))
two2d(axes[0], priorphys.rvs(1000)[:,(2,1)])
two2d(axes[1], cal_lin.thetadraw[:,(2,1)])
two2d(axes[2], cal_grav.thetadraw[:,(2,1)])
two2d(axes[3], cal_BMM.thetadraw[:,(2,1)])

fig, axes = plt.subplots(ncols=4, nrows=1, figsize=(21, 5))
plotpreds(axes[1], pred_lin)
plotpreds(axes[2], pred_grav)
plotpreds(axes[3], pred_BMM)