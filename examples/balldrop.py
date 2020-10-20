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
    for k in (20,40):
        inds = np.where(xtot[:,1] == k)[0]
        #uppercurve = preddict['mean'][inds] + 3*np.sqrt(preddict['var'][inds])
        #lowercurve = preddict['mean'][inds] - 3*np.sqrt(preddict['var'][inds])
        #axis.fill_between(xtot[inds,0], lowercurve, uppercurve, color='k', alpha=0.2)
        #for l in range(0,preddict['draws'].shape[0]):
        #    axis.plot(xtot[inds,0],preddict['draws'][l, inds],'k-', alpha=0.1,linewidth=0.1)
        uppercurve = np.quantile(preddict['draws'][:, inds],0.975,0)
        lowercurve = np.quantile(preddict['draws'][:, inds],0.025,0)
        #axis.plot(xtot[inds,0],uppercurve, 'k-', alpha=0.6,linewidth=0.5)
        axis.plot(xtot[inds,0],preddict['mean'][inds], 'k-', alpha=0.5,linewidth=0.5)
        axis.fill_between(xtot[inds,0], lowercurve, uppercurve, color='k', alpha=0.25)
    axis.plot(x,y, 'ro' ,markersize = 8)
    axis.set_xlim([0,3.8])
    axis.set_ylim([-2,42])
from scipy.stats import kde
def two2d(axis, theta):
    nbins = 50
    k = kde.gaussian_kde(theta.T)
    xi, yi = np.mgrid[0:18:nbins*1j, 0:40:nbins*1j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))
    axis.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='gouraud', cmap=plt.cm.BuGn_r)
    axis.contour(xi, yi, zi.reshape(xi.shape))



class priorphys:
    """ This defines the class instance of priors provided to the software. """
    def logpdf(theta):
        return np.squeeze(sps.norm.logpdf(theta[:, 0], 0, 5) +  # initial height deviation
                          sps.gamma.logpdf(theta[:, 1], 2, 0, 10) +  # terminal velocity
                          sps.gamma.logpdf(theta[:, 2], 2, 0, 5))  # gravity

    def rvs(n):
        return np.vstack((sps.norm.rvs(0, 5, size=n),  # initial height deviation
                          sps.gamma.rvs(2, 0, 10, size=n),  # terminal velocity
                          sps.gamma.rvs(2, 0, 5, size=n))).T  # gravity


tvec = np.concatenate((np.arange(0.1, 2.9, 0.1),
                       np.arange(0.1, 4.1, 0.1)))  # the time vector of interest
hvec = np.concatenate((20 * np.ones(28),
                       40 * np.ones(40)))  # the drop heights vector of interest
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

x = np.array([[ 0.1, 20. ],
              [ 0.2, 20. ],
        [ 0.3, 20. ],
        [ 0.4, 20. ],
        [ 0.5, 20. ],
        [ 0.7, 20. ],
        [ 2.2, 20. ],
        [ 0.1, 40. ],
        [ 0.2, 40. ],
        [ 0.3, 40. ],
        [ 0.4, 40. ],
        [ 0.6, 40. ],
        [ 0.8, 40. ],
        [ 2.3, 40. ],
        [ 2.5, 40. ],
        [ 2.7, 40. ],
        [ 2.9, 40. ],
        [3.1, 40. ],])
obsvar = 0.25*np.ones(x.shape[0])  # variance for the observations in 'y' below
y = balldroptrue(x) + sps.norm.rvs(0, np.sqrt(obsvar)) #observations at each row of 'x'


class priorstat_1model:
    def logpdf(phi):
        return np.squeeze(sps.norm.logpdf(phi, 2, 2))
    def rvs(n):
        return sps.norm.rvs(2,2, size = n).reshape((-1,1))

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


    
class priorstatdisc_modela:
    def logpdf(phi):
        return np.squeeze(sps.gamma.logpdf(np.exp(phi[:,0]), 6, 0, 1) + phi[:,0] +
                          sps.norm.logpdf(phi[:,1], -2, 0.1))
    def rvs(n):
        return np.vstack((np.log(sps.gamma.rvs(6,0,1, size = n )),
                         sps.norm.rvs(-2, 0.1, size = n))).T

class priorstatdisc_modelb:
    def logpdf(phi):
        return np.squeeze(sps.gamma.logpdf(np.exp(phi[:,0]), 6, 0, 1) + phi[:,0] +
                          sps.norm.logpdf(phi[:,1], 2, 0.1))
    def rvs(n):
        return np.vstack((np.log(sps.gamma.rvs(6, 0, 1, size = n )),
                         sps.norm.rvs(2, 0.1, size = n))).T
    
class priorstatdisc_models:
    def logpdf(phi):
        return np.squeeze(priorstatdisc_modela.logpdf(phi[:,:2]) +
                          priorstatdisc_modelb.logpdf(phi[:,2:]))
    def rvs(n):
        return np.hstack((priorstatdisc_modela.rvs(n),
                          priorstatdisc_modelb.rvs(n)))


def cov_delta(x,phi):
    C0 = np.exp(-4*np.abs(np.subtract.outer(np.sqrt(x[:,0]),np.sqrt(x[:,0]))) ** 1.5) #*\
        #(1+5*np.abs(np.subtract.outer(np.sqrt(x[:,0]),np.sqrt(x[:,0]))))
    if np.abs(phi[1]) < 0.00000001:
        adj = 1
    else:
        adj = 3 * phi[1] / (np.exp((phi[1])*3)-1)
    adj = np.minimum(np.exp(phi[0] + phi[1]*(x[:,0] ** 0.75)) * adj,
                     100)
    return (np.diag(adj) @ C0 @ np.diag(adj))

def cov_disc(x,k,phi):
    return cov_delta(x, phi[2*k:(2*k+2)])

cal_lin = calibrator(emu_lin, y, x,
                    thetaprior = priorphys,
                    phiprior = priorstatdisc_modela,
                    passoptions = {'obsvar': obsvar, 'cov_disc': cov_delta}, software = 'BDM')
pred_lin = cal_lin.predict(xtot)

cal_grav = calibrator(emu_grav, y, x,
                       thetaprior = priorphys,
                       phiprior = priorstatdisc_modelb,
                       passoptions = {'obsvar': obsvar, 'cov_disc': cov_delta}, software = 'BDM')
pred_grav = cal_grav.predict(xtot)

cal_BMM = calibrator((emu_lin,emu_grav), y, x,
                    thetaprior = priorphys,
                    phiprior = priorstatdisc_models,
                    passoptions = {'obsvar': obsvar, 'cov_disc': cov_disc}, software = 'BDM')
pred_BMM = cal_BMM.predict(xtot) #, theta = cal_lin.thetadraw, phi = cal_BMM.phidraw

fig, axes = plt.subplots(ncols=4, nrows=1, figsize=(21, 5))
two2d(axes[0], priorphys.rvs(1000)[:,(2,1)])
two2d(axes[1], cal_lin.thetadraw[:,(2,1)])
two2d(axes[2], cal_grav.thetadraw[:,(2,1)])
two2d(axes[3], cal_BMM.thetadraw[:,(2,1)])

fig, axes = plt.subplots(ncols=3, nrows=1, figsize=(21, 5))
plotpreds(axes[0], pred_lin)
plotpreds(axes[1], pred_grav)
plotpreds(axes[2], pred_BMM)

matchingvec = np.where(((x[:, None] > xtot - 1e-08) * (x[:, None] < xtot + 1e-08)).all(2))
xind = matchingvec[1][matchingvec[0]]
print(y - np.squeeze(emu_grav.predict(cal_grav.thetadraw[1],x)['mean'])[xind])
phi = cal_grav.phidraw[1]
adj = 2 * phi[1] / (np.exp((phi[1])*2)-1)
adj = np.minimum(np.exp(phi[0] + phi[1]*np.sqrt(x[:,0])) * adj,
             250)
print(adj)

cov_delta(x,phi)
phialt = cal_grav.phidraw
phialt[:,0] = 2