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
from base.emulation import emulator
from base.calibration import calibrator

import matplotlib.pyplot as plt 
def plotpreds(axis, preddict):
    for k in (25,50):
        inds = np.where(xtotv[:,1] == k)[0]
        #uppercurve = preddict['mean'][inds] + 3*np.sqrt(preddict['var'][inds])
        #lowercurve = preddict['mean'][inds] - 3*np.sqrt(preddict['var'][inds])
        #axis.fill_between(xtot[inds,0], lowercurve, uppercurve, color='k', alpha=0.2)
        for l in range(0,preddict['modeldraws'].shape[0]):
            axis.plot(xtotv[inds,0],preddict['modeldraws'][l, inds],'k-', alpha=0.01,linewidth=0.1)
        uppercurve = np.quantile(preddict['modeldraws'][:, inds],0.975,0)
        lowercurve = np.quantile(preddict['modeldraws'][:, inds],0.025,0)
        #axis.plot(xtot[inds,0],uppercurve, 'k-', alpha=0.6,linewidth=0.5)
        #axis.plot(xtot[inds,0],preddict['mean'][inds], 'k-', alpha=0.5,linewidth=0.5)
        axis.plot(xtotv[inds,0], balldroptrue(xtotv[inds,:]), 'k--',linewidth=2)
        axis.fill_between(xtotv[inds,0], lowercurve, uppercurve, color='k', alpha=0.25)
    axis.plot(xv,y, 'ro' ,markersize = 8)
    axis.set_xlim([0,4.2])
    axis.set_ylim([-5,55])
from scipy.stats import kde
def two2d(axis, theta):
    nbins = 50
    k = kde.gaussian_kde(theta.T)
    xi, yi = np.mgrid[0:18:nbins*1j, 0:50:nbins*1j]
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

#emaphsize they should contain the dashed lines
#show which model is being used
#template for the group now (later outside)
        

tvec = np.concatenate((np.arange(0.1, 4.3, 0.1),
                       np.arange(0.1, 4.3, 0.1)))  # the time vector of interest
hvec = np.concatenate((25 * np.ones(42),
                       50 * np.ones(42)))  # the drop heights vector of interest
xtot = (np.vstack((tvec, hvec)).T).astype('object')  # the input of interest
xtotv= xtot.astype('float')
xtot[xtot[:,1] == 25, 1] = 'lowdrop'
xtot[xtot[:,1] == 50, 1] = 'highdrop'
# each row is an individual vector of interest
# this should include those important to the study AND the data

# we now create a computer experiment to build an emulator
thetacompexp = priorphys.rvs(50)  # drawing 50 random parameters from the prior

linear_results = balldropmodel_linear(thetacompexp, xtotv)  # the value of the linear simulation
# This is for all vectors in the input of interest
emu_lin = emulator(thetacompexp, linear_results, xtot)  # this builds an emulator 
# for the linear simulation. this is a specialized class specific to ModCal.
grav_results = balldropmodel_grav(thetacompexp, xtotv)  # the value of the gravity simulation
emu_grav = emulator(thetacompexp, grav_results, xtot)  # this builds an
# emulator for the gravity simulation. this is a specialized class specific to ModCal.

x = np.array([[ 0.1, 25. ],
              [ 0.2, 25. ],
        [ 0.3, 25. ],
        [ 0.4, 25. ],
        [ 0.5, 25. ],
        [ 0.6, 25. ],
        [ 0.7, 25. ],
        [ 0.9, 25. ],
        [ 1.1, 25. ],
        [ 1.3, 25. ],
        [ 2.0, 25. ],
        [ 2.4, 25. ],
        [ 0.1, 50. ],
        [ 0.2, 50. ],
        [ 0.3, 50. ],
        [ 0.4, 50. ],
        [ 0.5, 50. ],
        [ 0.6, 50. ],
        [ 0.7, 50. ],
        [ 0.8, 50. ],
        [ 0.9, 50. ],
        [ 1.0, 50. ],
        [ 1.2, 50. ],
        [ 3.5, 50. ],
        [ 3.7, 50. ],
        [ 2.6, 50. ],
        [ 2.9, 50. ],
        [3.1, 50. ],
        [3.3, 50. ],]).astype('object')
xv = x.astype('float')
x[x[:,1] == 25, 1] = 'lowdrop'
x[x[:,1] == 50, 1] = 'highdrop'
obsvar = 4*np.ones(x.shape[0])  # variance for the observations in 'y' below
y = balldroptrue(xv) + sps.norm.rvs(0, np.sqrt(obsvar)) #observations at each row of 'x'

    
class priorstatdisc_modela:
    def logpdf(phi):
        return np.squeeze(sps.norm.logpdf(phi[:,0], 2, 0.5) +
                          sps.norm.logpdf(phi[:,1], 3, 0.5))
    def rvs(n):
        return np.vstack((sps.norm.rvs(2, 0.5, size = n ),
                         sps.norm.rvs(3, 0.5, size = n))).T

class priorstatdisc_modelb:
    def logpdf(phi):
        return np.squeeze(sps.norm.logpdf(phi[:,0], 2, 0.5) +
                          sps.norm.logpdf(phi[:,1], -4, 0.5))
    def rvs(n):
        return np.vstack((sps.norm.rvs(2, 0.5, size = n ),
                         sps.norm.rvs(-4, 0.5, size = n))).T
    
class priorstatdisc_models:
    def logpdf(phi):
        return np.squeeze(priorstatdisc_modela.logpdf(phi[:,:2]) +
                          priorstatdisc_modelb.logpdf(phi[:,2:]))
    def rvs(n):
        return np.hstack((priorstatdisc_modela.rvs(n),
                          priorstatdisc_modelb.rvs(n)))


def cov_delta(x,phi):
    xv = x[:,0].astype(float)
    C0 = np.exp(-1/2*np.abs(np.subtract.outer(np.sqrt(xv),np.sqrt(xv)))) *\
        (1+1/2*np.abs(np.subtract.outer(np.sqrt(xv),np.sqrt(xv))))
    adj = 20 / (1+np.exp(phi[1]*(xv - phi[0])))
    return (np.diag(adj) @ C0 @ np.diag(adj))

def cov_disc(x,k,phi):
    return cov_delta(x, phi[2*k:(2*k+2)])

cal_lin = calibrator(emu_lin, y, x,
                    thetaprior = priorphys,
                    software = 'BMA',
                    args = {'obsvar': obsvar, 'cov_disc': cov_delta,
                               'phiprior': priorstatdisc_modela})
pred_lin = cal_lin.predict(xtot)

cal_grav = calibrator(emu_grav, y, x,
                       thetaprior = priorphys,
                       software = 'BMA',
                    args = {'obsvar': obsvar, 'cov_disc': cov_delta,
                               'phiprior': priorstatdisc_modelb})
pred_grav = cal_grav.predict(xtot)

cal_BMA = calibrator((emu_lin,emu_grav), y, x,
                    thetaprior = priorphys,
                    software = 'BMA',
                    args = {'obsvar': obsvar, 'cov_disc': cov_disc,
                            'phiprior': priorstatdisc_models})
pred_BMA = cal_BMA.predict(xtot)

cal_BMM = calibrator((emu_lin,emu_grav), y, x,
                    thetaprior = priorphys,
                    software = 'BDM',
                    args = {'obsvar': obsvar, 'cov_disc': cov_disc,
                            'phiprior': priorstatdisc_models})
pred_BMM = cal_BMM.predict(xtot)

fig, axes = plt.subplots(ncols=4, nrows=1, figsize=(21, 5))
two2d(axes[0], cal_lin.theta.rvs(1000)[:,(2,1)])
two2d(axes[1], cal_grav.theta.rvs(1000)[:,(2,1)])
two2d(axes[2], cal_BMA.theta.rvs(1000)[:,(2,1)])
two2d(axes[3], cal_BMM.theta.rvs(1000)[:,(2,1)])

fig, axes = plt.subplots(ncols=4, nrows=1, figsize=(21, 5))
plotpreds(axes[0], pred_lin)
plotpreds(axes[1], pred_grav)
plotpreds(axes[2], pred_BMA)
plotpreds(axes[3], pred_BMM)