# -*- coding: utf-8 -*-
"""Header here."""
import numpy as np
import scipy.stats as sps
import sys
import os
import copy

#the following imports are sloppy until a module is actually built...
from balldroptestfuncs import balldropmodel_linear,\
    balldropmodel_grav, balldroptrue
SCRIPT_DIR = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, '..')))
from base.emulation import emulator
from base.calibration import calibrator
#sorry about that!

class priorphys_lin:
    """ This defines the class instance of priors provided to the software. """
    def lpdf(theta):
        return np.squeeze(sps.norm.logpdf(theta[:, 0], 0, 5) +  # initial height deviation
                          sps.gamma.logpdf(theta[:, 1], 2, 0, 10))   # terminal velocity

    def rnd(n):
        return np.vstack((sps.norm.rvs(0, 5, size=n),  # initial height deviation
                          sps.gamma.rvs(2, 0, 10, size=n))).T  # terminal velocity

class priorphys_grav:
    """ This defines the class instance of priors provided to the software. """
    def lpdf(theta):
        return np.squeeze(sps.gamma.logpdf(theta, 2, 0, 5))  # gravity

    def rnd(n):
        return np.reshape(sps.gamma.rvs(2, 0, 5, size=n), (-1,1))  # gravity

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
thetacompexp_lin = priorphys_lin.rnd(50)  # drawing 50 rndom parameters from the prior
linear_results = balldropmodel_linear(xtotv, thetacompexp_lin)  # the value of the linear simulation
# This is for all vectors in the input of interest
emu_lin = emulator(xtot, thetacompexp_lin, linear_results, software = 'PCGPwM')  # this builds an emulator 
# for the linear simulation. this is a specialized class specific to ModCal.

thetacompexp_grav = priorphys_grav.rnd(20)  # drawing 50 rndom parameters from the prior
grav_results = balldropmodel_grav(xtotv, thetacompexp_grav)  # the value of the gravity simulation
emu_grav = emulator(xtot, thetacompexp_grav, grav_results, software = 'PCGPwM')  # this builds an
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

    
class priorstatdisc_model:
    def lpdf(phi):
        return np.squeeze(sps.norm.logpdf(phi[:,0], 2, 2) +
                          sps.norm.logpdf(phi[:,1], 0, 2))
    def rnd(n):
        return np.vstack((sps.norm.rvs(2, 2, size = n ),
                         sps.norm.rvs(0, 2, size = n))).T
def cov_delta(x,phi):
    xv = x[:,0].astype(float)
    C0 = np.exp(-1/2*np.abs(np.subtract.outer(np.sqrt(xv),np.sqrt(xv)))) *\
        (1+1/2*np.abs(np.subtract.outer(np.sqrt(xv),np.sqrt(xv))))
    adj = 20 / (1+np.exp(phi[1]*(xv - phi[0])))
    return (np.diag(adj) @ C0 @ np.diag(adj))

cal_lin = calibrator(emu_lin, y, x, # need to build a calibrator
                    thetaprior = priorphys_lin,
                    software = 'BDM',
                    yvar = obsvar,
                    args = {'cov_disc': cov_delta,
                               'phiprior': priorstatdisc_model})# the arguments are being passed 
                                                                # to the BDM software
pred_lin = cal_lin.predict(xtot) # getting a prediction object

cal_grav = calibrator(emu_grav, y, x, # need to build a calibrator
                       thetaprior = priorphys_grav,
                       software = 'BDM',
                    yvar = obsvar,
                    args = { 'cov_disc': cov_delta,
                               'phiprior': priorstatdisc_model}) # the arguments are being passed 
                                                                # to the BDM software
pred_grav = cal_grav.predict(xtot) # getting a prediction object

import matplotlib.pyplot as plt 
from scipy.stats import kde
def two2d(axis, theta):
    nbins = 50
    k = kde.gaussian_kde(theta.T)
    xi, yi = np.mgrid[-10:10:nbins*1j, 0:20:nbins*1j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))
    axis.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='gouraud', cmap=plt.cm.BuGn_r)
    axis.contour(xi, yi, zi.reshape(xi.shape))

fig1, ax1 = plt.subplots()
two2d(ax1, cal_lin.theta(2000))
ax1.set_xlabel('drop offset')
ax1.set_ylabel('terminal velocity')
ax1.set_title('density plot for the parameter of the linear model')

fig2, ax2 = plt.subplots()
ax2.hist(cal_grav.theta(2000), bins=30)
ax2.set_xlabel('gravity')
ax2.set_ylabel('frequency')
ax2.set_title('histogram for gravity')


def plotpreds(axis, pred):
    preds = pred.rnd(1000)
    for k in (25,50):
        inds = np.where(xtotv[:,1] == k)[0]
        for l in range(0,1000):
            axis.plot(xtotv[inds,0],preds[l, inds],'k-', alpha=0.01,linewidth=0.1)
        uppercurve = np.quantile(preds[:, inds],0.975,0)
        lowercurve = np.quantile(preds[:, inds],0.025,0)
        p4 = axis.plot(xtotv[inds,0], balldroptrue(xtotv[inds,:]), 'k--',linewidth=2)
        axis.fill_between(xtotv[inds,0], lowercurve, uppercurve, color='k', alpha=0.25)
    p1 = axis.plot(np.NaN, np.NaN, color='k', linewidth=3)
    p2 = axis.fill(np.NaN, np.NaN, 'k', alpha=0.5)
    p3 = axis.plot(xv,y, 'ro' ,markersize = 8)
    axis.set_xlim([0,4.2])
    axis.set_ylim([-5,55])
    axis.set_xlabel('time')
    axis.set_ylabel('distance')
    axis.legend([p4[0],(p2[0], p1[0]), p3[0]], ['truth','prediction','observations'])
    
fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(10, 5))
plotpreds(axes[0], pred_lin)
axes[0].set_title('prediction using linear model')
plotpreds(axes[1], pred_grav)
axes[1].set_title('prediction using gravity model')