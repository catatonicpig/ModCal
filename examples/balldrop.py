# -*- coding: utf-8 -*-
"""Header here."""
import numpy as np
import scipy.stats as sps
import sys
import os

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
from statfuncs.sampling import postsampler
from testfuncs.balldrop import balldropmodel_linear,\
    balldropmodel_quad, balldropmodel_drag, balldroptrue

# t1 = np.array(range(1,12,1))/10
# t1 = np.reshape(t1, (t1.shape[0], -1))
# h1 = 10*np.ones(t1.shape)
# t2 =  np.array(range(1,30,4))/10
# t2 = np.reshape(t2, (t2.shape[0], -1))
# h2 = 40*np.ones(t2.shape)
# t3 =  np.array(range(1,20,3))/10
# t3 = np.reshape(t3, (t3.shape[0], -1))
# h3 = 20*np.ones(t3.shape)
# x1 = np.append(t1, h1, 1)
# x2 = np.append(t2, h2, 1)
# x3 = np.append(t3, h3, 1)
# x = np.vstack((x1,x2,x3))
# y = balldroptrue(x) + np.random.normal(0, np.sqrt(sigma2), x.shape[0])


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
       [ 5.8]])
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


import matplotlib.pyplot as plt

fig, ax = plt.subplots() 
ax.plot(x[:,0],y,'o')

y = np.squeeze(y)
sigma2 = 4
theta0 = np.reshape(np.array((10,10)), (1, 2))
theta_lin = np.reshape(np.array((1, 2.5, 5, 10, 15, 20)), (6, -1))
f_lin = balldropmodel_linear(theta_lin, x)

theta_quad = np.reshape(np.array((5, 10, 15)), (3, -1))
f_quad = balldropmodel_quad(theta_quad, x)

theta_drag = np.array(((1, 5), (10, 5),
                       (15, 1), (10, 20), (5, 20)))
f_drag = balldropmodel_drag(theta_drag, x)


#NEED TO EMULATE EACH MODEL


#NEED TO CALIBRATE EACH MODEL

print(y - balldropmodel_quad(np.array((10,40))[None,:], x))
print(y-balldroptrue(x))
#Prior on thetaasa
def lpriorphys(theta):
    return (sps.gamma.logpdf(theta[:,0], 20, 0, 1/2) +
            sps.gamma.logpdf(theta[:,1], 2, 0, 20))
def dpriorphys(n):
    return np.vstack((sps.gamma.rvs(20, 0, 1/2, size=n),
                     sps.gamma.rvs(2, 0, 20, size=n))).T
thetaprior = dpriorphys(1000)

def llik1(theta):
    return -0.5*np.sum((balldropmodel_quad(theta, x) - y) ** 2 / 1 ** 2,1)
thetapost1 = postsampler(dpriorphys(1000), lpriorphys, llik1)

def llikclosed(theta):
    resid1 = balldropmodel_quad(theta, x) - y
    resid2 = balldropmodel_linear(theta, x) - y
    resid3 = balldropmodel_drag(theta, x) - y
    term1 = -0.5*np.sum((resid1.T) ** 2 / np.sqrt(sigma2),0)
    term2 = -0.5*np.sum((resid2.T) ** 2 / np.sqrt(sigma2),0)
    term3 = -0.5*np.sum((resid3.T) ** 2 / np.sqrt(sigma2),0)
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
CorrMatDeltaT = np.exp(-2*np.abs(np.subtract.outer(x[:,0],x[:,0])))
CorrMatDeltaH = np.exp(-1/40*np.abs(np.subtract.outer(x[:,1],x[:,1])))
CorrMatDelta = CorrMatDeltaT * CorrMatDeltaH
W, V = np.linalg.eigh(CorrMatDelta)

def loglikopen(thetaphi):
    theta = thetaphi[:,:2]
    phi = thetaphi[:,2:]
    Sigobs = np.eye(x.shape[0])
    
    resid1 = balldropmodel_quad(theta, x) - y
    resid2 = balldropmodel_linear(theta, x) - y
    resid3 = balldropmodel_drag(theta, x) - y
    
    #speed up computation
    resid1half = resid1 @ V
    resid2half = resid2 @ V
    resid3half = resid3 @ V
    term1 = -0.5*np.sum((resid1.T / np.sqrt(np.abs(sigma2+np.multiply.outer(W, phi[:,0])))) ** 2,0)
    term2 = -0.5*np.sum((resid2.T / np.sqrt(np.abs(sigma2+np.multiply.outer(W, phi[:,1])))) ** 2,0)
    term3 = -0.5*np.sum((resid3.T / np.sqrt(np.abs(sigma2+np.multiply.outer(W, phi[:,2])))) ** 2,0)
    term1 -= 0.5*np.sum(np.log(np.abs(sigma2+np.multiply.outer(W, phi[:,0]))),0)
    term2 -= 0.5*np.sum(np.log(np.abs(sigma2+np.multiply.outer(W, phi[:,1]))),0)
    term3 -= 0.5*np.sum(np.log(np.abs(sigma2+np.multiply.outer(W, phi[:,2]))),0)
    terms = np.vstack((term1, term2, term3))
    tm = np.max(terms,0)
    termsadj = terms - tm
    logpost = tm + np.log(np.sum(np.exp(termsadj),0))
    return logpost

thetapostopen = postsampler(dprioropen(1000), lprioropen, loglikopen)

print(np.mean(thetapostopen,0))


fig, axes = plt.subplots(ncols=3, nrows=1, figsize=(21, 5))

from scipy.stats import kde
def two2d(axis, theta):
    nbins = 50
    k = kde.gaussian_kde(theta.T)
    xi, yi = np.mgrid[6:13:nbins*1j, 0:60:nbins*1j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))
    axis.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='gouraud', cmap=plt.cm.BuGn_r)
    axis.contour(xi, yi, zi.reshape(xi.shape))

two2d(axes[0], thetaprior)
two2d(axes[1], thetapostclosed)
two2d(axes[2], thetapostopen[:,:2])
#NEED TO DO SOME MODEL MIXING

#NEED TO DECIDE A NEW HEIGHT TO DROP IT AT