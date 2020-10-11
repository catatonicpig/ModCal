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
from statfuncs.emulation import emulator
from testfuncs.balldrop import balldropmodel_linear,\
    balldropmodel_quad, balldropmodel_drag, balldroptrue

def lpriorphys(theta):
    return (sps.gamma.logpdf(theta[:,0], 20, 0, 1/2) +
            sps.gamma.logpdf(theta[:,1], 2, 0, 20))
def dpriorphys(n):
    return np.vstack((sps.gamma.rvs(20, 0, 1/2, size=n),
                     sps.gamma.rvs(2, 0, 20, size=n))).T

xdata = np.array([[ 0.1, 10. ],
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

thetacompexp = dpriorphys(100)
fcompexp = balldropmodel_quad(thetacompexp, xdata) 

emu = emulator(thetacompexp, fcompexp)
thetacomptest = dpriorphys(100)
ftest = balldropmodel_quad(thetacomptest, xdata) 
Y = emu.predict(thetacomptest)[0]
print(np.sqrt(np.mean((Y-ftest)**2)))
# #NEED TO CALIBRATE EACH MODEL
# import matplotlib.pyplot as plt
# sigma2 = 4
# y = np.array([[11.5],
#        [11.2],
#        [ 9.6],
#        [ 7.9],
#        [ 9.6],
#        [ 6.8],
#        [ 7.3],
#        [ 7.9],
#        [ 6.5],
#        [ 2.2],
#        [ 2.8],
#        [21.6],
#        [20.3],
#        [18.8],
#        [12.1],
#        [11.8],
#        [10.6],
#        [ 3.9],
#        [40.4],
#        [36.2],
#        [36. ],
#        [32.6],
#        [24.8],
#        [19.7],
#        [13.1],
#        [ 5.8]])
# y = np.squeeze(y)
# fig, ax = plt.subplots() 
# ax.plot(x[:,0],y,'o')

# print(y - balldropmodel_quad(np.array((10,40))[None,:], x))
# print(y-balldroptrue(x))
# #Prior on thetaasa

# def llik1(theta):
#     return -0.5*np.sum((balldropmodel_quad(theta, x) - y) ** 2 / 1 ** 2,1)
# thetapost1 = postsampler(dpriorphys(1000), lpriorphys, llik1)

# def llikclosed(theta):
#     resid1 = balldropmodel_quad(theta, x) - y
#     resid2 = balldropmodel_linear(theta, x) - y
#     resid3 = balldropmodel_drag(theta, x) - y
#     term1 = -0.5*np.sum((resid1.T) ** 2 / np.sqrt(sigma2),0)
#     term2 = -0.5*np.sum((resid2.T) ** 2 / np.sqrt(sigma2),0)
#     term3 = -0.5*np.sum((resid3.T) ** 2 / np.sqrt(sigma2),0)
#     terms = np.vstack((term1,term2,term3))
#     tm = np.max(terms,0)
#     termsadj = terms - tm
#     logpost = tm + np.log(np.sum(np.exp(termsadj),0))
#     return logpost
# thetapostclosed = postsampler(dpriorphys(1000), lpriorphys, llikclosed)


# def lpriorstat(phi):
#     return (sps.gamma.logpdf(phi[:,0], 4, 0, 1) +
#             sps.gamma.logpdf(phi[:,1], 4, 0, 1) +
#             sps.gamma.logpdf(phi[:,2], 4, 0, 1))
# def dpriorstat(n):
#     return np.vstack((sps.gamma.rvs(4, 0, 1, size=n),
#                      sps.gamma.rvs(4, 0, 1, size=n),
#                      sps.gamma.rvs(4, 0, 1, size=n))).T

# def lprioropen(thetaphi):
#     return (lpriorphys(thetaphi[:,:2]) + lpriorstat(thetaphi[:,2:]))
# def dprioropen(n):
#     return np.vstack((dpriorphys(n).T, dpriorstat(n).T)).T


# #speed up computation
# CorrMatDeltaT = np.exp(-2*np.abs(np.subtract.outer(x[:,0],x[:,0])))
# CorrMatDeltaH = np.exp(-1/40*np.abs(np.subtract.outer(x[:,1],x[:,1])))
# CorrMatDelta = CorrMatDeltaT * CorrMatDeltaH
# W, V = np.linalg.eigh(CorrMatDelta)

# def loglikopen(thetaphi):
#     theta = thetaphi[:,:2]
#     phi = thetaphi[:,2:]
#     Sigobs = np.eye(x.shape[0])
    
#     resid1 = balldropmodel_quad(theta, x) - y
#     resid2 = balldropmodel_linear(theta, x) - y
#     resid3 = balldropmodel_drag(theta, x) - y
    
#     #speed up computation
#     resid1half = resid1 @ V
#     resid2half = resid2 @ V
#     resid3half = resid3 @ V
#     term1 = -0.5*np.sum((resid1.T / np.sqrt(np.abs(sigma2+np.multiply.outer(W, phi[:,0])))) ** 2,0)
#     term2 = -0.5*np.sum((resid2.T / np.sqrt(np.abs(sigma2+np.multiply.outer(W, phi[:,1])))) ** 2,0)
#     term3 = -0.5*np.sum((resid3.T / np.sqrt(np.abs(sigma2+np.multiply.outer(W, phi[:,2])))) ** 2,0)
#     term1 -= 0.5*np.sum(np.log(np.abs(sigma2+np.multiply.outer(W, phi[:,0]))),0)
#     term2 -= 0.5*np.sum(np.log(np.abs(sigma2+np.multiply.outer(W, phi[:,1]))),0)
#     term3 -= 0.5*np.sum(np.log(np.abs(sigma2+np.multiply.outer(W, phi[:,2]))),0)
#     terms = np.vstack((term1, term2, term3))
#     tm = np.max(terms,0)
#     termsadj = terms - tm
#     logpost = tm + np.log(np.sum(np.exp(termsadj),0))
#     return logpost

# thetapostopen = postsampler(dprioropen(1000), lprioropen, loglikopen)

# print(np.mean(thetapostopen,0))


# fig, axes = plt.subplots(ncols=3, nrows=1, figsize=(21, 5))

# from scipy.stats import kde
# def two2d(axis, theta):
#     nbins = 50
#     k = kde.gaussian_kde(theta.T)
#     xi, yi = np.mgrid[6:13:nbins*1j, 0:60:nbins*1j]
#     zi = k(np.vstack([xi.flatten(), yi.flatten()]))
#     axis.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='gouraud', cmap=plt.cm.BuGn_r)
#     axis.contour(xi, yi, zi.reshape(xi.shape))

# two2d(axes[0], thetaprior)
# two2d(axes[1], thetapostclosed)
# two2d(axes[2], thetapostopen[:,:2])
# #NEED TO DO SOME MODEL MIXING

# #NEED TO DECIDE A NEW HEIGHT TO DROP IT AT