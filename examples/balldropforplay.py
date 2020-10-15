# -*- coding: utf-8 -*-
"""Header here."""
import numpy as np
import scipy.stats as sps
import scipy.linalg as spla
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
            sps.norm.logpdf(theta[:,2], 0, 2))
    def rvs(n):
        return np.vstack((sps.gamma.rvs(2, 0, 5, size=n),
                     sps.gamma.rvs(1, 0, 40, size=n),
                     sps.norm.rvs(0, 2, size=n))).T

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
        [ 1.2, 40. ],
        [ 1.6, 40. ],
        [ 2.0, 40. ],
        [ 2.4, 40. ],
        [ 3.0, 40. ],
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


# yt = balldroptruealt(xtot)

# diffadj = (yt - ft) + (ft[-1] - yt[-1])

# plt.plot(xtot[:,0],diffadj,'.')
# plt.plot(xtot[:,0],-np.exp(5-xtot[:,0])/4,'.')

# yt = balldroptruealt(xtot)
# ft = np.squeeze(balldropmodel_quad(np.array([10,20,0]).reshape([1,-1]), xtot))

# diffadj = (yt - ft) 
# plt.plot(xtot[:,0],diffadj,'.')
# plt.plot(xtot[:,0],np.exp(xtot[:,0])/2,'.')
# asdad



def corr_f(x,k):
    corrdict = {}
    C0 = np.exp(-1/3*np.abs(np.subtract.outer(x[:,0],x[:,0])))*(1+1/3*np.abs(np.subtract.outer(x[:,0],x[:,0])))
    C1 = 0.1*(np.abs(np.subtract.outer(x[:,1],x[:,1]))<10**(-4))
    #C2 = np.exp(-np.abs(np.subtract.outer(x[:,0],x[:,0])))*(1+np.abs(np.subtract.outer(x[:,0],x[:,0])))
    #C3 = 0.0001*(np.abs(np.subtract.outer(x[:,1],x[:,1]))<10**(-4))
    if k == 0:
        adj = np.exp(5-x[:,0])/4
        corrdict['C'] = C1 + np.diag(adj) @ C0 @ np.diag(adj)
    if k == 1:
        adj = np.exp(x[:,0])/2
        corrdict['C'] = C1 + np.diag(adj) @ C0 @ np.diag(adj)
    return corrdict

R = corr_f(xtot,0)['C']/4 
R = 0.99*R + 0.01 * np.diag(np.diag(R))
Q = corr_f(xtot,1)['C']/4
Q = 0.99*Q + 0.01 * np.diag(np.diag(Q))




RQT = np.hstack((R, 0*R))
RQB = np.hstack((0*Q, Q))
RQ = np.vstack((RQT,RQB))
n = R.shape[0]
Jm = np.hstack((np.eye(n),-np.eye(n)))
RQu = RQ - (Jm @ RQ).T @ np.linalg.solve(Jm @ RQ @ Jm.T, Jm @ RQ)


RQu = RQ - (Jm @ RQ).T @ np.linalg.solve(R+Q, Jm @ RQ)

RQinv = np.linalg.inv(RQ)
RpQinv = np.linalg.inv(R+Q)

RET = np.hstack((spla.sqrtm(R), 0*R))
REB = np.hstack((0*Q, spla.sqrtm(Q)))
RE = np.vstack((RQT,RQB))
PERT = RE @ Jm.T

RQu = RE @ (np.eye(2*n) - PERT @  np.linalg.solve(PERT.T @ PERT, PERT.T)) @ RE

U,W,Vt = np.linalg.svd(PERT)

W = np.append(0.001 * np.ones(n), np.ones(n))
RQu = RE @ (U[:,n:]  @ U[:,n:].T) @ RE


n0 = xtot.shape[0]

inds = np.where(((x[:, None] > xtot-10**(-8)) *
                        (x[:, None] < xtot+10**(-8))).all(2))[1]
indst = np.hstack((inds,n0+inds))

ft1 = np.squeeze(balldropmodel_linear(np.array([10,20,0]).reshape([1,-1]), x))
ft2 = np.squeeze(balldropmodel_quad(np.array([10,20,0]).reshape([1,-1]), x))
yt = np.squeeze(balldroptruealt(x))

residval = np.hstack((yt - ft1,yt - ft2))
modval = np.hstack((ft1,ft2))

n0 = xtot.shape[0]

f1 = np.squeeze(balldropmodel_linear(np.array([10,20,0]).reshape([1,-1]), xtot))
f2 = np.squeeze(balldropmodel_quad(np.array([10,20,0]).reshape([1,-1]), xtot))

mval = np.hstack((f1,f2))
indst = np.hstack((inds,n0+inds))

W,V = np.linalg.eigh(RQu[indst,:][:,indst])

weg = np.where(np.abs(W) > (10 ** (-4)))[0]

dhat = RQu[:,indst] @ (V[:,weg] @ ((np.diag(1/W[weg]) @ V[:,weg].T) @ residval))
dhatR = R @ np.linalg.solve(R + Q, f1-f2)
print( dhat[:10]- dhat[ n0:(n0+10)])
print(f1-f2)
print(f1[:10] - f2[:10])
