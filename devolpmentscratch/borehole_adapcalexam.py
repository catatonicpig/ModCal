# -*- coding: utf-8 -*-
"""Header here."""
import numpy as np
import scipy.stats as sps
import sys
import os
import copy
import time
from line_profiler import LineProfiler
from boreholetestfunctions import borehole_model, borehole_true
SCRIPT_DIR = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, '..')))
from base.emulation import emulator
from base.calibration import calibrator
class thetaprior:
    """ This defines the class instance of priors provided to the methods. """
    def lpdf(theta):
        if theta.ndim > 1.5:
            return np.squeeze(np.sum(sps.norm.logpdf(theta, 1, 0.5),1))
        else:
            return np.squeeze(np.sum(sps.norm.logpdf(theta, 1, 0.5)))
    def rnd(n):
        return np.vstack((sps.norm.rvs(1, 0.5, size=(n,4))))

x = sps.uniform.rvs(0,1,[50,3])
x[:,2] = x[:,2] > 0.5
yt = np.squeeze(borehole_true(x))
yvar = (10 ** (-2)) * np.ones(yt.shape)
thetatot = (thetaprior.rnd(30))
f = (borehole_model(x, thetatot).T ).T
y = yt + sps.norm.rvs(0,np.sqrt(yvar))
emu = emulator(x, thetatot, f, method = 'PCGPwM')
emu.fit()
emu2 = emulator(passthroughfunc = borehole_model)
cal2 = calibrator( emu2, y, x, thetaprior, yvar, method = 'directbayes')
print(np.round(np.quantile(cal2.theta.rnd(10000), (0.01, 0.99), axis = 0),3))
cal = calibrator( emu, y, x, thetaprior, yvar, method = 'directbayes_wgrad')
print(np.round(np.quantile(cal.theta.rnd(10000), (0.01, 0.99), axis = 0),3))


def matrixmatching(mat1, mat2):
    #This is an internal function to do matching between two vectors
    #it just came up alot
    #It returns the where each row of mat2 is first found in mat1
    #If a row of mat2 is never found in mat1, then 'nan' is in that location
    if (mat1.shape[0] > (10 ** (4))) or (mat2.shape[0] > (10 ** (4))):
        raise ValueError('too many matchings attempted.  Don''t make the method work so hard!')
    if mat1.ndim != mat2.ndim and mat1.ndim == 1:
        if mat1.ndim == 1 and mat1.shape[0] == mat2.shape[1]:
            mat1 = np.reshape(mat1,(1,-1))
        else:
            raise ValueError('Somehow sent non-matching information to matrixmatching')
    if mat1.ndim == 1:
        matchingmatrix = np.isclose(mat1[:,None].astype('float'),
                 mat2.astype('float'))
    else:
        matchingmatrix = np.isclose(mat1[:,0][:,None].astype('float'),
                         mat2[:,0].astype('float'))
        for k in range(1,mat2.shape[1]):
            try:
                matchingmatrix *= np.isclose(mat1[:,k][:,None].astype('float'),
                         mat2[:,k].astype('float'))
            except:
                matchingmatrix *= np.equal(mat1[:,k],mat2[:,k])
    r, c = np.where(matchingmatrix.cumsum(axis=0).cumsum(axis=0) == 1)
    
    nc = np.array(list(set(range(0,mat2.shape[0])) - set(c))).astype('int')
    return nc, c, r


thetaqueue = np.zeros(0)

dosupp = False
numperbatch = 100
for k in range(0,10):
    if thetaqueue.shape[0] < numperbatch:
        thetanew, info = emu.supplement(size = 10, cal = cal, overwrite=True)
        thetatot = np.vstack((thetatot,thetanew))
        f = np.hstack((f,np.full((x.shape[0],thetanew.shape[0]),np.nan)))
        thetanewqueue = np.tile(thetanew, (x.shape[0],1))
        xnewqueue = np.repeat(x, thetanew.shape[0], axis =0)
        if thetaqueue.shape[0] == 0:
            thetaqueue = thetanewqueue
            xqueue = xnewqueue
        else:
            thetaqueue = np.vstack(thetaqueue,thetanewqueue)
            xqueue = np.vstack(xqueue,xnewqueue)
    
    for l in range(0, numperbatch):
        _, cx, _ = matrixmatching(xqueue[l,:], x)
        _, ctheta, _ = matrixmatching(thetaqueue[l,:], thetatot)
        f[cx, ctheta] = borehole_model(xqueue[l,:], thetaqueue[l,:])
    thetaqueue = np.delete(thetaqueue,range(0,numperbatch),0)
    xqueue = np.delete(xqueue,range(0,numperbatch),0)
    emu.update(thetatot, f)
    cal.fit()
    print(np.round(np.quantile(cal.theta.rnd(10000), (0.01, 0.99), axis = 0),3))