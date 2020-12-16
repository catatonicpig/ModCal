# -*- coding: utf-8 -*-
"""Header here."""
import numpy as np
import scipy.stats as sps
import sys
import os
import copy
import time
from line_profiler import LineProfiler
from boreholetestfunctions import borehole_model, borehole_failmodel, borehole_true
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
thetatot = (thetaprior.rnd(15))
f = (borehole_failmodel(x, thetatot).T ).T
y = yt + sps.norm.rvs(0,np.sqrt(yvar))
emu = emulator(x, thetatot, f, method = 'PCGPwM', options = {'xrmnan': 'all',
                                                             'thetarmnan': 'never',
                                                             'return_grad': True})
emu.fit()
emu2 = emulator(passthroughfunc = borehole_model)
cal2 = calibrator( emu2, y, x, thetaprior, yvar, method = 'directbayes')
print(np.round(np.quantile(cal2.theta.rnd(10000), (0.01, 0.99), axis = 0),3))
cal = calibrator( emu, y, x, thetaprior, yvar, method = 'directbayes')
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


thetaidqueue = np.zeros(0)
xidqueue = np.zeros(0)
pending = np.full(f.shape, False)
complete = np.full(f.shape, True)
cancelled = np.full(f.shape, False)
failures = np.logical_not(np.isfinite(f))
thetaspending = np.full(f.shape[0], False)
numperbatch = 250
for k in range(0,20):
    print('Percentage Cancelled: %0.2f ( %d / %d)' % (100*np.round(np.mean(1-pending-complete),4),
                                                    np.sum(1-pending-complete),
                                                    np.prod(pending.shape)))
    print('Percentage Pending: %0.2f ( %d / %d)' % (100*np.round(np.mean(pending),4),
                                                    np.sum(pending),
                                                    np.prod(pending.shape)))
    print('Percentage Complete: %0.2f ( %d / %d)' % (100*np.round(np.mean(complete),4),
                                                    np.sum(complete),
                                                    np.prod(pending.shape)))
    numnewtheta = 10
    keepadding = True
    while keepadding and (k>-1):
        numnewtheta += 2
        thetachoices = cal.theta(200)
        choicescost = np.ones(thetachoices.shape[0])
        thetaneworig, info = emu.supplement(size = numnewtheta, thetachoices = thetachoices, 
                                        choicescost = choicescost,
                                        cal = cal, overwrite=True,
                                        args = {'includepending': True,
                                              'costpending': 0.01+0.99*np.mean(pending,0),
                                              'pending': pending})
        thetaneworig = thetaneworig[:numnewtheta,:]
        thetanew = thetaneworig
        if info['obviatesugg'].shape[0] > 0:
            pending[:, info['obviatesugg']] = False
            print('obviating')
            print(info['obviatesugg'])
            for k in info['obviatesugg']:
                queue2delete = np.where(thetaidqueue == k)[0]
                if queue2delete.shape[0] > 0.5:
                    thetaidqueue = np.delete(thetaidqueue,queue2delete,0)
                    xidqueue = np.delete(xidqueue,queue2delete,0)
        
        if (thetanew.shape[0] > 0.5) and \
            (np.sum(np.hstack((pending,np.full((x.shape[0],thetanew.shape[0]),True)))) > 600):
            pending = np.hstack((pending,np.full((x.shape[0],thetanew.shape[0]),True)))
            complete = np.hstack((complete,np.full((x.shape[0],thetanew.shape[0]),False)))
            f = np.hstack((f,np.full((x.shape[0],thetanew.shape[0]),np.nan)))
            thetaidnewqueue = np.tile(np.arange(thetatot.shape[0],thetatot.shape[0]+
                                                thetanew.shape[0]), (x.shape[0]))
            thetatot = np.vstack((thetatot,thetanew))
            xidnewqueue = np.repeat(np.arange(0,x.shape[0]), thetanew.shape[0], axis =0)
            if thetaidqueue.shape[0] == 0:
                thetaidqueue = thetaidnewqueue
                xidqueue = xidnewqueue
            else:
                thetaidqueue = np.append(thetaidqueue,thetaidnewqueue)
                xidqueue = np.append(xidqueue,xidnewqueue)
            keepadding = False
    priorityscore = np.zeros(thetaidqueue.shape)
    priorityscore = np.random.choice(np.arange(0,priorityscore.shape[0]),
                                size=priorityscore.shape[0],replace=False)
    queuerearr = np.argsort(priorityscore)
    xidqueue = xidqueue[queuerearr]
    thetaidqueue = thetaidqueue[queuerearr]
    for l in range(0,np.minimum(xidqueue.shape[0],numperbatch)):
        f[xidqueue[l], thetaidqueue[l]] = borehole_failmodel(x[xidqueue[l],:],
                                                         thetatot[thetaidqueue[l],:])
        pending[xidqueue[l], thetaidqueue[l]] = False
        complete[xidqueue[l], thetaidqueue[l]] = True
    thetaidqueue = np.delete(thetaidqueue,range(0,numperbatch),0)
    xidqueue = np.delete(xidqueue,range(0,numperbatch),0)
    
    emu.update(theta = thetatot, f=f)
    cal.fit()
    # emu.remove(cal = cal)
    # cal.fit()
    # emu.update(theta = thetatot, f=f)
    print(np.round(np.quantile(cal.theta.rnd(10000), (0.01, 0.99), axis = 0),3))