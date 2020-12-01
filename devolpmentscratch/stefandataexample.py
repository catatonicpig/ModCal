# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 10:56:34 2020

@author: Plumlee
"""
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
import matplotlib.pyplot as plt

thetaeval = np.loadtxt('stefandata/thetavals.csv',delimiter=',')
feval = np.loadtxt('stefandata/functionevals.csv',delimiter=',')
inputeval = np.loadtxt('stefandata/inputdata.csv',delimiter=',',dtype='object')
failmat = np.genfromtxt('stefandata/failval.csv', delimiter=',')

feval[failmat > 0.5] = np.nan



class thetaprior:
    """ This defines the class instance of priors provided to the methods. """
    def lpdf(theta):
        if theta.ndim > 1.5:
            logprior = -50 * np.sum((theta - 0.6) ** 2, axis=1)
            logprior += 0.5*np.log(2-np.sum(np.abs(theta - 0.5), axis=1))
            flag = np.sum(np.abs(theta - 0.5), axis=1) > 2
            logprior[flag] = -np.inf
            logprior = np.array(logprior,ndmin=1)
        else:
            logprior = -50 * np.sum((theta - 0.6) ** 2)
            logprior += 0.5*np.log(2-np.sum(np.abs(theta - 0.5)))
            if np.sum(np.abs(theta - 0.5)) > 2:
                logprior = -np.inf
            logprior = np.array(logprior,ndmin=1)
        return logprior
    def rnd(n):
        if n > 1:
            rndval = np.vstack((sps.norm.rvs(0.6, 0.1, size=(n,13))))
            flag = np.sum(np.abs(rndval - 0.5), axis=1) > 2
            while np.any(flag):
                rndval[flag,:] = np.vstack((sps.norm.rvs(0.6, 0.1, size=(np.sum(flag),13))))
                flag = np.sum(np.abs(rndval - 0.5), axis=1) > 2
        else:
            rndval = sps.norm.rvs(0.6, 0.1, size =13)
            while np.sum(np.abs(rndval - 0.5)) > 2:
                rndval = np.vstack((sps.norm.rvs(0.6, 0.1,size = 13)))
        return np.array(rndval)

x = inputeval
f = feval[np.arange(0,800,16),:].T
theta = thetaeval[np.arange(0,800,16),:]
emu = emulator(x = x, theta=theta, f=f, method = 'PCGPwM')  # this builds an emulator 

y = np.zeros(f.shape[0])
yvar = np.ones(f.shape[0])
cal = calibrator( emu, y, x, thetaprior, yvar, method = 'directbayes')
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

thetapost = cal.theta(1000)
posstheta = np.where(cal.theta.lpdf(thetaeval) > -250)[0]
nc, c, r = matrixmatching(cal.emu._emulator__theta, thetaeval[posstheta,:])
posstheta = posstheta[nc]
lpdfexisting = cal.theta.lpdf(cal.emu._emulator__theta)
spread0 = np.quantile(thetapost,(0.95),0)-np.quantile(thetapost,(0.05),0)
print(np.round(spread0,3))
print(round(np.linalg.slogdet(np.cov(thetapost.T))[1],2))
fpred = emu.predict(theta = thetaeval)
print(np.nanmean(np.abs(fpred.mean()- feval.T)))
for k in range(0,4):
    numnewtheta = np.round(cal.emu._emulator__theta.shape[0]*0.33).astype('int')
    print('choosing new %d thetas out of %d thetas...'% (numnewtheta,posstheta.shape[0]) )
    thetanew, _ = emu.supplement(size = numnewtheta, 
                                 thetachoices = thetaeval[posstheta,:],
                                 cal = cal,
                                 overwrite = True)
    # print('chose new %d thetas to add to %d thetas...'% (thetanew.shape[0],emu._emulator__theta.shape[0]) )
    nc, c, r = matrixmatching(thetanew, thetaeval)
    print('number of failures: %d / %d' % (np.sum(np.isnan(feval[c,:].T)),np.prod(feval[c,:].shape)))
    emu.update(theta=thetaeval[c,:], f=feval[c,:].T)
    # print('updated the emulator...')
    cal = calibrator( emu, y, x, thetaprior, yvar, method = 'directbayes')
    print('calibrated the model...')
    thetapost = cal.theta(1000)
    # print(cal.emu._emulator__theta.shape)
    spread = (np.quantile(thetapost,(0.95),0)-np.quantile(thetapost,(0.05),0)) 
    # print(np.round(spread,3))
    # print(np.round(spread/spread0,2))
    print(round(np.linalg.slogdet(np.cov(thetapost.T))[1],2))
    fpred = emu.predict(theta = thetaeval)
    print(np.nanmean(np.abs(fpred.mean()- feval.T)))
    posstheta = np.where(cal.theta.lpdf(thetaeval) > -250)[0]
    nc, c, r = matrixmatching(cal.emu._emulator__theta, thetaeval[posstheta,:])
    posstheta = posstheta[nc]
    if posstheta.shape[0] < 1.2*numnewtheta:
        print('could not find any more reasonable values')
        break
indsstar= np.where(cal.theta.lpdf(thetaeval) > -20)[0]

#print(np.nanmean(feval[indsstar,:] ** 2,1))
#print(np.nanmean(feval ** 2))

Lfull = np.nanmean(feval ** 2,1)
Lfull = -1 / 2 * np.nansum(feval ** 2,1) - (1 ** 2)/2*np.sum(np.isnan(feval),1)
mLfull = np.max(Lfull[np.isfinite(Lfull)])
indfocus = np.where(np.logical_and(Lfull > np.max(Lfull) - 80, np.sum(np.isnan(feval),1) < 20))[0]

plt.plot(Lfull[indfocus], cal.theta.lpdf(thetaeval[indfocus]),'.')
plt.plot(Lfull[indfocus], Lfull[indfocus]-Lfull[indfocus[5]]+cal.theta.lpdf(thetaeval[indfocus])[5],'-')

np.mean(np.isnan(feval[indsstar,:]))
np.mean(np.isnan(feval))
np.mean(np.isnan(feval[indsstar,:]),1)


# import pandas as pd
# fayans_cols = [r'$\rho_{\mathrm{eq}}$', r'$E/A$', r'$K$', r'$J$',
#                     r'$L$', '$h^{\mathrm{v}}_{2{-}}$',
#                     r'$a^{\mathrm{s}}_{+}$',
#                     r'$h^{\mathrm{s}}_{\nabla}$',
#                     r'$\kappa$', r'$\kappa^\prime$',
#                     r'$f^{\xi}_{\mathrm{ex}}$',
#                     r'$h^{\xi}_{+}$', r'$h^{\xi}_{\nabla}$',
#                     'cat']
# post_samples = pd.DataFrame(thetapost)
# post_samples['cat'] = 'posterior'
# plot_thetas(post_samples, include=['posterior'], fname='imagename')