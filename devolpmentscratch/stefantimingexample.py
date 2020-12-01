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


from base.calibrationmethods.directbayes import loglik_grad as fit2
from base.utilitiesmethods.plumleeMCMC_wgrad import plumleepostsampler_wgrad as sampler1
from base.emulationmethods.PCGPwM import predict as predict1
from base.emulationmethods.PCGPwM import __covmat as covfunc1

def emulation_test_stefan():
    thetaeval = np.loadtxt('stefandata/thetavals.csv',delimiter=',')
    feval = np.loadtxt('stefandata/functionevals.csv',delimiter=',')
    inputeval = np.loadtxt('stefandata/inputdata.csv',delimiter=',',dtype='object')
    failmat = np.genfromtxt('stefandata/failval.csv', delimiter=',')
    
    feval[failmat > 0.5] = np.nan
    
    
    
    class thetaprior:
        """ This defines the class instance of priors provided to the methods. """
        def lpdf(theta):
            if theta.ndim > 1.5:
                logprior = -8 * np.sum((theta - 0.6) ** 2, axis=1)
                logprior += 0.5*np.log(2-np.sum(np.abs(theta - 0.5), axis=1))
                flag = np.sum(np.abs(theta - 0.5), axis=1) > 2
                logprior[flag] = -np.inf
                logprior = np.array(logprior,ndmin=1)
            else:
                logprior = -8 * np.sum((theta - 0.6) ** 2)
                logprior += 0.5*np.log(2-np.sum(np.abs(theta - 0.5)))
                if np.sum(np.abs(theta - 0.5)) > 2:
                    logprior = -np.inf
                logprior = np.array(logprior,ndmin=1)
            return logprior
        def rnd(n):
            if n > 1:
                rndval = np.vstack((sps.norm.rvs(0.6, 0.25, size=(n,13))))
                flag = np.sum(np.abs(rndval - 0.5), axis=1) > 2
                while np.any(flag):
                    rndval[flag,:] = np.vstack((sps.norm.rvs(0.6, 0.25, size=(np.sum(flag),13))))
                    flag = np.sum(np.abs(rndval - 0.5), axis=1) > 2
            else:
                rndval = sps.norm.rvs(0.6, 0.25, size =13)
                while np.sum(np.abs(rndval - 0.5)) > 2:
                    rndval = np.vstack((sps.norm.rvs(0.6, 0.25,size = 13)))
            return np.array(rndval)
    
    x = inputeval
    f = feval[np.arange(0,800,12),:].T
    theta = thetaeval[np.arange(0,800,12),:]
    emu = emulator(x = x, theta=theta, f=f, method = 'PCGPwM')  # this builds an emulator 
    
    y = np.zeros(f.shape[0])
    yvar = np.ones(f.shape[0])
    cal = calibrator( emu, y, x, thetaprior, yvar, method = 'directbayes')
    thetapost = cal.theta(1000)
    spread0 = np.quantile(thetapost,(0.95),0)-np.quantile(thetapost,(0.05),0)
    print(np.round(spread0,3))
    print(round(np.linalg.slogdet(np.cov(thetapost.T))[1],2))
    fpred = emu.predict(theta = thetaeval)
    print(np.nanmean(np.abs(fpred.mean()- feval.T)))
    

#emulation_test_borehole()
lp = LineProfiler()
lp_wrapper = lp(emulation_test_stefan)    
lp.add_function(predict1)
#lp.add_function(sampler1)
lp.add_function(fit2)
#lp.add_function(covfunc1)
lp_wrapper()
lp.print_stats()