# -*- coding: utf-8 -*-
"""Header here."""
import numpy as np


def plumleepostsampler(thetastart, logpostfunc, numsamp, tarESS):
    """
    Return draws from the posterior.

    Parameters
    ----------
    thetastart : array of float
        Some matrix of parameters where function evaluations as starting points.
    logpost : function
        A function call describing the log of the prior distribution
    numsamp : integer
        Number of samples returned
    tarESS : integer
        Minimum effective sample size desired in the returned samples
    Returns
    -------
    theta : matrix of sampled paramter values
    """
    numchain = 100
    maxiters = 10
    keepgoing = True
    while keepgoing:
        logpost = logpostfunc(thetastart)
        logpost = logpost - np.max(logpost)
        logpost -= np.log(np.sum(np.exp(logpost)))
        post = np.exp(logpost)
        post = post/np.sum(post)
        startingv = np.random.choice(range(0, thetastart.shape[0]),
                                     size=1000,
                                     p=post)
        thetasave = thetastart[startingv, :]
        if np.max(post) > 0.5:
            thetastart = thetastart[startingv[0], :] +\
                (thetastart - thetastart[startingv[0], :]) / 2
        else:
            keepgoing = False
    rho = 0.5
    jitter = 0.01
    numsamppc = np.ceil(numsamp/numchain).astype('int')
    for iters in range(0,maxiters):
        covmat0 = np.cov(thetasave.T)
        Wc,Vc = np.linalg.eigh(covmat0)
        covmat0 = (1-jitter)*covmat0 + jitter*np.diag(np.diag(covmat0))
        Wc,Vc = np.linalg.eigh(covmat0)
        hc = (Vc @ np.diag(np.sqrt(Wc)) @ Vc.T)
        thetac = thetasave[np.random.choice(range(0,thetasave.shape[0]),size = numchain),:]
        logpostc = logpostfunc(thetac)
        
        thetasave = np.zeros((numchain,numsamppc,thetac.shape[1]))
        numtimes = 0
        for k in range(0,numsamppc):
            thetap = 1*thetac
            for l in range(0,numchain):
                thetap[l,:] = thetac[l,:] +rho * (np.random.normal(0,1,thetac.shape[1]) @ hc)
            logpostp = logpostfunc(thetap)
            for l in range(0,numchain):
                if np.log(np.random.uniform()) < (logpostp[l]-logpostc[l]):
                    numtimes = numtimes+(1/numchain)
                    thetac[l,:] = 1*thetap[l,:]
                    logpostc[l] = 1*logpostp[l]
                thetasave[l, k,:] = 1*thetac[l,:]
        W = np.mean(np.var(thetasave,1),0)
        mut = np.mean(np.mean(thetasave,1),0)
        B = np.zeros(W.shape)
        autocorr = np.zeros(W.shape)
        for l in range(0,numchain):
            muv = np.mean(thetasave[l,:,:],0)
            varc = np.var(thetasave[l,:,:],0)
            autocorr += np.mean((thetasave[l,0:(numsamppc-1),:] - muv.T)*(thetasave[l,1:,:] - muv.T),0)
            B += numsamppc/(numchain-1) *((muv-mut) **2)
        varplus = (numsamppc-1)/(numsamppc)*W + 1/numsamppc * B
        rhohat = (1-(W-autocorr/numchain)/varplus)
        ESS = numchain * numsamppc * (1 -np.abs(rhohat))
        thetasave = np.reshape(thetasave,(-1,thetac.shape[1]))
        accr = numtimes / numsamppc
        if iters > 0.5:
            if accr > 0.1 and (np.mean(ESS) > tarESS or numsamppc > 800):
                break
            if (accr < 0.23):
                rho = rho*np.max((np.exp((np.log(accr+0.01)-np.log(0.26))*2),0.25))
            elif (accr > 0.28):
                rho = rho/np.max((np.exp((np.log(1.01-accr)-np.log(0.76))),0.25))
            if accr < 0.3*numsamppc and accr > 0.2 and np.mean(ESS) < tarESS/2 and numsamppc < 200:
                numsamppc = (np.array(numsamppc*np.min((tarESS/np.mean(ESS),4)))).astype('int')
            if accr < 0.3*numsamppc and accr > 0.2 and np.mean(ESS) > 2*tarESS and np.mean(ESS) > 2*tarESS:
                numsamppc = (np.array(numsamppc*np.max((tarESS/np.mean(ESS),0.25)))).astype('int')
                numsamppc = np.max(numsamppc,np.ceil(numsamp/numchain))
    return thetasave[np.random.choice(range(0,thetasave.shape[0]),size = numsamp),:]