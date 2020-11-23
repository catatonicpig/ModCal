# -*- coding: utf-8 -*-
"""Header here."""
import numpy as np
import scipy.optimize as spo

def plumleepostsampler_wgrad(thetastart, logpostfunc, numsamp, tarESS):
    """
    Return draws from the post.

    Parameters
    ----------
    thetastart : array of float
        Some matrix of parameters where function evaluations as starting points.
    logpost : function
        A function call describing the log of the posterior distribution
    numsamp : integer
        Number of samples returned
    tarESS : integer
        Minimum effective sample size desired in the returned samples
    Returns
    -------
    theta : matrix of sampled paramter values
    """
    
    def logpostfuncnew(thetaph):
        logpost, _ = logpostfunc(thetaph)
        return logpost
    
    logpost = logpostfuncnew(thetastart)
    logpost = logpost - np.max(logpost)
    logpost -= np.log(np.sum(np.exp(logpost)))
    post = np.exp(logpost)
    post = post/np.sum(post)
    size = np.minimum(50, np.sum(post>10 ** (-6)))
    startingv = np.random.choice(range(0, thetastart.shape[0]),
                                 size=size,
                                 p=post, replace=False)
    startingv = np.unique(startingv)
    thetaop = thetastart[startingv,:]
    
    
    
    thetac = np.mean(thetastart,0)
    thetas = np.std(thetastart,0)
    for k in range(0,startingv.shape[0]):
        print(thetaop[k,:])
        LB,UB = test1dboundarys(thetaop[k,:], logpostfuncnew)
        thetas = np.maximum(thetas, 4*(UB-LB))
        thetastart = np.vstack((thetastart,LB))
        thetastart = np.vstack((thetastart,UB))
    bounds = spo.Bounds(-10*np.ones(thetastart.shape[1]), 10*np.ones(thetastart.shape[1]))
    def neglogpostfunc2(thetap):
        theta = thetac + thetas * thetap
        logpost, _ = logpostfunc(theta)
        return -logpost
    for k in range(0,startingv.shape[0]):
        theta0 = (np.squeeze(thetaop[k,:]) - thetac) / thetas
        opval = spo.minimize(neglogpostfunc2, theta0, method='L-BFGS-B',
                              bounds = bounds)
        thetaop[k,:] = thetac + thetas * opval.x
        
    #test backward and forward
    thetastart = np.vstack((thetastart,thetaop))
    numchain = 100
    maxiters = 30
    keepgoing = True
    while keepgoing:
        logpost = logpostfuncnew(thetastart)
        logpost = logpost - np.max(logpost)
        logpost -= np.log(np.sum(np.exp(logpost)))
        post = np.exp(logpost)
        post = post/np.sum(post)
        startingv = np.random.choice(range(0, thetastart.shape[0]),
                                      size=1000,
                                      p=post)
        thetasave = thetastart[startingv, :]
        if np.max(post) > 1/(thetastart.shape[1] +2)/2:
            thetastart = thetastart[startingv[0], :] +\
                (thetastart - thetastart[startingv[0], :]) / 2
        else:
            keepgoing = False
    rho = 0.5
    numsamppc = np.minimum(1000,np.maximum(25,np.ceil(numsamp/numchain))).astype('int')
    for iters in range(0,maxiters):
        covmat0 = np.cov(thetasave.T)
        Wc, Vc = np.linalg.eigh(covmat0)
        if iters < 2:
            Wc = Wc + (10 ** (-8))
        else:
            Vc = Vc[:,Wc> 10 **(-20) * np.max(Wc)]
            Wc = Wc[Wc> 10 **(-20) * np.max(Wc)]
        hc = (Vc @ np.diag(np.sqrt(Wc)) @ Vc.T)
        hc2 = (Vc @ np.diag(Wc) @ Vc.T)
        hc2i = (Vc @ np.diag(1 / Wc) @ Vc.T)
        hc2hi = (Vc @ np.diag(1 / np.sqrt(Wc)) @ Vc.T)
        thetac = thetasave[np.random.choice(range(0,thetasave.shape[0]),size = numchain),:]
        fval, dfval = logpostfunc(thetac) 
        
        thetasave = np.zeros((numchain,numsamppc,thetac.shape[1]))
        numtimes = 0
    
        for k in range(0,numsamppc):
            thetap = thetac + rho ** 2 * (dfval @ hc2) +  np.sqrt(2) * rho * (np.random.normal(0,1,thetac.shape) @ hc)
            fvalp, dfvalp = logpostfunc(thetap)
            qxpxk = -1 / (4 * (rho ** 2)) * np.sum( ((thetap - thetac - rho ** 2 * (dfval @ hc2)) @ hc2hi) ** 2, 1)
            qxkxp = -1 / (4 * (rho ** 2)) * np.sum( ((thetac - thetap - rho ** 2 * (dfvalp @ hc2)) @ hc2hi) ** 2, 1)
            for l in range(0,numchain):
                if np.log(np.random.uniform()) < (fvalp[l]-fval[l] + qxkxp[l] - qxpxk[l]):
                    numtimes = numtimes+(1/numchain)
                    thetac[l,:] = 1*thetap[l,:]
                    fval[l] = 1*fvalp[l]
                    dfval[l] = 1*dfvalp[l]
                thetasave[l, k,:] = 1*thetac[l,:]
        mut = np.mean(np.mean(thetasave,1),0)
        B = np.zeros(mut.shape)
        autocorr = np.zeros(mut.shape)
        W = np.zeros(mut.shape)
        for l in range(0,numchain):
            muv = np.mean(thetasave[l,:,:],0)
            varc = np.var(thetasave[l,:,:],0)
            autocorr +=1/numchain * np.mean((thetasave[l,0:(numsamppc-1),:] - muv.T)*(thetasave[l,1:,:] - muv.T),0)
            W += 1/numchain * np.mean((thetasave[l,0:(numsamppc-1),:] - muv.T) ** 2,0)
            B += numsamppc/(numchain-1) *((muv-mut) **2)
        varplus = W + 1/numsamppc * B
        rhohat = (1-(W-autocorr)/varplus)
        ESS = numchain * numsamppc * (1 -np.abs(rhohat))
        thetasave = np.reshape(thetasave,(-1,thetac.shape[1]))
        accr = numtimes / numsamppc
        print('accr')
        print(accr)
        print('ESS')
        print(ESS)
        print(numsamppc)
        if  iters > 2.5 and accr > 0.4 and accr < 0.6 and (np.mean(ESS) > tarESS):
            break
        if iters > 0.5:
            if (accr < 0.45):
                rho = rho*np.max((np.exp((np.log(accr+0.01)-np.log(0.5))*2),0.75))
            elif (accr > 0.55):
                rho = rho/np.max((np.exp((np.log(1.01-accr)-np.log(0.5))),0.75))
        if accr < 0.6 and accr > 0.4 and np.mean(ESS) < tarESS and numsamppc < 250:
            numsamppc = (np.array(numsamppc*np.min((tarESS/np.mean(ESS),4)))).astype('int')
        elif accr < 0.6 and accr > 0.4 and numsamppc > 250:
            break
        print('update')
        print(np.max((np.exp((np.log(1.01-accr)-np.log(0.5))),0.5)))
        print('rho')
        print(rho)
    return thetasave[np.random.choice(range(0,thetasave.shape[0]),size = numsamp),:]

def test1dboundarys(theta0, logpostfunchere):
    L0 = logpostfunchere(theta0)
    thetaminsave = np.zeros(theta0.shape)
    thetamaxsave = np.zeros(theta0.shape)
    for k in range(0,theta0.shape[0]):
        notfarenough = 0
        farenough = 0
        eps = 10 ** (-4)
        keepgoing = True
        while keepgoing:
            thetaadj = 1 * theta0
            thetaadj[k] += eps
            L1 = logpostfunchere(thetaadj)
            if (L0-L1) < 4:
                eps = eps * 2
                notfarenough += 1
                thetamaxsave[k] = 1 * thetaadj[k]
            else:
                eps = eps / 2
                farenough += 1
            if notfarenough > 1.5 and farenough > 1.5:
                keepgoing = False
        notfarenough = 0
        farenough = 0
        keepgoing = True
        while keepgoing:
            thetaadj = 1 * theta0
            thetaadj[k] -= eps
            L1 = logpostfunchere(thetaadj)
            if (L0-L1) < 4:
                eps = eps * 2
                notfarenough += 1
                thetaminsave[k] = 1 * thetaadj[k]
            else:
                eps = eps / 2
                farenough += 1
            if notfarenough > 1.5 and farenough > 1.5:
                keepgoing = False
    return thetaminsave, thetamaxsave