# -*- coding: utf-8 -*-
"""Header here."""
import numpy as np
import scipy.optimize as spo
import inspect

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
    testout = logpostfunc(thetastart[0:2,:])
    if type(testout) is tuple:
        if len(testout) is not 2:
            raise ValueError('log density does not return 1 or 2 elements')
        if testout[1].shape[1] is not thetastart.shape[1]:
            raise ValueError('derivative appears to be the wrong shape')
        logpostf = logpostfunc
        def logpostf_grad(theta):
            return logpostfunc(theta)[1]
        try: 
            testout = logpostfunc(thetastart[0:2,:], return_grad = False)
            if type(testout) is tuple:
                raise ValueError('it seems like we cannot stop returning a grad')
            def logpostf_nograd(theta):
                return logpostfunc(theta, return_grad = False)
        except:
            def logpostf_nograd(theta):
                return logpostfunc(theta)[0]
    else:
        logpostf_grad = None
        logpostf = logpostfunc
        logpostf_nograd = logpostfunc
    logpost = logpostf_nograd(thetastart)
    logpost = logpost - np.max(logpost)
    logpost -= np.log(np.sum(np.exp(logpost)))
    post = np.exp(logpost)
    post = post/np.sum(post)
    size = np.minimum(10, np.sum(post>10 ** (-6)))
    startingv = np.random.choice(range(0, thetastart.shape[0]),
                                 size=size,
                                 p=post, replace=False)
    startingv = np.unique(startingv)
    thetaop = thetastart[startingv,:]
    
    thetac = np.mean(thetastart,0)
    thetas = np.std(thetastart,0)
    def neglogpostf_nograd(thetap):
        theta = thetac + thetas * thetap
        return -logpostf_nograd(theta)
    if logpostf_grad is not None:
        def neglogpostf_grad(thetap):
            theta = thetac + thetas * thetap
            return -thetas * logpostf_grad(theta)
        
    bounds = spo.Bounds(-10*np.ones(thetastart.shape[1]), 10*np.ones(thetastart.shape[1]))
    keeptryingwithgrad = True
    failureswithgrad = 0
    for k in range(0,startingv.shape[0]):
        theta0 = (np.squeeze(thetaop[k,:]) - thetac) / thetas
        if logpostf_grad is None:
            opval = spo.minimize(neglogpostf_nograd, theta0, method='L-BFGS-B',
                                 bounds = bounds, options={'maxiter': 4,'maxfun':100})
        else:
            if keeptryingwithgrad:
                opval = spo.minimize(neglogpostf_nograd, theta0,method='L-BFGS-B',
                                 jac = neglogpostf_grad,bounds = bounds,
                                 options={'maxiter':10,'maxfun':100})
                thetaop[k,:] = thetac + thetas * opval.x
            if not keeptryingwithgrad or not opval.success:
                if keeptryingwithgrad:
                    failureswithgrad += 1
                    alpha = failureswithgrad+0.25
                    beta = (k-failureswithgrad+1)
                    stdtest = np.sqrt(alpha * beta / ((alpha+beta+1) *((alpha+beta) ** 2)))
                    meantest = alpha/(alpha+beta)
                    if meantest - 3*stdtest > 0.25:
                        keeptryingwithgrad = False
                        #print('gave up on optimizing with grad, maybe it is approximate..')
                #opval = spo.minimize(neglogpostf_nograd, theta0,method='L-BFGS-B',
                #                 bounds = bounds, options={'maxiter': 2,'maxfun':100})
    if keeptryingwithgrad or logpostf_grad is None:
        thetastart = np.vstack((thetastart,thetaop))
    
    numchain = 50
    maxiters = 30
    keepgoing = True
    while keepgoing:
        logpost = logpostf_nograd(thetastart)
        logpost = logpost - np.max(logpost)
        logpost -= np.log(np.sum(np.exp(logpost)))
        post = np.exp(logpost)
        post = post/np.sum(post)
        if np.max(post) > 1/(thetastart.shape[1] +2)/2:
            thetastart = thetastart[startingv[0], :] +\
                (thetastart - thetastart[startingv[0], :]) / 2
        else:
            startingv = np.random.choice(range(0, thetastart.shape[0]),
                                          size=1000,
                                          p=post)
            thetasave = thetastart[startingv, :]
            keepgoing = False
    if logpostf_grad is None:
        rho = 2 / thetastart.shape[1] ** (1/2)
        taracc = 0.25
    else:
        rho = 2 / thetastart.shape[1] ** (1/6)
        taracc = 0.60
    numsamppc = np.minimum(1000,np.maximum(10,np.ceil(numsamp/numchain))).astype('int')
    for iters in range(0,maxiters):
        covmat0 = np.cov(thetasave.T)
        Wc, Vc = np.linalg.eigh(covmat0)
        if iters < 2:
            Wc = Wc + (10 ** (-8))
        else:
            Vc = Vc[:,Wc> 10 **(-20) * np.max(Wc)]
            Wc = Wc[Wc> 10 **(-20) * np.max(Wc)]
        hc = (Vc @ np.diag(np.sqrt(Wc)) @ Vc.T)
        thetac = thetasave[np.random.choice(range(0,thetasave.shape[0]),size = numchain),:]
        if logpostf_grad is not None:
            fval, dfval = logpostf(thetac)
        else:
            fval = logpostf_nograd(thetac)
        thetasave = np.zeros((numchain,numsamppc,thetac.shape[1]))
        numtimes = 0
        for k in range(0,numsamppc):
            rvalo = np.random.normal(0,1,thetac.shape)
            rval = np.sqrt(2) * rho * (rvalo @ hc)
            thetap = thetac + rval
            if logpostf_grad is not None:
                diffval = rho ** 2 * (dfval @ covmat0)
                thetap += diffval
                fvalp, dfvalp = logpostf(thetap)
                term1= rvalo / np.sqrt(2)
                term2= (dfval+dfvalp) @ hc * rho / 2
                qadj = -(2 * np.sum( term1 * term2, 1) + np.sum(term2 ** 2, 1))
            else:
                fvalp = logpostf(thetap)
                qadj = 0 * fvalp
            swaprnd = np.log(np.random.uniform(size = fval.shape[0]))
            whereswap = np.where(swaprnd < (fvalp-fval + qadj))[0]
            if whereswap.shape[0] > 0:
                numtimes = numtimes+(whereswap.shape[0]/numchain)
                thetac[whereswap,:] = 1*thetap[whereswap,:]
                fval[whereswap] = 1*fvalp[whereswap]
                if logpostf_grad is not None:
                    dfval[whereswap,:] = 1*dfvalp[whereswap,:]
            thetasave[:, k,:] = 1*thetac
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
        if  iters > 2.5 and accr > taracc*0.66 and accr < taracc*1.55 and (np.mean(ESS) > tarESS):
            break
        if iters > 0.5:
            if (accr < taracc*1.15):
                rho = rho*np.max((np.exp((np.log(accr+0.01)-np.log(taracc))*2),0.33))
            elif (accr > taracc/1.15):
                rho = rho/np.max((np.exp((np.log(1.01-accr)-np.log(1-taracc))),0.33))
        if accr < taracc*1.55 and accr > taracc*0.66 and np.mean(ESS) < tarESS and numsamppc < 250:
            numsamppc = (np.array(numsamppc*np.min((tarESS/np.mean(ESS),4)))).astype('int')
        elif accr < taracc*1.55 and accr > taracc*0.66 and numsamppc > 250:
            break
    return thetasave[np.random.choice(range(0,thetasave.shape[0]),size = numsamp),:]