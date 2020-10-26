"""Header here."""
import numpy as np
import scipy.stats as sps
from base.utilities import postsampler

def fit(info, emu, y, x, args=None):
    """
    Return draws from the posterior.

    Parameters
    ----------
    thetaprior : array of float
        Some matrix of parameters where function evaluations as starting points.
    logpriorfunc : function
        A function call describing the log of the prior distribution
    loglikfunc : function
        A function call describing the log of the likelihood function
    Returns
    -------
    theta : matrix of sampled paramter values
    """
    
    if 'obsvar' in args.keys():
        obsvar = args['obsvar']
    else:
        raise ValueError('Must provide a prior on statistical parameters in this software.')
    
    if 'phiprior' in args.keys():
        phiprior = args['phiprior']
    else:
        raise ValueError('Must provide a prior on statistical parameters in this software.')
    
    thetaprior = info['thetaprior']
    theta = thetaprior.rvs(1000)
    thetadim = theta[0].shape[0]
    if phiprior.rvs(1) is None:
        phidim = 0
        thetaphi = theta
    else:
        phi = phiprior.rvs(1000)
        phidim = (phiprior.rvs(1)).shape[1]
        thetaphi = np.hstack((theta,phi))
        
        
    if type(emu) is tuple:
        emux = emu[0].x
    else:
        emux = emu.x
    
    matchingvec = np.where(((x[:, None] > emux - 1e-08) *\
                            (x[:, None] < emux + 1e-08)).all(2))
    xind = matchingvec[1][matchingvec[0]]
    def logpostfull(thetaphi):
        if phidim > 0.5:
            theta = thetaphi[:, :thetadim]
            phi = thetaphi[:, thetadim:]
        else:
            theta = thetaphi
            phi = None
        logpost =thetaprior.logpdf(theta) + phiprior.logpdf(phi)
        
        inds = np.where(np.isfinite(logpost))[0]
        if phi is None:
            logpost[inds] += loglik(emu, theta[inds], None, y, xind, args)
        else:
            logpost[inds] += loglik(emu, theta[inds], phi[inds], y, xind, args)
        return logpost
    
    numsamp = 1000
    tarESS = np.max((100, 10 * thetaphi.shape[1]))
    thetaphi = postsampler(thetaphi, logpostfull)
    
    if phidim > 0.5:
        theta = thetaphi[:, :thetadim]
        phi = thetaphi[:, thetadim:]
    else:
        theta = thetaphi
        phi = None
    
    info['theta'] = theta
    info['phi'] = phi
    info['xind'] = xind
    info['y'] = y
    info['x'] = x
    info['emux'] = emux
    return

def predict(x, emu, info, args):
    """
    Return posterior of function evaluation at the new parameters.

    Parameters
    ----------
    emumodel : Pred
        A fitted emu model defined as an emulation class.
    theta : array of float
        Some matrix of parameters where function evaluations as starting points.
    y : Observations
        A vector of the same length as x with observations. 'None' is equivlent to a vector of
        zeros.
    Sinv : Observation Precision Matrix
        A matrix of the same length as \"emu.x\" with observations. 'None' is equivlent to the
        identity matrix.

    Returns
    -------
    post: vector of unnormlaized log posterior
    """
    xind = info['xind']
    y = info['y']
    theta = info['theta']
    phi = info['phi']
    
    matchingvec = np.where(((x[:, None] > info['emux'] - 1e-08) *\
                            (x[:, None] < info['emux'] + 1e-08)).all(2))
    xindnew = matchingvec[1][matchingvec[0]]
    
    if theta.ndim == 1:
        theta = theta.reshape((1, theta.shape[0]))
    elif 'obsvar' in args.keys():
        obsvar = args['obsvar']
    else:
        raise ValueError('Must provide obsvar at this moment.')
    preddict = {}
    if type(emu) is tuple:
        predinfo = [dict() for x in range(len(emu))]
        for k in range(0, len(emu)):
            predinfo[k] = emu[k].predict(theta)
        preddict['meanfull'] = predinfo[0]['mean']
        preddict['varfull'] = predinfo[0]['var']
        preddict['draws'] = predinfo[0]['mean']
        preddict['modeldraws'] = predinfo[0]['mean']
    else:
        predinfo = emu.predict(theta)
        preddict['meanfull'] = predinfo['mean']
        preddict['full'] = predinfo['mean']
        preddict['draws'] = predinfo['mean']
        preddict['modeldraws'] = predinfo['mean']
        preddict['varfull'] = predinfo['var']
    
    for k in range(0, theta.shape[0]):
        if type(emu) is tuple:
            logliklocal = np.zeros(len(emu))
            predlocal = np.zeros((xindnew.shape[0], len(emu)))
            sslocal = np.zeros((xindnew.shape[0], len(emu)))
            for l in range(0, len(emu)):
                mu = predinfo[l]['mean'][(k, xind)]
                A1 = np.squeeze(predinfo[l]['covdecomp'][k, :, xind])
                S0 = A1 @ A1.T
                if 'cov_disc' in args.keys():
                    S0 += args['cov_disc'](emu[l].x[xind,:], l, phi[k,:])
                W, V = np.linalg.eigh(np.diag(obsvar) + S0)
                Sinvu = V @ np.diag(1/W) @ V.T
                ldetS = np.sum(np.log(W))
                logliklocal[l] = -0.5*(np.squeeze(y)-mu).T @ (Sinvu @ (np.squeeze(y)-mu))
                logliklocal[l] += -0.5*ldetS
                A2 = np.squeeze(predinfo[l]['covdecomp'][k, :, xindnew])
                S10 = A2 @ A1.T
                S11 = A2 @ A2.T
                if 'cov_disc' in args.keys():
                    C = args['cov_disc'](info['emux'], l, phi[k,:])
                    S10 += C[xindnew,:][:,xind]
                    S11 += C[xindnew,:][:,xindnew]
                mus0 = predinfo[l]['mean'][(k, xindnew)]
                predlocal[:,l] = mus0 + S10 @ np.linalg.solve(np.diag(obsvar) + S0, y-mu)
                sslocal[:,l] = np.diag(S11 - S10 @ np.linalg.solve(np.diag(obsvar) + S0, S10.T)) + predlocal[:,l] ** 2
            lm = np.max(logliklocal)
            logpostcalc = np.log(np.sum(np.exp(logliklocal-lm))) + lm
            post = np.exp(logliklocal - logpostcalc)
            
            preddict['meanfull'][k,:] = np.sum(post * predlocal,1)
            preddict['varfull'][k,:] = np.sum(post * sslocal,1) -\
                preddict['meanfull'][k,:] ** 2
            rc = np.random.choice(len(emu), p = post)
            A1 = np.squeeze(predinfo[rc]['covdecomp'][k, :, xind])
            A2 = np.squeeze(predinfo[rc]['covdecomp'][k, :, xindnew])
            S0 = A1 @ A1.T
            S0 += np.diag(obsvar)
            if 'cov_disc' in args.keys():
                C = args['cov_disc'](info['emux'], rc, phi[k,:])
                S0 += C[xind,:][:,xind]
                S10 += C[xindnew,:][:,xind]
                S11 += C[xindnew,:][:,xindnew]
            Wmat, Vmat = np.linalg.eigh(S11 - S10 @ np.linalg.solve(S0, S10.T))
            re = Vmat @ np.diag(np.sqrt(np.abs(Wmat))) @ Vmat.T @\
                sps.norm.rvs(0,1,size=(Vmat.shape[1]))
            preddict['draws'][k,:] = preddict['meanfull'][k, :]  + re
            preddict['modeldraws'][k,:] = mus0
            re = predinfo[rc]['covdecomp'][k,:,:].T @\
                sps.norm.rvs(0,1,size=(predinfo[rc]['covdecomp'].shape[1]))
            preddict['draws'][k,:] = predinfo[rc]['mean'][k,xindnew] + re[xindnew]
            preddict['modeldraws'][k,:] = predinfo[rc]['mean'][k,xindnew] + re[xindnew]

        else:
            mu = predinfo['mean'][(k, xind)]
            A1 = np.squeeze(predinfo['covdecomp'][k, :, xind])
            A2 = np.squeeze(predinfo['covdecomp'][k, :, xindnew])
            S0 = A1 @ A1.T
            S10 = A2 @ A1.T
            S11 = A2 @ A2.T
            if 'cov_disc' in args.keys():
                C = args['cov_disc'](info['emux'], phi[k,:])
                S0 += C[xind,:][:,xind]
                S10 += C[xindnew,:][:,xind]
                S11 += C[xindnew,:][:,xindnew]
            S0 += np.diag(obsvar)
            mus0 = predinfo['mean'][(k, xindnew)]
            preddict['meanfull'][k,:] = mus0 + S10 @ np.linalg.solve(S0, y-mu)
            preddict['varfull'][k,:] = np.diag(S11 - S10 @ np.linalg.solve(S0, S10.T))
            Wmat, Vmat = np.linalg.eigh(S11 - S10 @ np.linalg.solve(S0, S10.T))
            re = Vmat @ np.diag(np.sqrt(np.abs(Wmat))) @ Vmat.T @\
                sps.norm.rvs(0,1,size=(Vmat.shape[1]))
            preddict['modeldraws'][k,:] = mus0
            re = predinfo['covdecomp'][k,:,:].T @\
                sps.norm.rvs(0,1,size=(predinfo['covdecomp'].shape[1]))
            preddict['draws'][k,:] = predinfo['mean'][k,xindnew] + re[xindnew]
                
    preddict['mean'] = np.mean(preddict['meanfull'], 0)
    varterm1 = np.var(preddict['meanfull'], 0)
    preddict['var'] = np.mean(preddict['varfull'], 0) + varterm1
    return preddict

def loglik(emu, theta, phi, y, xind, args):
    r"""
    Return posterior of function evaluation at the new parameters.

    Parameters
    ----------
    emumodel : Pred
        A fitted emu model defined as an emulation class.
    theta : array of float
        Some matrix of parameters where function evaluations as starting points.
    y : Observations
        A vector of the same length as x with observations. 'None' is equivlent to a vector of
        zeros.
    Sinv : Observation Precision Matrix
        A matrix of the same length as \"emu.x\" with observations. 'None' is equivlent to the
        identity matrix.

    Returns
    -------
    post: vector of unnormlaized log posterior
    """
    if theta.ndim == 1:
        theta = theta.reshape((1, theta.shape[0]))
    elif 'obsvar' in args.keys():
        obsvar = args['obsvar']
    else:
        raise ValueError('Must provide obsvar at this moment.')
    if type(emu) is tuple:
        predinfo = [dict() for x in range(len(emu))]
        for k in range(0, len(emu)):
            predinfo[k] = emu[k].predict(theta)
        loglik = np.zeros(predinfo[0]['mean'].shape[0])
    else:
        predinfo = emu.predict(theta)
        loglik = np.zeros(predinfo['mean'].shape[0])
        
    for k in range(0, theta.shape[0]):
        if type(emu) is tuple:
            logliklocal = np.zeros(len(emu))
            for l in range(0, len(emu)):
                mu = predinfo[l]['mean'][(k, xind)]
                A1 = np.squeeze(predinfo[l]['covdecomp'][k, :, xind])
                S0 = A1 @ A1.T
                if 'cov_disc' in args.keys():
                    S0 += args['cov_disc'](emu[l].x[xind,:], l, phi[k,:])
                W, V = np.linalg.eigh(np.diag(obsvar) + S0)
                Sinvu = V @ np.diag(1/W) @ V.T
                ldetS = np.sum(np.log(W))
                logliklocal[l] = -0.5*(np.squeeze(y)-mu).T @ (Sinvu @ (np.squeeze(y)-mu))
                logliklocal[l] += -0.5*ldetS
            lm = np.max(logliklocal)
            loglik[k] = np.log(np.sum(np.exp(logliklocal-lm))) + lm
        else:
            mu = predinfo['mean'][(k, xind)]
            A1 = np.squeeze(predinfo['covdecomp'][k, :, xind])
            S0 = A1 @ A1.T
            if 'cov_disc' in args.keys():
                S0 += args['cov_disc'](emu.x[xind,:], phi[k,:])
            W, V = np.linalg.eigh(np.diag(obsvar) + S0)
            Sinvu = V @ np.diag(1/W) @ V.T
            ldetS = np.sum(np.log(W))
            loglik[k] = -0.5*(np.squeeze(y)-mu).T @ (Sinvu @ (np.squeeze(y)-mu))
            loglik[k] += -0.5*ldetS
    return loglik

