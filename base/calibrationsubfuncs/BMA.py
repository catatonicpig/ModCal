"""Header here."""
import numpy as np
import scipy.stats as sps
from base.utilities import postsampler
import copy
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
    theta = thetaprior.rnd(1000)
    thetadim = theta[0].shape[0]
    if phiprior.rnd(1) is None:
        phidim = 0
        thetaphi = theta
    else:
        phi = phiprior.rnd(1000)
        phidim = (phiprior.rnd(1)).shape[1]
        thetaphi = np.hstack((theta,phi))
        
        
    if type(emu) is tuple:
        emux = emu[0].x
    else:
        emux = emu.x
    
    def logpostfull(thetaphi):
        if phidim > 0.5:
            theta = thetaphi[:, :thetadim]
            phi = thetaphi[:, thetadim:]
        else:
            theta = thetaphi
            phi = None
        logpost =thetaprior.lpdf(theta) + phiprior.lpdf(phi)
        
        inds = np.where(np.isfinite(logpost))[0]
        if phi is None:
            logpost[inds] += loglik(emu, theta[inds], None, y, x, args)
        else:
            logpost[inds] += loglik(emu, theta[inds], phi[inds], y, x, args)
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
    
    info['thetarnd'] = theta
    info['phirnd'] = phi
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
    y = info['y']
    theta = info['thetarnd']
    phi = info['phirnd']
    
    
    if theta.ndim == 1:
        theta = theta.reshape((1, theta.shape[0]))
    elif 'obsvar' in args.keys():
        obsvar = args['obsvar']
    else:
        raise ValueError('Must provide obsvar at this moment.')
    preddict = {}
    
    xtot = np.vstack((info['x'],x))
    mx =info['x'].shape[0]
    if type(emu) is tuple:
        predinfo = [dict() for x in range(len(emu))]
        for k in range(0, len(emu)):
            predinfo[k] = emu[k].predict(theta, xtot)
        preddict['meanfull'] = copy.deepcopy(predinfo[0]['mean'][:,mx:])
        preddict['varfull'] = copy.deepcopy(predinfo[0]['var'][:,mx:])
        preddict['rnd'] = copy.deepcopy(predinfo[0]['mean'][:,mx:])
        preddict['modelrnd'] = copy.deepcopy(predinfo[0]['mean'][:,mx:])
    else:
        predinfo = emu.predict(theta, xtot)
        preddict['meanfull'] = copy.deepcopy(predinfo['mean'][:,mx:])
        preddict['varfull'] = copy.deepcopy(predinfo['var'][:,mx:])
        preddict['rnd'] = copy.deepcopy(predinfo['mean'][:,mx:])
        preddict['modelrnd'] = copy.deepcopy(predinfo['mean'][:,mx:])
    
    for k in range(0, theta.shape[0]):
        if type(emu) is tuple:
            logliklocal = np.zeros(len(emu))
            predlocal = np.zeros((x.shape[0], len(emu)))
            sslocal = np.zeros((x.shape[0], len(emu)))
            for l in range(0, len(emu)):
                mu = predinfo[l]['mean'][k, :mx]
                A1 = np.squeeze(predinfo[l]['covdecomp'][k, :, :mx])
                S0 = A1.T @ A1
                if 'cov_disc' in args.keys():
                    S0 += args['cov_disc'](xtot[:mx,:], l, phi[k,:])
                W, V = np.linalg.eigh(np.diag(obsvar) + S0)
                Sinvu = V @ np.diag(1/W) @ V.T
                ldetS = np.sum(np.log(W))
                logliklocal[l] = -0.5*(np.squeeze(y)-mu).T @ (Sinvu @ (np.squeeze(y)-mu))
                logliklocal[l] += -0.5*ldetS
                A2 = np.squeeze(predinfo[l]['covdecomp'][k, :, mx:])
                S10 = A2.T @ A1
                S11 = A2.T @ A2
                if 'cov_disc' in args.keys():
                    C = args['cov_disc'](xtot, l, phi[k,:])
                    S10 += C[mx:,:][:,:mx]
                    S11 += C[mx:,:][:,mx:]
                mus0 = predinfo[l]['mean'][k, mx:]
                predlocal[:,l] = mus0 + S10 @ np.linalg.solve(np.diag(obsvar) + S0, y-mu)
                sslocal[:,l] = np.diag(S11 - S10 @ np.linalg.solve(np.diag(obsvar) + S0, S10.T)) + predlocal[:,l] ** 2
            lm = np.max(logliklocal)
            logpostcalc = np.log(np.sum(np.exp(logliklocal-lm))) + lm
            post = np.exp(logliklocal - logpostcalc)
            
            preddict['meanfull'][k,:] = np.sum(post * predlocal,1)
            preddict['varfull'][k,:] = np.sum(post * sslocal,1) -\
                preddict['meanfull'][k,:] ** 2
            rc = np.random.choice(len(emu), p = post)
            A1 = np.squeeze(predinfo[rc]['covdecomp'][k, :, :mx])
            A2 = np.squeeze(predinfo[rc]['covdecomp'][k, :, mx:])
            
            S0 = A1.T @ A1
            S0 += np.diag(obsvar)
            if 'cov_disc' in args.keys():
                C = args['cov_disc'](xtot, rc, phi[k,:])
                S0 += C[:mx,:][:,:mx]
                S10 += C[mx:,:][:,:mx]
                S11 += C[mx:,:][:,mx:]
            Wmat, Vmat = np.linalg.eigh(S11 - S10 @ np.linalg.solve(S0, S10.T))
            re = Vmat @ np.diag(np.sqrt(np.abs(Wmat))) @ Vmat.T @\
                sps.norm.rvs(0,1,size=(Vmat.shape[1]))
            preddict['rnd'][k,:] = preddict['meanfull'][k, :]  + re
            preddict['modelrnd'][k,:] = 1*mus0
            re = predinfo[rc]['covdecomp'][k,:,:].T @\
                sps.norm.rvs(0,1,size=(predinfo[rc]['covdecomp'].shape[1]))
            preddict['rnd'][k,:] = predinfo[rc]['mean'][k,mx:] + re[mx:]
            preddict['modelrnd'][k,:] = predinfo[rc]['mean'][k,mx:] + re[mx:]

        else:
            mu = predinfo['mean'][k, :mx]
            A1 = np.squeeze(predinfo['covdecomp'][k, :, :mx])
            A2 = np.squeeze(predinfo['covdecomp'][k, :, mx:])
            S0 = A1.T @ A1
            S10 = A2.T @ A1
            S11 = A2.T @ A2
            if 'cov_disc' in args.keys():
                C = args['cov_disc'](xtot, phi[k,:])
                S0 += C[:mx,:][:,:mx]
                S10 += C[mx:,:][:,:mx]
                S11 += C[mx:,:][:,mx:]
            S0 += np.diag(obsvar)
            mus0 = 1*predinfo['mean'][k, mx:]
            preddict['meanfull'][k,:] = mus0 + S10 @ np.linalg.solve(S0, y-mu)
            preddict['varfull'][k,:] = np.diag(S11 - S10 @ np.linalg.solve(S0, S10.T))
            Wmat, Vmat = np.linalg.eigh(S11 - S10 @ np.linalg.solve(S0, S10.T))
            re = Vmat @ np.diag(np.sqrt(np.abs(Wmat))) @ Vmat.T @\
                sps.norm.rvs(0,1,size=(Vmat.shape[1]))
            preddict['modelrnd'][k,:] = predinfo['mean'][k, mx:]
            re = predinfo['covdecomp'][k,:,:].T @\
                sps.norm.rvs(0,1,size=(predinfo['covdecomp'].shape[1]))
            preddict['rnd'][k,:] = predinfo['mean'][k,mx:] + re[mx:]
                
    preddict['mean'] = np.mean(preddict['meanfull'], 0)
    varterm1 = np.var(preddict['meanfull'], 0)
    preddict['var'] = np.mean(preddict['varfull'], 0) + varterm1
    return preddict

def loglik(emu, theta, phi, y, x, args):
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
            predinfo[k] = emu[k].predict(theta, x)
        loglik = np.zeros(predinfo[0]['mean'].shape[0])
    else:
        predinfo = emu.predict(theta, x)
        loglik = np.zeros(predinfo['mean'].shape[0])
        
    for k in range(0, theta.shape[0]):
        if type(emu) is tuple:
            logliklocal = np.zeros(len(emu))
            for l in range(0, len(emu)):
                mu = predinfo[l]['mean'][k, :]
                A1 = np.squeeze(predinfo[l]['covdecomp'][k, :, :])
                S0 = A1.T @ A1
                if 'cov_disc' in args.keys():
                    S0 += args['cov_disc'](x, l, phi[k,:])
                W, V = np.linalg.eigh(np.diag(obsvar) + S0)
                Sinvu = V @ np.diag(1/W) @ V.T
                ldetS = np.sum(np.log(W))
                logliklocal[l] = -0.5*(np.squeeze(y)-mu).T @ (Sinvu @ (np.squeeze(y)-mu))
                logliklocal[l] += -0.5*ldetS
            lm = np.max(logliklocal)
            loglik[k] = np.log(np.sum(np.exp(logliklocal-lm))) + lm
        else:
            mu = predinfo['mean'][k, :]
            A1 = np.squeeze(predinfo['covdecomp'][k, :, :])
            S0 = A1.T @ A1
            if 'cov_disc' in args.keys():
                S0 += args['cov_disc'](x, phi[k,:])
            W, V = np.linalg.eigh(np.diag(obsvar) + S0)
            Sinvu = V @ np.diag(1/W) @ V.T
            ldetS = np.sum(np.log(W))
            loglik[k] = -0.5*(np.squeeze(y)-mu).T @ (Sinvu @ (np.squeeze(y)-mu))
            loglik[k] += -0.5*ldetS
    return loglik

