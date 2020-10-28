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

def predict(x, emu, calinfo, args):
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
    y = calinfo['y']
    theta = calinfo['thetarnd']
    phi = calinfo['phirnd']
    
    
    if theta.ndim == 1:
        theta = theta.reshape((1, theta.shape[0]))
    elif 'obsvar' in args.keys():
        obsvar = args['obsvar']
    else:
        raise ValueError('Must provide obsvar at this moment.')
    info = {}
    
    xtot = np.vstack((calinfo['x'],x))
    mx = calinfo['x'].shape[0]
    if type(emu) is tuple:
        emupredict = [dict() for x in range(len(emu))]
        for k in range(0, len(emu)):
            emupredict[k] = emu[k].predict(theta, xtot)
        info['meanfull'] = copy.deepcopy(emupredict[0]()[:,mx:])
        info['varfull'] = copy.deepcopy(emupredict[0]()[:,mx:])
        info['rnd'] = copy.deepcopy(emupredict[0]()[:,mx:])
        info['modelrnd'] = copy.deepcopy(emupredict[0]()[:,mx:])
    else:
        emupredict = emu.predict(theta, xtot)
        info['meanfull'] = copy.deepcopy(emupredict()[:,mx:])
        info['varfull'] = copy.deepcopy(emupredict()[:,mx:])
        info['rnd'] = copy.deepcopy(emupredict()[:,mx:])
        info['modelrnd'] = copy.deepcopy(emupredict()[:,mx:])
    
    for k in range(0, theta.shape[0]):
        if type(emu) is tuple:
            logliklocal = np.zeros(len(emu))
            predlocal = np.zeros((x.shape[0], len(emu)))
            sslocal = np.zeros((x.shape[0], len(emu)))
            for l in range(0, len(emu)):
                mu = emupredict[l]()[k, :mx]
                A1 = np.squeeze(emupredict[l].info['covdecomp'][k, :, :mx])
                S0 = A1.T @ A1
                if 'cov_disc' in args.keys():
                    S0 += args['cov_disc'](xtot[:mx,:], l, phi[k,:])
                W, V = np.linalg.eigh(np.diag(obsvar) + S0)
                Sinvu = V @ np.diag(1/W) @ V.T
                ldetS = np.sum(np.log(W))
                logliklocal[l] = -0.5*(np.squeeze(y)-mu).T @ (Sinvu @ (np.squeeze(y)-mu))
                logliklocal[l] += -0.5*ldetS
                A2 = np.squeeze(emupredict[l].info['covdecomp'][k, :, mx:])
                S10 = A2.T @ A1
                S11 = A2.T @ A2
                if 'cov_disc' in args.keys():
                    C = args['cov_disc'](xtot, l, phi[k,:])
                    S10 += C[mx:,:][:,:mx]
                    S11 += C[mx:,:][:,mx:]
                mus0 = emupredict[l]()[k, mx:]
                predlocal[:,l] = mus0 + S10 @ np.linalg.solve(np.diag(obsvar) + S0, y-mu)
                sslocal[:,l] = np.diag(S11 - S10 @ np.linalg.solve(np.diag(obsvar) + S0, S10.T)) + predlocal[:,l] ** 2
            lm = np.max(logliklocal)
            logpostcalc = np.log(np.sum(np.exp(logliklocal-lm))) + lm
            post = np.exp(logliklocal - logpostcalc)
            
            info['meanfull'][k,:] = np.sum(post * predlocal,1)
            info['varfull'][k,:] = np.sum(post * sslocal,1) -\
                info['meanfull'][k,:] ** 2
            rc = np.random.choice(len(emu), p = post)
            A1 = np.squeeze(emupredict[rc].info['covdecomp'][k, :, :mx])
            A2 = np.squeeze(emupredict[rc].info['covdecomp'][k, :, mx:])
            
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
            info['rnd'][k,:] = info['meanfull'][k, :]  + re
            info['modelrnd'][k,:] = 1*mus0
            re = emupredict[rc].info['covdecomp'][k,:,:].T @\
                sps.norm.rvs(0,1,size=(emupredict[rc].info['covdecomp'].shape[1]))
            info['rnd'][k,:] = emupredict[rc]()[k,mx:] + re[mx:]
            info['modelrnd'][k,:] = emupredict[rc]()[k,mx:] + re[mx:]

        else:
            mu = emupredict()[k, :mx]
            A1 = np.squeeze(emupredict.info['covdecomp'][k, :, :mx])
            A2 = np.squeeze(emupredict.info['covdecomp'][k, :, mx:])
            S0 = A1.T @ A1
            S10 = A2.T @ A1
            S11 = A2.T @ A2
            if 'cov_disc' in args.keys():
                C = args['cov_disc'](xtot, phi[k,:])
                S0 += C[:mx,:][:,:mx]
                S10 += C[mx:,:][:,:mx]
                S11 += C[mx:,:][:,mx:]
            S0 += np.diag(obsvar)
            mus0 = 1*emupredict()[k, mx:]
            info['meanfull'][k,:] = mus0 + S10 @ np.linalg.solve(S0, y-mu)
            info['varfull'][k,:] = np.diag(S11 - S10 @ np.linalg.solve(S0, S10.T))
            Wmat, Vmat = np.linalg.eigh(S11 - S10 @ np.linalg.solve(S0, S10.T))
            re = Vmat @ np.diag(np.sqrt(np.abs(Wmat))) @ Vmat.T @\
                sps.norm.rvs(0,1,size=(Vmat.shape[1]))
            info['modelrnd'][k,:] = emupredict()[k, mx:]
            re = emupredict.info['covdecomp'][k,:,:].T @\
                sps.norm.rvs(0,1,size=(emupredict.info['covdecomp'].shape[1]))
            info['rnd'][k,:] = emupredict()[k,mx:] + re[mx:]
                
    info['mean'] = np.mean(info['meanfull'], 0)
    varterm1 = np.var(info['meanfull'], 0)
    info['var'] = np.mean(info['varfull'], 0) + varterm1
    return info

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
        emupredict = [dict() for x in range(len(emu))]
        for k in range(0, len(emu)):
            emupredict[k] = emu[k].predict(theta, x)
        loglik = np.zeros(emupredict[0]().shape[0])
    else:
        emupredict = emu.predict(theta, x)
        loglik = np.zeros(emupredict().shape[0])
        
    for k in range(0, theta.shape[0]):
        if type(emu) is tuple:
            logliklocal = np.zeros(len(emu))
            for l in range(0, len(emu)):
                mu = emupredict[l]()[k, :]
                A1 = np.squeeze(emupredict[l].info['covdecomp'][k, :, :])
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
            mu = emupredict()[k, :]
            A1 = np.squeeze(emupredict.info['covdecomp'][k, :, :])
            S0 = A1.T @ A1
            if 'cov_disc' in args.keys():
                S0 += args['cov_disc'](x, phi[k,:])
            W, V = np.linalg.eigh(np.diag(obsvar) + S0)
            Sinvu = V @ np.diag(1/W) @ V.T
            ldetS = np.sum(np.log(W))
            loglik[k] = -0.5*(np.squeeze(y)-mu).T @ (Sinvu @ (np.squeeze(y)-mu))
            loglik[k] += -0.5*ldetS
    return loglik

