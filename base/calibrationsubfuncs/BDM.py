"""Header here."""
import numpy as np
import scipy.stats as sps
from base.utilities import postsampler

def fit(thetaprior, emu, y, x, args=None):
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
    
    theta = thetaprior.rvs(1000)
    phi = phiprior.rvs(1000)
    
    thetadim = theta[0].shape[0]
    if phi is None:
        phidim = 0
        thetaphi = theta
    elif phi[0] is None:
        phidim = 0
        thetaphi = theta
    else:
        phidim = phi[0].shape[0]
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
    
    info = {}
    info['theta'] = theta
    info['phi'] = phi
    info['xind'] = xind
    info['y'] = y
    info['x'] = x
    info['emux'] = emux
    return info


def predict(x, emu, info, args = None):
    """
    Return posterior of function evaluation at the new parameters.

    Parameters
    ----------
    emu : dict
        A fitted emu model defined as an emulation class.
    info : array of float
        Some matrix of parameters where function evaluations as starting points.
    y : Observations
        A vector of the same length as x with observations. 'None' is equivlent to a vector of
        zeros.
    Sinv : Observation Precision Matrix
        A matrix of the same length as "emu.x" with observations. 'None' is equivlent to the
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
    
    theta = info['theta']
    phi = info['phi']
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
            covmats = [np.array(0) for x in range(len(emu))]
            covmatsB = [np.array(0) for x in range(len(emu))]
            covmatsC = [np.array(0) for x in range(len(emu))]
            covmatsinv = [np.array(0) for x in range(len(emu))]
            mus = [np.array(0) for x in range(len(emu))]
            totInv = np.zeros((emu[0].x.shape[0], emu[0].x.shape[0]))
            term2 = np.zeros(emu[0].x.shape[0])
            for l in range(0, len(emu)):
                mus[l] = predinfo[l]['mean'][k, :]
            for l in reversed(range(0, len(emu))):
                A1 = np.squeeze(predinfo[l]['covdecomp'][k, :, :])
                covmats[l] = A1.T @ A1
                if 'cov_disc' in args.keys():
                    covmats[l] += args['cov_disc'](info['emux'], l, phi[k,:])
                covmats[l] += np.diag(np.diag(covmats[l])) * (10 ** (-8))
                covmatsinv[l] = np.linalg.inv(covmats[l])
                totInv += covmatsinv[l]
                term2 += covmatsinv[l] @ mus[l]

            S0 = np.linalg.inv(totInv)
            m0 = np.linalg.solve(totInv, term2)
            m00 = m0[xind]
            m10 = m0[xindnew]
            S0inv = np.linalg.inv(np.diag(obsvar) + S0[xind,:][:,xind])
            S10 = S0[xindnew, :][:, xind]
            Mat1 = S10 @ S0inv
            resid = np.squeeze(y)
            preddict['meanfull'][k, :] =  m10 +  Mat1 @ (np.squeeze(y) - m00)
            preddict['varfull'][k, :] = (np.diag(S0)[xindnew] -\
                np.sum(S10 * Mat1,1))
            Wmat, Vmat = np.linalg.eigh(S0[xindnew,:][:,xindnew] - S10 @ Mat1.T)
            
            re = Vmat @ np.diag(np.sqrt(np.abs(Wmat))) @ Vmat.T @\
                sps.norm.rvs(0,1,size=(Vmat.shape[1]))
            preddict['draws'][k,:] = preddict['meanfull'][k, :]  + re
            Wmat, Vmat = np.linalg.eigh(S0[xindnew,:][:,xindnew])
            re = Vmat @ np.diag(np.sqrt(np.abs(Wmat))) @ Vmat.T @\
                sps.norm.rvs(0,1,size=(Vmat.shape[1]))
            preddict['modeldraws'][k,:] = m10 + re
        else:
            m0 = np.squeeze(y) * 0
            mut = np.squeeze(y) - predinfo['mean'][(k, xind)]
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
            preddict['meanfull'][k, :] = mus0 + S10 @ np.linalg.solve(S0, mut)
            preddict['varfull'][k, :] = np.diag(S11 - S10 @ np.linalg.solve(S0, S10.T))
            Wmat, Vmat = np.linalg.eigh(S11 - S10 @ np.linalg.solve(S0, S10.T))
            re = Vmat @ np.diag(np.sqrt(np.abs(Wmat))) @ Vmat.T @\
                sps.norm.rvs(0,1,size=(Vmat.shape[1]))
            preddict['draws'][k,:] = preddict['meanfull'][k, :]  + re
            preddict['modeldraws'][k,:] = mus0

    preddict['mean'] = np.mean(preddict['meanfull'], 0)
    varterm1 = np.var(preddict['meanfull'], 0)
    preddict['var'] = np.mean(preddict['varfull'], 0) + varterm1
    return preddict


def loglik(emu, theta, phi, y, xind, args):
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
        A matrix of the same length as "emu.x" with observations. 'None' is equivlent to the
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
    else:
        predinfo = emu.predict(theta)
    
    loglik = np.zeros(theta.shape[0])
    for k in range(0, theta.shape[0]):
        if type(emu) is tuple:
            covmats = [np.array(0) for x in range(len(emu))]
            covmatsinv = [np.array(0) for x in range(len(emu))]
            mus = [np.array(0) for x in range(len(emu))]
            resid = np.zeros(len(emu) * xind.shape[0])
            totInv = np.zeros((xind.shape[0], xind.shape[0]))
            term2 = np.zeros(xind.shape[0])
            for l in range(0, len(emu)):
                mus[l] = predinfo[l]['mean'][k, xind]
                A1 = np.squeeze(predinfo[l]['covdecomp'][k, :, xind])
                covmats[l] = (A1 @ A1.T)
                if 'cov_disc' in args.keys():
                    covmats[l] += args['cov_disc'](emu[l].x[xind,:], l, phi[k,:])
                covmats[l] += np.diag(np.diag(covmats[l])) * (10 ** (-8))
                covmatsinv[l] = np.linalg.inv(covmats[l])
                totInv += covmatsinv[l]
                term2 += covmatsinv[l] @ mus[l]
            m0 = np.linalg.solve(totInv, term2)
            W, V = np.linalg.eigh(np.diag(obsvar) + np.linalg.inv(totInv))
        else:
            m0 = predinfo['mean'][(k, xind)]
            A1 = np.squeeze(predinfo['covdecomp'][k, :, xind])
            S0 = A1 @ A1.T
            if 'cov_disc' in args.keys():
                S0 += args['cov_disc'](emu.x[xind,:], phi[k,:])
            W, V = np.linalg.eigh(np.diag(obsvar) + S0)
        muadj = V.T @ (np.squeeze(y) - m0)
        loglik[k] = -0.5 * np.sum((muadj ** 2) / W)-0.5 * np.sum(np.log(W))
    return loglik