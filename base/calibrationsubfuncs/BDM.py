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
    
    thetaprior = info['thetaprior']
    theta = thetaprior.rvs(1000)
    
    if 'obsvar' in args.keys():
        obsvar = args['obsvar']
    else:
        raise ValueError('Must provide a prior on statistical parameters in this software.')
    
    if 'phiprior' in args.keys():
        phiprior = args['phiprior']
    else:
        raise ValueError('Must provide a prior on statistical parameters in this software.')
    
    
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
    
    info['theta'] = theta
    info['phi'] = phi
    info['y'] = y
    info['x'] = x
    info['emux'] = emux
    return


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
    y = info['y']
    theta = info['theta']
    phi = info['phi']
    
    
    theta = info['theta']
    phi = info['phi']
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
        preddict['draws'] = copy.deepcopy(predinfo[0]['mean'][:,mx:])
        preddict['modeldraws'] = copy.deepcopy(predinfo[0]['mean'][:,mx:])
    else:
        predinfo = emu.predict(theta, xtot)
        preddict['meanfull'] = copy.deepcopy(predinfo['mean'][:,mx:])
        preddict['varfull'] = copy.deepcopy(predinfo['var'][:,mx:])
        preddict['draws'] = copy.deepcopy(predinfo['mean'][:,mx:])
        preddict['modeldraws'] = copy.deepcopy(predinfo['mean'][:,mx:])
    
    
    xind = range(0,mx)
    xindnew = range(mx,xtot.shape[0])
    for k in range(0, theta.shape[0]):
        if type(emu) is tuple:
            covmats = [np.array(0) for l in range(len(emu))]
            covmatsinv = [np.array(0) for l in range(len(emu))]
            mus = [np.array(0) for l in range(len(emu))]
            totInv = np.zeros((xtot.shape[0], xtot.shape[0]))
            term2 = np.zeros(xtot.shape[0])
            for l in range(0, len(emu)):
                mus[l] = predinfo[l]['mean'][k, :]
            for l in reversed(range(0, len(emu))):
                A1 = np.squeeze(predinfo[l]['covdecomp'][k, :, :])
                covmats[l] = A1.T @ A1
                if 'cov_disc' in args.keys():
                    covmats[l] += args['cov_disc'](xtot, l, phi[k,:])
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

def thetarvs(emu, info, args, n):
    """
    Return posterior of function evaluation at the new parameters.

    """
    return info['theta'][
        np.random.choice(info['theta'].shape[0],
                         size=n), :]


def loglik(emu, theta, phi, y, x, args):
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
            predinfo[k] = emu[k].predict(theta,x)
    else:
        predinfo = emu.predict(theta,x)
    
    loglik = np.zeros(theta.shape[0])
    for k in range(0, theta.shape[0]):
        if type(emu) is tuple:
            covmats = [np.array(0) for x in range(len(emu))]
            covmatsinv = [np.array(0) for x in range(len(emu))]
            mus = [np.array(0) for x in range(len(emu))]
            resid = np.zeros(len(emu) * x.shape[0])
            totInv = np.zeros((x.shape[0], x.shape[0]))
            term2 = np.zeros(x.shape[0])
            for l in range(0, len(emu)):
                mus[l] = predinfo[l]['mean'][k, :]
                A1 = np.squeeze(predinfo[l]['covdecomp'][k, :, :])
                covmats[l] = A1.T @ A1
                if 'cov_disc' in args.keys():
                    covmats[l] += args['cov_disc'](x, l, phi[k,:])
                covmats[l] += np.diag(np.diag(covmats[l])) * (10 ** (-8))
                covmatsinv[l] = np.linalg.inv(covmats[l])
                totInv += covmatsinv[l]
                term2 += covmatsinv[l] @ mus[l]
            m0 = np.linalg.solve(totInv, term2)
            W, V = np.linalg.eigh(np.diag(obsvar) + np.linalg.inv(totInv))
        else:
            m0 = predinfo['mean'][k, :]
            A1 = np.squeeze(predinfo['covdecomp'][k, :, :])
            S0 = A1 @ A1.T
            if 'cov_disc' in args.keys():
                S0 += args['cov_disc'](x, phi[k,:])
            W, V = np.linalg.eigh(np.diag(obsvar) + S0)
        muadj = V.T @ (np.squeeze(y) - m0)
        loglik[k] = -0.5 * np.sum((muadj ** 2) / W)-0.5 * np.sum(np.log(W))
    return loglik