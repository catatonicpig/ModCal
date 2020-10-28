"""Header here."""
import numpy as np
import scipy.stats as sps
from base.utilities import postsampler
import copy

##############################################################################
##############################################################################
###################### THIS BEGINS THE REQUIRED PORTION ######################
######### THE NEXT FUNCTIONS REQUIRED TO BE CALLED BY CALIBRATION ############
##############################################################################
##############################################################################

def fit(info, emu, y, x, args=None):
    r"""
    This is a optional docstring for an internal function.
    """
    
    if type(emu) is not tuple:
        raise ValueError('Must provide a tuple of emulators to BDM_BMA.')
    
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
    return

def predict(x, emu, calinfo, args):
    r"""
    This is a optional docstring for an internal function.
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
    emumean = [np.ones(1) for x in range(len(emu))]
    emucov = [np.ones(1) for x in range(len(emu))]
    for k in range(0, len(emu)):
        emupredict = emu[k].predict(theta, xtot)
        emumean[k] = emupredict.mean()
        emucov[k] = emupredict.cov()
    
    meanfull = np.ones((emumean[k].shape[0],x.shape[0]))
    varfull = np.ones((emumean[k].shape[0],x.shape[0]))
    info['rnd'] = np.ones((emumean[k].shape[0],x.shape[0]))
    info['modelrnd'] = np.ones((emumean[k].shape[0],x.shape[0]))
    for k in range(0, theta.shape[0]):
        logliklocal = np.zeros(len(emu))
        predlocal = np.zeros((x.shape[0], len(emu)))
        sslocal = np.zeros((x.shape[0], len(emu)))
        for l in range(0, len(emu)):
            mu = emumean[l][k][:mx]
            S0 = emucov[l][k][:,:mx][:mx,:]
            if 'cov_disc' in args.keys():
                S0 += args['cov_disc'](xtot[:mx,:], l, phi[k,:])
            W, V = np.linalg.eigh(np.diag(obsvar) + S0)
            Sinvu = V @ np.diag(1/W) @ V.T
            ldetS = np.sum(np.log(W))
            logliklocal[l] = -0.5*(np.squeeze(y)-mu).T @ (Sinvu @ (np.squeeze(y)-mu))
            logliklocal[l] += -0.5*ldetS
            S10 = emucov[l][k][mx:,:][:,:mx]
            S11 = emucov[l][k][:,mx:][mx:,:]
            if 'cov_disc' in args.keys():
                C = args['cov_disc'](xtot, l, phi[k,:])
                S10 += C[mx:,:][:,:mx]
                S11 += C[mx:,:][:,mx:]
            mus0 = emumean[l][k][mx:]
            predlocal[:,l] = mus0 + S10 @ np.linalg.solve(np.diag(obsvar) + S0, y-mu)
            sslocal[:,l] = np.diag(S11 - S10 @ np.linalg.solve(np.diag(obsvar) + S0, S10.T)) + predlocal[:,l] ** 2
        lm = np.max(logliklocal)
        logpostcalc = np.log(np.sum(np.exp(logliklocal-lm))) + lm
        post = np.exp(logliklocal - logpostcalc)
        
        meanfull[k,:] = np.sum(post * predlocal,1)
        varfull[k,:] = np.sum(post * sslocal,1) -\
            meanfull[k,:] ** 2
        rc = np.random.choice(len(emu), p = post)
        
        S0 = emucov[rc][k][:,:mx][:mx,:]
        S0 += np.diag(obsvar)
        S10 = emucov[rc][k][mx:,:][:,:mx]
        S11 = emucov[rc][k][:,mx:][mx:,:]
        if 'cov_disc' in args.keys():
            C = args['cov_disc'](xtot, rc, phi[k,:])
            S0 += C[:mx,:][:,:mx]
            S10 += C[mx:,:][:,:mx]
            S11 += C[mx:,:][:,mx:]
        Wmat, Vmat = np.linalg.eigh(S11 - S10 @ np.linalg.solve(S0, S10.T))
        re = Vmat @ np.diag(np.sqrt(np.abs(Wmat))) @ Vmat.T @\
            sps.norm.rvs(0,1,size=(Vmat.shape[1]))
        info['rnd'][k,:] = meanfull[k, :]  + re
        info['modelrnd'][k,:] = 1*mus0
    
    info['mean'] = np.mean(meanfull, 0)
    varterm1 = np.var(meanfull, 0)
    info['var'] = np.mean(varfull, 0) + varterm1
    return info


##############################################################################
##############################################################################
####################### THIS ENDS THE REQUIRED PORTION #######################
###### THE NEXT FUNCTIONS ARE OPTIONAL TO BE CALLED BY CALIBRATION ###########
##############################################################################
##############################################################################

##############################################################################
##############################################################################
####################### THIS ENDS THE OPTIONAL PORTION #######################
######### USE SPACE BELOW FOR ANY SUPPORTING FUNCTIONS YOU DESIRE ############
##############################################################################
##############################################################################

def loglik(emu, theta, phi, y, x, args):
    r"""
    This is a optional docstring for an internal function.
    """
    
    if theta.ndim == 1:
        theta = theta.reshape((1, theta.shape[0]))
    elif 'obsvar' in args.keys():
        obsvar = args['obsvar']
    else:
        raise ValueError('Must provide obsvar at this moment.')
    
    emumean = [np.ones(1) for x in range(len(emu))]
    emucov = [np.ones(1) for x in range(len(emu))]
    for k in range(0, len(emu)):
        emupredict = emu[k].predict(theta, x)
        emumean[k] = emupredict.mean()
        emucov[k] = emupredict.cov()
    loglik = np.zeros(emumean[0].shape[0])
        
    for k in range(0, theta.shape[0]):
        logliklocal = np.zeros(len(emu))
        for l in range(0, len(emu)):
            mu = emumean[l][k]
            S0 = emucov[l][k]
            if 'cov_disc' in args.keys():
                S0 += args['cov_disc'](x, l, phi[k,:])
            W, V = np.linalg.eigh(np.diag(obsvar) + S0)
            Sinvu = V @ np.diag(1/W) @ V.T
            ldetS = np.sum(np.log(W))
            logliklocal[l] = -0.5*(np.squeeze(y)-mu).T @ (Sinvu @ (np.squeeze(y)-mu))
            logliklocal[l] += -0.5*ldetS
        lm = np.max(logliklocal)
        loglik[k] = np.log(np.sum(np.exp(logliklocal-lm))) + lm
    return loglik

