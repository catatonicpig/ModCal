"""Header here."""
import numpy as np
import scipy.stats as sps
from base.utilities import postsampler
import copy

"""
##############################################################################
##############################################################################
###################### THIS BEGINS THE REQUIRED PORTION ######################
######### THE NEXT FUNCTIONS REQUIRED TO BE CALLED BY CALIBRATION ############
##############################################################################
##############################################################################
"""

"""
##############################################################################
################################### fit ######################################
### The purpose of this is to take an emulator _emu_ and plug all of our fit
### information into _info_, which is a python dictionary. Example emu functions
### emupredict = emu(theta, x).predict()
### emupredict.mean(): an array of size (theta.shape[0], x.shape[0]) containing the mean
###             of the target function at theta and x
### emupredict.var(): an array of size (theta.shape[0], x.shape[0]) containing the variance
###             of the target function at theta and x
### emupredict.cov(): an array of size (theta.shape[0], x.shape[0], x.shape[0]) containing the
###             covariance matrix in x at each theta.
### emupredict.rand(s): an array of size (s, theta.shape[0], x.shape[0]) containing s
###             random draws from the emulator at theta and x.
### Not all of these will work, it depends on your emulation software.
##############################################################################
##############################################################################
"""
def fit(fitinfo, emu, x, y, args=None):
    r"""
    Fits a calibration model.

    Parameters
    ----------
    fitinfo : dict
        An arbitary dictionary where you should place all of your fitting information once complete.
        This dictionary is pass by reference, so there is no reason to return anything. Keep
        only stuff that will be used by predict below. Note that the following are preloaded
        fitinfo['thetaprior'].rnd(s) : Get s random draws from the prior predictive distribution on
            theta.
        fitinfo['thetaprior'].lpdf(theta) : Get the logpdf at theta(s).
        In addition, calibration can directly use:
        fitinfo['thetamean'] : the mean of the prediction of theta
        fitinfo['thetavar'] : the var of the predictive variance on theta
        fitinfo['thetarand'] : some number draws from the predictive distribution on theta
    emu : tuple of instances of emulator class
        An emulator class instatance as defined in emulation
    x : array of objects
        An array of x  that represent the inputs.
    y : array of float
        A one demensional array of observed values at x
    args : dict
        A dictionary containing options passed to you.
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
    
    thetaprior = fitinfo['thetaprior']
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
    
    fitinfo['thetarnd'] = theta
    fitinfo['phirnd'] = phi
    fitinfo['y'] = y
    fitinfo['x'] = x
    return

"""
##############################################################################
################################### predict ##################################
### The purpose of this is to take an emulator emu alongside fitinfo, and 
### predict at x. You shove all your information into the dictionary predinfo.
##############################################################################
##############################################################################
"""
def predict(predinfo, fitinfo, emu, x, args=None):
    r"""
    Finds prediction at x given the emulator emu and dictionary fitinfo.

    Parameters
    ----------
    predinfo : dict
        An arbitary dictionary where you should place all of your prediction information once complete. 
        This dictionary is pass by reference, so there is no reason to return anything. Keep
        only stuff that will be used by predict.  Key elements
        predinfo['mean'] : the mean of the prediction
        predinfo['var'] : the variance of the prediction
        predinfo['rand'] : some number draws from the predictive distribution on theta.
    fitinfo : dict
        An arbitary dictionary where you placed all your important fitting information from the 
        fit function above.
    emu : tuple of instances of emulator class
        An emulator class instatance as defined in emulation
    x : array of float
        An array of x values where you want to predict.
    args : dict
        A dictionary containing options passed to you.
    """
    y = fitinfo['y']
    theta = fitinfo['thetarnd']
    phi = fitinfo['phirnd']
    
    if theta.ndim == 1:
        theta = theta.reshape((1, theta.shape[0]))
    elif 'obsvar' in args.keys():
        obsvar = args['obsvar']
    else:
        raise ValueError('Must provide obsvar at this moment.')
    
    xtot = np.vstack((fitinfo['x'],x))
    mx = fitinfo['x'].shape[0]
    emumean = [np.ones(1) for x in range(len(emu))]
    emucov = [np.ones(1) for x in range(len(emu))]
    for k in range(0, len(emu)):
        emupredict = emu[k].predict(theta, xtot)
        emumean[k] = emupredict.mean()
        emucov[k] = emupredict.cov()
    
    meanfull = np.ones((emumean[k].shape[0],x.shape[0]))
    varfull = np.ones((emumean[k].shape[0],x.shape[0]))
    predinfo['rnd'] = np.ones((emumean[k].shape[0],x.shape[0]))
    predinfo['modelrnd'] = np.ones((emumean[k].shape[0],x.shape[0]))
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
        predinfo['rnd'][k,:] = meanfull[k, :]  + re
        predinfo['modelrnd'][k,:] = 1*mus0
    
    predinfo['mean'] = np.mean(meanfull, 0)
    varterm1 = np.var(meanfull, 0)
    predinfo['var'] = np.mean(varfull, 0) + varterm1
    return

"""
##############################################################################
##############################################################################
####################### THIS ENDS THE REQUIRED PORTION #######################
###### THE NEXT FUNCTIONS ARE OPTIONAL TO BE CALLED BY CALIBRATION ###########
## If this project works, there will be a list of useful calibration functions
## to provide as you want.
##############################################################################
##############################################################################
"""
"""
##############################################################################
##############################################################################
####################### THIS ENDS THE OPTIONAL PORTION #######################
######### USE SPACE BELOW FOR ANY SUPPORTING FUNCTIONS YOU DESIRE ############
##############################################################################
##############################################################################
"""

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

