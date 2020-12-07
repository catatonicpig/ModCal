"""Header here."""
import numpy as np
import scipy.stats as sps
from base.utilities import postsampler
import copy

"""
##############################################################################
################################### fit ######################################
This [calibrationfitinfo] automatically filled by docinfo.py when running updatedocs.py
##############################################################################
##############################################################################
"""

def fit(fitinfo, emu, x, y,  args=None):
    r"""
    Fits a calibration model.
    This [calibrationfitdocstring] automatically filled by docinfo.py when running updatedocs.py

    Parameters
    ----------
    fitinfo : dict
        An arbitary dictionary where you should place all of your fitting information once complete.
        This dictionary is pass by reference, so there is no reason to return anything. Keep
        only stuff that will be used by predict below. Note that the following are preloaded
        fitinfo['thetaprior'].rnd(s) : Get s random draws from the prior predictive distribution on
            theta.
        fitinfo['thetaprior'].lpdf(theta) : Get the logpdf at theta(s).
        The following are optional preloads based on user input
        fitinfo[yvar] : The vector of observation variances at y
        In addition, calibration can directly use and communicate back to the user if you include:
        fitinfo['thetamean'] : the mean of the prediction of theta
        fitinfo['thetavar'] : the var of the predictive variance on theta
        fitinfo['thetarnd'] : some number draws from the predictive distribution on theta
    emu : instance of emulator class
        An emulator class instatance as defined in emulation
    x : array of objects
        An array of x  that represent the inputs.
    y : array of float
        A one demensional array of observed values at x
    args : dict
        A dictionary containing options passed to you.
    """
    
    thetaprior = fitinfo['thetaprior']
    theta = thetaprior.rnd(1000)
    
    if 'yvar' in fitinfo.keys():
        obsvar = np.squeeze(fitinfo['yvar'])
    else:
        raise ValueError('Must provide yvar in this software.')
    
    if 'phiprior' in args.keys():
        phiprior = args['phiprior']
    else:
        raise ValueError('Must provide a prior on statistical parameters in this software.')
    
    
    if type(emu) is tuple:
        raise ValueError('Cannot provide a tuple of emulators to BDM.')
    
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
        skipprioradj = False
        if thetaphi.ndim < 1.5:
            thetaphi = np.reshape(thetaphi,(1,-1))
            skipprioradj = True
        if phidim > 0.5:
            theta = thetaphi[:, :thetadim]
            phi = thetaphi[:, thetadim:]
        else:
            theta = thetaphi
            phi = None
        logpost =thetaprior.lpdf(theta) + phiprior.lpdf(phi)
        inds = np.where(np.isfinite(logpost))[0]
        if not skipprioradj:
            if phi is None:
                logpost[inds] += loglik(fitinfo, emu, theta[inds], None, y, x, args)
            else:
                logpost[inds] += loglik(fitinfo,emu, theta[inds], phi[inds], y, x, args)
        else:
            logpost += loglik(fitinfo,emu, theta, phi, y, x, args)
        return logpost
    
    numsamp = 1000
    tarESS = np.max((100, 10 * thetaphi.shape[1]))
    thetaphi = postsampler(thetaphi, logpostfull, args)
    
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
This [calibrationpredictinfo] automatically filled by docinfo.py when running updatedocs.py
##############################################################################
################################### predict ##################################
### The purpose of this is to take an emulator emu alongside fitinfo, and 
### predict at x. You shove all your information into the dictionary predinfo.
##############################################################################
##############################################################################
"""
def predict(predinfo, fitinfo, emu, x, args=None):
    r"""
    Finds prediction at x given the emulator _emu_ and dictionary fitinfo.
    This [calibrationpredictdocstring] automatically filled by docinfo.py when running updatedocs.py

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
    emu : instance of emulator class
        An emulator class instatance as defined in emulation.
    x : array of float
        An array of x values where you want to predict.
    args : dict
        A dictionary containing options passed to you.
    """
    
    y = fitinfo['y']
    
    theta = fitinfo['thetarnd']
    phi = fitinfo['phirnd']
    if theta.ndim == 1 and fitinfo['theta'].shape[1] > 1.5:
        theta = theta.reshape((1, theta.shape[0]))
    
    if 'yvar' in fitinfo.keys():
        obsvar = np.squeeze(fitinfo['yvar'])
    else:
        raise ValueError('Must provide yvar in this software.')
    
    if 'cov_disc' in args.keys():
        cov_disc = args['cov_disc']
    else:
        raise ValueError('Must provide obsvar at this moment.')
    
    xtot = np.vstack((fitinfo['x'],x))
    mx =fitinfo['x'].shape[0]
    emupredict = emu.predict(xtot, theta)
    meanfull = copy.deepcopy(emupredict()[mx:,:]).T
    varfull = copy.deepcopy(emupredict()[mx:,:]).T
    predinfo['rnd'] = copy.deepcopy(emupredict()[mx:,:]).T
    predinfo['modelrnd'] = copy.deepcopy(emupredict()[mx:,:]).T
    
    
    emupredict = emu.predict(xtot, theta)
    emumean = emupredict.mean()
    emucov = emupredict.covx()
    xind = range(0,mx)
    xindnew = range(mx,xtot.shape[0])
    for k in range(0, theta.shape[0]):
        m0 = np.squeeze(y) * 0
        mut = np.squeeze(y) - emupredict()[(xind,k)]
        m0 = emumean[:,k]
        St = emucov[:,k,:]
        St[np.isnan(St)] = 0
        S0 = St[xind,:][:,xind]
        S10 = St[xindnew,:][:,xind]
        S11 =St[xindnew,:][:,xindnew]
        C = args['cov_disc'](xtot, phi[k,:])
        S0 += C[xind,:][:,xind]
        S10 += C[xindnew,:][:,xind]
        S11 += C[xindnew,:][:,xindnew]
        S0 += np.diag(obsvar)
        mus0 = emupredict()[(xindnew, k)]
        meanfull[k, :] = mus0 + S10 @ np.linalg.solve(S0, mut)
        varfull[k, :] = np.diag(S11 - S10 @ np.linalg.solve(S0, S10.T))
        Wmat, Vmat = np.linalg.eigh(S11 - S10 @ np.linalg.solve(S0, S10.T))
        re = Vmat @ np.diag(np.sqrt(np.abs(Wmat))) @ Vmat.T @\
            sps.norm.rvs(0,1,size=(Vmat.shape[1]))
        predinfo['rnd'][k,:] = meanfull[k, :]  + re
        predinfo['modelrnd'][k,:] = mus0

    predinfo['mean'] = np.mean(meanfull, 0)
    varterm1 = np.var(meanfull, 0)
    predinfo['var'] = np.mean(varfull, 0) + varterm1
    return

"""
This [calibrationadditionalfuncsinfo] automatically filled by docinfo.py when running updatedocs.py
##############################################################################
##############################################################################
####################### THIS ENDS THE REQUIRED PORTION #######################
###### THE NEXT FUNCTIONS ARE OPTIONAL TO BE CALLED BY CALIBRATION ###########
## If this project works, there will be a list of useful calibration functions
## to provide as you want.
##############################################################################
##############################################################################
"""

def thetarnd(fitinfo, s=100, args=None):
    """
    Return s draws from the predictive distribution of theta.  Not required.
    """
    return fitinfo['thetarnd'][np.random.choice(fitinfo['thetarnd'].shape[0], size=s), :]

"""
This [endfunctionsflag] is automatically filled by docinfo.py when running updatedocs.py
##############################################################################
##############################################################################
####################### THIS ENDS THE OPTIONAL PORTION #######################
######### USE SPACE BELOW FOR ANY SUPPORTING FUNCTIONS YOU DESIRE ############
##############################################################################
##############################################################################
"""
def loglik(fitinfo, emu, theta, phi, y, x, args):
    r"""
    This is a optional docstring for an internal function.
    """
    
    
    if 'yvar' in fitinfo.keys():
        obsvar = np.squeeze(fitinfo['yvar'])
    else:
        raise ValueError('Must provide yvar in this software.')
    
    if 'cov_disc' in args.keys():
        cov_disc = args['cov_disc']
    else:
        raise ValueError('Must provide cov_disc at this moment.')
    emupredict = emu.predict(x, theta)
    emumean = emupredict.mean()
    emucov = emupredict.covx()
    loglik = np.zeros(theta.shape[0])
    for k in range(0, theta.shape[0]):
        m0 = emumean[:,k]
        S0 = emucov[:,k]#Shalf.T @ Shalf
        S0 += cov_disc(x, phi[k,:])
        W, V = np.linalg.eigh(np.diag(obsvar) + S0)
        muadj = V.T @ (np.squeeze(y) - m0)
        loglik[k] = -0.5 * np.sum((muadj ** 2) / W)-0.5 * np.sum(np.log(W))
    return loglik