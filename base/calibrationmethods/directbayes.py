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
    
    if 'yvar' in fitinfo.keys():
        obsvar = fitinfo['yvar']
    else:
        raise ValueError('Must provide yvar in this software.')
    
    thetaprior = fitinfo['thetaprior']
    theta = thetaprior.rnd(2500)
    thetadim = theta[0].shape[0]
    
    def logpostfull(theta):
        logpost =thetaprior.lpdf(theta)
        if theta.ndim > 1.5:
            inds = np.where(np.isfinite(logpost))[0]
            logpost[inds] += loglik(fitinfo, emu, theta[inds], y, x, args)
        else:
            if np.isfinite(logpost):
                logpost += loglik(fitinfo, emu, theta, y, x, args)
        return logpost
    numsamp = 10000
    tarESS = np.max((100, 10 * theta.shape[1]))
    theta = postsampler(theta, logpostfull)
    Lm = np.max(logpostfull(theta))
    fitinfo['thetarnd'] = theta
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
    if theta.ndim == 1 and fitinfo['theta'].shape[1] > 1.5:
        theta = theta.reshape((1, theta.shape[0]))
    
    if 'yvar' in fitinfo.keys():
        obsvar = fitinfo['yvar']
    else:
        raise ValueError('Must provide yvar in this software.')
    
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
# def approxhess(fitinfo, emu, theta, y, x, args):
#     r"""
#     This is a optional docstring for an internal function.
#     """
    
#     if 'yvar' in fitinfo.keys():
#         obsvar = np.squeeze(fitinfo['yvar'])
#     else:
#         raise ValueError('Must provide yvar in this software.')
#     #y = np.squeeze(y)
#     p = theta.shape[1]
    
#     thetastd = np.std(theta,0)
#     L0 = loglik(fitinfo, emu, theta, y, x, args)
#     theta0 = theta[np.argmin(L0)]
#     emumean = emu.predict(x, theta0).mean()
#     emucovhalf = emu.predict(x, theta0).covxhalf()
    
#     fdir = np.zeros((emumean.shape[0], theta.shape[1]))
#     obsvaradj = obsvar + 1 * np.sum(emucovhalf ** 2,0)
#     for p in range(0,theta.shape[1]):
#         thetadj = copy.copy(theta0)
#         thetadj[p] += 10 ** (-4) * thetastd[p]
#         emumeanadj1 = emu.predict(x, thetadj).mean()
#         thetadj[p] -= 2*10 **(-4) * thetastd[p]
#         emumeanadj2 = emu.predict(x, thetadj).mean()
#         fdir[:,p] = np.squeeze((emumeanadj1-emumeanadj2) * (10 ** (4)) / 2)
#     fdiradj = (fdir.T / np.sqrt(obsvaradj)).T
#     D, W, _ = np.linalg.svd(fdir.T)
#     W = W ** 2
#     print(W)
#     Amat = np.diag(np.sqrt(10**(-12) + W[:3])) @ D[:,:3].T
#     bvec = - Amat @ theta0
#     Cmat = D[:,:3] @ np.diag(1/np.sqrt(10**(-12) +W[:3]))
#     dvec = Cmat @ Amat @ theta0
    
#     return Amat, bvec, Cmat, dvec


def loglik(fitinfo, emu, theta, y, x, args):
    r"""
    This is a optional docstring for an internal function.
    """
    
    
    if 'yvar' in fitinfo.keys():
        obsvar = np.squeeze(fitinfo['yvar'])
    else:
        raise ValueError('Must provide yvar in this software.')
    #y = np.squeeze(y)
    emupredict = emu.predict(x, theta)
    emumean = emupredict.mean()
    emucovhalf = emupredict.covxhalf()
    loglik = np.zeros(emumean.shape[1])
    for k in range(0, emumean.shape[1]):
        m0 = emumean[:,k]
        S0 = np.squeeze(emucovhalf[:,k,:])
        obsvaradj = np.squeeze(obsvar + 0.0001*np.squeeze( np.sum(S0 ** 2,0)))
        stndresid = (np.squeeze(y) - m0) / np.sqrt(obsvaradj)
        term1 = np.sum(stndresid ** 2)
        #loglik[k] = - 0.5 * term1
        J = 1/ np.sqrt(obsvaradj) * S0
        if J.ndim < 1.5:
            J = J[:,None].T
        J2 =  J @ stndresid
        if J2.shape[0] == 1:
            term2 =  J2** 2 / (1 + np.sum(J **2))
            term3 = np.log(1 + np.sum(J **2))
        else:
            W, V = np.linalg.eigh(np.eye(J.shape[0]) + J @ J.T)
            J3 = np.squeeze(V).T @ np.squeeze(J2)
            term2 = np.sum((J3 ** 2) / W)
            term3 = np.sum(np.log(W))
        residsq = term1 - term2
        loglik[k] += -0.5 * (m0.shape[0]+0.1) * np.log(residsq+0.1) - 0.5 * term3
    
    return loglik