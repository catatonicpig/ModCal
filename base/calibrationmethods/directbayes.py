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
    try: 
        theta = thetaprior.rnd(10)
        emupredict = emu.predict(x, theta, args={'return_grad': True})
        emupredict.mean_gradtheta()
        emureturn_grad = True
    except:
        emureturn_grad = False
    if emureturn_grad and 'lpdf_grad' not in dir(thetaprior):
        def lpdf_grad(theta):
            f_base = thetaprior.lpdf(theta)
            if theta.ndim > 1.5:
                inds = np.where(np.isfinite(f_base))[0]
                grad = np.zeros((theta.shape[0], theta.shape[1]))
                for k in range(0,theta.shape[1]):
                    thetaprop = copy.copy(theta)
                    thetaprop[:,k] += 10 ** (-6)
                    f_base2 = thetaprior.lpdf(thetaprop[inds,:])
                    grad[inds,k] = 10 ** (6) * (f_base2 - f_base[inds])
            else:
                if np.isfinite(f_base):
                    grad = np.zeros(theta.shape[0])
                    for k in range(0,theta.shape[0]):
                        thetaprop = copy.copy(theta)
                        thetaprop[k] += 10 ** (-6)
                        f_base2 = thetaprior.lpdf(thetaprop)
                        grad[k] = 10 ** (6) * (f_base2 - f_base)
                else:
                    grad = 0
            return grad
        thetaprior.lpdf_grad = lpdf_grad
    
    def logpostfull_wgrad(theta, return_grad = True):
        logpost = thetaprior.lpdf(theta)
        if emureturn_grad and return_grad:
            dlogpost =  thetaprior.lpdf_grad(theta)
        if logpost.ndim > 0.5 and logpost.shape[0] > 1.5:
            inds = np.where(np.isfinite(logpost))[0]
            if emureturn_grad and return_grad:
                loglikinds, dloglikinds = loglik_grad(fitinfo, emu, theta[inds], y, x, args)
                logpost[inds] += loglikinds
                dlogpost[inds] += dloglikinds
            else:
                logpost[inds] += loglik(fitinfo, emu, theta[inds], y, x, args)
        else:
            if np.isfinite(logpost):
                if emureturn_grad and return_grad:
                    loglikinds, dloglikinds = loglik_grad(fitinfo, emu, theta, y, x, args)
                    logpost += loglikinds
                    dlogpost +=np.squeeze(dloglikinds)
                else:
                    logpost += loglik(fitinfo, emu, theta, y, x, args)
        if emureturn_grad and return_grad:
            return logpost, dlogpost
        else:
            return logpost
    
    # thetatest = thetaprior.rnd(2)
    # logpost, dlogpost = logpostfull_wgrad(thetatest, return_grad = True)
    # for k in range(0,thetatest.shape[1]):
    #     thetaadj = copy.copy(thetatest)
    #     thetaadj[:,k] += 10 ** (-6)
    #     logpostadj, dlogpostadj = logpostfull_wgrad(thetaadj, return_grad = True)
    #     print((logpostadj - logpost) * 10 ** (6))
    #     print(dlogpost[:,k])
    # asdada
    theta = thetaprior.rnd(1000)
    if 'thetarnd' in fitinfo:
        theta = np.vstack((fitinfo['thetarnd'],theta))
    if '_emulator__theta' in dir(emu):
        theta = np.vstack((theta,copy.copy(emu._emulator__theta)))
    theta = postsampler(theta, logpostfull_wgrad, options={'sampler': 'plumlee'})
    ladj = logpostfull_wgrad(theta, return_grad = False)
    mladj = np.max(ladj)
    fitinfo['lpdfapproxnorm'] = np.log(np.mean(np.exp(ladj - mladj))) + mladj
    fitinfo['thetarnd'] = theta
    fitinfo['y'] = y
    fitinfo['x'] = x
    fitinfo['emu'] = emu
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
    emupredict = emu.predict(x, theta)
    predinfo['rnd'] = copy.deepcopy(emupredict()).T
    predinfo['modelrnd'] = copy.deepcopy(emupredict()).T
    
    emupredict = emu.predict(x, theta)
    emumean = emupredict.mean()
    emucovxhalf = emupredict.covxhalf()
    varfull = np.sum(np.square(emucovxhalf), axis = 2)
    for k in range(0, theta.shape[0]):
        re = emucovxhalf[:,k,:] @ sps.norm.rvs(0,1,size=(emucovxhalf.shape[2]))
        predinfo['rnd'][k,:] += re
        predinfo['modelrnd'][k,:] += re
        
    predinfo['mean'] = np.mean(emumean, 0)
    varterm1 = np.var(emumean, 0)
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


def thetalpdf(fitinfo, theta, args=None):
    """
    Return logposterior draws from the predictive distribution of theta.  Not required.
    """
    emu = fitinfo['emu']
    y = fitinfo['y']
    x = fitinfo['x']
    thetaprior = fitinfo['thetaprior']
    logpost = thetaprior.lpdf(theta)
    if logpost.ndim > 0.5 and logpost.shape[0] > 1.5:
        inds = np.where(np.isfinite(logpost))[0]
        logpost[inds] += loglik(fitinfo, emu, theta[inds], y, x, args)
    elif np.isfinite(logpost):
        logpost += loglik(fitinfo, emu, theta, y, x, args)
    return (logpost-fitinfo['lpdfapproxnorm'])

"""
This [endfunctionsflag] is automatically filled by docinfo.py when running updatedocs.py
##############################################################################
##############################################################################
####################### THIS ENDS THE OPTIONAL PORTION #######################
######### USE SPACE BELOW FOR ANY SUPPORTING FUNCTIONS YOU DESIRE ############
##############################################################################
##############################################################################
"""


def loglik(fitinfo, emu, theta, y, x, args):
    r"""
    This is a optional docstring for an internal function.
    """
    
    
    if 'yvar' in fitinfo.keys():
        obsvar = 1*np.squeeze(fitinfo['yvar'])
    else:
        raise ValueError('Must provide yvar in this software.')
    emupredict = emu.predict(x, theta)
    emumean = emupredict.mean()
    emuvar = emupredict.var()
    emucovxhalf = emupredict.covxhalf()
    loglik = np.zeros(emumean.shape[1])
    if np.any(np.abs(emuvar/(10 ** (-4) + (1+10 ** (-4))*np.sum(np.square(emucovxhalf),2))) > 1):
        emuoldpredict = emu.predict(x)
        emuoldvar = emuoldpredict.var()
        emuoldcxh = emuoldpredict.covxhalf()
        obsvar += np.mean(np.abs(emuoldvar- np.sum(np.square(emuoldcxh),2)),1)
    for k in range(0, emumean.shape[1]):
        m0 = emumean[:,k]
        
        #### MATT: This part does not work with 1d theta!!!
        
        S0 = np.squeeze(emucovxhalf[:,k,:]) # np.reshape(emucovxhalf[:,k,:], (emumean.shape[0], theta.shape[1])) # np.squeeze(emucovxhalf[:,k,:])
        stndresid = (np.squeeze(y) - m0) / np.sqrt(obsvar)
        term1 = np.sum(stndresid ** 2)
        J = (S0.T / np.sqrt(obsvar)).T
        if J.ndim < 1.5:
            J = J[:,None]
            stndresid = stndresid[:,None]
        J2 =  J.T @ stndresid
        W, V = np.linalg.eigh(np.eye(J.shape[1]) + J.T @ J)
        if W.shape[0] > 1:
            J3 = V @ np.diag(1/W) @ V.T @ J2 # np.squeeze(V) @ np.diag(1/W) @ np.squeeze(V).T @ np.squeeze(J2)
        else:
            J3 = ((V ** 2)/ W) * J2 # np.squeeze(V) @ np.diag(1/W) @ np.squeeze(V).T @ np.squeeze(J2)
        term2 = np.sum(J3 * J2)
        residsq = term1 - term2
        loglik[k] = -0.5 * residsq - 0.5 * np.sum(np.log(W))
     
    #Replace line 307 with 303--306  
    if emumean.shape[1] == 1:
        return float(loglik)
    else:
        return loglik
    #return loglik


def loglik_gradapprox(fitinfo, emu, theta, y, x, args):
    r"""
    This is a optional docstring for an internal function.
    """
    
    
    if 'yvar' in fitinfo.keys():
        obsvar = 1*np.squeeze(fitinfo['yvar'])
    else:
        raise ValueError('Must provide yvar in this software.')
    #y = np.squeeze(y)
    emupredict = emu.predict(x, theta, args={'return_grad': True})
    emumean = emupredict.mean()
    emuvar = emupredict.var()
    emucovxhalf = emupredict.covxhalf()
    emumean_grad = emupredict.mean_gradtheta()
    loglik = np.zeros(emumean.shape[1])
    dloglik = np.zeros((emumean.shape[1],emu._info['theta'].shape[1]))
    dterm1 = np.zeros(emu._info['theta'].shape[1])
    dterm2 = np.zeros(emu._info['theta'].shape[1])
    if np.any(np.abs(emuvar/(10 ** (-4) + (1+10 ** (-4))*np.sum(np.square(emucovxhalf),2))) > 1):
        emuoldpredict = emu.predict(x)
        emuoldvar = emuoldpredict.var()
        emuoldcxh = emuoldpredict.covxhalf()
        obsvar += np.mean(np.abs(emuoldvar- np.sum(np.square(emuoldcxh),2)),1)
    if '_info' in dir(emu) and 'extravar' in emu._info:
        obsvar = obsvar + np.mean(emu._info['extravar'])
    for k in range(0, emumean.shape[1]):
        m0 = emumean[:,k]
        dm0 = np.squeeze(emumean_grad[:, k, :])
        S0 = np.squeeze(emucovxhalf[:,k,:])
        
        stndresid = (np.squeeze(y) - m0) / np.sqrt(obsvar)
        term1 = np.sum(stndresid ** 2)
        stndresid_grad = - (dm0.T / np.sqrt(obsvar)).T
        dterm1 = 2 * np.sum(stndresid * stndresid_grad.T, 1)
        J = (S0.T / np.sqrt(obsvar)).T
        if J.ndim < 1.5:
            J = J[:,None].T
        J2 =  J.T @ stndresid
        W, V = np.linalg.eigh(np.eye(J.shape[1]) + J.T @ J)
        J3 = np.squeeze(V) @ np.diag(1/W) @ np.squeeze(V).T @ np.squeeze(J2)
        term2 = np.sum(J3 * J2)
        for l in range(0,stndresid_grad.shape[1]):
            dJ2 = J.T @ np.squeeze(stndresid_grad[:,l])
            dterm2[l] = 2*np.sum(dJ2 * J3)
        term3 = np.sum(np.log(W))
        residsq = term1 - term2
        loglik[k] = -0.5 * residsq - 0.5 * term3
        dloglik[k, :] = -0.5 * (dterm1-dterm2)
    return loglik, dloglik


def loglik_grad(fitinfo, emu, theta, y, x, args):
    r"""
    This is a optional docstring for an internal function.
    """
    
    
    if 'yvar' in fitinfo.keys():
        obsvar = 1*np.squeeze(fitinfo['yvar'])
    else:
        raise ValueError('Must provide yvar in this software.')
    #y = np.squeeze(y)
    emupredict = emu.predict(x, theta, args={'return_grad': True})
    emumean = emupredict.mean()
    emuvar = emupredict.var()
    emucovxhalf = emupredict.covxhalf()
    emumean_grad = emupredict.mean_gradtheta()
    emucovxhalf_grad  = emupredict.covxhalf_gradtheta()
    loglik = np.zeros(emumean.shape[1])
    dloglik = np.zeros((emumean.shape[1],emu._info['theta'].shape[1]))
    dterm1 = np.zeros(emu._info['theta'].shape[1])
    dterm2 = np.zeros(emu._info['theta'].shape[1])
    dterm3 = np.zeros(emu._info['theta'].shape[1])
    
    #adj for any unmodled variance:    
    if np.any(np.abs(emuvar/(10 ** (-4) + (1+10 ** (-4))*np.sum(np.square(emucovxhalf),2))) > 1):
        emuoldpredict = emu.predict(x)
        emuoldvar = emuoldpredict.var()
        emuoldcxh = emuoldpredict.covxhalf()
        obsvar += np.mean(np.abs(emuoldvar- np.sum(np.square(emuoldcxh),2)),1)
    for k in range(0, emumean.shape[1]):
        m0 = emumean[:,k]
        dm0 = np.squeeze(emumean_grad[:, k, :])
        S0 = np.squeeze(emucovxhalf[:,k,:])
        
        stndresid = (np.squeeze(y) - m0) / np.sqrt(obsvar)
        term1 = np.sum(stndresid ** 2)
        stndresid_grad = - (dm0.T / np.sqrt(obsvar)).T
        dterm1 = 2 * np.sum(stndresid * stndresid_grad.T, 1)
        J = (S0.T / np.sqrt(obsvar)).T
        if J.ndim < 1.5:
            J = J[:,None]
        J2 =  J.T @ stndresid
        W, V = np.linalg.eigh(np.eye(J.shape[1]) + J.T @ J)
        J3 = V @ np.diag(1/W) @ V.T @ J2
        term2 = np.sum(J3 * J2)
        for l in range(0,stndresid_grad.shape[1]):
            dJ = (np.squeeze(emucovxhalf_grad[:,k,:,l]).T * (1/ np.sqrt(obsvar)))
            dJ2 = J.T @ stndresid_grad[:,l] + dJ @ stndresid
            exmat = dJ @ J
            exmat = (exmat + exmat.T)
            dJ3 = V @ np.diag(1/W) @ V.T @ (dJ2 - exmat @ J3)
            dterm2[l] = np.sum(J2 * dJ3) + np.sum(dJ2 * J3)
        if W.shape[0] > 1:
            V2 =  1/obsvar * (((V * (1/W)) @ V.T) @ S0.T)
        else:
            V2 =  (1/obsvar * ((V**2/W) * S0))
        for l in range(0,stndresid_grad.shape[1]):
            dterm3[l] = 2 * np.sum(V2 * np.squeeze(emucovxhalf_grad[:,k,:,l]).T)
        term3 = np.sum(np.log(W))
        residsq = term1 - term2
        loglik[k] = - 0.5 * term3-0.5 * residsq #
        dloglik[k, :] = - 0.5 * dterm3 -0.5 * (dterm1-dterm2) #
    return loglik, dloglik