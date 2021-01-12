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
    
    thetaprior = fitinfo['thetaprior']
    if 'clf_method' in args.keys():
        clf_method = args['clf_method']
    else:
        clf_method = None

    def logpostfull(theta):
        #print(theta)
        if theta.ndim < 1.5:
            theta = theta.reshape((1, len(theta)))
        logpost = thetaprior.lpdf(theta)
        if logpost.ndim > 0.5 and logpost.shape[0] > 1.5:
            inds = np.where(np.isfinite(logpost))[0]
            logpost[inds] += loglik(fitinfo, emu, theta[inds], y, x, args)
        else:
            if np.isfinite(logpost):
                if clf_method is None:
                    logpost += loglik(fitinfo, emu, theta, y, x, args)
                else:
                    logpost += loglik(fitinfo, emu, theta, y, x, args) 
                    
                    ml_probability = clf_method.predict_proba(theta)[0][1]
                    ml_logprobability = np.log(ml_probability) if ml_probability > 0 else np.inf
                    #print(clf_method.predict_proba(theta)[0][1])
                    if np.isfinite(ml_logprobability):
                        logpost += ml_logprobability
                    else:
                        logpost = np.inf

        return logpost
    
    # Obtain an initial theta value for plumlee (i think we should define this within sampler)    
    if 'sampler' in args.keys():
        if args['sampler'] == 'plumlee':
            thetastart = thetaprior.rnd(1000)
    else:
        thetastart = None
    ### ### ### ### ### ### 
        
    # Call the sampler
    theta = postsampler(thetastart, logpostfull, options = args)

    fitinfo['thetarnd'] = theta
    fitinfo['y'] = y
    fitinfo['x'] = x
    return

def loglik(fitinfo, emulator, theta, y, x, args):
    
    '''
    Parameters
    ----------
    fitinfo : TYPE
        DESCRIPTION.
    emulator : TYPE
        DESCRIPTION.
    theta : TYPE
        DESCRIPTION.
    y : TYPE
        DESCRIPTION.
    x : TYPE
        DESCRIPTION.
    args : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    # if theta.ndim == 1:
    #     theta = theta.reshape((1, theta.shape[0]))
    
    if 'yvar' in fitinfo.keys():
        obsvar = fitinfo['yvar']
    else:
        raise ValueError('Must provide obsvar at this moment.')
    
    # Obtain emulator results
    emupredict = emulator.predict(x, theta)
    emumean = emupredict.mean()
    #emucov = emupredict.var()
    
    try:
        emucov = emupredict.covx()
        is_cov = True
    except ValueError:
        emucov = emupredict.var()
        is_cov = False
        
    p = emumean.shape[1]
    n = emumean.shape[0]
    y = y.reshape((n, 1))
    
    loglikelihood = np.zeros(p)

    for k in range(0, p):
        m0 = emumean[:, k].reshape((n, 1))
    
        # Compute the covariance matrix
        if is_cov == True:
            s0 = emucov[:, k, :].reshape((n, n))
            CovMat = s0 + np.diag(np.squeeze(obsvar))
        else:
            # if n == 1:
            #     s0 = emucov[:, k].reshape((n, 1))
            #     CovMat = np.diag(s0) + np.diag(obsvar)
            # else:     
            s0 = emucov[:, k].reshape((n, 1))
            CovMat = np.diag(np.squeeze(s0)) + np.diag(np.squeeze(obsvar))
            
        # Get the decomposition of covariance matrix
        CovMatEigS, CovMatEigW = np.linalg.eigh(CovMat)
        
        # Calculate residuals
        resid = m0 - y
        
        # 
        CovMatEigInv = CovMatEigW @ np.diag(1/CovMatEigS) @ CovMatEigW.T
        
        #
        #print(theta)
        #print(-0.5 * (resid.T @ CovMat @ resid))
        #print(float(-0.5 * resid.T @ CovMatEigInv @ resid - 0.5 * np.sum(np.log(CovMatEigS))))
        #loglikelihood[k] = -0.5 * (resid.T @ CovMat @ resid)
        
        loglikelihood[k] = float(-0.5 * resid.T @ CovMatEigInv @ resid - 0.5 * np.sum(np.log(CovMatEigS)))
        #print(loglikelihood[k])
    if p == 1:
        return float(loglikelihood)
    else:
        return loglikelihood

