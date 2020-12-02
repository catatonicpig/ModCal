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
    clf_method = args['clf_method']
    

    def logpostfull(theta):

        logpost = thetaprior.lpdf(theta) 
        print('theta', theta)
        print('prior', logpost)
        
        if np.isfinite(logpost):
            if clf_method is None:
                logpost += loglik(fitinfo, emu, theta, y, x, args)
            else:
                logpost += loglik(fitinfo, emu, theta, y, x, args) 
                print('post', loglik(fitinfo, emu, theta, y, x, args) )
                
                # here construct feature later
                ml_probability = clf_method.predict_proba(theta)[0][1]
                ml_logprobability = np.log(ml_probability) if ml_probability > 0 else float('inf')
                print(clf_method.predict_proba(theta)[0][1])
                
                if np.isfinite(ml_logprobability):
                    logpost += ml_logprobability
                else:
                    logpost = float('inf')
        else:
            logpost = float('inf')
                
        return logpost
    
    # Obtain an initial theta value 
    #thetastart = np.array([0.4]).reshape(1, 1)
    theta_initial = np.mean(thetaprior.rnd(1000), axis = 0)
    thetastart = theta_initial.reshape(1, len(theta_initial))
    
    # Call the sampler
    theta = postsampler(thetastart, logpostfull)

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
    if theta.ndim == 1:
        theta = theta.reshape((1, theta.shape[0]))
    
    if 'yvar' in fitinfo.keys():
        obsvar = fitinfo['yvar']
    else:
        raise ValueError('Must provide obsvar at this moment.')
    
    # Obtain emulator results
    emupredict = emulator.predict(x, theta)
    emumean = emupredict.mean()
    emucov = emupredict.var()

    # Compute the covariance matrix
    CovMat = np.diag(np.squeeze(emucov)) + np.diag(np.squeeze(obsvar))
    
    # Get the decomposition of covariance matrix
    CovMatEigS, CovMatEigW = np.linalg.eigh(CovMat)
    
    # Calculate residuals
    resid = emumean - np.reshape(y, (len(y), 1))
    
    # 
    CovMatEigInv = CovMatEigW @ np.diag(1/CovMatEigS) @ CovMatEigW.T
    
    #
    loglikelihood = float(-1/2 * resid.T @ CovMat @ resid)
    #loglikelihood = float(-1/2 * resid.T @ CovMatEigInv @ resid - 1/2 * np.sum(np.log(CovMatEigS)))

    return loglikelihood

