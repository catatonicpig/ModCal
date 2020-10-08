# -*- coding: utf-8 -*-
"""Header here."""
import numpy as np
from statfuncs.samplingsubfuncs.plumleesamp import plumleepostsampler




def semu_pred(semumodel, theta, options=None):
    """
    Return predicted values of function evaluation at the new parameters.

    Parameters
    ----------
    semumodel : dict
        Simple emulator model, as described above.
    thetanew : array of float
        New parameter where function evaluations should be sampled.
    options : dict, optional
        Prediction options, as a dictionary with entries

        * `'returnvariance'`: boolean if predictive variance is returned.
        The default is False.

        * `'returnvariance'`: boolean if predictive covariance matrix is 
        returned. The default is False.
        
    Returns
    -------
    fhat : array of float
        Predicted values of recovered function evaluation, of size
        (`Nnew`, `n`) where `Nnew` is the number of new
        parameters, `n` is the number of input.
    predvar_inclremoval : array of float
        Predictive variances/covariances for the function evaluations.
    """
    if theta.ndim == 1:
        theta = theta.reshape((1,theta.shape[0]))
    
    predmean, predvar = emulation_smart_prediction(semumodel, theta, options)
    return predmean, predvar


def postsampler(thetastart, logpriorfunc, loglikfunc, options=None):
    """
    Return draws from the posterior.

    Parameters
    ----------
    thetastart : array of float
        Some matrix of parameters where function evaluations as starting points.
    logpriorfunc : function
        A function call describing the log of the prior distribution
    loglikfunc : function
        A function call describing the log of the likelihood function
    Returns
    -------
    theta : matrix of sampled paramter values
    """
    def logpostfunc(thetaval):
        return logpriorfunc(thetaval) + loglikfunc(thetaval)

    if options is None:
        numsamp = 5000
        tarESS = np.max((100, 10 * thetastart.shape[1]))

        def postsamplefunc(thetastart, logpostfunc):
            return plumleepostsampler(thetastart, logpostfunc, numsamp, tarESS)
    else:
        print('options are not yet avaiable for this function')
    return postsamplefunc(thetastart, logpostfunc)
