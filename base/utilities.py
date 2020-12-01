# -*- coding: utf-8 -*-
"""Header here."""
import numpy as np
from base.utilitiesmethods.plumleeMCMC import plumleepostsampler
from base.utilitiesmethods.plumleeMCMC_wgrad import plumleepostsampler_wgrad
from base.utilitiesmethods.metropolis_hasting import metropolis_hasting

def postsampler(thetastart, logpostfunc, options= {}):
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
    if 'method' in options.keys():
        method = options['method']
    else:
        method = 'default'
    if 'numsamp' in options.keys():
        numsamp = options['numsamp']
    else:
        numsamp = 2000
    
    def postsamplefunc(thetastart, logpostfunc):
        if method is 'plumlee':
            tarESS = np.max((150, 10 * thetastart.shape[1]))
            return plumleepostsampler_wgrad(thetastart, logpostfunc, numsamp, tarESS)
        else:
            return metropolis_hasting(thetastart, logpostfunc)
    return postsamplefunc(thetastart, logpostfunc)