# -*- coding: utf-8 -*-
"""Header here."""
import numpy as np
from base.utilitiesmethods.plumleeMCMC import plumleepostsampler
from base.utilitiesmethods.plumleeMCMC_wgrad import plumleepostsampler_wgrad

def postsampler(thetastart, logpostfunc, options=None):
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

    if options is None:
        numsamp = 500
        tarESS = np.max((200, 10 * thetastart.shape[1]))
        def postsamplefunc(thetastart, logpostfunc):
            return plumleepostsampler_wgrad(thetastart, logpostfunc, numsamp, tarESS)
    else:
        print('options are not yet avaiable for this function')
    return postsamplefunc(thetastart, logpostfunc)