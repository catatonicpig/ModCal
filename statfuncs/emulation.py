# -*- coding: utf-8 -*-
"""Header here."""




def semu_logpost(semumodel, theta, options=None):
    """
    Return posterior of function evaluation at the new parameters.

    Parameters
    ----------
    semumodel : dict
        Simple emulator model of (f(x_i,\theta) - y_i) / \sigma_i
    thetaorig : array of float
        Some matrix of parameters where function evaluations as starting points.
    Returns
    -------
    post: vector of unnormlaized log posterior
    """
    if theta.ndim == 1:
        theta = theta.reshape((1,theta.shape[0]))
    loglik = emulation_smart_loglik(semumodel, theta)
    #logprior = -4 * np.sum((theta-0.5) ** 2,1)
    
    logpost = loglik# + logprior
    
    #sumofExp = post - np.log(np.sum(np.exp(post - np.max(post))))
    
    return logpost