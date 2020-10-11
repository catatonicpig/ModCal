# -*- coding: utf-8 -*-
"""Header here."""
import numpy as np


class emulation(object):
    """Emulator."""

    def __init__(self):
        self._means = None
        self._stds = None

    def build(self, codepointer):
        if self._means is None:  # during training only
            self._means = np.mean(data, axis=0)

        if self._stds is None:  # during training only
            self._stds = np.std(data, axis=0)
            if not self._stds.all():
                raise ValueError('At least one column has standard deviation of 0.')
        
        return (data - self._means) / self._stds

    def preprocess(self, data):
        if self._means is None:  # during training only
            self._means = np.mean(data, axis=0)

        if self._stds is None:  # during training only
            self._stds = np.std(data, axis=0)
            if not self._stds.all():
                raise ValueError('At least one column has standard deviation of 0.')
        
        return (data - self._means) / self._stds


def loglik(emumodel, theta, y=None, S=None):
    """
    Return posterior of function evaluation at the new parameters.

    Parameters
    ----------
    emumodel : Pred
        A fitted emulator model defined as an emulation class.
    theta : array of float
        Some matrix of parameters where function evaluations as starting points.
    y : Observations
        A vector of the same length as x with observations. 'None' is equivlent to a vector of
        zeros.
    S : Observation Covariance Matrix
        A matrix of the same length as x with observations. 'None' is equivlent to the
        identity matrix.

    Returns
    -------
    post: vector of unnormlaized log posterior
    """
    if theta.ndim == 1:
        theta = theta.reshape((1,theta.shape[0]))
    
    loglik = emulation_smart_loglik(semumodel, theta)
    
    logpost = loglik
    
    return logpost