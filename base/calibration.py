# -*- coding: utf-8 -*-
"""Header here."""
import numpy as np

def loglik(emulator, theta, y=None, Sinv=None):
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
    Sinv : Observation Precision Matrix
        A matrix of the same length as \"emulator.x\" with observations. 'None' is equivlent to the
        identity matrix.

    Returns
    -------
    post: vector of unnormlaized log posterior
    """
    if theta.ndim == 1:
        theta = theta.reshape((1,theta.shape[0]))
        
        
        
    predinfo = emulator.predict(theta)
    loglik = np.zeros(predinfo['mean'].shape[0])
    for k in range(0, predinfo['mean'].shape[0]):
        mu = predinfo['mean'][k,:]
        U = predinfo['covdecomp'][k,:,:]
        Am = Sinv @ U.T
        W, V = np.linalg.eigh(np.eye(U.shape[0]) + U @ Am)
        Amp = y.T @ Am @ V
        loglik[k] += -0.5*(Amp @ (Amp * (1/W)).T)
        loglik[k] = np.sum(np.log(W))
    return loglik


def loglik(emulator, theta, y=None, Sinv=None, ldetS=None):
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
    Sinv : Observation Precision Matrix
        A matrix of the same length as \"emulator.x\" with observations. 'None' is equivlent to the
        identity matrix.

    Returns
    -------
    post: vector of unnormlaized log posterior
    """
    if theta.ndim == 1:
        theta = theta.reshape((1,theta.shape[0]))
        
        
        
    predinfo = emulator.predict(theta)
    loglik = np.zeros(predinfo['mean'].shape[0])
    for k in range(0, predinfo['mean'].shape[0]):
        mu = np.squeeze(predinfo['mean'][k,:])
        U = predinfo['covdecomp'][k,:,:]
        Am = Sinv @ U.T
        W, V = np.linalg.eigh(np.eye(U.shape[0]) + U @ Am)
        Amp = (np.squeeze(y)-mu).T @ Am @ V.T
        loglik[k] += -0.5*(np.squeeze(y)-mu).T @ (Sinv @ (np.squeeze(y)-mu))
        loglik[k] += -0.5*ldetS
        loglik[k] += 0.5*(Amp @ (Amp * (1/W)).T)
        loglik[k] += -0.5*np.sum(np.log(W))
    return loglik