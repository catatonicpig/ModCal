# -*- coding: utf-8 -*-
"""Header here."""
import numpy as np

def loglik(emulator, theta, phi, y, xind, options):
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
    
    
    if 'obsvar' in options.keys():
        obsvar = options['obsvar']
    else:
        raise ValueError('Must provide obsvar at this moment.')
    
    Sinv = np.diag(1/obsvar)
    ldetS = np.sum(np.log(obsvar))
    
    if 'covhalf' in options.keys():
        Bm = Sinv @ ((options['covhalf'])[:,xind]).T
        W2, V2 = np.linalg.eigh(options['covhalf'][:,xind] @ Bm)
    
    predinfo = emulator.predict(theta)
    loglik = np.zeros(predinfo['mean'].shape[0])
    for k in range(0, predinfo['mean'].shape[0]):
        if phi is not None and 'covhalf' in options.keys():
            W3 = 1/np.abs(phi[k]) + W2
            T1 = (np.diag(1/np.sqrt(np.abs(W3))) @ (V2.T)) @ Bm.T
            Sinvu = Sinv - T1.T @ T1
            ldetSu = ldetS + np.sum(np.log(1+np.abs(phi[k])*W2))
        else:
            Sinvu = Sinv
            ldetSu = ldetS
            
        mu = np.squeeze(predinfo['mean'][k,:])[xind]
        U = (predinfo['covdecomp'][k,:,:])[:,xind]
        Am = Sinvu @ U.T
        W, V = np.linalg.eigh(np.eye(U.shape[0]) + U @ Am)
        Amp = (np.squeeze(y)-mu).T @ Am @ V.T
        loglik[k] = -0.5*(np.squeeze(y)-mu).T @ (Sinvu @ (np.squeeze(y)-mu))
        loglik[k] += -0.5*ldetSu
        loglik[k] += 0.5*(Amp @ (Amp * (1/W)).T)
        loglik[k] += -0.5*np.sum(np.log(W))
    return loglik

def predict(xindnew, emulator, theta, phi, y, xind, options):
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
        
    if 'obsvar' in options.keys():
        obsvar = options['obsvar']
    else:
        raise ValueError('Must provide obsvar at this moment.')
        
    Sinv = np.diag(1/obsvar)
    ldetS = np.sum(np.log(obsvar))
    
    predinfo = emulator.predict(theta)
    preddict = {}
    preddict['mean'] = np.mean(predinfo['mean'],0)
    varterm1 = np.var(predinfo['mean'],0)
    preddict['var'] = np.mean(predinfo['var'],0) + varterm1
    return preddict