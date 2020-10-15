# -*- coding: utf-8 -*-
"""Header here."""
import numpy as np

def loglik(emulator, theta, phi, y, xind, modelnum, options):
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
    
    
    
    predinfo = emulator.predict(theta)
    loglik = np.zeros(predinfo['mean'].shape[0])
    for k in range(0, predinfo['mean'].shape[0]):
        Sinv = np.diag(1/(obsvar))
        ldetS = np.sum(np.log(obsvar))
        if 'corrf' in options.keys() and phi is not None:
            obsvarm = obsvar
            Snew = np.diag(obsvarm) + phi[k,modelnum]*options['corrf'](emulator.x, modelnum)['C'][xind,:][:,xind]
            #print(Snew)
            Sinvu = np.linalg.inv(Snew)
            ldetSu = np.linalg.slogdet(Snew)[1]
        elif phi is not None:
            obsvarm = obsvar + phi[k,modelnum]
            Sinvu = np.diag(1/(obsvarm))
            ldetSu = np.sum(np.log(obsvarm))
        else:
            Sinvu = 1*Sinv
            ldetSu = 1*ldetS 
        mu = np.squeeze(predinfo['mean'][k,:])[xind]
        U = (predinfo['covdecomp'][k,:,:])[:,xind]
        Am = Sinvu @ U.T
        W, V = np.linalg.eigh(np.eye(U.shape[0]) + U @ Am)
        Amp = (np.squeeze(y)-mu).T @ Am @ V.T
        loglik[k] = -0.5*(np.squeeze(y)-mu).T @ (Sinvu @ (np.squeeze(y)-mu))
        loglik[k] += -0.5*ldetSu
        loglik[k] += 0.5*(Amp @ (Amp * (1/W)).T)
        loglik[k] += -0.5*np.sum(np.log(W))
    #print(loglik)
    return loglik

def predict(xindnew, emulator, theta, phi, y, xind, modelnum,  options):
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
    
    predinfo = emulator.predict(theta)
    preddict = {}
    
            
    preddict['meanfull'] = predinfo['mean']
    preddict['varfull'] = predinfo['var'] 
    
    Sinv = np.diag(1/obsvar)
    for k in range(0, predinfo['mean'].shape[0]):
        Sinv = np.diag(1/(obsvar))
        ldetS = np.sum(np.log(obsvar))
        if 'corrf' in options.keys() and phi is not None:
            obsvarm = obsvar 
            Snew = np.diag(obsvarm) + phi[k,modelnum]*options['corrf'](emulator.x, modelnum)['C'][xind,:][:,xind]
            Sinvu = np.linalg.inv(Snew)
            C =  phi[k,modelnum]*options['corrf'](emulator.x, modelnum)['C']
            mu = np.squeeze(predinfo['mean'][k,:])[xind]
            preddict['meanfull'][k,:] += C[xindnew,:][:,xind] @ Sinvu @ (np.squeeze(y) - mu) 
            preddict['varfull'][k,:] += np.diag(C[xindnew,:][:,xindnew] -
                                                C[xindnew,:][:,xind] @ Sinvu @\
                                                    C[xind,:][:,xindnew])
                
    preddict['mean'] = np.mean(preddict['meanfull'],0)
    varterm1 = np.var(predinfo['mean'],0)
    preddict['var'] = np.mean(preddict['varfull'],0) + varterm1
    return preddict