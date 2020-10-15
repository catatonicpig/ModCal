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
    
    if type(emulator) is tuple:
        predinfo = [dict() for x in range(len(emulator))]
        for k in range(0,len(emulator)):
            predinfo[k] = emulator[k].predict(theta)
    else:
        predinfo = emulator.predict(theta)
    
    loglikr = np.zeros(theta.shape[0])
    for k in range(0, theta.shape[0]):
        if type(emulator) is tuple:
            covmats = [np.array(0) for x in range(len(emulator))]
            covmatsinv = [np.array(0) for x in range(len(emulator))]
            mus = [np.array(0) for x in range(len(emulator))]
            resid= np.zeros(len(emulator)*xind.shape[0])
            totInv = np.zeros((xind.shape[0],xind.shape[0]))
            term1 = np.zeros((xind.shape[0],xind.shape[0]))
            term2 = np.zeros((xind.shape[0],xind.shape[0]))
            term3 = np.zeros((xind.shape[0]))
            for l in range(0,len(emulator)):
                mus[l] = np.squeeze(y) - predinfo[l]['mean'][k,xind]
                A1 = np.squeeze(predinfo[l]['covdecomp'][k,:,xind])
                covmats[l] = np.diag(obsvar)
                covmats[l] += A1 @ A1.T
                if 'corrf' in options.keys():
                    covmats[l] += phi[k,l]*options['corrf'](emulator[l].x[xind,:], l)['C']
                else:
                    covmats[l] += phi[k,l]*np.eye(obsvar.shape[0])
                covmatsinv[l] = np.linalg.inv(covmats[l])
                totInv += covmatsinv[l]
                if l > 0.5:
                    term1 += covmats[0] @ covmatsinv[l] @ covmats[0]
                    term2 += covmats[0] @ covmatsinv[l]
                    term3 += -covmatsinv[l] @ (mus[l] - mus[0])
            S0 = covmats[0] - (term1 - term2 @ np.linalg.solve(totInv, term2.T))
            m0 = covmats[0] @ term3 - term2 @ np.linalg.solve(totInv, term3)
            mut = mus[0]
        else:
            m0 = np.squeeze(y)*0
            mut = np.squeeze(y) - predinfo[l]['mean'][k,xind]
            A1 = np.squeeze(predinfo['covdecomp'][k,:,xind])
            S0 = np.diag(obsvar)
            S0 += A1 @ A1.T
            if 'corrf' in options.keys():
                covmats[l] += phi[k,l] * options['corrf'](emulator[l].x[xind,:], l)['C']
            else:
                covmats[l] += phi[k,l]*np.eye(obsvar.shape[0])
        loglikr[k] += -0.5*np.sum((mut-m0) * np.linalg.solve(S0,mut-m0))
        loglikr[k] += -0.5*np.linalg.slogdet(S0)[0]
    return loglikr

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
        
    if type(emulator) is tuple:
        predinfo = [dict() for x in range(len(emulator))]
        for k in range(0,len(emulator)):
            predinfo[k] = emulator[k].predict(theta)
    else:
        predinfo = emulator.predict(theta)
    
    for k in range(0, theta.shape[0]):
        if type(emulator) is tuple:
            covmats = [np.array(0) for x in range(len(emulator))]
            covmatsinv = [np.array(0) for x in range(len(emulator))]
            mus = [np.array(0) for x in range(len(emulator))]
            resid= np.zeros(len(emulator)*xind.shape[0])
            totInv = np.zeros((xind.shape[0],xind.shape[0]))
            term1 = np.zeros((xind.shape[0],xind.shape[0]))
            term2 = np.zeros((xind.shape[0],xind.shape[0]))
            term3 = np.zeros((xind.shape[0]))
            for l in range(0,len(emulator)):
                mus[l] = np.squeeze(y) - predinfo[l]['mean'][k,xind]
                A1 = np.squeeze(predinfo[l]['covdecomp'][k,:,xind])
                covmats[l] = np.diag(obsvar)
                covmats[l] += A1 @ A1.T
                if 'corrf' in options.keys():
                    covmats[l] += phi[k,l] * options['corrf'](emulator[l].x[xind,:], l)['C']
                else:
                    covmats[l] += phi[k,l]*np.eye(obsvar.shape[0])
                covmatsinv[l] = np.linalg.inv(covmats[l])
                totInv += covmatsinv[l]
                if l > 0.5:
                    term1 += covmats[0] @ covmatsinv[l] @ covmats[0]
                    term2 += covmats[0] @ covmatsinv[l]
                    term3 += covmatsinv[l] @ (mus[l] - mus[0])
            S0 = covmats[0] - (term1 - term2 @ np.linalg.solve(totInv, term2.T))
            m0 = covmats[0] @ term3 - term2 @ np.linalg.solve(totInv, term3)
            m0 = covmats[0] @ np.linalg.solve(covmats[0] +covmats[1],mus[0] - mus[l])
            mut = m0
        else:
            m0 = np.squeeze(y)*0
            mut = np.squeeze(y) - predinfo[l]['mean'][k,xind]
            A1 = np.squeeze(predinfo['covdecomp'][k,:,xind])
            S0 = np.diag(obsvar)
            S0 += A1 @ A1.T
            if 'corrf' in options.keys():
                covmats[l] += phi[k,l] * options['corrf'](emulator[l].x[xind,:], l)['C']
            else:
                covmats[l] += phi[k,l]*np.eye(obsvar.shape[0])
        loglikr[k] += -0.5*np.sum((mut-m0) * np.linalg.solve(S0,mut-m0))
        loglikr[k] += -0.5*np.linalg.slogdet(S0)[0]
    return loglikr