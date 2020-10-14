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
        if 'corrf' in options.keys():
            if type(emulator) is tuple:
                TotMatInv = np.diag(1/obsvar)
                SInv = np.diag(1/obsvar)
                ldetSu = np.sum(np.log(obsvar))
                term1 = 0
                resid = np.zeros((y.shape[0],len(emulator)))
                residinv = np.zeros((y.shape[0],len(emulator)))
                for l in range(0,len(emulator)):
                    Sdisc = options['corrf'](emulator[l].x, l)['C'][xind,:][:,xind]
                    A1 = np.squeeze(predinfo[l]['covdecomp'][k,:,xind])
                    Sdisc +=  A1 @ A1.T
                    Sdinv = np.linalg.inv(Sdisc)
                    TotMatInv += Sdinv
                    mu = np.squeeze(predinfo[l]['mean'][k,:])[xind]
                    resid[:, l] = np.squeeze(y)-mu
                    residinv[:, l] =Sdinv @ resid[:, l]
                    ldetSu += np.linalg.slogdet(Sdisc)[1]
                    term1 += np.sum(residinv[:, l] * resid[:, l])
                Qv2 = np.sum(residinv,1)
                ldetSu += np.linalg.slogdet(TotMatInv)[1]
                term1 -= Qv2.T @ np.linalg.solve(TotMatInv,Qv2)
            else:
                TotMatInv = np.diag(1/obsvar)
                SInv = np.diag(1/obsvar)
                ldetSu = np.sum(np.log(obsvar))
                term1 = 0
                resid = np.zeros((y.shape[0],len(emulator)))
                residinv = np.zeros((y.shape[0],len(emulator)))
                Sdisc = options['corrf'](emulator.x, l)['C'][xind,:][:,xind]
                A1 = np.squeeze(predinfo['covdecomp'][k,:,xind])
                Sdisc +=  A1 @ A1.T
                Sdinv = np.linalg.inv(Sdisc)
                TotMatInv += Sdinv
                mu = np.squeeze(predinfo['mean'][k,:])[xind]
                resid[:, l] = np.squeeze(y)-mu
                residinv[:, l] =Sdinv @ resid[:, l]
                ldetSu += np.linalg.slogdet(Sdisc)[1]
                term1 += np.sum(residinv[:, l] * resid[:, l])
                Qv2 = np.sum(residinv,1)
                ldetSu += np.linalg.slogdet(TotMatInv)[1]
                term1 -= Qv2.T @ np.linalg.solve(TotMatInv,Qv2)
        loglikr[k] += -0.5 * term1
        loglikr[k] += -0.5 * ldetSu
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
    meanvec = np.zeros((theta.shape[0],xindnew.shape[0]))
    varvec = np.zeros((theta.shape[0],xindnew.shape[0]))
    for k in range(0, theta.shape[0]):
        Sinv = np.diag(1/(obsvar))
        ldetS = np.sum(np.log(obsvar))
        if 'corrf' in options.keys():
            if type(emulator) is tuple:
                TotMatInv = np.diag(1/obsvar)
                SInv = np.diag(1/obsvar)
                resid = np.zeros((y.shape[0],len(emulator)))
                residinv = np.zeros((y.shape[0],len(emulator)))
                T1 = np.zeros((xindnew.shape[0],y.shape[0]))
                T2 = np.zeros((xindnew.shape[0],xindnew.shape[0]))
                for l in range(0,len(emulator)):
                    Sdisc = options['corrf'](emulator[l].x, l)['C'][xind,:][:,xind]
                    A1 = np.squeeze(predinfo[l]['covdecomp'][k,:,xind])
                    Sdisc +=  A1 @ A1.T
                    Sdinv = np.linalg.inv(Sdisc)
                    Sdisc2 = options['corrf'](emulator[l].x, l)['C'][xindnew,:][:,xind]
                    A12 = np.squeeze(predinfo[l]['covdecomp'][k,:,xindnew])
                    Sdisc2 +=  A12 @ A1.T
                    Sdisc3 = options['corrf'](emulator[l].x, l)['C'][xindnew,:][:,xindnew]
                    Sdisc3 +=  A12 @ A12.T
                    TotMatInv += Sdinv
                    mu = np.squeeze(predinfo[l]['mean'][k,:])[xind]
                    mun = np.squeeze(predinfo[l]['mean'][k,:])[xindnew]
                    resid[:, l] = np.squeeze(y)-mu
                    residinv[:, l] =Sdinv @ resid[:, l]
                    meanvec[k,:] += mun + Sdisc2 @ residinv[:, l]
                    T1 += Sdisc2 @ Sdinv
                    T2 += Sdisc3 - Sdisc2 @ Sdinv @ Sdisc2.T
                Qv2 = np.sum(residinv,1)
                meanvec[k,:] -= T1 @ np.linalg.solve(TotMatInv,Qv2)
                varadj = np.diag(T2 + T1 @ np.linalg.solve(TotMatInv,T1.T))
    preddict = {}
    preddict['mean'] = np.mean(meanvec,0)
    return preddict