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
    
    obsvar = 100*obsvar
    
    if type(emulator) is tuple:
        predinfo = [dict() for x in range(len(emulator))]
        for k in range(0,len(emulator)):
            predinfo[k] = emulator[k].predict(theta)
    else:
        predinfo = emulator.predict(theta)
    
    loglikr = np.zeros(theta.shape[0])
    for k in range(0, theta.shape[0]):
        if 'corrf' in options.keys():
            covmatinv = np.zeros((len(emulator)*xind.shape[0],
                                 len(emulator)*xind.shape[0]))
            resid= np.zeros(len(emulator)*xind.shape[0])
            for l in range(0,len(emulator)):
                resid[l*xind.shape[0]:(l+1)*xind.shape[0]] = np.squeeze(y) - predinfo[l]['mean'][k,xind]
                A1 = np.squeeze(predinfo[l]['covdecomp'][k,:,xind])
                covmat = 1000*A1 @ A1.T
                covmatinv[l*xind.shape[0]:(l+1)*xind.shape[0],
                          l*xind.shape[0]:(l+1)*xind.shape[0]] =\
                              np.linalg.inv(0.5*covmat + 0.5*
                                            np.diag(np.diag(covmat)))
            R = np.diag(obsvar)
            Q = np.diag(obsvar)
            RQT = np.hstack((R, 0*R))
            RQB = np.hstack((0*Q, Q))
            RQ = np.vstack((RQT,RQB))
            n = R.shape[0]
            Jm = np.hstack((np.eye(n),-np.eye(n)))
            RQu = RQ - (Jm @ RQ).T @ np.linalg.solve(Jm @ RQ @ Jm.T, Jm @ RQ)
            
            dhat = 
            print(np.linalg.eigh(CovMatInv3)[0])
            #print(np.linalg.eigh(np.linalg.inv(covmatinv) )[0])

            print(1/np.linalg.eigh(np.linalg.inv(covmatinv) + Jm.T @ np.diag(obsvar) @ Jm)[0])

            print(np.sum(resid * (CovMatInv3 @ resid)))
            asdasd
            asdasda
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