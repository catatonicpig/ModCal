# -*- coding: utf-8 -*-
"""Header here."""
import numpy as np
import scipy.stats as sps

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
        theta = theta.reshape((1, theta.shape[0]))
    elif 'obsvar' in options.keys():
        obsvar = options['obsvar']
    else:
        raise ValueError('Must provide obsvar at this moment.')
    if type(emulator) is tuple:
        predinfo = [dict() for x in range(len(emulator))]
        for k in range(0, len(emulator)):
            predinfo[k] = emulator[k].predict(theta)
        loglik = np.zeros(predinfo[0]['mean'].shape[0])
    else:
        predinfo = emulator.predict(theta)
        loglik = np.zeros(predinfo['mean'].shape[0])
        
    for k in range(0, theta.shape[0]):
        if type(emulator) is tuple:
            logliklocal = np.zeros(len(emulator))
            for l in range(0, len(emulator)):
                Sinv = np.diag(1 / (np.exp(phi[k,l]) * obsvar))
                ldetS = np.sum(phi[k,l] + np.log(obsvar))
                Sinvu = Sinv
                ldetSu = ldetS
                mu = np.squeeze(predinfo[l]['mean'][k,:])[xind]
                U = (predinfo[l]['covdecomp'][k,:,:])[:,xind]
                Am = Sinvu @ U.T
                W, V = np.linalg.eigh(np.eye(U.shape[0]) + U @ Am)
                Amp = (np.squeeze(y)-mu).T @ Am @ V.T
                logliklocal[l] = -0.5*(np.squeeze(y)-mu).T @ (Sinvu @ (np.squeeze(y)-mu))
                logliklocal[l] += -0.5*ldetSu
                logliklocal[l] += 0.5*(Amp @ (Amp * (1/W)).T)
                logliklocal[l] += -0.5*np.sum(np.log(W))
            lm = np.max(logliklocal)
            loglik[k] = np.log(np.sum(np.exp(logliklocal-lm))) + lm
        else:
            Sinv = np.diag(1 / (np.exp(phi[k]) * obsvar))
            ldetS = np.sum(phi[k] + np.log(obsvar))
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
        theta = theta.reshape((1, theta.shape[0]))
    elif 'obsvar' in options.keys():
        obsvar = options['obsvar']
    else:
        raise ValueError('Must provide obsvar at this moment.')
    preddict = {}
    if type(emulator) is tuple:
        predinfo = [dict() for x in range(len(emulator))]
        for k in range(0, len(emulator)):
            predinfo[k] = emulator[k].predict(theta)
        preddict['meanfull'] = predinfo[0]['mean']
        preddict['varfull'] = predinfo[0]['var']
        preddict['draws'] = predinfo[0]['mean']
        preddict['modeldraws'] = predinfo[0]['mean']
    else:
        predinfo = emulator.predict(theta)
        preddict['meanfull'] = predinfo['mean']
        preddict['full'] = predinfo['mean']
        preddict['draws'] = predinfo['mean']
        preddict['modeldraws'] = predinfo['mean']
        preddict['varfull'] = predinfo['var']
    
    for k in range(0, theta.shape[0]):
        if type(emulator) is tuple:
            logliklocal = np.zeros(len(emulator))
            predlocal = np.zeros((xindnew.shape[0], len(emulator)))
            sslocal = np.zeros((xindnew.shape[0], len(emulator)))
            for l in range(0, len(emulator)):
                Sinv = np.diag(1 / (np.exp(phi[k,l]) * obsvar))
                ldetS = np.sum(phi[k,l] + np.log(obsvar))
                Sinvu = Sinv
                ldetSu = ldetS
                mu = np.squeeze(predinfo[l]['mean'][k,:])[xind]
                U = (predinfo[l]['covdecomp'][k,:,:])[:,xind]
                Am = Sinvu @ U.T
                W, V = np.linalg.eigh(np.eye(U.shape[0]) + U @ Am)
                Amp = (np.squeeze(y)-mu).T @ Am @ V.T
                logliklocal[l] = -0.5*(np.squeeze(y)-mu).T @ (Sinvu @ (np.squeeze(y)-mu))
                logliklocal[l] += -0.5*ldetSu
                logliklocal[l] += 0.5*(Amp @ (Amp * (1/W)).T)
                logliklocal[l] += -0.5*np.sum(np.log(W))
                predlocal[:, l] = np.squeeze(predinfo[l]['mean'][k,xindnew])
                sslocal[:, l] = np.sum((predinfo[l]['covdecomp'][k,:,xindnew]) ** 2) +\
                    np.squeeze(predlocal[:, l] ** 2)
            lm = np.max(logliklocal)
            logpostcalc = np.log(np.sum(np.exp(logliklocal-lm))) + lm
            post = np.exp(logliklocal - logpostcalc)
            preddict['meanfull'][k,:] = np.sum(post * predlocal,1)
            preddict['varfull'][k,:] = np.sum(post * sslocal,1) -\
                preddict['meanfull'][k,:] ** 2
            rc = np.random.choice(len(emulator), p = post)
            re = predinfo[rc]['covdecomp'][k,:,:].T @\
                sps.norm.rvs(0,1,size=(predinfo[rc]['covdecomp'].shape[1]))
            preddict['draws'][k,:] = predinfo[rc]['mean'][k,xindnew] + re[xindnew]
            preddict['modeldraws'][k,:] = predinfo[rc]['mean'][k,xindnew] + re[xindnew]
        else:
            preddict['meanfull'][k,:] = np.squeeze(predinfo['mean'][k,xindnew])
            preddict['varfull'][k,:] = np.sum((predinfo['covdecomp'][k,:,xindnew]) ** 2)
            #rc = np.random.choice(len(emulator), p = post)
            re = predinfo['covdecomp'][k,:,:].T @\
                sps.norm.rvs(0,1,size=(predinfo['covdecomp'].shape[1]))
            preddict['draws'][k,:] = preddict['meanfull'][k,:] + re[xindnew]
            preddict['modeldraws'][k,:] = predinfo['mean'][k,xindnew] + re[xindnew]
                
    preddict['mean'] = np.mean(preddict['meanfull'], 0)
    varterm1 = np.var(preddict['meanfull'], 0)
    preddict['var'] = np.mean(preddict['varfull'], 0) + varterm1
    return preddict