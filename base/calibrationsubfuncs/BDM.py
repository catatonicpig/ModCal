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
        A matrix of the same length as "emulator.x" with observations. 'None' is equivlent to the
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

    else:
        predinfo = emulator.predict(theta)
    loglikr1 = np.zeros(theta.shape[0])
    loglikr2 = np.zeros(theta.shape[0])
    for k in range(0, theta.shape[0]):
        if type(emulator) is tuple:
            covmats = [np.array(0) for x in range(len(emulator))]
            covmatsinv = [np.array(0) for x in range(len(emulator))]
            mus = [np.array(0) for x in range(len(emulator))]
            resid = np.zeros(len(emulator) * xind.shape[0])
            totInv = np.zeros((xind.shape[0], xind.shape[0]))
            term2 = np.zeros(xind.shape[0])
            for l in range(0, len(emulator)):
                mus[l] = predinfo[l]['mean'][k, xind]
                A1 = np.squeeze(predinfo[l]['covdecomp'][k, :, xind])
                covmats[l] = (A1 @ A1.T)
                if 'cov_disc' in options.keys():
                    covmats[l] += options['cov_disc'](emulator[l].x[xind,:], l, phi[k,:])
                covmats[l] += np.diag(np.diag(covmats[l])) * (10 ** (-8))
                covmatsinv[l] = np.linalg.inv(covmats[l])
                totInv += covmatsinv[l]
                term2 += covmatsinv[l] @ mus[l]
            m0 = np.linalg.solve(totInv, term2)
            W, V = np.linalg.eigh(np.diag(obsvar) + np.linalg.inv(totInv))
        else:
            m0 = predinfo['mean'][(k, xind)]
            A1 = np.squeeze(predinfo['covdecomp'][k, :, xind])
            S0 = A1 @ A1.T
            if 'cov_disc' in options.keys():
                S0 += options['cov_disc'](emulator.x[xind,:], phi[k,:])
            #S0 += np.diag(np.diag(S0)) * 0.00000001
            W, V = np.linalg.eigh(np.diag(obsvar) + S0)
        muadj = V.T @ (np.squeeze(y) - m0)
        loglikr1[k] = -0.5 * np.sum((muadj ** 2) / W)
        if np.min(W) < 10 ** (-9):
            print('lik is all messed up')
            print(W)
            print(phi[k,:])
            print(S0)
            adsadas
        loglikr2[k] = -0.5 * np.sum(np.log(W))
    return loglikr1 + loglikr2


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
        A matrix of the same length as "emulator.x" with observations. 'None' is equivlent to the
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
            covmats = [np.array(0) for x in range(len(emulator))]
            covmatsB = [np.array(0) for x in range(len(emulator))]
            covmatsC = [np.array(0) for x in range(len(emulator))]
            covmatsinv = [np.array(0) for x in range(len(emulator))]
            mus = [np.array(0) for x in range(len(emulator))]
            totInv = np.zeros((emulator[0].x.shape[0], emulator[0].x.shape[0]))
            term2 = np.zeros(emulator[0].x.shape[0])
            for l in range(0, len(emulator)):
                mus[l] = predinfo[l]['mean'][k, :]
            for l in reversed(range(0, len(emulator))):
                A1 = np.squeeze(predinfo[l]['covdecomp'][k, :, :])
                covmats[l] = A1.T @ A1
                if 'cov_disc' in options.keys():
                    covmats[l] += options['cov_disc'](emulator[l].x, l, phi[k,:])
                covmats[l] += np.diag(np.diag(covmats[l])) * (10 ** (-8))
                covmatsinv[l] = np.linalg.inv(covmats[l])
                totInv += covmatsinv[l]
                term2 += covmatsinv[l] @ mus[l]

            S0 = np.linalg.inv(totInv)
            m0 = np.linalg.solve(totInv, term2)
            m00 = m0[xind]
            m10 = m0[xindnew]
            S0inv = np.linalg.inv(np.diag(obsvar) + S0[xind,:][:,xind])
            S10 = S0[xindnew, :][:, xind]
            if 'pred_para' in options.keys():
                S10 = options['pred_para'] * S10
            Mat1 = S10 @ S0inv
            resid = np.squeeze(y)
            preddict['meanfull'][k, :] =  m10 +  Mat1 @ (np.squeeze(y) - m00)
            preddict['varfull'][k, :] = (np.diag(S0)[xindnew] -\
                np.sum(S10 * Mat1,1))
            Wmat, Vmat = np.linalg.eigh(S0[xindnew,:][:,xindnew] - S10 @ Mat1.T)
            
            re = Vmat @ np.diag(np.sqrt(np.abs(Wmat))) @ Vmat.T @\
                sps.norm.rvs(0,1,size=(Vmat.shape[1]))
            preddict['draws'][k,:] = preddict['meanfull'][k, :]  + re
            preddict['modeldraws'][k,:] = m10
        else:
            m0 = np.squeeze(y) * 0
            mut = np.squeeze(y) - predinfo['mean'][(k, xind)]
            A1 = np.squeeze(predinfo['covdecomp'][k, :, xind])
            A2 = np.squeeze(predinfo['covdecomp'][k, :, xindnew])
            S0 = A1 @ A1.T
            S10 = A2 @ A1.T
            S11 = A2 @ A2.T
            if 'cov_disc' in options.keys():
                C = options['cov_disc'](emulator.x, phi[k,:])
                S0 += C[xind,:][:,xind]
                S10 += C[xindnew,:][:,xind]
                S11 += C[xindnew,:][:,xindnew]
            #S0 += np.diag(np.diag(S0)) * 0.00000001
            if 'pred_para' in options.keys():
                S10 = options['pred_para'] * S10
            S0 += np.diag(obsvar)
            mus0 = predinfo['mean'][(k, xindnew)]
            preddict['meanfull'][k, :] = mus0 + S10 @ np.linalg.solve(S0, mut)
            preddict['varfull'][k, :] = np.diag(S11 - S10 @ np.linalg.solve(S0, S10.T))
            Wmat, Vmat = np.linalg.eigh(S11 - S10 @ np.linalg.solve(S0, S10.T))
            re = Vmat @ np.diag(np.sqrt(np.abs(Wmat))) @ Vmat.T @\
                sps.norm.rvs(0,1,size=(Vmat.shape[1]))
            preddict['draws'][k,:] = preddict['meanfull'][k, :]  + re
            preddict['modeldraws'][k,:] = mus0

    preddict['mean'] = np.mean(preddict['meanfull'], 0)
    varterm1 = np.var(preddict['meanfull'], 0)
    preddict['var'] = np.mean(preddict['varfull'], 0) + varterm1
    return preddict