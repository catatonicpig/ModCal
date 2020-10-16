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
            totInv = np.zeros((emulator[0].x.shape[0], emulator[0].x.shape[0]))
            term2 = np.zeros(emulator[0].x.shape[0])
            for l in range(0, len(emulator)):
                mus[l] = predinfo[l]['mean'][k, :]
                A1 = np.squeeze(predinfo[l]['covdecomp'][k, :, :])
                covmats[l] = A1.T @ A1
                if 'corrf' in options.keys():
                    covmats[l] += phi[(k, l)] * options['corrf'](emulator[l].x, l)['C']
                else:
                    covmats[l] += phi[(k, l)] * np.eye(obsvar.shape[0])
                covmats[l] += np.mean(obsvar) * 0.0001 * np.eye(covmats[l].shape[0])
                covmatsinv[l] = np.linalg.inv(covmats[l])
                totInv += covmatsinv[l]
                term2 += covmatsinv[l] @ mus[l]
            S0 = np.linalg.inv(totInv)
            m0 = np.linalg.solve(totInv, term2)
            
            m0 = m0[xind]
            S0 = S0[xind, :][:, xind] + np.diag(obsvar)
            S00altinv = np.diag(1/obsvar) -\
                np.linalg.inv(np.diag(1/obsvar) + totInv[xind,:][:,xind])
            S0 = np.linalg.inv(S00altinv)
        else:
            m0 = predinfo['mean'][(k, xind)]
            A1 = np.squeeze(predinfo['covdecomp'][k, :, xind])
            S0 = np.diag(obsvar)
            S0 += A1 @ A1.T
            if 'corrf' in options.keys():
                S0 += phi[k] * options['corrf'](emulator.x[xind, :])['C']
            else:
                S0 += phi[k] * np.eye(obsvar.shape[0])
            S0 += 0.0001 * np.diag(obsvar)
        W, V = np.linalg.eigh(S0)
        muadj = V.T @ (np.squeeze(y) - m0)
        loglikr1[k] = -0.5 * np.sum(muadj ** 2 / W)
        if loglikr1[k] > 0:
            print('lik is all messed up')
        loglikr2[k] = -0.5 * np.sum(np.log(W)) * 0

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
        preddict['meanfull'] = predinfo[1]['mean']
        preddict['varfull'] = predinfo[1]['var']
    else:
        predinfo = emulator.predict(theta)
        preddict['meanfull'] = predinfo['mean']
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
                if 'corrf' in options.keys():
                    covmats[l] = phi[k, l] * options['corrf'](emulator[l].x, l)['C']
                else:
                    covmats[l] = phi[k, l] * np.eye(emulator[l].x.shape[0])
                covmats[l] += np.mean(obsvar) * 0.0001 * np.eye(covmats[l].shape[0])
                Cinv = np.linalg.inv(covmats[l])
                A1 = np.squeeze(predinfo[l]['covdecomp'][k, :, :])
                covmatsinv[l] = Cinv - Cinv @ A1.T @ np.linalg.solve( A1 @ Cinv @ A1.T + np.eye(A1.shape[0]), A1 @ Cinv)
                totInv += covmatsinv[l]
                term2 += covmatsinv[l] @ mus[l]

            S0 = np.linalg.inv(totInv)
            m0 = np.linalg.solve(totInv, term2)
            m00 = m0[xind]
            m10 = m0[xindnew]
           # print(m0[xind])
            m00a = np.linalg.solve(totInv[xind,:] @ totInv[xind,:].T, totInv[xind,:] @term2)
            #print(m00a)
            #asdad
            #S00 = S0[xind, :][:, xind] + np.diag(obsvar)
            S00altinv = np.diag(1/obsvar) -\
                np.linalg.inv(np.diag(1/obsvar) + totInv[xind,:][:,xind])
            S01 = S0[xind, :][:, xindnew]
            S11 = S0[xindnew, :][:, xindnew]
            resid = np.squeeze(y)
            preddict['meanfull'][k, :] =  m10 + S01.T @ S00altinv @ (resid - m00)
            preddict['varfull'][k, :] = np.diag(S11 -S01.T @ S00altinv @ S01)
        else:
            m0 = np.squeeze(y) * 0
            mut = np.squeeze(y) - predinfo['mean'][(k, xind)]
            A1 = np.squeeze(predinfo['covdecomp'][k, :, xind])
            A2 = np.squeeze(predinfo['covdecomp'][k, :, xindnew])
            S0 = np.diag(obsvar)
            S0 += A1 @ A1.T
            mus0 = predinfo['mean'][(k, xindnew)]
            S10 = A2 @ A1.T
            S11 = A2 @ A2.T
            if 'corrf' in options.keys():
                Cm = options['corrf'](emulator.x)['C']
                S10 += phi[k] * Cm[xindnew, :][:, xind]
                S11 += phi[k] * Cm[xindnew, :][:, xindnew]
                S0 += phi[k] * options['corrf'](emulator.x[xind, :])['C']
            else:
                S0 += phi[k] * np.eye(obsvar.shape[0])
            S0 += 0.0001 * np.diag(obsvar)
            preddict['meanfull'][k, :] = mus0 + S10 @ np.linalg.solve(S0, mut)
            preddict['varfull'][k, :] = np.diag(S11 - S10 @ np.linalg.solve(S0, S10.T))

    preddict['mean'] = np.mean(preddict['meanfull'], 0)
    varterm1 = np.var(preddict['meanfull'], 0)
    preddict['var'] = np.mean(preddict['varfull'], 0) + varterm1
    return preddict