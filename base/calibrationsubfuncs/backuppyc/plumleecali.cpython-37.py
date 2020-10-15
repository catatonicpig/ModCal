# uncompyle6 version 3.7.4
# Python bytecode 3.7 (3394)
# Decompiled from: Python 3.7.6 (default, Jan  8 2020, 20:23:39) [MSC v.1916 64 bit (AMD64)]
# Embedded file name: C:\Users\FloYd\Documents\GithubFiles\ModCal\base\calibrationsubfuncs\plumleecali.py
# Compiled at: 2020-10-15 17:06:27
# Size of source mod 2**32: 7872 bytes
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
            totInv = np.zeros((xind.shape[0], xind.shape[0]))
            term2 = np.zeros(xind.shape[0])
            for l in range(0, len(emulator)):
                mus[l] = np.squeeze(y) - predinfo[l]['mean'][(k, xind)]
                A1 = np.squeeze(predinfo[l]['covdecomp'][k, :, xind])
                covmats[l] = A1 @ A1.T
                if 'corrf' in options.keys():
                    covmats[l] += phi[(k, l)] * options['corrf'](emulator[l].x[xind, :], l)['C']
                else:
                    covmats[l] += phi[(k, l)] * np.eye(obsvar.shape[0])
                covmats[l] += 0.0001 * np.diag(obsvar)
                covmatsinv[l] = np.linalg.inv(covmats[l])
                totInv += covmatsinv[l]
                if l > 0.5:
                    term2 += covmatsinv[l] @ (mus[0] - mus[l])

            S0 = np.linalg.inv(totInv)
            S0 += np.diag(obsvar)
            m0 = np.linalg.solve(totInv, term2)
            mut = mus[0]
        else:
            m0 = np.squeeze(y) * 0
            mut = np.squeeze(y) - predinfo['mean'][(k, xind)]
            A1 = np.squeeze(predinfo['covdecomp'][k, :, xind])
            S0 = np.diag(obsvar)
            S0 += A1 @ A1.T
            if 'corrf' in options.keys():
                S0 += phi[k] * options['corrf'](emulator.x[xind, :])['C']
            else:
                S0 += phi[k] * np.eye(obsvar.shape[0])
            S0 += 0.0001 * np.diag(obsvar)
        W, V = np.linalg.eigh(S0)
        muadj = V.T @ (mut - m0)
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
            if k == 0:
                preddict['meanfull'] = predinfo[0]['mean']
                preddict['varfull'] = predinfo[0]['var']

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
                A1 = np.squeeze(predinfo[l]['covdecomp'][k, :, :])
                covmats[l] = A1.T @ A1
                if 'corrf' in options.keys():
                    covmats[l] += phi[(k, l)] * options['corrf'](emulator[l].x, l)['C']
                else:
                    covmats[l] += phi[(k, l)] * np.eye(emulator[l].x.shape[0])
                covmats[l] += np.mean(obsvar) * 0.0001 * np.eye(covmats[l].shape[0])
                covmatsinv[l] = np.linalg.inv(covmats[l])
                totInv += covmatsinv[l]
                if l > 0.5:
                    term2 += covmatsinv[l] @ (mus[0] - mus[l])

            S0 = np.linalg.inv(totInv)
            m0 = -np.linalg.solve(totInv, term2)
            m00 = m0[xind]
            m10 = m0[xindnew]
            S00 = S0[xind, :][:, xind] + np.diag(obsvar)
            S01 = S0[xind, :][:, xindnew]
            S11 = S0[xindnew, :][:, xindnew]
            resid = np.squeeze(y) - mus[0][xind]
            preddict['meanfull'][k, :] = mus[0][xindnew] + m10 + S01.T @ np.linalg.solve(S00, resid - m00)
            preddict['varfull'][k, :] = np.diag(S11 - S01.T @ np.linalg.solve(S00, S01))
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
            elif 'corrf' in options.keys():
                S0 += phi[k] * options['corrf'](emulator.x[xind, :])['C']
            else:
                S0 += phi[k] * np.eye(obsvar.shape[0])
            preddict['meanfull'][k, :] = mus0 + S10 @ np.linalg.solve(S0, mut)
            preddict['varfull'][k, :] = np.diag(S11 - S10 @ np.linalg.solve(S0, S10.T))

    preddict['mean'] = np.mean(preddict['meanfull'], 0)
    varterm1 = np.var(preddict['meanfull'], 0)
    preddict['var'] = np.mean(preddict['varfull'], 0) + varterm1
    return preddict