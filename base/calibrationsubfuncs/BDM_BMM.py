"""Header here."""
import numpy as np
import scipy.stats as sps
from base.utilities import postsampler
import copy


##############################################################################
##############################################################################
###################### THIS BEGINS THE REQUIRED PORTION ######################
######### THE NEXT FUNCTIONS REQUIRED TO BE CALLED BY CALIBRATION ############
##############################################################################
##############################################################################

def fit(info, emu, y, x, args=None):
    r"""
    This is a optional docstring for an internal function.
    """
    
    
    if type(emu) is not tuple:
        raise ValueError('Must provide a tuple of emulators to BDM_BMM.')
        
    thetaprior = info['thetaprior']
    theta = thetaprior.rnd(1000)
    
    if 'obsvar' in args.keys():
        obsvar = args['obsvar']
    else:
        raise ValueError('Must provide a prior on statistical parameters in BDM_BMM.')
    
    if 'phiprior' in args.keys():
        phiprior = args['phiprior']
    else:
        raise ValueError('Must provide a prior on statistical parameters in BDM_BMM.')
    
    
    theta = thetaprior.rnd(1000)
    thetadim = theta[0].shape[0]
    if phiprior.rnd(1) is None:
        phidim = 0
        thetaphi = theta
    else:
        phi = phiprior.rnd(1000)
        phidim = (phiprior.rnd(1)).shape[1]
        thetaphi = np.hstack((theta,phi))
        
        
    
    def logpostfull(thetaphi):
        if phidim > 0.5:
            theta = thetaphi[:, :thetadim]
            phi = thetaphi[:, thetadim:]
        else:
            theta = thetaphi
            phi = None
        logpost =thetaprior.lpdf(theta) + phiprior.lpdf(phi)
        
        inds = np.where(np.isfinite(logpost))[0]
        if phi is None:
            logpost[inds] += loglik(emu, theta[inds], None, y, x, args)
        else:
            logpost[inds] += loglik(emu, theta[inds], phi[inds], y, x, args)
        return logpost
    
    numsamp = 1000
    tarESS = np.max((100, 10 * thetaphi.shape[1]))
    thetaphi = postsampler(thetaphi, logpostfull)
    
    if phidim > 0.5:
        theta = thetaphi[:, :thetadim]
        phi = thetaphi[:, thetadim:]
    else:
        theta = thetaphi
        phi = None
    
    info['thetarnd'] = theta
    info['phirnd'] = phi
    info['y'] = y
    info['x'] = x
    return


def predict(x, emu, calinfo, args = None):
    r"""
    This is a optional docstring for an internal function.
    """
    
    theta = calinfo['thetarnd']
    phi = calinfo['phirnd']
    if theta.ndim == 1:
        theta = theta.reshape((1, theta.shape[0]))
    elif 'obsvar' in args.keys():
        obsvar = args['obsvar']
    else:
        raise ValueError('Must provide obsvar at this moment.')
    info = {}
    xtot = np.vstack((calinfo['x'],x))
    mx =calinfo['x'].shape[0]

    emumean = [np.ones(1) for x in range(len(emu))]
    emucov = [np.ones(1) for x in range(len(emu))]
    for k in range(0, len(emu)):
        emupredict = emu[k].predict(theta, xtot)
        emumean[k] = emupredict.mean()
        emucov[k] = emupredict.cov()
        
    
    meanfull = np.ones((emumean[k].shape[0],x.shape[0]))
    varfull = np.ones((emumean[k].shape[0],x.shape[0]))
    info['rnd'] = np.ones((emumean[k].shape[0],x.shape[0]))
    info['modelrnd'] = np.ones((emumean[k].shape[0],x.shape[0]))
    xind = range(0,mx)
    xindnew = range(mx,xtot.shape[0])
    covmats = [np.array(0) for l in range(len(emu))]
    covmatsinv = [np.array(0) for l in range(len(emu))]
    mus = [np.array(0) for l in range(len(emu))]
    for k in range(0, theta.shape[0]):
        totInv = np.zeros((xtot.shape[0], xtot.shape[0]))
        term2 = np.zeros(xtot.shape[0])
        for l in range(0, len(emu)):
            mus[l] = emumean[l][k]
        for l in reversed(range(0, len(emu))):
            #A1 = np.squeeze(emupredict[l].info['covdecomp'][k, :, :])
            covmats[l] =  emucov[l][k]
            if 'cov_disc' in args.keys():
                covmats[l] += args['cov_disc'](xtot, l, phi[k,:])
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
        Mat1 = S10 @ S0inv
        resid = np.squeeze(calinfo['y'])
        meanfull[k, :] =  m10 +  Mat1 @ (resid - m00)
        varfull[k, :] = (np.diag(S0)[xindnew] -\
            np.sum(S10 * Mat1,1))
        Wmat, Vmat = np.linalg.eigh(S0[xindnew,:][:,xindnew] - S10 @ Mat1.T)
        
        re = Vmat @ np.diag(np.sqrt(np.abs(Wmat))) @ Vmat.T @\
            sps.norm.rvs(0,1,size=(Vmat.shape[1]))
        info['rnd'][k,:] = meanfull[k, :]  + re
        Wmat, Vmat = np.linalg.eigh(S0[xindnew,:][:,xindnew])
        re = Vmat @ np.diag(np.sqrt(np.abs(Wmat))) @ Vmat.T @\
            sps.norm.rvs(0,1,size=(Vmat.shape[1]))
        info['modelrnd'][k,:] = m10 + re

    info['mean'] = np.mean(meanfull, 0)
    varterm1 = np.var(meanfull, 0)
    info['var'] = np.mean(varfull, 0) + varterm1
    return info


##############################################################################
##############################################################################
####################### THIS ENDS THE REQUIRED PORTION #######################
###### THE NEXT FUNCTIONS ARE OPTIONAL TO BE CALLED BY CALIBRATION ###########
##############################################################################
##############################################################################

##############################################################################
##############################################################################
####################### THIS ENDS THE OPTIONAL PORTION #######################
######### USE SPACE BELOW FOR ANY SUPPORTING FUNCTIONS YOU DESIRE ############
##############################################################################
##############################################################################

def loglik(emu, theta, phi, y, x, args):
    r"""
    This is a optional docstring for an internal function.
    """
    if theta.ndim == 1:
        theta = theta.reshape((1, theta.shape[0]))
    elif 'obsvar' in args.keys():
        obsvar = args['obsvar']
    else:
        raise ValueError('Must provide obsvar at this moment.')
    emumean = [np.ones(1) for x in range(len(emu))]
    emucov = [np.ones(1) for x in range(len(emu))]
    for k in range(0, len(emu)):
        emupredict = emu[k].predict(theta, x)
        emumean[k] = emupredict.mean()
        emucov[k] = emupredict.cov()
    
    
    loglik = np.zeros(theta.shape[0])
    covmats = [np.array(0) for x in range(len(emu))]
    covmatsinv = [np.array(0) for x in range(len(emu))]
    mus = [np.array(0) for x in range(len(emu))]
    resid = np.zeros(len(emu) * x.shape[0])
    for k in range(0, theta.shape[0]):
        totInv = np.zeros((x.shape[0], x.shape[0]))
        term2 = np.zeros(x.shape[0])
        for l in range(0, len(emu)):
            mus[l] = emumean[l][k]
            #A1 = np.squeeze(emupredict[l].info['covdecomp'][k, :, :])
            covmats[l] = emucov[l][k] #A1.T @ A1
            if 'cov_disc' in args.keys():
                covmats[l] += args['cov_disc'](x, l, phi[k,:])
            covmats[l] += np.diag(np.diag(covmats[l])) * (10 ** (-8))
            covmatsinv[l] = np.linalg.inv(covmats[l])
            totInv += covmatsinv[l]
            term2 += covmatsinv[l] @ mus[l]
        m0 = np.linalg.solve(totInv, term2)
        W, V = np.linalg.eigh(np.diag(obsvar) + np.linalg.inv(totInv))
        muadj = V.T @ (np.squeeze(y) - m0)
        loglik[k] = -0.5 * np.sum((muadj ** 2) / W)-0.5 * np.sum(np.log(W))
    return loglik