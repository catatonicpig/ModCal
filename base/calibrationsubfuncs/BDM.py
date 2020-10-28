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
    
    thetaprior = info['thetaprior']
    theta = thetaprior.rnd(1000)
    
    if 'obsvar' in args.keys():
        obsvar = args['obsvar']
    else:
        raise ValueError('Must provide a prior on statistical parameters in this software.')
    
    if 'phiprior' in args.keys():
        phiprior = args['phiprior']
    else:
        raise ValueError('Must provide a prior on statistical parameters in this software.')
    
    
    if type(emu) is tuple:
        raise ValueError('Cannot provide a tuple of emulators to BDM.')
    
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
    y = calinfo['y']
    theta = calinfo['thetarnd']
    phi = calinfo['phirnd']
    if theta.ndim == 1:
        theta = theta.reshape((1, theta.shape[0]))
    elif 'obsvar' in args.keys():
        obsvar = args['obsvar']
    else:
        raise ValueError('Must provide obsvar at this moment.')
    
    if 'cov_disc' in args.keys():
        cov_disc = args['cov_disc']
    else:
        raise ValueError('Must provide obsvar at this moment.')
    
    info = {}
    xtot = np.vstack((calinfo['x'],x))
    mx =calinfo['x'].shape[0]
    emupredict = emu.predict(theta, xtot)
    meanfull = copy.deepcopy(emupredict()[:,mx:])
    varfull = copy.deepcopy(emupredict()[:,mx:])
    info['rnd'] = copy.deepcopy(emupredict()[:,mx:])
    info['modelrnd'] = copy.deepcopy(emupredict()[:,mx:])
    
    
    emupredict = emu.predict(theta, xtot)
    emumean = emupredict.mean()
    emucov = emupredict.cov()
    
    
    xind = range(0,mx)
    xindnew = range(mx,xtot.shape[0])
    for k in range(0, theta.shape[0]):
        m0 = np.squeeze(y) * 0
        mut = np.squeeze(y) - emupredict()[(k, xind)]
        m0 = emumean[k]
        St = emucov[k]
        S0 = St[xind,:][:,xind]
        S10 = St[xindnew,:][:,xind]
        S11 =St[xindnew,:][:,xindnew]
        C = args['cov_disc'](xtot, phi[k,:])
        S0 += C[xind,:][:,xind]
        S10 += C[xindnew,:][:,xind]
        S11 += C[xindnew,:][:,xindnew]
        S0 += np.diag(obsvar)
        mus0 = emupredict()[(k, xindnew)]
        meanfull[k, :] = mus0 + S10 @ np.linalg.solve(S0, mut)
        varfull[k, :] = np.diag(S11 - S10 @ np.linalg.solve(S0, S10.T))
        Wmat, Vmat = np.linalg.eigh(S11 - S10 @ np.linalg.solve(S0, S10.T))
        re = Vmat @ np.diag(np.sqrt(np.abs(Wmat))) @ Vmat.T @\
            sps.norm.rvs(0,1,size=(Vmat.shape[1]))
        info['rnd'][k,:] = meanfull[k, :]  + re
        info['modelrnd'][k,:] = mus0

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

def thetarnd(calinfo, s=100, args=None):
    """
    Return posterior of function evaluation at the new parameters.
    """
    return calinfo['thetarnd'][np.random.choice(calinfo['thetarnd'].shape[0], size=s), :]

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
    
    if 'cov_disc' in args.keys():
        cov_disc = args['cov_disc']
    else:
        raise ValueError('Must provide obsvar at this moment.')
        
    emupredict = emu.predict(theta, x)
    emumean = emupredict.mean()
    emucov = emupredict.cov()
    loglik = np.zeros(theta.shape[0])
    for k in range(0, theta.shape[0]):
        m0 = emumean[k]
        S0 = emucov[k]#Shalf.T @ Shalf
        S0 += cov_disc(x, phi[k,:])
        W, V = np.linalg.eigh(np.diag(obsvar) + S0)
        muadj = V.T @ (np.squeeze(y) - m0)
        loglik[k] = -0.5 * np.sum((muadj ** 2) / W)-0.5 * np.sum(np.log(W))
    return loglik