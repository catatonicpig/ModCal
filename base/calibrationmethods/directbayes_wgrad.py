"""Header here."""
import numpy as np
import scipy.stats as sps
from base.utilities import postsampler
import copy
from base.utilitiesmethods.nuts import nuts6

"""
##############################################################################
################################### fit ######################################
This [calibrationfitinfo] automatically filled by docinfo.py when running updatedocs.py
##############################################################################
##############################################################################
"""

def fit(fitinfo, emu, x, y,  args=None):
    r"""
    Fits a calibration model.
    This [calibrationfitdocstring] automatically filled by docinfo.py when running updatedocs.py

    Parameters
    ----------
    fitinfo : dict
        An arbitary dictionary where you should place all of your fitting information once complete.
        This dictionary is pass by reference, so there is no reason to return anything. Keep
        only stuff that will be used by predict below. Note that the following are preloaded
        fitinfo['thetaprior'].rnd(s) : Get s random draws from the prior predictive distribution on
            theta.
        fitinfo['thetaprior'].lpdf(theta) : Get the logpdf at theta(s).
        The following are optional preloads based on user input
        fitinfo[yvar] : The vector of observation variances at y
        In addition, calibration can directly use and communicate back to the user if you include:
        fitinfo['thetamean'] : the mean of the prediction of theta
        fitinfo['thetavar'] : the var of the predictive variance on theta
        fitinfo['thetarnd'] : some number draws from the predictive distribution on theta
    emu : instance of emulator class
        An emulator class instatance as defined in emulation
    x : array of objects
        An array of x  that represent the inputs.
    y : array of float
        A one demensional array of observed values at x
    args : dict
        A dictionary containing options passed to you.
    """
    
    if 'yvar' in fitinfo.keys():
        obsvar = fitinfo['yvar']
    else:
        raise ValueError('Must provide yvar in this software.')
    
    thetaprior = fitinfo['thetaprior']
    def logpostfull(theta, return_grad = False):
        logpost = thetaprior.lpdf(theta)
        if theta.ndim > 1.5:
            inds = np.where(np.isfinite(logpost))[0]
            loglikinds, dloglikinds = loglik_grad(fitinfo, emu, theta[inds], y, x, args)
            logpost[inds] += loglikinds
            #dlogpost[inds] += dloglikinds
        else:
            if np.isfinite(logpost):
                loglikinds, dloglikinds = loglik_grad(fitinfo, emu, theta, y, x, args)
                logpost += loglikinds
                #dlogpost += dloglikinds
        #if return_grad:
        #    return logpost#, dlogpost
        #else:
        return logpost
    # thetatest = thetaprior.rnd(10)
    # logpost, dlogpost = logpostfull(thetatest, return_grad = True)
    # for k in range(0,thetatest.shape[1]):
    #     thetaadj = copy.copy(thetatest)
    #     thetaadj[:,k] += 10 ** (-4)
    #     thetaadj[0,k] = thetatest[0,k]
    #     logpostadj, dlogpostadj = logpostfull(thetaadj, return_grad = True)
    #     print((logpostadj - logpost) * 10 ** (4))
    #     print(dlogpost[:,k])
    
    theta = thetaprior.rnd(500)
    theta = postsampler(theta, logpostfull)
    
    mtheta = np.mean(theta,0)
    covmat0 = np.cov(theta.T)
    Wc, Vc = np.linalg.eigh(covmat0)
    
    AdjMatInv = (1/np.sqrt(Wc) * Vc.T)
    AdjMat = Vc  * np.sqrt(Wc)
    def logpostfull1d(thetain, return_grad = True):
        #theta = np.reshape(theta,(1,-1))
        theta = mtheta + AdjMat @ thetain 
        logpost = thetaprior.lpdf(theta)
        if return_grad:
            dlogpost =  thetaprior.lpdf_grad(theta) @ AdjMat
        if theta.ndim > 1.5:
            inds = np.where(np.isfinite(logpost))[0]
            loglikinds, dloglikinds = loglik_grad(fitinfo, emu, theta[inds], y, x, args)
            logpost[inds] += loglikinds
            dlogpost[inds] += dloglikinds
        else:
            if np.isfinite(logpost):
                loglikinds, dloglikinds = loglik_grad(fitinfo, emu, theta, y, x, args)
                logpost += loglikinds
                dlogpost +=np.squeeze(dloglikinds) @ AdjMat
        if return_grad:
            return logpost, dlogpost
        else:
            return logpost
    
    theta0 = AdjMatInv @ (np.squeeze(thetaprior.rnd(1)) - mtheta)
    samples, lnprob, epsilon  = nuts6(logpostfull1d,1000,1000, theta0)
    print(mtheta)
    print(np.sqrt(np.diag(covmat0)))
    print(AdjMat.shape)
    theta = (AdjMat @ samples.T).T + mtheta
    mtheta = np.mean(theta,0)
    covmat0 = np.cov(theta.T)
    print(mtheta)
    print(np.sqrt(np.diag(covmat0)))
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        import pylab as plt
    plt.subplot(1,3,1)
    plt.plot( samples[:, 0],'.')
    plt.plot( samples[:, 1],'.')
    plt.plot( samples[:, 2],'.')
    asdasda
    
    theta = thetaprior.rnd(2500)
    numsamp = 10000
    tarESS = np.max((100, 10 * theta.shape[1]))
    theta = postsampler(theta, logpostfull)
    Lm = np.max(logpostfull(theta))
    fitinfo['thetarnd'] = theta
    fitinfo['y'] = y
    fitinfo['x'] = x
    return


"""
This [calibrationpredictinfo] automatically filled by docinfo.py when running updatedocs.py
##############################################################################
################################### predict ##################################
### The purpose of this is to take an emulator emu alongside fitinfo, and 
### predict at x. You shove all your information into the dictionary predinfo.
##############################################################################
##############################################################################
"""
def predict(predinfo, fitinfo, emu, x, args=None):
    r"""
    Finds prediction at x given the emulator _emu_ and dictionary fitinfo.
    This [calibrationpredictdocstring] automatically filled by docinfo.py when running updatedocs.py

    Parameters
    ----------
    predinfo : dict
        An arbitary dictionary where you should place all of your prediction information once complete. 
        This dictionary is pass by reference, so there is no reason to return anything. Keep
        only stuff that will be used by predict.  Key elements
        predinfo['mean'] : the mean of the prediction
        predinfo['var'] : the variance of the prediction
        predinfo['rand'] : some number draws from the predictive distribution on theta.
    fitinfo : dict
        An arbitary dictionary where you placed all your important fitting information from the 
        fit function above.
    emu : instance of emulator class
        An emulator class instatance as defined in emulation.
    x : array of float
        An array of x values where you want to predict.
    args : dict
        A dictionary containing options passed to you.
    """
    
    y = fitinfo['y']
    
    theta = fitinfo['thetarnd']
    if theta.ndim == 1 and fitinfo['theta'].shape[1] > 1.5:
        theta = theta.reshape((1, theta.shape[0]))
    
    if 'yvar' in fitinfo.keys():
        obsvar = fitinfo['yvar']
    else:
        raise ValueError('Must provide yvar in this software.')
    
    xtot = np.vstack((fitinfo['x'],x))
    mx =fitinfo['x'].shape[0]
    emupredict = emu.predict(xtot, theta)
    meanfull = copy.deepcopy(emupredict()[mx:,:]).T
    varfull = copy.deepcopy(emupredict()[mx:,:]).T
    predinfo['rnd'] = copy.deepcopy(emupredict()[mx:,:]).T
    predinfo['modelrnd'] = copy.deepcopy(emupredict()[mx:,:]).T
    
    
    emupredict = emu.predict(xtot, theta)
    emumean = emupredict.mean()
    emucov = emupredict.covx()
    xind = range(0,mx)
    xindnew = range(mx,xtot.shape[0])
    for k in range(0, theta.shape[0]):
        m0 = np.squeeze(y) * 0
        mut = np.squeeze(y) - emupredict()[(xind,k)]
        m0 = emumean[:,k]
        St = emucov[:,k,:]
        St[np.isnan(St)] = 0
        S0 = St[xind,:][:,xind]
        S10 = St[xindnew,:][:,xind]
        S11 =St[xindnew,:][:,xindnew]
        S0 += np.diag(obsvar)
        mus0 = emupredict()[(xindnew, k)]
        meanfull[k, :] = mus0 + S10 @ np.linalg.solve(S0, mut)
        varfull[k, :] = np.diag(S11 - S10 @ np.linalg.solve(S0, S10.T))
        Wmat, Vmat = np.linalg.eigh(S11 - S10 @ np.linalg.solve(S0, S10.T))
        re = Vmat @ np.diag(np.sqrt(np.abs(Wmat))) @ Vmat.T @\
            sps.norm.rvs(0,1,size=(Vmat.shape[1]))
        predinfo['rnd'][k,:] = meanfull[k, :]  + re
        predinfo['modelrnd'][k,:] = mus0

    predinfo['mean'] = np.mean(meanfull, 0)
    varterm1 = np.var(meanfull, 0)
    predinfo['var'] = np.mean(varfull, 0) + varterm1
    return

"""
This [calibrationadditionalfuncsinfo] automatically filled by docinfo.py when running updatedocs.py
##############################################################################
##############################################################################
####################### THIS ENDS THE REQUIRED PORTION #######################
###### THE NEXT FUNCTIONS ARE OPTIONAL TO BE CALLED BY CALIBRATION ###########
## If this project works, there will be a list of useful calibration functions
## to provide as you want.
##############################################################################
##############################################################################
"""

def thetarnd(fitinfo, s=100, args=None):
    """
    Return s draws from the predictive distribution of theta.  Not required.
    """
    return fitinfo['thetarnd'][np.random.choice(fitinfo['thetarnd'].shape[0], size=s), :]

"""
This [endfunctionsflag] is automatically filled by docinfo.py when running updatedocs.py
##############################################################################
##############################################################################
####################### THIS ENDS THE OPTIONAL PORTION #######################
######### USE SPACE BELOW FOR ANY SUPPORTING FUNCTIONS YOU DESIRE ############
##############################################################################
##############################################################################
"""
# def approxhess(fitinfo, emu, theta, y, x, args):
#     r"""
#     This is a optional docstring for an internal function.
#     """
    
#     if 'yvar' in fitinfo.keys():
#         obsvar = np.squeeze(fitinfo['yvar'])
#     else:
#         raise ValueError('Must provide yvar in this software.')
#     #y = np.squeeze(y)
#     p = theta.shape[1]
    
#     thetastd = np.std(theta,0)
#     L0 = loglik(fitinfo, emu, theta, y, x, args)
#     theta0 = theta[np.argmin(L0)]
#     emumean = emu.predict(x, theta0).mean()
#     emucovhalf = emu.predict(x, theta0).covxhalf()
    
#     fdir = np.zeros((emumean.shape[0], theta.shape[1]))
#     obsvaradj = obsvar + 1 * np.sum(emucovhalf ** 2,0)
#     for p in range(0,theta.shape[1]):
#         thetadj = copy.copy(theta0)
#         thetadj[p] += 10 ** (-4) * thetastd[p]
#         emumeanadj1 = emu.predict(x, thetadj).mean()
#         thetadj[p] -= 2*10 **(-4) * thetastd[p]
#         emumeanadj2 = emu.predict(x, thetadj).mean()
#         fdir[:,p] = np.squeeze((emumeanadj1-emumeanadj2) * (10 ** (4)) / 2)
#     fdiradj = (fdir.T / np.sqrt(obsvaradj)).T
#     D, W, _ = np.linalg.svd(fdir.T)
#     W = W ** 2
#     print(W)
#     Amat = np.diag(np.sqrt(10**(-12) + W[:3])) @ D[:,:3].T
#     bvec = - Amat @ theta0
#     Cmat = D[:,:3] @ np.diag(1/np.sqrt(10**(-12) +W[:3]))
#     dvec = Cmat @ Amat @ theta0
    
#     return Amat, bvec, Cmat, dvec


def loglik_grad(fitinfo, emu, theta, y, x, args):
    r"""
    This is a optional docstring for an internal function.
    """
    
    
    if 'yvar' in fitinfo.keys():
        obsvar = np.squeeze(fitinfo['yvar'])
    else:
        raise ValueError('Must provide yvar in this software.')
    #y = np.squeeze(y)
    emupredict = emu.predict(x, theta, args={'return_grad': True})
    emumean = emupredict.mean()
    emucovxhalf = emupredict.covxhalf()
    emumean_grad = emupredict.mean_gradtheta()
    emucovxhalf_grad  = emupredict.covxhalf_gradtheta()
    loglik = np.zeros(emumean.shape[1])
    dloglik = np.zeros((emumean.shape[1],emu._info['theta'].shape[1]))
    dterm1 = np.zeros(emu._info['theta'].shape[1])
    dterm2 = np.zeros(emu._info['theta'].shape[1])
    dterm3 = np.zeros(emu._info['theta'].shape[1])
    for k in range(0, emumean.shape[1]):
        m0 = emumean[:,k]
        dm0 = np.squeeze(emumean_grad[:, k, :])
        S0 = np.squeeze(emucovxhalf[:,k,:])
        
        stndresid = (np.squeeze(y) - m0) / np.sqrt(obsvar)
        term1 = np.sum(stndresid ** 2)
        stndresid_grad = - (dm0.T / np.sqrt(obsvar)).T
        dterm1 = 2 * np.sum(stndresid * stndresid_grad.T, 1)
        J = (S0.T / np.sqrt(obsvar)).T
        if J.ndim < 1.5:
            J = J[:,None].T
        J2 =  J.T @ stndresid
        W, V = np.linalg.eigh(np.eye(J.shape[1]) + J.T @ J)
        J3 = np.squeeze(V) @ np.diag(1/W) @ np.squeeze(V).T @ np.squeeze(J2)
        term2 = np.sum(J3 * J2)
        for l in range(0,stndresid_grad.shape[1]):
            dJ = (np.squeeze(emucovxhalf_grad[:,k,:,l]) @ np.diag(1/ np.sqrt(obsvar)))
            dJ2 = J.T @ np.squeeze(stndresid_grad[:,l]) + dJ @ stndresid
            exmat = (np.squeeze(emucovxhalf_grad[:,k,:,l]) @ np.diag(1/ np.sqrt(obsvar))) @ J
            exmat = (exmat + exmat.T)
            dJ3 = np.squeeze(V) @ np.diag(1/W) @ np.squeeze(V).T @ (dJ2 - exmat @ J3)
            dterm2[l] = np.sum(J2 * dJ3) + np.sum(dJ2 * J3)
        V2 =  1/obsvar * (((np.squeeze(V) * (1/W)) @ np.squeeze(V).T) @ S0.T)
        for l in range(0,stndresid_grad.shape[1]):
            V3 = np.squeeze(emucovxhalf_grad[:,k,:,l])
            dterm3[l] = 2 * np.sum(V2 * V3)
        term3 = np.sum(np.log(W))
        residsq = term1 - term2
        loglik[k] = -0.5 * (m0.shape[0]+0.1) * np.log(residsq+0.1) - 0.5 * term3
        dloglik[k, :] = -0.5 * (m0.shape[0]+0.1) * (dterm1-dterm2) / (residsq+0.1) - 0.5 * dterm3
    return loglik, dloglik