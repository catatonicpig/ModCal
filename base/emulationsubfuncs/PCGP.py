# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 14:40:38 2020

@author: Plumlee
"""

import numpy as np
import scipy.optimize as spo

##############################################################################
##############################################################################
###################### THIS BEGINS THE REQUIRED PORTION ######################
########## THE NEXT FUNCTIONS REQUIRED TO BE CALLED BY EMULATION #############
##############################################################################
##############################################################################

def fit(info, theta, f, x=None,  args=None):
    """Return a Gaussian Process emulator model."""
    info['offset'] = np.zeros(f.shape[1])
    info['scale'] = np.ones(f.shape[1])
    info['theta'] = theta
    info['x'] = x
    
    fstand = 1*f
    for k in range(0, f.shape[1]):
        info['offset'][k] = np.mean(f[:, k])
        info['scale'][k] = 0.9*np.std(f[:, k]) + 0.1*np.std(f)
    fstand = (fstand - info['offset']) / info['scale']
    Vecs, Vals, _ = np.linalg.svd((fstand / np.sqrt(fstand.shape[0])).T)
    Vals = np.append(Vals, np.zeros(Vecs.shape[1] - Vals.shape[0]))
    Valssq = (fstand.shape[0]*(Vals ** 2) + 0.01) /\
        (fstand.shape[0] + 0.01)
    numVals = 1 + np.sum(np.cumsum(Valssq) < 0.9995*np.sum(Valssq))
    numVals = np.maximum(np.minimum(2,fstand.shape[1]),numVals)
    info['Cs'] = Vecs * np.sqrt(Valssq)
    info['PCs'] = info['Cs'][:, :numVals]
    info['PCsi'] = Vecs[:,:numVals] * np.sqrt(1 / Valssq[:numVals])
    pcaval = fstand @ info['PCsi']
    fhat= pcaval @ info['PCs'].T
    info['extravar'] = np.mean((fstand-fhat) ** 2,0) *\
        (info['scale'] ** 2)
        
    info['var0'] = np.ones(numVals)
    hypinds = np.zeros(numVals)
    emulist = [dict() for x in range(0, numVals)]
    
    for pcanum in range(0, numVals):
        if pcanum > 0.5:
            hypwhere = np.where(hypinds == np.array(range(0, numVals)))[0]
            emulist[pcanum] = emulation_smart_fit(theta,
                                                  pcaval[:, pcanum],
                                                  hypstarts[hypwhere,:],
                                                  hypwhere)
        else:
            emulist[pcanum] = emulation_smart_fit(theta,
                                                  pcaval[:, pcanum])
            hypstarts = np.zeros((numVals,
                                  emulist[pcanum]['hyp'].shape[0]))
        hypstarts[pcanum, :] = emulist[pcanum]['hyp']
        if emulist[pcanum]['hypind'] < -0.5:
            emulist[pcanum]['hypind'] = pcanum
        hypinds[pcanum] = emulist[pcanum]['hypind']
    info['emulist'] = emulist
    return

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

def predict(info, theta, x=None, args=None):
    infos = info['emulist']
    predvecs = np.zeros((theta.shape[0], len(infos)))
    predvars = np.zeros((theta.shape[0], len(infos)))
    
    if x is not None:
        matchingmatrix = np.ones((x.shape[0], info['x'].shape[0]))
        for k in range(0,x[0].shape[0]):
            try:
                matchingmatrix *= np.isclose(x[:,k][:,None].astype('float'),
                         info['x'][:,k].astype('float'))
            except:
                matchingmatrix *= np.equal(x[:,k][:,None],info['x'][:,k])
        xind = np.argwhere(matchingmatrix > 0.5)[:,1]
    else:
        xind = range(0,info['x'].shape[0])
    rsave = np.array(np.ones(len(infos)), dtype=object)
    for k in range(0, len(infos)):
        if infos[k]['hypind'] == k:
            rsave[k] = (1-infos[k]['nug']) *\
                emulation_smart_covmat(theta, info['theta'], 
                                       infos[k]['hypcov'])
        r = np.squeeze(rsave[infos[k]['hypind']])
        Rinv = 1*infos[(infos[k]['hypind'])]['Rinv']
        predvecs[:, k] = r @ infos[k]['pw']
        predvars[:, k] = info['var0'][k] - np.sum(r.T * (Rinv @ r.T), 0)

    predmean = (predvecs @ info['PCs'][xind,:].T)*info['scale'][xind] +\
        info['offset'][xind]
    predvar = info['extravar'][xind] + (predvars @ (info['PCs'][xind,:] ** 2).T) *\
        (info['scale'][xind] ** 2)
    
    preddict = {}
    preddict['mean'] = 1*predmean
    preddict['var'] = 1*predvar
    CH = (np.sqrt(np.abs(predvars))[:,:,np.newaxis] *
                             (info['PCs'][xind,:].T)[np.newaxis,:,:])
    preddict['covhalf'] = CH
    # CH = preddict['covhalf']
    # C = np.ones((CH.shape[0],CH.shape[2],CH.shape[2]))
    # for k in range(0,CH.shape[0]):
    #     C[k,:,:] = CH[k].T @ CH[k]
    # preddict['cov'] = C
    return preddict


def emulation_covmat(x1, x2, gammav, returndir = False):
    d = gammav.shape[0]
    x1 = x1.reshape(1,d) if x1.ndim < 1.5 else x1
    x2 = x2.reshape(1,d) if x2.ndim < 1.5 else x2
    
    V = np.zeros([x1.shape[0], x2.shape[0]])
    R = np.ones([x1.shape[0], x2.shape[0]])
    if returndir:
        dR = np.zeros([x1.shape[0], x2.shape[0],d])
    for k in range(0, d):
        S = np.abs(np.subtract.outer(x1[:,k],x2[:,k])/np.exp(gammav[k]))
        R *= (1 + S)
        V -= S
        if returndir:
            dR[:,:, k] = (S * S) / (1 + S)
    R *= np.exp(V)
    if returndir:
        for k in range(0, d):
            dR[:,:,k] = R * dR[:,:,k]
        return R, dR
    else:
        return R

def emulation_smart_fit(theta, pcaval, hypstarts=None, hypinds=None):
    """Return a fitted model from the emulator model using smart method."""
    subinfo = {}
    subinfo['hypregmean'] = np.append(0.5 + np.log(np.std(theta, 0)), (0, -10))
    subinfo['hypregLB'] = np.append(-1 + np.log(np.std(theta, 0)), (-10, -20))
    subinfo['hypregUB'] = np.append(3 + np.log(np.std(theta, 0)), (1, -4))
    subinfo['hypregstd'] = (subinfo['hypregUB'] - subinfo['hypregLB']) / 3
    subinfo['hypregstd'][-2] = 2
    subinfo['hypregstd'][-1] = 0.5
    subinfo['hyp'] = 1*subinfo['hypregmean']
    nhyptrain = np.min((20*theta.shape[1], theta.shape[0]))
    thetac = np.random.choice(theta.shape[0], nhyptrain, replace=False)
    subinfo['theta'] = theta[thetac, :]
    subinfo['f'] = pcaval[thetac]
    hypind0 = -1
    if hypstarts is not None:
        L0 = emulation_smart_negloglik(subinfo['hyp'], subinfo)
        for k in range(0, hypstarts.shape[0]):
            L1 = emulation_smart_negloglik(hypstarts[k, :], subinfo)
            if L1 < L0:
                subinfo['hyp'] = hypstarts[k, :]
                L0 = 1* L1
                hypind0 = hypinds[k]
    opval = spo.minimize(emulation_smart_negloglik,
                         1*subinfo['hyp'], args=(subinfo), method='L-BFGS-B',
                         options={'gtol': 0.5 / (subinfo['hypregUB'] -
                                                  subinfo['hypregLB'])},
                         jac=emulation_smart_negloglikgrad,
                         bounds=spo.Bounds(subinfo['hypregLB'],
                                           subinfo['hypregUB']))
    if hypind0 > -0.5 and 2 * (L0-opval.fun) < \
        (subinfo['hyp'].shape[0] + 3 * np.sqrt(subinfo['hyp'].shape[0])):
        subinfo['hypcov'] = subinfo['hyp'][:-1]
        subinfo['hypind'] = hypind0
        subinfo['nug'] = np.exp(subinfo['hyp'][-1])/(1+np.exp(subinfo['hyp'][-1]))
        R = emulation_smart_covmat(theta, theta, subinfo['hypcov'])
        R =  (1-subinfo['nug'])*R + subinfo['nug'] * np.eye(R.shape[0])
        W, V = np.linalg.eigh(R)
        Rinv = V @ np.diag(1/W) @ V.T
    else:
        subinfo['hyp'] = opval.x[:]
        subinfo['hypind'] = -1
        subinfo['hypcov'] = subinfo['hyp'][:-1]
        subinfo['nug'] = np.exp(subinfo['hyp'][-1])/(1+np.exp(subinfo['hyp'][-1]))
        R = emulation_smart_covmat(theta, theta, subinfo['hypcov'])
        subinfo['R'] =  (1-subinfo['nug'])*R + subinfo['nug'] * np.eye(R.shape[0])
        W, V = np.linalg.eigh(subinfo['R'])
        subinfo['Rinv'] = V @ np.diag(1/W) @ V.T
        Rinv = subinfo['Rinv']
    subinfo['pw'] = Rinv @ pcaval
    return subinfo


def emulation_smart_covmat(x1, x2, gammav, returndir=False):
    """Return the covariance between x1 and x2 given parameter gammav."""
    x1 = 1*x1.reshape(1, gammav.shape[0]-1) if x1.ndim < 1.5 else x1
    x2 = 1*x2.reshape(1, gammav.shape[0]-1) if x2.ndim < 1.5 else x2
    V = np.zeros([x1.shape[0], x2.shape[0]])
    R = np.ones([x1.shape[0], x2.shape[0]])
    x1 = x1/np.exp(gammav[:-1])
    x2 = x2/np.exp(gammav[:-1])
    if returndir:
        dR = np.zeros([x1.shape[0], x2.shape[0], gammav.shape[0]])
    for k in range(0, gammav.shape[0]-1):
        S = np.abs(np.subtract.outer(x1[:, k], x2[:, k]))
        R *= (1 + S)
        V -= S
        if returndir:
            dR[:, :, k] = (S ** 2) / (1 + S)
    R *= np.exp(V)
    RT = R * 1/(1+np.exp(gammav[-1])) + np.exp(gammav[-1])/(1+np.exp(gammav[-1]))
    if returndir:
        dR = R[:, :, None] * dR * 1/(1+np.exp(gammav[-1]))
        dR[:, :, -1] = np.exp(gammav[-1]) / ((1+np.exp(gammav[-1])) ** 2) *\
            (1-R)
    if returndir:
        return RT, dR
    else:
        return RT

def emulation_smart_negloglik(hyp, info):
    """Return penalized log likelihood of single demensional GP model."""
    R0 = emulation_smart_covmat(info['theta'], info['theta'], hyp[:-1])
    nug = np.exp(hyp[-1])/(1+np.exp(hyp[-1]))
    R = (1-nug)* R0 + nug * np.eye(info['theta'].shape[0])
    W, V = np.linalg.eigh(R)
    Vh = V / np.sqrt(np.abs(W))
    fcenter = Vh.T @ info['f']
    negloglik = 1/2 * np.sum(np.log(np.abs(W))) +1/2 * np.sum(fcenter ** 2)
    negloglik += 0.5*np.sum(((hyp-info['hypregmean']) ** 2) /
                            (info['hypregstd'] ** 2))
    return negloglik


def emulation_smart_negloglikgrad(hyp, info):
    """Return gradient of the penalized log likelihood of single demensional GP model."""
    R0, dR = emulation_smart_covmat(info['theta'], info['theta'], hyp[:-1], True)
    nug = np.exp(hyp[-1])/(1+np.exp(hyp[-1]))
    R = (1-nug)* R0 + nug * np.eye(info['theta'].shape[0])
    dR = (1-nug) * dR
    dRappend = nug/((1+np.exp(hyp[-1]))) *\
        (-R0+np.eye(info['theta'].shape[0]))
    dR = np.append(dR, dRappend[:,:,None], axis=2)
    W, V = np.linalg.eigh(R)
    Vh = V / np.sqrt(np.abs(W))
    fcenter = Vh.T @ info['f']
    dnegloglik = np.zeros(dR.shape[2])
    Rinv = Vh @ (np.eye(Vh.shape[0]) - np.multiply.outer(fcenter, fcenter)) @ Vh.T
    for k in range(0, dR.shape[2]):
        dnegloglik[k] = 0.5*np.sum(Rinv * dR[:, :, k])
    dnegloglik += (hyp-info['hypregmean'])/(info['hypregstd'] ** 2)
    return dnegloglik
