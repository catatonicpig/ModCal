# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 14:40:38 2020

@author: Plumlee
"""

import numpy as np
import scipy.optimize as spo

def build(theta, f, x=None,  options=None):
    """Return a Gaussian Process emulator model."""
    emuinfo = {}
    emuinfo['offset'] = np.zeros(f.shape[1])
    emuinfo['scale'] = np.ones(f.shape[1])
    emuinfo['theta'] = theta
    
    fstand = 1*f
    for k in range(0, f.shape[1]):
        emuinfo['offset'][k] = np.mean(f[:, k])
        emuinfo['scale'][k] = 0.9*np.std(f[:, k]) + 0.1*np.std(f)
    fstand = (fstand - emuinfo['offset']) / emuinfo['scale']
    Vecs, Vals, _ = np.linalg.svd((fstand / np.sqrt(fstand.shape[0])).T)
    Vals = np.append(Vals, np.zeros(Vecs.shape[1] - Vals.shape[0]))
    Valssq = (fstand.shape[0]*(Vals ** 2) + 0.01) /\
        (fstand.shape[0] + 0.01)
    numVals = 1 + np.sum(np.cumsum(Valssq) < 0.9995*np.sum(Valssq))
    numVals = np.maximum(np.minimum(2,fstand.shape[1]),numVals)
    emuinfo['Cs'] = Vecs * np.sqrt(Valssq)
    emuinfo['PCs'] = emuinfo['Cs'][:, :numVals]
    emuinfo['PCsi'] = Vecs[:,:numVals] * np.sqrt(1 / Valssq[:numVals])
    pcaval = fstand @ emuinfo['PCsi']
    fhat= pcaval @ emuinfo['PCs'].T
    emuinfo['extravar'] = np.mean((fstand-fhat) ** 2,0) *\
        (emuinfo['scale'] ** 2)
        
    emuinfo['var0'] = np.ones(numVals)
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
    emuinfo['emulist'] = emulist
    return emuinfo


def predict(emumodel, theta,  options=None):
    emumodels = emumodel['emulist']
    predvecs = np.zeros((theta.shape[0], len(emumodels)))
    predvars = np.zeros((theta.shape[0], len(emumodels)))
    rsave = np.array(np.ones(len(emumodels)), dtype=object)
    for k in range(0, len(emumodels)):
        if emumodels[k]['hypind'] == k:
            rsave[k] = (1-emumodels[k]['nug']) *\
                emulation_smart_covmat(theta, emumodel['theta'], 
                                       emumodels[k]['hypcov'])
        r = np.squeeze(rsave[emumodels[k]['hypind']])
        Rinv = 1*emumodels[(emumodels[k]['hypind'])]['Rinv']
        predvecs[:, k] = r @ emumodels[k]['pw']
        predvars[:, k] = emumodel['var0'][k] - np.sum(r.T * (Rinv @ r.T), 0)

    predmean = (predvecs @ emumodel['PCs'].T)*emumodel['scale'] + emumodel['offset']
    predvar = emumodel['extravar'] + (predvars @ (emumodel['PCs'] ** 2).T) *\
        (emumodel['scale'] ** 2)
    
    preddict = {}
    preddict['mean'] = predmean
    preddict['var'] = predvar
    preddict['covdecomp'] = (np.sqrt(np.abs(predvars))[:,:,np.newaxis] *
                             (emumodel['PCs'].T)[np.newaxis,:,:])
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
    subemuinfo = {}
    subemuinfo['hypregmean'] = np.append(0.5 + np.log(np.std(theta, 0)), (0, -10))
    subemuinfo['hypregLB'] = np.append(-1 + np.log(np.std(theta, 0)), (-10, -20))
    subemuinfo['hypregUB'] = np.append(3 + np.log(np.std(theta, 0)), (1, -4))
    subemuinfo['hypregstd'] = (subemuinfo['hypregUB'] - subemuinfo['hypregLB']) / 3
    subemuinfo['hypregstd'][-2] = 2
    subemuinfo['hypregstd'][-1] = 0.5
    subemuinfo['hyp'] = 1*subemuinfo['hypregmean']
    nhyptrain = np.min((20*theta.shape[1], theta.shape[0]))
    thetac = np.random.choice(theta.shape[0], nhyptrain, replace=False)
    subemuinfo['theta'] = theta[thetac, :]
    subemuinfo['f'] = pcaval[thetac]
    hypind0 = -1
    if hypstarts is not None:
        L0 = emulation_smart_negloglik(subemuinfo['hyp'], subemuinfo)
        for k in range(0, hypstarts.shape[0]):
            L1 = emulation_smart_negloglik(hypstarts[k, :], subemuinfo)
            if L1 < L0:
                subemuinfo['hyp'] = hypstarts[k, :]
                L0 = 1* L1
                hypind0 = hypinds[k]
    opval = spo.minimize(emulation_smart_negloglik,
                         1*subemuinfo['hyp'], args=(subemuinfo), method='L-BFGS-B',
                         options={'gtol': 0.5 / (subemuinfo['hypregUB'] -
                                                  subemuinfo['hypregLB'])},
                         jac=emulation_smart_negloglikgrad,
                         bounds=spo.Bounds(subemuinfo['hypregLB'],
                                           subemuinfo['hypregUB']))
    if hypind0 > -0.5 and 2 * (L0-opval.fun) < \
        (subemuinfo['hyp'].shape[0] + 3 * np.sqrt(subemuinfo['hyp'].shape[0])):
        subemuinfo['hypcov'] = subemuinfo['hyp'][:-1]
        subemuinfo['hypind'] = hypind0
        subemuinfo['nug'] = np.exp(subemuinfo['hyp'][-1])/(1+np.exp(subemuinfo['hyp'][-1]))
        R = emulation_smart_covmat(theta, theta, subemuinfo['hypcov'])
        R =  (1-subemuinfo['nug'])*R + subemuinfo['nug'] * np.eye(R.shape[0])
        W, V = np.linalg.eigh(R)
        Rinv = V @ np.diag(1/W) @ V.T
    else:
        subemuinfo['hyp'] = opval.x[:]
        subemuinfo['hypind'] = -1
        subemuinfo['hypcov'] = subemuinfo['hyp'][:-1]
        subemuinfo['nug'] = np.exp(subemuinfo['hyp'][-1])/(1+np.exp(subemuinfo['hyp'][-1]))
        R = emulation_smart_covmat(theta, theta, subemuinfo['hypcov'])
        subemuinfo['R'] =  (1-subemuinfo['nug'])*R + subemuinfo['nug'] * np.eye(R.shape[0])
        W, V = np.linalg.eigh(subemuinfo['R'])
        subemuinfo['Rinv'] = V @ np.diag(1/W) @ V.T
        Rinv = subemuinfo['Rinv']
    subemuinfo['pw'] = Rinv @ pcaval
    return subemuinfo


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

def emulation_smart_negloglik(hyp, emuinfo):
    """Return penalized log likelihood of single demensional GP model."""
    R0 = emulation_smart_covmat(emuinfo['theta'], emuinfo['theta'], hyp[:-1])
    nug = np.exp(hyp[-1])/(1+np.exp(hyp[-1]))
    R = (1-nug)* R0 + nug * np.eye(emuinfo['theta'].shape[0])
    W, V = np.linalg.eigh(R)
    Vh = V / np.sqrt(np.abs(W))
    fcenter = Vh.T @ emuinfo['f']
    negloglik = 1/2 * np.sum(np.log(np.abs(W))) +1/2 * np.sum(fcenter ** 2)
    negloglik += 0.5*np.sum(((hyp-emuinfo['hypregmean']) ** 2) /
                            (emuinfo['hypregstd'] ** 2))
    return negloglik


def emulation_smart_negloglikgrad(hyp, emuinfo):
    """Return gradient of the penalized log likelihood of single demensional GP model."""
    R0, dR = emulation_smart_covmat(emuinfo['theta'], emuinfo['theta'], hyp[:-1], True)
    nug = np.exp(hyp[-1])/(1+np.exp(hyp[-1]))
    R = (1-nug)* R0 + nug * np.eye(emuinfo['theta'].shape[0])
    dR = (1-nug) * dR
    dRappend = nug/((1+np.exp(hyp[-1]))) *\
        (-R0+np.eye(emuinfo['theta'].shape[0]))
    dR = np.append(dR, dRappend[:,:,None], axis=2)
    W, V = np.linalg.eigh(R)
    Vh = V / np.sqrt(np.abs(W))
    fcenter = Vh.T @ emuinfo['f']
    dnegloglik = np.zeros(dR.shape[2])
    Rinv = Vh @ (np.eye(Vh.shape[0]) - np.multiply.outer(fcenter, fcenter)) @ Vh.T
    for k in range(0, dR.shape[2]):
        dnegloglik[k] = 0.5*np.sum(Rinv * dR[:, :, k])
    dnegloglik += (hyp-emuinfo['hypregmean'])/(emuinfo['hypregstd'] ** 2)
    return dnegloglik

