# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 14:40:38 2020

@author: Plumlee
"""

import numpy as np
import scipy.optimize as spo

def build(theta, f, x=None,  options=None):    
    nummodel = f.shape[1]
    
    nhyptrain = np.min((200,theta.shape[0]))
    
    modellist = [dict() for x in range(0,nummodel)]
    
    for pcanum in range(0,nummodel):
        covhyp0 = np.log(np.std(theta,0)*3)+1
        covhypLB = covhyp0-2
        covhypUB = covhyp0+3
        
        nughyp0 = -4
        nughypLB = -8
        nughypUB = 1
    
        emuinfo = {}
        emuinfo['theta'] =  1*theta[:nhyptrain,:]
        emuinfo['f'] =  1*f[:nhyptrain,pcanum]
        emuinfo['n'] =  emuinfo['f'].shape[0]
        emuinfo['p'] =  covhyp0.shape[0]
        
        emuinfo['hypregmean'] =  np.append(covhyp0,nughyp0)
        emuinfo['hypregstd'] =  np.append((covhypUB-covhypLB)/3,1)
        
        L1 = emulation_negloglik(np.append(covhyp0+1,nughyp0), emuinfo)
        dL1 = emulation_negloglikgrad(np.append(covhyp0+1,nughyp0), emuinfo)
        #check our derivative
        # for k in range(0,theta.shape[1]):
        #     covhyp = 1*covhyp0+1
        #     nughyp = 1*nughyp0
        #     if k < theta.shape[1]:
        #         covhyp[k] = 1*covhyp[k] + 10 **(-4)
        #     else:
        #         nughyp = 1*nughyp + 10 **(-4)
        #     L2 = emulation_negloglik(np.append(covhyp,nughyp), emuinfo)
        #     print((L2-L1) * (10 **(4)))
        #     print(dL1[k])
        
        bounds = spo.Bounds(np.append(covhypLB,nughypLB), np.append(covhypUB,nughypUB))
        #print('was here')
        #print(emulation_negloglik(np.append(covhyp0,nughyp0),emuinfo))
        opval = spo.minimize(emulation_negloglik, np.append(covhyp0,nughyp0),
                             args=(emuinfo),
                             method='L-BFGS-B',
                             options={'disp': True},
                             jac=emulation_negloglikgrad)
        #print(opval)
        hypcov = opval.x[:emuinfo['p']]
        hypnug = opval.x[emuinfo['p']]
        #print('went here')
        #print(emulation_negloglik(np.append(hypcov,hypnug),emuinfo))
        n = theta.shape[0]
        R = emulation_covmat(theta, theta, hypcov)
        R = R + np.exp(hypnug)*np.diag(np.ones(R.shape[0]))
        W, V = np.linalg.eigh(R)
        fspin = V.T @ f[:,pcanum]
        onespin = V.T @ np.ones(f[:,pcanum].shape)
        
        muhat = np.sum(V @ (1/W * fspin)) / np.sum(V @ (1/W * onespin))
        fcenter = fspin - muhat * onespin
        sigma2hat = np.mean((fcenter) ** 2 / W)
        Rinv = V @ np.diag(1/W) @ V.T
        
        modellist[pcanum]['hypcov'] = 1*hypcov
        modellist[pcanum]['hypnug'] = 1*hypnug
        modellist[pcanum]['R'] = 1*R
        modellist[pcanum]['Rinv'] = 1*Rinv
        modellist[pcanum]['pw'] = 1*Rinv @ (f[:,pcanum] - muhat)
        modellist[pcanum]['muhat'] = 1*muhat
        modellist[pcanum]['sigma2hat'] = 1*sigma2hat
        modellist[pcanum]['theta'] = 1*theta
    model = modellist
    return model


def predict(emumodel, thetanew,  options=None):
    predvec = np.zeros((thetanew.shape[0],len(emumodel)))
    predvar = np.zeros((thetanew.shape[0],len(emumodel)))
    for k in range(0,len(emumodel)):
        r = emulation_covmat(thetanew, emumodel[k]['theta'], emumodel[k]['hypcov'])
        predvec[:,k] = emumodel[k]['muhat'] + r @ emumodel[k]['pw']
        predvar[:,k] = emumodel[k]['sigma2hat'] * (1 + np.exp(emumodel[k]['hypnug']) - np.sum(r.T * (emumodel[k]['Rinv'] @ r.T),0))
    return predvec, predvar


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

def emulation_negloglik(hyp, emuinfo):
    covhyp = hyp[0:emuinfo['p']]
    nughyp = hyp[emuinfo['p']]
    
    R = emulation_covmat(emuinfo['theta'], emuinfo['theta'], covhyp)
    R = R + np.exp(nughyp)*np.diag(np.ones(emuinfo['n']))
    W, V = np.linalg.eigh(R)
    
    fspin = V.T @ emuinfo['f']
    onespin = V.T @ np.ones(emuinfo['f'].shape)
    
    muhat = np.sum(V @ (1/W * fspin)) / np.sum(V @ (1/W * onespin))
    fcenter = fspin - muhat * onespin
    sigma2hat = np.mean((fcenter) ** 2 / W)
    
    negloglik = 1/2 * np.sum(np.log(W)) + emuinfo['n']/2 * np.log(sigma2hat)
    negloglik += 1/2* np.sum((hyp - emuinfo['hypregmean']) **2 /(emuinfo['hypregstd'] ** 2))
    return 1*negloglik

def emulation_negloglikgrad(hyp, emuinfo):
    covhyp = hyp[0:emuinfo['p']]
    nughyp = hyp[emuinfo['p']]
    
    R, dR = emulation_covmat(emuinfo['theta'], emuinfo['theta'], covhyp, True)
    R = R + np.exp(nughyp)*np.diag(np.ones(emuinfo['n']))
    dRappend = np.exp(nughyp)*np.diag(np.ones(emuinfo['n'])).reshape(R.shape[0],R.shape[1],1)
    dR = np.append(dR, dRappend, axis=2)
    W, V = np.linalg.eigh(R)
    
    fspin = V.T @ emuinfo['f']
    onespin = V.T @ np.ones(emuinfo['f'].shape)
    
    mudenom = np.sum(V @ (1/W * onespin))
    munum = np.sum(V @ (1/W * fspin))
    muhat = munum / mudenom
    fcenter = fspin - muhat * onespin
    sigma2hat = np.mean((fcenter) ** 2 / W)
    
    negloglik = emuinfo['n']/2 * np.log(sigma2hat) + 1/2 * np.sum(np.log(W))
    
    dmuhat = np.zeros(emuinfo['p']+1)
    dsigma2hat = np.zeros(emuinfo['p']+1)
    dfcentercalc = (fcenter / W) @ V.T
    dfspincalc = (fspin / W) @ V.T
    donespincalc = (onespin / W) @ V.T
    Rinv = V @ np.diag(1/W) @ V.T
    dlogdet = np.zeros(emuinfo['p']+1)
    for k in range(0, dR.shape[2]):
        dRnorm =  np.squeeze(dR[:,:,k])
        dmuhat[k] = -np.sum(donespincalc @ dRnorm @ dfspincalc) / mudenom + muhat * (np.sum(donespincalc @ dRnorm @ donespincalc)/mudenom)
        dsigma2hat[k] = -1/emuinfo['n'] * (dfcentercalc.T @ dRnorm @ dfcentercalc) + 2*dmuhat[k] * np.mean((fcenter * onespin) / W)
        dlogdet[k] = np.sum(Rinv * dRnorm)
        
    dnegloglik = 1/2 * dlogdet + emuinfo['n']/2 * 1/sigma2hat * dsigma2hat
    dnegloglik += ((hyp - emuinfo['hypregmean'])  /(emuinfo['hypregstd'] ** 2))
    return 1*dnegloglik


