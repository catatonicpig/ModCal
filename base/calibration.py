# -*- coding: utf-8 -*-
"""Header here."""
import numpy as np
import importlib
import base.emulation
import copy
from base.utilities import postsampler



class calibrator(object):
    """Calibrator."""

    def __init__(self, emu=None,
                 y=None, x=None,
                 thetaprior=None,
                 phiprior=None,
                 passoptions={},
                 options=None,
                 software='plumleecali'):
        
        if y is None:
            raise ValueError('You have not provided any y.')
        
        if emu is None:
            raise ValueError('You have not provided any emulator.')
        
        if thetaprior is None:
            print('You have not provided any prior function, stopping...')
            
        if type(emu) is tuple:
            print('You have provided a tuple of emulators')
            for k in range(0,len(emu)):
                try:
                    ftry = emu[k].predict(emu[k].theta[0,:])['mean']
                except:
                    raise ValueError('Your provided emulator failed to predict.')
                emu[k].modelnum = k
            #WIP: need to ensure they have the same x's (same thetas not required)
            self.emu0 = emu[-1]
            self.modelcount = len(emu)
        else:
            try:
                ftry = emu.predict(emu.theta[0,:])['mean']
            except:
                raise ValueError('Your provided emulator failed to predict.')
            emu.modelnum = 0
            self.emu0 = emu
            self.modelcount = 1
            
        self.emu = emu
        self.y = y        
        if x is None and (y.shape[0] != ftry.shape[0]):
            raise ValueError('If x is not provided, predictions must ' +
                             'align with y and emu.predict()')
        
        if x is not None and (x.shape[0] != y.shape[0]):
            raise ValueError('If x is provided, predictions must align with y and emu.predict()')
        elif x is not None:
            matchingvec = np.where(((x[:, None] > self.emu0.x-10**(-8)) *
                                    (x[:, None] < self.emu0.x+10**(-8))).all(2))
            xind = matchingvec[1][matchingvec[0]]
            if xind.shape[0] < x.shape[0]:
                raise ValueError('If x is provided, it must be a subset of emu.x')
            self.xind = xind
            self.x = x
        else:
            self.xind = range(0, x.shape[0])
            self.x = emu0.x
        
        try:
            self.calsoftware = importlib.import_module('base.calibrationsubfuncs.' + software)
        except:
            raise ValueError('Module not found!')
        
        self.calsoftware.loglik
        self.calsoftware.predict
        self.passoptions = passoptions
        
        if phiprior is None:
            class phiprior:
                def logpdf(phi):
                    return 0
                def rvs(n):
                    return None
            self.phiprior = phiprior
        else:
            self.phiprior = phiprior
        
        self.thetaprior = thetaprior
        if phiprior.rvs(1) is not None:
            self.thetaphidraw = postsampler(np.hstack((self.thetaprior.rvs(1000),
                                                       self.phiprior.rvs(1000))),
                                            self.logpostfull)
            self.thetadraw = self.thetaphidraw[:,:self.emu0.theta.shape[1]]
            self.phidraw = self.thetaphidraw[:,(self.emu0.theta.shape[1]):]
        else:
            self.thetadraw = postsampler(self.thetaprior.rvs(1000), self.logpostfull)
            self.phidraw = None
        
        if self.modelcount > 1.5:
            if passoptions is None:
                passoptions = self.passoptions
            L0 = self.logprior(self.thetadraw , self.phidraw) 
            La = np.tile(L0.reshape((-1,1)), (1,self.modelcount))
            for k in range(0, self.modelcount):
                if self.phidraw is not None:
                    La[:,k] += self.calsoftware.loglik(self.emu[k], 
                                                   self.thetadraw , self.phidraw, 
                                                   self.y, self.xind, k,
                                                   passoptions)
                else:
                    La[:,k] += self.calsoftware.loglik(self.emu[k], 
                                                   self.thetadraw, None, 
                                                   self.y, self.xind, k,
                                                   passoptions)
            Lm = np.max(La, 1)
            G = np.sum(np.exp(La-Lm[:,None]),1)
            L0 = np.log(G) + Lm
            self.probdraw = np.exp(La - L0[:,None])
    def logprior(self, theta, phi):
        return (self.thetaprior.logpdf(copy.deepcopy(theta)) +
                self.phiprior.logpdf(copy.deepcopy(phi)))
    
    def logpost(self, theta, phi, passoptions=None):
        if passoptions is None:
            passoptions = self.passoptions
        L0 = self.logprior(theta, phi) 
        inds = np.where(np.isfinite(L0))[0]
        if self.modelcount < 1.5:
            if phi is not None:
                L0[inds] += self.calsoftware.loglik(self.emu0, 
                                               theta[inds,:], phi[inds,:], 
                                               self.y, self.xind, 0,
                                               passoptions)
            else:
                L0[inds] += self.calsoftware.loglik(self.emu0, 
                                               theta[inds,:], None, 
                                               self.y, self.xind, 0,
                                               passoptions)
        else:
            La = np.tile(L0.reshape((-1,1)), (1,self.modelcount))
            for k in range(0, self.modelcount):
                if phi is not None:
                    La[inds,k] += self.calsoftware.loglik(self.emu[k], 
                                                   theta[inds,:], phi[inds,:], 
                                                   self.y, self.xind, k,
                                                   passoptions)
                else:
                    La[inds,k] += self.calsoftware.loglik(self.emu[k], 
                                                   theta[inds,:], None, 
                                                   self.y, self.xind, k,
                                                   passoptions)
            Lm = np.max(La[inds], 1)
            G = np.exp(La[inds,:]-Lm[:,None])
            L0[inds] = np.log(np.sum(G,1)) + Lm
        return L0
    
    def logpostfull(self, thetaphi, passoptions=None):
        if self.phiprior.rvs(1) is not None:
            theta = thetaphi[:,:(self.emu0.theta.shape[1])]
            phi = thetaphi[:,(self.emu0.theta.shape[1]):]
        else:
            theta = thetaphi
            phi = None
        if passoptions is None:
            passoptions = self.passoptions
        return self.logpost(theta, phi, passoptions)
    
    def predict(self, x, theta = None, phi = None, passoptions=None):
        matchingvec = np.where(((x[:, None] > self.emu0.x-10**(-8)) *
                                (x[:, None] <  self.emu0.x+10**(-8))).all(2))
        xind = matchingvec[1][matchingvec[0]]
        if theta is None:
            theta = self.thetadraw
            phi = self.phidraw
            if self.modelcount > 1.5:
                probdraw = self.probdraw
        if passoptions is None:
            passoptions = self.passoptions
        if self.modelcount < 1.5:
            return (self.calsoftware.predict(xind, self.emu, theta, phi, 
                                             self.y, self.xind, 0, passoptions))
        else:
            preddict = {}
            emupreddict = self.calsoftware.predict(xind, self.emu[0], theta, 
                                                   phi, self.y, self.xind, 
                                                   0, passoptions)
            meancalcvec = (probdraw[:,0])[:,None] * emupreddict['meanfull']
            varcalcvec = (probdraw[:,0])[:,None]  * (
                    emupreddict['meanfull'] ** 2 + emupreddict['varfull'])
            for k in range(1, self.modelcount):
                emupreddict = self.calsoftware.predict(xind, self.emu[k], theta, 
                                                   phi, self.y, self.xind, 
                                                   k, passoptions)
                meancalcvec += (probdraw[:,k])[:,None] * emupreddict['meanfull']
                varcalcvec += (probdraw[:,k])[:,None]  * (
                    emupreddict['meanfull'] ** 2 + emupreddict['varfull'])
            preddict['mean'] = np.mean(meancalcvec,0)
            preddict['var'] = np.mean(varcalcvec- meancalcvec ** 2,0) +\
                        np.var(meancalcvec,0)
            return preddict
             