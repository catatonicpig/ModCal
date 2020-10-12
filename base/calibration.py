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
                 passoptions=None,
                 options=None,
                 software='plumleecali'):
        
        if y is None:
            raise ValueError('You have not provided any y.')
        
        if emu is None:
            raise ValueError('You have not provided any emulator.')
        
        if thetaprior is None:
            print('You have not provided any prior function, stopping...')
        
        try:
            ftry = emu.predict(emu.theta[0,:])['mean']
        except:
            raise ValueError('Your provided emulator failed to predict.')
        
        self.y = y
        self.emu = emu
        
        if x is None and (y.shape[0] != ftry.shape[0]):
            raise ValueError('If x is not provided, predictions must ' +
                             'align with y and emu.predict()')
        
        if x is not None and (x.shape[0] != y.shape[0]):
            raise ValueError('If x is provided, predictions must align with y and emu.predict()')
        elif x is not None:
            matchingvec = np.where(((x[:, None] > emu.x-10**(-8)) *
                                    (x[:, None] < emu.x+10**(-8))).all(2))
            xind = matchingvec[1][matchingvec[0]]
            if xind.shape[0] < x.shape[0]:
                raise ValueError('If x is provided, it must be a subset of emu.x')
            self.xind = xind
            self.x = x
        else:
            self.xind = range(0, x.shape[0])
            self.x = emu.x
        
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
            self.thetadraw = self.thetaphidraw[:,:self.emu.theta.shape[1]]
            self.phidraw = self.thetaphidraw[:,(self.emu.theta.shape[1]):]
        else:
            self.thetadraw = postsampler(self.thetaprior.rvs(1000), self.logpostfull)
            self.phidraw = None
    
    def logprior(self, theta, phi):
        return (self.thetaprior.logpdf(copy.deepcopy(theta)) +
                self.phiprior.logpdf(copy.deepcopy(phi)))
    
    def logpost(self, theta, phi, passoptions=None):
        if passoptions is None:
            passoptions = self.passoptions
        return (self.logprior(theta, phi) +
            self.calsoftware.loglik(self.emu, theta, phi, self.y, self.xind, passoptions))
    
    def logpostfull(self, thetaphi, passoptions=None):
        if self.phiprior.rvs(1) is not None:
            theta = thetaphi[:,:(self.emu.theta.shape[1])]
            phi = thetaphi[:,(self.emu.theta.shape[1]):]
        else:
            theta = thetaphi
            phi = None
        if passoptions is None:
            passoptions = self.passoptions
        return self.logpost(theta, phi, passoptions)
    
    def predict(self, x, theta = None, phi = None, passoptions=None):
        matchingvec = np.where(((x[:, None] > self.emu.x-10**(-8)) *
                                (x[:, None] <  self.emu.x+10**(-8))).all(2))
        xind = matchingvec[1][matchingvec[0]]
        if theta is None:
            theta = self.thetadraw
            phi = self.phidraw
        if passoptions is None:
            passoptions = self.passoptions
        return (self.calsoftware.predict(xind, self.emu, theta, phi, self.y, self.xind, passoptions))