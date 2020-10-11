# -*- coding: utf-8 -*-
"""Header here."""
import numpy as np
import importlib
import copy

class emulator(object):
    """Emulator."""

    def __init__(self, theta=None, f=None, x=None, software='plumleeemu',
                 buildfirst=True, testpred=True):
        if f is None or theta is None:
            raise ValueError('You have no provided any theta and/or f.')
        
        if f.ndim < 0.5 or f.ndim > 2.5:
            raise ValueError('f must have either 1 or 2 demensions.')
            
        if theta.ndim < 0.5 or theta.ndim > 2.5:
            raise ValueError('theta must have either 1 or 2 demensions.')
        
        if theta.shape[0] < 2 * theta.shape[1]:
            raise ValueError('theta should have at least 2 more' +
                             'rows than columns.')
            
        if f.shape[0] is not theta.shape[0]:
            raise ValueError('The rows in f must match' +
                             ' the rows in theta')
            
        if f.ndim == 1 and x is not None:
            raise ValueError('Cannot use x if f has a single column.')
        
        if f.ndim == 2 and x is not None and\
            f.shape[1] is not x.shape[0]:
            raise ValueError('The columns in f must match' +
                             ' the rows in x')
        
        self.theta = theta
        self.f = f
        self.x = x
        
        try:
            emusoftware = importlib.import_module('statfuncs.emulationsubfuncs.' + software)
        except:
            raise ValueError('Module not found!')
            
            
        if "build" not in dir(emusoftware):
            raise ValueError('Function \"build\" not found in module!')
        
        self.builder = emusoftware.build
            
        if buildfirst:
            self.build()
            self.modelbuilt = True
            
        if "predict" not in dir(emusoftware):
            raise ValueError('Function \"predict\" not found in module!')
            
        self.predictor = emusoftware.predict
        
        if testpred:
            if self.modelbuilt is False:
                print('Cannot test prediction because \"buildfirst\" is False.')
            else:
                self.predictor(self.model,
                               copy.deepcopy(self.theta),
                               options=None)
        
    def build(self, options=None):
        self.model = self.builder(copy.deepcopy(self.theta),
                                  copy.deepcopy(self.f), 
                                  copy.deepcopy(self.x),
                                  copy.deepcopy(options))
        
    def predict(self, theta, options=None):
        return self.predictor(self.model,
                              copy.deepcopy(theta),
                              copy.deepcopy(options))

def loglik(emumodel, theta, y=None, S=None):
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
    S : Observation Covariance Matrix
        A matrix of the same length as x with observations. 'None' is equivlent to the
        identity matrix.

    Returns
    -------
    post: vector of unnormlaized log posterior
    """
    if theta.ndim == 1:
        theta = theta.reshape((1,theta.shape[0]))
    
    loglik = emulation_smart_loglik(semumodel, theta)
    
    logpost = loglik
    
    return logpost