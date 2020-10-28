# -*- coding: utf-8 -*-
"""Header here."""
import numpy as np
import importlib
import copy


class emulator(object):
    """A class used to represent an emulator or surrogate model."""

    def __init__(self, theta=None, f=None, x=None, software='PCGP', args={}):
        r"""
        Intitalizes an emulator or surrogate.

        It directly calls "emulationsubfuncs.[software]" where [software] is
        the user option with default listed above. If you would like to change this software, just
        drop a new file in the "emulationsubfuncs" folder with the required formatting.

        Parameters
        ----------
        theta : array of float
            An n-by-d matrix of parameters. n should be at least 2 times m. Each row in theta should
            correspond to a row in f.
        f : array of float
            An n-by-m matrix of responses with 'nan' representing responses not yet available. Each
            row in f should correspond to a row in theta. Each column should correspond to a row in
            x.
        x : array of objects
            An m-by-p matrix of inputs. Each column should correspond to a row in f.
        software : str
            A string that points to the file located in "emulationsubfuncs" you would
            like to use.
        args : dict
            Optional dictionary containing options you would like to pass to
            [software].fit(theta, f, x, args)
            or
            [software].predict(theta, args)

        Returns
        -------
        emu : instance of emulation class
            An instance of the emulation class that can be used with the functions listed below.
        """
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
        self.args = args
        try:
            self.software = importlib.import_module('base.emulationsubfuncs.' + software)
        except:
            raise ValueError('Module not found!')
            
            
        if "fit" not in dir(self.software):
            raise ValueError('Function \"fit\" not found in module!')
        
        if "predict" not in dir(self.software):
            raise ValueError('Function \"predict\" not found in module!')
        
        self.info = {}
        self.fit()
        
        self.software.predict(self.info, copy.deepcopy(self.theta))


    def fit(self, args= None):
        r"""
        Fits an emulator or surrogate.
        
        Calls
        emu.info = [software].fit(emu.theta, emu.f, emu.x, args = args)

        Parameters
        ----------
        emu.theta : array of float
            An n-by-d matrix of parameters. n should be at least 2 times m. Each row in theta should
            correspond to a row in f.
        emu.f : array of float
            An n-by-m matrix of responses with 'nan' representing responses not yet available. Each
            row in f should correspond to a row in theta. Each column should correspond to a row in
            x.
        emu.x : array of objects
            An m-by-p matrix of inputs. Each column should correspond to a row in f.
        args : dict
            Optional dictionary containing options you would like to pass to
            [software].fit(theta, phi, args)
            Defaults to the one used to build emu.

        Returns
        -------
        emu.info : dict
            An arbitrary dictionary that can be used with [software].pred
        """
        if args is None:
            args = self.args
        self.software.fit(self.info, copy.deepcopy(self.theta), copy.deepcopy(self.f), 
                                      copy.deepcopy(self.x), args = args)


    def predict(self, theta, x=None, args=None):
        r"""
        Fits an emulator or surrogate.
        
        Calls
        preddict = [software].predict(emu.theta, args = args)

        Parameters
        ----------
        theta : array of float
            An n'-by-d array of parameters.
        x : array or list of inputs of interest
            An m'-by-p array of objects (optional). Defaults to emu.x
        args : dict
            A dictionary containing options you would like to pass to
            [software].fit(theta, phi, args). 
            Defaults to the one used to build emu.

        Returns
        -------
        preddict : dict of prediction objects
            preddict['mean'] : Mean of prediction at each point theta and emu.x
            preddict['var'] : Variance of prediction at each point theta and emu.x
            preddict['covdecomp'] : A matrix such that when A = preddict['covdecomp'][k,:,:]
                then A.T @ A = covariance matrix of prediction errors at theta[k,:] and emu.x
        """
        if args is None:
            args = self.args
        if theta.ndim < 1.5:
            theta = theta.reshape([-1, theta.shape[0]])
        if theta[0].shape[0] is not self.theta[0].shape[0]:
            raise ValueError('The new parameters do not match old parameters.')
            
        if x is None:
            x = copy.deepcopy(self.x)
        else:
            if x[0].shape[0] is not self.x[0].shape[0]:
                raise ValueError('The new inputs do not match old inputs.')
        
        info = self.software.predict(self.info, copy.deepcopy(theta),
                              copy.deepcopy(x), args)
        return prediction(info, self)


class prediction(object):
    r"""
    A class to represent an emulation prediction.  
    predict.info will give the dictionary from the software.
    """

    def __init__(self, info, emu):
        self.info = info
        self.emu = emu

    def __repr__(self):
        object_methods = [method_name for method_name in dir(self)
                  if callable(getattr(self, method_name))]
        object_methods = [x for x in object_methods if not x.startswith('_')]
        object_methods = [x for x in object_methods if not x.startswith('emu')]
        strrepr = ('A emulation prediction object predict where the code in located in the file '
                   + ' emulation.  The main methods are predict.' +
                   ', predict.'.join(object_methods) + '.  Default of predict() is' +
                   ' predict.mean() and ' +
                   'predict(s) will run pred.rnd(s).  Run help(predict) for the document' +
                   ' string.')
        return strrepr

    def __call__(self, s=None, args=None):
        if s is None:
            return self.mean(args)
        else:
            return self.rnd(s, args)
        

    def __softwarenotfoundstr(self, pfstr, opstr):
        print(pfstr + opstr + ' functionality not in software... \n' +
              ' Key labeled ' + opstr + ' not ' +
              'provided in ' + pfstr + '.info... \n' +
              ' Key labeled rnd not ' +
              'provided in ' + pfstr + '.info...')
        return 'Could not reconsile a good way to compute this value in current software.'

    def mean(self, args = None):
        r"""
        Returns the mean at theta and x in when building the prediction.
        """
        pfstr = 'predict' #prefix string
        opstr = 'mean' #operation string
        if (pfstr + opstr) in dir(self.emu.software):
            if args is None:
                args = self.emu.args
            return self.emu.software.predictmean(self.info, args)
        elif opstr in self.info.keys():
            return self.info[opstr]
        elif 'rnd' in self.info.keys():
            return np.mean(self.info['rnd'], 0)
        else:
            raise ValueError(self.__softwarenotfoundstr(pfstr, opstr))

    def var(self, args = None):
        r"""
        Returns the variance at theta and x in when building the prediction.
        """
        pfstr = 'predict' #prefix string
        opstr = 'var' #operation string
        if (pfstr + opstr) in dir(self.emu.software):
            if args is None:
                args = self.emu.args
            return self.emu.software.predictvar(self.info, args)
        elif opstr in self.info.keys():
            return self.info[opstr]
        elif 'rnd' in self.info.keys():
            return np.mean(self.info['rnd'], 0)
        else:
            raise ValueError(self.__softwarenotfoundstr(pfstr, opstr))
    
    def rnd(self, s=100, args=None):
        r"""
        Returns s random draws at theta and x in when building the prediction.
        """
        pfstr = 'predict' #prefix string
        opstr = 'rnd' #operation string
        if (pfstr + opstr) in dir(self.emu.software):
            if args is None:
                args = self.emu.args
            return self.emu.software.predictrnd(self.info, args)
        elif 'rnd' in self.info.keys():
            return self.info['rnd'][np.random.choice(self.info['rnd'].shape[0], size=s), :]
        else:
            raise ValueError(self.__softwarenotfoundstr(pfstr, opstr))

    def lpdf(self, f=None, args=None):
        r"""
        Returns a log pdf at theta and x 
        """
        raise ValueError('lpdf functionality not in software')