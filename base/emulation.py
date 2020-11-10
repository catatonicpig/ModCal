# -*- coding: utf-8 -*-
"""Header here."""
import numpy as np
import importlib
import copy


class emulator(object):
    """A class used to represent an emulator or surrogate model."""

    def __init__(self, theta, f, x, software='PCGP', args={}, options={}):
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
        options : dict
            Optional options dictionary containing options you would like emulation
            to have.  This does not get passed to the software

        Returns
        -------
        emu : instance of emulation class
            An instance of the emulation class that can be used with the functions listed below.
        """
        if f is None or theta is None:
            raise ValueError('You have no provided any theta and/or f.')
        
        if f.ndim < 0.5 or f.ndim > 2.5:
            raise ValueError('f must have either 1 or 2 demensions.')
        
        isinff = np.isinf(f)
        if np.any(isinff):
            print('All infs were converted to nans.')
            f[isinff] = float("NaN")
        
        isnanf = np.isnan(f)
        rowallnanf = np.all(isnanf,1)
        if np.any(rowallnanf):
            print('Row(s) %s removed due to nans.' % np.array2string(np.where(rowallnanf)[0]))
            j = np.where(np.logical_not(rowallnanf))[0]
            f = f[j,:]
            theta = theta[j,:]
        
        numthetamin = 2 * theta.shape[1]
        if theta.ndim < 0.5 or theta.ndim > 2.5:
            raise ValueError('theta must have either 1 or 2 demensions.')
        
        if theta.shape[0] < numthetamin:
            raise ValueError('theta should have at least 2 more' +
                             'rows than columns.')
        
        colnumdone = np.sum(1-isnanf,0)
        notenoughvals = (colnumdone < numthetamin)
        if np.any(notenoughvals):
            print('Column(s) %s removed due to not enough completed values.'
                  % np.array2string(np.where(notenoughvals)[0]))
            j = np.where((colnumdone >= numthetamin))[0]
            f = f[:,j]
            x = x[j,:]
        
        if f.shape[0] is not theta.shape[0]:
            raise ValueError('The rows in f must match' +
                             ' the rows in theta')
            
        if f.ndim == 1 and x is not None:
            raise ValueError('Cannot use x if f has a single column.')
        
        if f.ndim == 2 and x is not None and\
            f.shape[1] is not x.shape[0]:
            raise ValueError('The columns in f must match' +
                             ' the rows in x')
        
        self.__theta = copy.deepcopy(theta)
        self.__f = copy.deepcopy(f)
        self.__x = copy.deepcopy(x)
        self._options = copy.deepcopy(options)
        self._args = copy.deepcopy(args)
        try:
            self.software = importlib.import_module('base.emulationsubfuncs.' + software)
        except:
            raise ValueError('Module not found!')
            
            
        if "fit" not in dir(self.software):
            raise ValueError('Function \"fit\" not found in module!')
        
        if "predict" not in dir(self.software):
            raise ValueError('Function \"predict\" not found in module!')
        
        self._info = {}
        self.fit()

    def __repr__(self):
        object_methods = [method_name for method_name in dir(self)
                  if callable(getattr(self, method_name))]
        object_methods = [x for x in object_methods if not x.startswith('__')]
        strrepr = ('An emulation object where the code in located in the file '
                   + ' emulation.  The main methods are emu.' +
                   ', emu.'. join(object_methods) + '.  Default of emu(theta) is' +
                   ' emu.predict(theta).  Run help(emu) for the document string.')
        return strrepr
    
    def __call__(self, x=None):
        return self.predict(self.__theta, x)

    def fit(self, args= None):
        r"""
        Fits an emulator or surrogate.
        
        Calls
        emu._info = [software].fit(emu.theta, emu.f, emu.x, args = args)

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
        emu._info : dict
            An arbitrary dictionary that can be used with [software].pred
        """
        if args is None:
            args = self._args
        self.software.fit(self._info, copy.deepcopy(self.__theta), copy.deepcopy(self.__f), 
                                      copy.deepcopy(self.__x), args = args)


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
        prediction : an instance of emulation class prediction
            prediction._info : Gives the dictionary of what was produced by the software.
        """
        if args is None:
            args = self._args
        if theta.ndim < 1.5:
            theta = theta.reshape([-1, theta.shape[0]])
        if theta[0].shape[0] is not self.__theta[0].shape[0]:
            raise ValueError('The new parameters do not match old parameters.')
            
        if x is None:
            x = copy.deepcopy(self.__x)
        else:
            if x[0].shape[0] is not self.__x[0].shape[0]:
                raise ValueError('The new inputs do not match old inputs.')
        _info = {}
        self.software.predict(_info, self._info, copy.deepcopy(theta),
                              copy.deepcopy(x), args)
        return prediction(_info, self)
    
    def supplement(self, n, cal=None, theta=None, args=None, append=False):
        r"""
        Chooses a new theta to be investigated.
        
        It can either come from the software or is automatted to use fit and
        predict from the software to complete the operation.

        Parameters
        ----------
        n : option array of float
            The number of thetas you would like to use
        theta : optional array of float
            An array of parameters where you would like to predict
        cal : optional calibrator object
            A calibrator object that contains information about calibration.
        args : optional dict
            A dictionary containing options you would like to pass to
            [software].selecttheta(theta, phi, args). 
            Defaults to the one used to build emu.
        append : optional dict
            Do you want to append emu.__supptheta?  If append is False, 
            emu.__supptheta is replaced.

        Returns
        -------
        theta : at most n new values of parameters to include
        """
        
        allowreps = False
        if 'reps' in self.__options.keys():
            allowreps = self.__options.keys('reps')
                
        if args is None:
            args = self._args
        if n < 0.5:
            raise ValueError('The number of new parameters must be a positive integer.')
        if cal is None and theta is None:
            raise ValueError('Either a calibrator or thetas must be provided.')
        if cal is not None:
            try:
                thetadraw = cal.theta(np.minimum(10*n,np.maximum(1000, 2*n)))
            except:
                raise ValueError('cal.theta(1000) failed.')
        else:
            if self.__theta.shape[1] != theta.shape[1]:
                raise ValueError('theta has the wrong shape, it does not match emu.theta.')
            if theta.shape[0] < n:
                raise ValueError('you want to predict at less than n values,' + 
                                 'just run them you silly goose')
            if theta.shape[0] > 10000:
                print('To stop memory issues, supply less than 10000 thetas...')
            thetadraw = theta[:10000,:]
        supptheta = copy.deepcopy(thetadraw)
        if append and self.__supptheta is not None:
            cutoff = (10 ** (-8)) * np.std(np.vstack((self.__theta,
                                                      self.__supptheta,
                                                      supptheta)),0)
            if not allowreps:
                MAT = np.array(np.all(np.abs(self.__supptheta[:,None,:]-
                                       supptheta[None,:,:]) / cutoff < 1,axis=-1).nonzero()).T
                keepinds = set(range(0,supptheta.shape[0])) - set(np.unique(MAT[:,1]))
            else:
                keepinds = set(range(0,supptheta.shape[0]))
            if len(keepinds) < 0.5:
                print('Was not able to assign any new values because everything ' +
                      'was a replication of emu.__supptheta.')
                return self.supptheta
            elif len(keepinds) < supptheta.shape[0]:
                print('Had to remove replications versus previous emu.__supptheta.')
                supptheta = supptheta[np.array(list(keepinds)),:]
        else:
            cutoff = (10 ** (-8)) * np.std(np.vstack((self.__theta,
                                                      supptheta)),0)
        if not allowreps:
            MAT = np.array(np.all(np.abs(self.__theta[:,None,:]-
                                   supptheta[None,:,:]) / cutoff < 1,axis=-1).nonzero()).T
            keepinds = set(range(0,supptheta.shape[0])) - set(np.unique(MAT[:,1]))
        else:
            keepinds = set(range(0,supptheta.shape[0]))
        if len(keepinds) < 0.5:
            print('Was not able to assign any new values because everything ' +
                  'was a replication of emu.__theta.')
            if not append:
                self.__supptheta = None
        else:
            if len(keepinds) < supptheta.shape[0]:
                print('Had to remove replications versus emu.__theta.')
                supptheta = supptheta[np.array(list(keepinds)),:]
            if append:
                self.__supptheta = np.vstack((self.__supptheta,supptheta[:n,:]))
            else:
                self.__supptheta = supptheta[:n,:]
        return copy.deepcopy(self.__supptheta)
    
    def update(self, f, theta=None, x=None, args=None, options=None):
        r"""
        Chooses a new theta to be investigated.
        
        It can either come from the software or is automatted to use fit and
        predict from the software to complete the operation.

        Parameters
        ----------
        f : new f values
            A 2d array of responses at (theta, emu.__supptheta and/or emu.__theta)
            and (x, emu.__x).
        theta : optional array of float
            thetas you would like to append. Defaults to emu.__supptheta.
            Will attempt to resolve if using all theta and supptheta.
        x : optional array of float
            xs you would like to append. Defaults to emu.__x.
            Will attempt to resolve if using all x and emu.__x.
        args : optional dict
            A dictionary containing options you would like to pass to
            [software].update(f,theta,x,args). 
            Defaults to the one used to build emu.
        options : optional dict
            A dictionary containing options you would like to keep around
            to build the emulator.  Modify with update when you want to change
            it.

        Returns
        -------
        """
        f = copy.deepcopy(f)
        if args is not None:
            self._args = copy.deepcopy(args)        
        if options is not None:
            self.options = copy.deepcopy(options)
        
        if f.shape[1] == self.__f.shape[1]:
            if theta is None:
                if self.__supptheta is not None:
                    if f.shape[0] == self.__supptheta.shape[0]:
                        self.__theta = np.vstack((self.__theta, self.__supptheta))
                        self.__f = np.vstack((self.__f,f))
                        self.__supptheta = None
                elif f.shape[0] == self.__theta.shape[0]:
                    self.__f = f
                else:
                    raise ValueError('Could not resolve absense of theta,' +
                                     'please provide theta')
            else:
                if np.array_equal(theta, self.__supptheta):
                    self.__f = np.vstack((self.__f,f))
                if np.arrayequal
        return
        
class prediction(object):
    r"""
    A class to represent an emulation prediction.  
    predict._info will give the dictionary from the software.
    """

    def __init__(self, _info, emu):
        self._info = _info
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
              'provided in ' + pfstr + '._info... \n' +
              ' Key labeled rnd not ' +
              'provided in ' + pfstr + '._info...')
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
            return copy.deepcopy(self.emu.software.predictmean(self._info, args))
        elif opstr in self._info.keys():
            return self._info[opstr]
        elif 'rnd' in self._info.keys():
            return copy.deepcopy(np.mean(self._info['rnd'], 0))
        else:
            raise ValueError(self.__softwarenotfoundstr(pfstr, opstr))

    def var(self, args = None):
        r"""
        Returns the pointwise variance at theta and x in when building the prediction.
        """
        pfstr = 'predict' #prefix string
        opstr = 'var' #operation string
        if (pfstr + opstr) in dir(self.emu.software):
            if args is None:
                args = self.emu.args
            return copy.deepcopy(self.emu.software.predictvar(self._info, args))
        elif opstr in self._info.keys():
            return copy.deepcopy(self._info[opstr])
        elif 'rnd' in self._info.keys():
            return copy.deepcopy(np.var(self._info['rnd'], 0))
        else:
            raise ValueError(self.__softwarenotfoundstr(pfstr, opstr))

    def cov(self, args = None):
        r"""
        Returns the covariance matrix at theta and x in when building the prediction.
        """
        pfstr = 'predict' #prefix string
        opstr = 'cov' #operation string
        if (pfstr + opstr) in dir(self.emu.software):
            if args is None:
                args = self.emu.args
            return copy.deepcopy(self.emu.software.predictcov(self._info, args))
        elif opstr in self._info.keys():
            return copy.deepcopy(self._info[opstr])
        elif 'covhalf' in self._info.keys():
            if self._info['covhalf'].ndim == 2:
                return self._info['covhalf'].T @ self._info['covhalf']
            else:
                am = self._info['covhalf'].shape
                cov = np.ones((am[0],am[2],am[2]))
                for k in range(0, self._info['covhalf'].shape[0]):
                    A = self._info['covhalf'][k]
                    cov[k,:,:] = A.T @ A
            self._info['cov'] = cov
            return copy.deepcopy(self._info[opstr])
        else:
            raise ValueError(self.__softwarenotfoundstr(pfstr, opstr))

    def covhalf(self, args = None):
        r"""
        Returns the sqrt of the covariance matrix at theta and x in when building the prediction.
        That is, if this returns A = predict.covhalf(.)[k], than A.T @ A = predict.cov(.)[k]
        """
        pfstr = 'predict' #prefix string
        opstr = 'covhalf' #operation string
        if (pfstr + opstr) in dir(self.emu.software):
            if args is None:
                args = self.emu.args
            return copy.deepcopy(self.emu.software.predictcov(self._info, args))
        elif opstr in self._info.keys():
            return copy.deepcopy(self._info[opstr])
        elif 'cov' in self._info.keys():
            covhalf = np.ones(self._info['cov'].shape)
            if self._info['cov'].ndim == 2:
                W, V = np.linalg.eigh(self._info['cov'])
                covhalf = (V @ (np.sqrt(np.abs(W)) * V.T))
            else:
                for k in range(0, self._info['cov'].shape[0]):
                    W, V = np.linalg.eigh(self._info['cov'][k])
                    covhalf[k,:,:] = (V @ (np.sqrt(np.abs(W)) * V.T))
            self._info['covhalf'] = covhalf
            return copy.deepcopy(self._info[opstr])
        else:
            raise ValueError(self.__softwarenotfoundstr(pfstr, opstr))

    def rnd(self, s=100, args=None):
        r"""
        Returns a rnd draws of size s at theta and x 
        """
        raise ValueError('rnd functionality not in software')

    def lpdf(self, f=None, args=None):
        r"""
        Returns a log pdf at theta and x 
        """
        raise ValueError('lpdf functionality not in software')