# -*- coding: utf-8 -*-
"""Header here."""
import numpy as np
import importlib
import scipy.stats as sps
import copy

class emulator(object):
    """A class used to represent an emulator or surrogate model."""

    def __init__(self, x=None, theta=None, f=None,  software='PCGP', args={}, options={}):
        r"""
        Intitalizes an emulator or surrogate.

        It directly calls "emulationmethods.[software]" where [software] is
        the user option with default listed above. If you would like to change this software, just
        drop a new file in the "emulationmethods" folder with the required formatting.

        Parameters
        ----------
        x : array of objects
            An array of inputs. Each row should correspond to a row in f. We will attempt
            to resolve size differences.
        theta : array of objects
            Anarray of parameters. Each row should correspond to a column in f.   We will attempt
            to resolve size differences.
        f : array of float
            An array of responses with 'nan' representing responses not yet available. Each
            column in f should correspond to a row in x. Each row should correspond to a row in
            f. We will attempt to resolve if these are flipped.
        software : str
            A string that points to the file located in "emulationmethods" you would
            like to use.
        args : dict
            Optional dictionary containing options you would like to pass to
            [software].fit(x, theta, f, args)
            or
            [software].predict(x, theta args)
        options : dict
            Optional options dictionary containing options you would like emulation
            to have.  This does not get passed to the software.  Some options are below:
                

        Returns
        -------
        emu : instance of emulation class
            An instance of the emulation class that can be used with the functions listed below.
        """
        self._args = copy.deepcopy(args)
        
        if f is not None:
            if f.ndim < 1 or f.ndim > 2:
                raise ValueError('f must have either 1 or 2 demensions.')
            if (x is None) and (theta is None):
                raise ValueError('You have not provided any theta or x, no emulator' +
                                 ' inference possible.')
            if x is not None:
                if x.ndim < 0.5 or x.ndim > 2.5:
                    raise ValueError('x must have either 1 or 2 demensions.')
            if theta is not None:
                if theta.ndim < 0.5 or theta.ndim > 2.5:
                    raise ValueError('theta must have either 1 or 2 demensions.')
        else:
            print('You have not provided f, ignoring everything and just warming up.')
            if (x is not None) and (theta is not None):
                raise ValueError('You have not provided f, cannot include theta or x.')
        
        if x is not None and (f.shape[0] != x.shape[0]):
            if theta is not None:
                if f.ndim == 2 and f.shape[1] == x.shape[0] and f.shape[0] == theta.shape[0]:
                    print('transposing f to try to get agreement....')
                    self.__f = copy.copy(f).T
                else:
                    raise ValueError('The number of rows in f must match the number of rows in x.')
            else:
                if f.ndim == 2 and f.shape[1] == x.shape[0]:
                    print('transposing f to try to get agreement....')
                    self.__f = copy.copy(f).T
                else:
                    raise ValueError('The number of rows in f must match the number of rows in x.')
        
        if theta is not None and (f.shape[1] != theta.shape[0]):
            if x is not None:
                raise ValueError('The number of columns in f must match the number of rows in theta.')
            else:
                if f.ndim == 2 and f.shape[0] == theta.shape[0]:
                    print('transposing f to try to get agreement....')
                    self.__f = copy.copy(f).T
                elif f.ndim == 1 and f.shape[0] == theta.shape[0]:
                    print('transposing f to try to get agreement....')
                    self.__f = np.reshape(copy.copy(f),(1,-1))
                raise ValueError('The number of columns in f must match the number of rows in theta.')
            
        if theta is not None and (f.shape[1] != theta.shape[0]):
            if f.shape[1] == theta.shape[0] and f.shape[0] == x.shape[0]:
                self.__f = copy.copy(f).T
            else:
                raise ValueError('The columns in f must match the rows in theta')
        
        if x is not None:
            self.__x = copy.copy(x)
        else:
            self.__x = None
        self.__suppx = None
        if theta is not None:
            self.__theta = copy.copy(theta)
        else:
            self.__theta = None
        self.__supptheta = None
        
        if f is not None:
            self.__f = copy.copy(f)
        else:
            self.__f = None
        
        try:
            self.software = importlib.import_module('base.emulationmethods.' + software)
        except:
            raise ValueError('Module not loaded correctly.')
        if "fit" not in dir(self.software):
            raise ValueError('Function fit not found in module!')
        if "predict" not in dir(self.software):
            raise ValueError('Function predict not found in module!')
        if "supplement" not in dir(self.software):
            print('Function supplement not found in module!')
        self.__options = {}
        self.__optionsset(options)
        self._info = {}
        
        if self.__f is not None and self.__options['autofit']:
            self.fit()

    def __repr__(self):
        object_methods = [method_name for method_name in dir(self)
                  if callable(getattr(self, method_name))]
        object_methods = [x for x in object_methods if not x.startswith('__')]
        strrepr = ('An emulation object where the code in located in the file '
                   + ' emulation.  The main methods are emu.' +
                   ', emu.'. join(object_methods) + '.  Default of emu(x,theta) is' +
                   ' emu.predict(x,theta).  Run help(emu) for the document string.')
        return strrepr
    
    def __call__(self, x=None, theta=None, args=None):
        return self.predict(x, theta, args)

    def fit(self, args= None):
        r"""
        Fits an emulator or surrogate and places that in emu._info
        
        Calls
        emu._info = [software].fit(emu.__theta, emu.__f, emu.__x, args = args)

        Parameters
        ----------
        args : dict
            Optional dictionary containing options you would like to pass to fit function. It will
            add/modify those in emu._args.
        """
        if args is not None:
            argstemp = {**self._args, **copy.deepcopy(args)} #properly merge the arguments
        else:
            argstemp = copy.copy(self._args)
        x, theta, f = self.__preprocess()
        self.software.fit(self._info, x, theta, f, args = argstemp)


    def predict(self, x=None, theta=None, args=None):
        r"""
        Fits an emulator or surrogate.
        
        Calls
        preddict = [software].predict(emu.theta, args = args)

        Parameters
        ----------
        x : array of objects
            A matrix of inputs. Each row should correspond to a row in f. We will attempt
            to resolve if these are flipped.
        theta : array of objects
            An m-by-d matrix of parameters. Each row should correspond to a column in f.  We will 
            attempt to resolve if these are flipped.
        args : dict
            A dictionary containing options you would like to pass to
            [software].fit(theta, phi, args). 
            Defaults to the one used to build emu.

        Returns
        -------
        prediction : an instance of emulation class prediction
            prediction._info : Gives the dictionary of what was produced by the software.
        """
        if args is not None:
            argstemp = {**self._args, **copy.deepcopy(args)} #properly merge the arguments
        else:
            argstemp = copy.copy(self._args)
        if x is None:
            x = copy.copy(self.__x)
        else:
            x = copy.copy(x)
            if x.ndim == 2 and self.__x.ndim == 1:
                raise ValueError('Your x shape seems to not agree with the emulator build.')
            elif x.ndim == 1 and self.__x.ndim == 2 and x.shape[0] == self.__x.shape[1]:
                x = reshape(x, (-1,1))
            elif x.ndim == 1 and self.__x.ndim == 2:
                raise ValueError('Your x shape seems to not agree with the emulator build.')
            elif x.shape[1] != self.__x.shape[1] and x.shape[0] == self.__x.shape[1]:
                x = x.T
            elif x.shape[1] != self.__x.shape[1] and x.shape[0] != self.__x.shape[1]:
                raise ValueError('Your x shape seems to not agree with the emulator build.')
        if theta is None:
            theta = copy.copy(self.__theta)
        else:
            theta = copy.copy(theta)
            if theta.ndim == 2 and self.__theta.ndim == 1:
                raise ValueError('Your theta shape seems to not agree with the emulator build.')
            elif theta.ndim == 1 and self.__theta.ndim == 2 and theta.shape[0] == self.__theta.shape[1]:
                theta = reshape(theta, (-1,1))
            elif theta.ndim == 1 and self.__theta.ndim == 2:
                raise ValueError('Your theta shape seems to not agree with the emulator build.')
            elif theta.shape[1] != self.__theta.shape[1] and theta.shape[0] == self.__theta.shape[1]:
                theta = theta.T
            elif theta.shape[1] != self.__theta.shape[1] and theta.shape[0] != self.__theta.shape[1]:
                raise ValueError('Your theta shape seems to not agree with the emulator build.')
        
        info = {}
        self.software.predict(info, self._info, copy.copy(x), copy.copy(theta),
                              copy.deepcopy(args))
        return prediction(info, self)
    
    def supplement(self, size, theta=None, x=None, cal=None, args=None, overwrite=False):
        r"""
        Chooses a new theta to be investigated.
        
        It can either come from the software or is automatted to use fit and
        predict from the software to complete the operation.

        Parameters
        ----------
        size : option array of float
            The number of of new supplements you would like to choose. If only theta is supplied, 
            it will return at most size of those. If only x is supplied, it will return at most
            size of those. If  both x and theta are supplied, then size will be less than 
            the product of the returned theta and x.
        theta : optional array of float
            An array of parameters where you would like to predict. YYou must provide either theta, 
            x or both or another object like cal.
        x : optional array of float
            An array of parameters where you would like to predict. You must provide either theta, 
            x or both or another object like cal.
        cal : optional calibrator object
            A calibrator object that contains information about calibration. You must provide either 
            theta, x or both or another object like cal.
        args : optional dict
            A dictionary containing options you would like to pass to  [software].supplement(theta, 
            phi, args).  Defaults to the one used to build emu.
        overwrite : boolean
            Do you want to replace existing supplement?  If not, and one exists, it will return 
            without doing anything.
        Returns
        -------
        theta : at most n new values of parameters to include
        """
        
        if args is None:
            args = self._args
        
        if size < 0.5:
            if size == 0:
                print('since size is zero, we presume you just want to return current supp.  If'+
                      ' supptheta exists, returning ')
                return copy.deepcopy(self.__supptheta)
            else:
                raise ValueError('The number of new parameters must be a positive integer.')
        if cal is None and theta is None:
            raise ValueError('Either a calibrator or thetas must be provided.')
            
        if cal is not None:
            try:
                print('overwriting given theta because cal is provided...')
                theta = cal.theta(1000)
            except:
                raise ValueError('cal.theta(1000) failed.')
        
        if x is not None and self.__suppx is not None and (not overwrite):
            raise ValueError('You must either evaulate the stuff in emu._emulator__suppx  or select'
                            + ' overwrite = True.')
        if theta is not None and self.__supptheta is not None and (not overwrite):
            raise ValueError('You must either evaulate the stuff in emu._emulator__supptheta or select'
                            + ' overwrite = True.')
        
        else:
            if self.__theta.shape[1] != theta.shape[1]:
                raise ValueError('theta has the wrong shape, it does not match emu.theta.')
            if theta.shape[0] > 10000:
                print('To stop memory issues, supply less than 10000 thetas...')
            thetadraw = theta[:10000,:]
        
        supptheta, suppx, suppinfo = self.software.supplement(self._info,size,
                                                              copy.deepcopy(self.__x),
                                                              copy.deepcopy(thetadraw),
                                                              cal)
        if not self.__options['thetareps']:
            nc, c, r = _matrixmatching(self.__theta, supptheta)
        else:
            nc = np.array(range(0,supptheta.shape[0])).astype('int')
        if nc.shape[0] < 0.5:
            print('Was not able to assign any new values because everything ' +
                  'was a replication of emu.__theta.')
            self.__supptheta = None
        else:
            if nc.shape[0] < supptheta.shape[0]:
                print('Had to remove replications versus emu.__theta.')
                supptheta = supptheta[nc,:]
            self.__supptheta = supptheta
        
        if self.__supptheta is not None and self.__suppx is None:
            return copy.deepcopy(self.__supptheta), suppinfo
        elif self.__suppx is not None and self.__supptheta is None:
            return copy.deepcopy(self.__suppx), suppinfo
        elif self.__supptheta is not None and self.__suppx is not None:
            return copy.deepcopy(self.__suppx), copy.deepcopy(self.__supptheta), suppinfo
        else:
            raise ValueError('Nothing to return.')
        
    
    def update(self,theta=None, f=None,  x=None, args=None, options=None):
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
        if theta is not None:
            theta = copy.deepcopy(theta)
        if f is not None:
            f = copy.deepcopy(f)
        if x is not None:
            x = copy.deepcopy(x)
        if options is not None:
            self.__optionsset(copy.deepcopy(options))
        if args is not None:
            self._args = {**self._args, **copy.deepcopy(args)} #properly merge the arguments
        if f is not None and theta is None and x is None:
            if f.shape[0] == self.__f.shape[0]:
                if self.__supptheta is not None:
                    if f.shape[1] == self.__supptheta.shape[0]:
                        self.__theta = np.vstack((self.__theta, self.__supptheta))
                        self.__f = np.hstack((self.__f,f))
                        self.__supptheta = None
                    elif f.shape[1] == (self.__theta.shape[0] + self.__supptheta.shape[0]):
                        self.__theta = np.vstack((self.__theta, self.__supptheta))
                        self.__f = f
                        self.__supptheta = None
                    else:
                        raise ValueError('Could not resolve absense of theta,' +
                                     'please provide theta')
                elif f.shape[1] == self.__theta.shape[1] and self.__x is None:
                    self.__f = f
                else:
                    raise ValueError('Could not resolve absense of theta,' +
                                     'please provide theta')
            else:
                raise ValueError('Could not resolve absense of x,' +
                                 'please provide x')
        if (x is not None) and (f is None):
            if x.shape[0] != self.__f.shape[0]:
                print('you have change the number of x, but not provided a new f...')
            else:
                self.__x = x
        if (theta is not None) and (f is None):
            if theta.shape[1] != self.__f.shape[1]:
                print('you have change the number of theta, but not provided a new f...')
            else:
                self.__theta = theta
        
        if (f is not None) and (theta is not None) and (x is None):
            if theta.shape[0] != f.shape[1]:
                raise ValueError('number of rows between theta and columns in f do not align.')
            if theta.shape[1] != self.__theta.shape[1]:
                raise ValueError('theta shape does not match old one,'
                                 + ' use emu.update(theta = theta) to update it first if' + 
                                 ' you changed your parameterization.')
            if f.shape[0] != self.__f.shape[0]:
                raise ValueError('Row of f are different than those provided originally,' +
                                 'please provide x to allow for alignment')
            if self.__options['reps']:
                    self.__theta = np.vstack((self.__theta, theta))
                    self.__f = np.hstack((self.__f,f))
            else:
                nc, c, r = _matrixmatching(self.__theta, theta)
                self.__f[:, r] = f[:,c]
                if nc.shape[0] > 0.5:
                    f = f[:,nc]
                    theta = theta[:,nc]
                    nc, c, r = _matrixmatching(self.__supptheta, theta)
                    self.__f = np.hstack((self.__f, f[c,:]))
                    self.__theta = np.vstack((self.__theta, theta[c,:]))
                    self.__supptheta = np.delete(self.__supptheta, r, axis = 0)
                if nc.shape[0] > 0.5:
                    f = f[:, nc]
                    theta = theta[nc, :]
                    self.__f = np.hstack(self.__f,f[:,c])
                    self.__theta = np.vstack(self.__f,theta[c,:])
        
        if (f is not None) and (theta is None) and (x is not None):
            if x.shape[0] != f.shape[1]:
                raise ValueError('number of rows in f and rows in x does not align.')
            if x.shape[1] != self.__x.shape[1]:
                raise ValueError('x shape does not match old one,'
                                 + ' use emu.update(x = x) to update it first if' + 
                                 ' you changed your description of x.')
            if f.shape[1] != self.__f.shape[1]:
                raise ValueError('Rows of f are different than those provided originally,' +
                                 'please provide theta to allow for alignment')
            if options['reps']:
                self.__x = np.vstack((self.__x, x))
                self.__f = np.vstack((self.__f,f))
            else:
                nc, c, r = _matrixmatching(self.__x, x)
                self.__f[r, :] = f[c, :]
                if nc.shape[0] > 0.5:
                    self.__f = np.vstack(self.__f, f[c,:])
        
        if (f is not None) and (theta is not None) and (x is not None):
                raise ValueError('Simultaneously adding new theta and x at once is currently'+
                                 'not supported.  Please supply either theta OR x.')
        
        self.fit()
        return
    
    
    def __optionsset(self, options=None):
        options = copy.deepcopy(options)
        options =  {k.lower(): v for k, v in options.items()} #options will always be lowercase
        
        if 'thetareps' in options.keys():
            if type(options['thetareps']) is bool:
                self.__options['thetareps'] = options['thetareps']
            else:
                raise ValueError('option thetareps must be true or false')
        
        if 'xreps' in options.keys():
            if type(options['xreps']) is bool:
                self.__options['xreps'] = options['xreps']
            else:
                raise ValueError('option xreps must be true or false')
        
        if 'thetarmnan' in options.keys():
            if type(options['thetarmnan']) is bool:
                if options['thetarmnan']:
                    self.__options['thetarmnan'] = 0
                else:
                    self.__options['thetarmnan'] =  1 + (10** (-12))
            elif options['thetarmnan'] is str and options['thetarmnan']=='any':
                    self.__options['thetarmnan'] = 0
            elif options['thetarmnan'] is str and options['thetarmnan']=='some':
                    self.__options['thetarmnan'] = 0.2
            elif options['thetarmnan'] is str and options['thetarmnan']=='most':
                    self.__options['thetarmnan'] = 0.5
            elif options['thetarmnan'] is str and options['thetarmnan']=='alot':
                    self.__options['thetarmnan'] = 0.8
            elif options['thetarmnan'] is str and options['thetarmnan']=='all':
                    self.__options['thetarmnan'] = 1 - (10** (-12))
            elif options['thetarmnan'] is str and options['thetarmnan']=='never':
                    self.__options['thetarmnan'] = 1 + (10** (-12))
            elif np.isfinite(options['thetarmnan']) and options['thetarmnan']>=0\
                and options['thetarmnan']<=1:
                self.__options['thetarmnan'] = options['thetarmnan']
            else:
                print(options['thetarmnan'])
                raise ValueError('option thetarmnan must be True, False, ''any'', ''some''' +
                                 ', ''most'', ''alot'', ''all'', ''never'' or an scaler bigger'+
                                 'than zero and less than one.')
        if 'xrmnan' in options.keys():
            if type(options['xrmnan']) is bool:
                if options['xrmnan']:
                    self.__options['xrmnan'] = 0
                else:
                    self.__options['xrmnan'] =  1 + (10** (-12))
            elif options['xrmnan'] is str and options['xrmnan']=='any':
                    self.__options['xrmnan'] = 0
            elif options['xrmnan'] is str and options['xrmnan']=='some':
                    self.__options['xrmnan'] = 0.2
            elif options['xrmnan'] is str and options['xrmnan']=='most':
                    self.__options['xrmnan'] = 0.5
            elif options['xrmnan'] is str and options['xrmnan']=='alot':
                    self.__options['xrmnan'] = 0.8
            elif options['xrmnan'] is str and options['xrmnan']=='all':
                    self.__options['xrmnan'] = 1- (10** (-12))
            elif options['xrmnan'] is str and options['xrmnan']=='never':
                    self.__options['xrmnan'] = 1 + (10** (-12))
            elif np.isfinite(options['xrmnan']) and options['xrmnan']>=0\
                and options['xrmnan']<=1:
                self.__options['xrmnan'] = options['xrmnan']
            else:
                raise ValueError('option xrmnan must be True, False, ''any'', ''some'''+
                                 ', ''most'', ''alot'', ''all'', ''never'' or an scaler bigger'+
                                 'than zero and less than one.')
        
        if 'rmthetafirst' in options.keys():
            if type(options['rmthetafirst']) is bool:
                self.__options['rmthetafirst'] = options['rmthetafirst']
            else:
                raise ValueError('option rmthetafirst must be True or False.')
        
        if 'autofit' in options.keys():
            if type(options['autofit']) is bool:
                self.__options['minsampsize'] = options['autofit']
            else:
                raise ValueError('option autofit must be of type bool.')
        
        if 'thetareps' not in self.__options.keys():
            self.__options['thetareps'] = False
        if 'xreps' not in self.__options.keys():
            self.__options['xreps'] = False
        if 'thetarmnan' not in self.__options.keys():
            self.__options['thetarmnan'] =0.8
        if 'xrmnan' not in self.__options.keys():
            self.__options['xrmnan'] = 0.8
        if 'autofit' not in self.__options.keys():
            self.__options['autofit'] = True
        if 'rmthetafirst' not in self.__options.keys():
            self.__options['rmthetafirst'] = True

    def __preprocess(self):
        x = copy.copy(self.__x)
        theta = copy.copy(self.__theta)
        f = copy.copy(self.__f)
        options = self.__options
        isinff = np.isinf(f)
        if np.any(isinff):
            print('All infs were converted to nans.')
            f[isinff] = float("NaN")
        isnanf = np.isnan(f)
        if self.__options['rmthetafirst']:
            j = np.where(np.mean(isnanf, 0) < self.__options['thetarmnan'])[0]
            f = f[:,j]
            theta = theta[j,:]
        j = np.where(np.mean(isnanf, 1) < self.__options['xrmnan'])[0]
        f = f[j,:]
        x = x[j,:]
        if not self.__options['rmthetafirst']:
            j = np.where(np.mean(isnanf, 0) < self.__options['thetarmnan'])[0]
            f = f[:,j]
            theta = theta[j,:]
        return x, theta, f

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

    def covx(self, args = None):
        r"""
        Returns the covariance matrix at theta and x in when building the prediction.
        """
        pfstr = 'predict' #prefix string
        opstr = 'covx' #operation string
        if (pfstr + opstr) in dir(self.emu.software):
            if args is None:
                args = self.emu.args
            return copy.deepcopy(self.emu.software.predictcov(self._info, args))
        elif opstr in self._info.keys():
            return copy.deepcopy(self._info[opstr])
        elif 'covxhalf' in self._info.keys():
            if self._info['covxhalf'].ndim == 2:
                return self._info['covxhalf'] @ self._info['covxhalf'].T
            else:
                am = self._info['covxhalf'].shape
                covx = np.ones((am[2],am[1],am[2]))
                for k in range(0, self._info['covxhalf'].shape[1]):
                    A = self._info['covxhalf'][:,k,:]
                    covx[:,k,:] = A.T @ A
            self._info['covx'] = covx
            return copy.deepcopy(self._info[opstr])
        else:
            raise ValueError(self.__softwarenotfoundstr(pfstr, opstr))

    def covxhalf(self, args = None):
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
        elif 'covx' in self._info.keys():
            covxhalf = np.ones(self._info['covx'].shape)
            if self._info['covx'].ndim == 2:
                W, V = np.linalg.eigh(self._info['covx'])
                covxhalf = (V @ (np.sqrt(np.abs(W)) * V.T))
            else:
                for k in range(0, self._info['covx'].shape[0]):
                    W, V = np.linalg.eigh(self._info['covx'][k])
                    covxhalf[k,:,:] = (V @ (np.sqrt(np.abs(W)) * V.T))
            self._info['covxhalf'] = covxhalf
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

#### Below are some functions that I found useful.

def _matrixmatching(mat1, mat2):
    #This is an internal function to do matching between two vectors
    #it just came up alot
    #It returns the where each row of mat2 is first found in mat1
    #If a row of mat2 is never found in mat1, then 'nan' is in that location
    
    
    if (mat1.shape[0] > (10 ** (4))) or (mat2.shape[0] > (10 ** (4))):
        raise ValueError('too many matchings attempted.  Don''t make the software work so hard!')
    if mat1.ndim != mat2.ndim:
        raise ValueError('Somehow sent non-matching information to _matrixmatching')
    if mat1.ndim == 1:
        matchingmatrix = np.isclose(mat1[:,None].astype('float'),
                 mat2.astype('float'))
    else:
        matchingmatrix = np.isclose(mat1[:,0][:,None].astype('float'),
                         mat2[:,0].astype('float'))
        for k in range(1,mat2.shape[1]):
            try:
                matchingmatrix *= np.isclose(mat1[:,k][:,None].astype('float'),
                         mat2[:,k].astype('float'))
            except:
                matchingmatrix *= np.equal(mat1[:,k],mat2[:,k])
    r, c = np.where(matchingmatrix.cumsum(axis=0).cumsum(axis=0) == 1)
    
    nc = np.array(list(set(range(0,mat2.shape[0])) - set(c))).astype('int')
    return nc, c, r