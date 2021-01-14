# -*- coding: utf-8 -*-
"""Header here."""
import numpy as np
import importlib
import scipy.stats as sps
import copy

class emulator(object):
    """A class used to represent an emulator or surrogate model."""

    def __init__(self, x=None, theta=None, f=None,  method='PCGP', passthroughfunc = None,
                 args={}, options={}):
        r"""
        Intitalizes an emulator or surrogate.

        It directly calls "emulationmethods.[method]" where [method] is
        the user option with default listed above. If you would like to change this method, just
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
        method : str
            A string that points to the file located in "emulationmethods" you would
            like to use.
        args : dict
            Optional dictionary containing options you would like to pass to
            [method].fit(x, theta, f, args)
            or
            [method].predict(x, theta args)
        options : dict
            Optional options dictionary containing options you would like emulation
            to have.  This does not get passed to the method.  Some options are below:
                

        Returns
        -------
        emu : instance of emulation class
            An instance of the emulation class that can be used with the functions listed below.
        """
        self.__ptf = passthroughfunc
        if self.__ptf is not None:
            return
        self._args = copy.deepcopy(args)
        
        if f is not None:
            if f.ndim < 1 or f.ndim > 2:
                raise ValueError('f must have either 1 or 2 dimensions.')
            if (x is None) and (theta is None):
                raise ValueError('You have not provided any theta or x, no emulator inference possible.')
            if x is not None:
                if x.ndim < 0.5 or x.ndim > 2.5:
                    raise ValueError('x must have either 1 or 2 dimensions.')
            if theta is not None:
                if theta.ndim < 0.5 or theta.ndim > 2.5:
                    raise ValueError('theta must have either 1 or 2 dimensions.')
        else:
            raise ValueError('You have not provided f, cannot include theta or x.')

        
        if x is not None and (f.shape[0] != x.shape[0]):
            if theta is not None:
                if f.ndim == 2 and f.shape[1] == x.shape[0] and f.shape[0] == theta.shape[0]:
                    print('transposing f to try to get agreement...')
                    self.__f = copy.copy(f).T
                    f = copy.copy(f).T
                else:
                    raise ValueError('The number of rows in f must match the number of rows in x.')
            else:
                if f.ndim == 2 and f.shape[1] == x.shape[0]:
                    print('transposing f to try to get agreement...')
                    self.__f = copy.copy(f).T
                    f = copy.copy(f).T
                else:
                    raise ValueError('The number of rows in f must match the number of rows in x.')

        if theta is not None and (f.shape[1] != theta.shape[0]):
            if x is not None:
                if not (f.ndim == 2 and f.shape[0] == theta.shape[0] and f.shape[1] == x.shape[0]):
                    raise ValueError('The number of columns in f must match the number of rows in theta.')
            else:
                if f.ndim == 2 and f.shape[0] == theta.shape[0]:
                    print('transposing f to try to get agreement....')
                    self.__f = copy.copy(f).T
                    f = copy.copy(f).T
                elif f.ndim == 1 and f.shape[0] == theta.shape[0]:
                    print('transposing f to try to get agreement....')
                    self.__f = np.reshape(copy.copy(f),(1,-1))
                    f = np.reshape(copy.copy(f),(1,-1))
                else:
                    raise ValueError('The number of columns in f must match the number of rows in theta.')
            
        
        if x is not None:
            self.__x = copy.copy(x)
        else:
            self.__x = None

        if theta is not None:
            self.__theta = copy.copy(theta)
        else:
            self.__theta = None
            raise ValueError('This feature has not developed yet.')
            
        self.__suppx = None
        self.__supptheta = None
        self.__f = copy.copy(f)

        try:
            self.method = importlib.import_module('base.emulationmethods.' + method)
        except:
            raise ValueError('Module not loaded correctly.')
        if "fit" not in dir(self.method):
            raise ValueError('Function fit not found in module!')
        if "predict" not in dir(self.method):
            raise ValueError('Function predict not found in module!')
        if "supplementtheta" not in dir(self.method):
            print('Function supplementtheta not found in module!')
        self.__options = {}
        self.__optionsset(options)
        self._info = {}
        self._info = {'method': method}

        if (self.__f is not None) and (self.__options['autofit']):
            self.fit()

    def __repr__(self):
        object_method = [method_name for method_name in dir(self)
                  if callable(getattr(self, method_name))]
        object_method = [x for x in object_method if not x.startswith('__')]
        strrepr = ('An emulation object where the code in located in the file '
                   + ' emulation.  The main method are emu.' +
                   ', emu.'. join(object_method) + '.  Default of emu(x,theta) is' +
                   ' emu.predict(x,theta).  Run help(emu) for the document string.')
        return strrepr
    
    def __call__(self, x=None, theta=None, args=None):
        return self.predict(x, theta, args)

    def fit(self, args= None):
        r"""
        Fits an emulator or surrogate and places that in emu._info
        
        Calls
        emu._info = [method].fit(emu.__theta, emu.__f, emu.__x, args = args)

        Parameters
        ----------
        args : dict
            Optional dictionary containing options you would like to pass to fit function. It will
            add/modify those in emu._args.
        """
        
        # note: not sure if args here in fit(self, args= None) makes sense--it is not necessary and I cant test it with the current setting
        #if args is not None:
        #    argstemp = {**self._args, **copy.deepcopy(args)} #properly merge the arguments
        #else:
        argstemp = copy.copy(self._args)
        x, theta, f = self.__preprocess()
        self.method.fit(self._info, x, theta, f, args = argstemp)


    def predict(self, x=None, theta=None, args={}):
        r"""
        Fits an emulator or surrogate.
        
        Calls
        preddict = [method].predict(x, theta, args = args)

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
            [method].fit(theta, phi, args). 
            Defaults to the one used to build emu.

        Returns
        -------
        prediction : an instance of emulation class prediction
            prediction._info : Gives the dictionary of what was produced by the method.
        """
    
        if self.__ptf is not None:
            info = {}
            if theta is not None:
                info['mean'] = self.__ptf(x, theta)
            else:
                info['mean'] = self.__ptf(x, self.__theta)
            info['var'] = 0 *  info['mean'] 
            info['covxhalf'] = 0 *  np.stack((info['mean'],info['mean']), 2)
            return prediction(info, self)
        if args is not None:
            argstemp = {**self._args, **copy.deepcopy(args)} #properly merge the arguments
        else:
            argstemp = copy.copy(self._args)
        if x is None:
            x = copy.copy(self.__x)
        else:
            x = copy.copy(x)
            if x.ndim == 1:
                if self.__x.ndim == 2 and x.shape[0] == self.__x.shape[1]:
                    x = np.reshape(x, (1,-1))
                elif self.__x.ndim == 2:
                    raise ValueError('Your x shape seems to not agree with the emulator build.')
            elif x.ndim ==2:
                if self.__x.ndim == 1:
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
            #note: dont understand why we have this statement
            elif theta.ndim == 1 and self.__theta.ndim == 2 and theta.shape[0] == self.__theta.shape[1]:
                theta = np.reshape(theta, (1,-1))
            elif theta.ndim == 1 and self.__theta.ndim == 2:
                raise ValueError('Your theta shape seems to not agree with the emulator build.')
            elif theta.shape[1] != self.__theta.shape[1] and theta.shape[0] == self.__theta.shape[1]:
                theta = theta.T
            elif theta.shape[1] != self.__theta.shape[1] and theta.shape[0] != self.__theta.shape[1]:
                raise ValueError('Your theta shape seems to not agree with the emulator build.')
        
        info = {}
        self.method.predict(info, self._info, x, theta, args = argstemp)
        return prediction(info, self)
    
    def supplement(self, size,  x=None, xchoices=None,  theta=None, thetachoices=None, choicescost=None,
                   cal=None, args=None, overwrite=False, removereps = None):
        r"""
        Chooses a new theta to be investigated.
        
        It can either come from the method or is automatted to use fit and
        predict from the method to complete the operation.

        Parameters
        ----------
        size : option array of float
            The number of of new supplements you would like to choose. If only theta is supplied, 
            it will return at most size of those. If only x is supplied, it will return at most
            size of those. If  both x and theta are supplied, then size will be less than 
            the product of the number of returned theta and the number of x.
        x : optional array of float
            An array of parameters where you would like to predict. A user must provide either x, 
            theta or both or another object like cal.
        xchoices : optional array of float
            An  array of inputs where you would like to select from.  If not provided, we
            will use a subset of x.
        theta : optional array of float
            An array of parameters where you would like to predict. A user must provide either x, 
            theta or both or another object like cal.
        thetachoices : optional array of float
            An  array of parameters where you would like to select from.  If not provided, we
            will use a subset of theta.
        choicescost : optional array of values
            An array of positive cost of each element in choice
        cal : optional calibrator object
            A calibrator object that contains information about calibration. A user must provide 
            either x, theta or both or another object like cal.
        args : optional dict
            A dictionary containing options you would like to pass to the method.
            Defaults to the one used to build emu.
        overwrite : boolean
            Do you want to replace existing supplement?  If False, and one exists, it will return 
            without doing anything.
        removereps : boolean
            Do you want to remove any replications existing supplement? Will default to options.
        Returns
        -------
        theta, info : If returning supplemented thetas
        x, info : If returning supplemented xs
        x, theta, info : If returning both x and theta
        """
        
        if args is not None:
            argstemp = {**self._args, **args} #properly merge the arguments
        else:
            argstemp = self._args
        
        if removereps is None:
            if x is not None:
                removereps = not self.__options['xreps']
            if theta is not None:
                removereps = not self.__options['thetareps']
        
        if size < 0.5:
            if size == 0:
                print('since size is zero, we presume you just want to return current supp.  If'+
                      ' supptheta exists, we are returning that now.')
                return copy.deepcopy(self.__supptheta)
            else:
                raise ValueError('The number of new values must be a positive integer.')
        
        if cal is None and theta is None and x is None:
            raise ValueError('Either x or (theta or cal) must be provided.')
            
        if cal is not None:
            try:
                if theta is None:
                    theta = cal.theta(2000)
            except:
                raise ValueError('cal.theta(2000) failed.')
        
        if x is not None and theta is not None:
            raise ValueError('You must either provide either x or (theta or cal).')
        
        if x is not None and self.__suppx is not None and (not overwrite):
            raise ValueError('You must either evaluate the stuff in emu._emulator__suppx  or select'
                            + ' overwrite = True.')
        elif x is not None:
            x = copy.copy(x)
            if self.__x.shape[1] != x.shape[1]:
                raise ValueError('x has the wrong shape, it does not match emu._emulator__x.')
        else:
            x = None
        
        if xchoices is not None:
            raise ValueError('selection of x is not yet supported.')
            
        
        
        if thetachoices is not None and theta is None:
            raise ValueError('You must provide theta (or a cal) if you give thetachoices.')
        if theta is not None and thetachoices is None:
            if theta.shape[0] > 30 * size:
                thetachoices = theta[np.random.choice(theta.shape[0], 30 * size, replace=False),:]
            else:
                thetachoices = copy.copy(theta)
        
        if choicescost is None and thetachoices is not None:
            choicescost = np.ones(thetachoices.shape[0])
        #elif choicescost is None and xchoices is not None:
        #    choicescost = np.ones(xchoices.shape[0])
        elif thetachoices is not None and thetachoices.shape[0] != choicescost.shape[0]:
            raise ValueError('choicecost is not the right shape.')
        #elif xchoices is not None and xchoices.shape[0] != choicescost.shape[0]:
        #    raise ValueError('choicecost is not the right shape.')


        if thetachoices is not None:
            if thetachoices.shape[1] != theta.shape[1]:
                raise ValueError('Your demensions of choices and predictions are not aligning.')
        
        if theta is not None and self.__supptheta is not None and (not overwrite):
            raise ValueError('You must either evaulate the stuff in emu._emulator__supptheta or select'
                            + ' overwrite = True.')
        elif theta is not None:
            theta = copy.copy(theta)
            if self.__theta.shape[1] != theta.shape[1]:
                raise ValueError('theta has the wrong shape, it does not match emu._emulator__theta.')
        elif cal is not None:
            theta = copy.copy(cal.theta(1000))
            if self.__theta.shape[1] != theta.shape[1]:
                raise ValueError('cal.theta(n) produces the wrong shape.')
        else:
            theta = None
        
        # adding those two lines below
        supptheta = None
        suppx = None
        if thetachoices is not None:
            supptheta, suppinfo = self.method.supplementtheta(self._info, copy.copy(size),
                                                              copy.copy(theta),
                                                              copy.copy(thetachoices),
                                                              copy.copy(choicescost),
                                                              copy.copy(cal),
                                                              argstemp)
            suppx = None
        #elif xchoices is not None:
            # fixing that part, too
        #    suppx, suppinfo = self.method.supplementx(self._info, copy.copy(size),
        #                                                      copy.copy(x),
        #                                                      copy.copy(xchoices),
        #                                                      copy.copy(choicescost),
        #                                                      copy.copy(cal),
        #                                                      argstemp)
        #    supptheta = None
        
        if supptheta is not None and removereps:
            nctheta, ctheta, rtheta = _matrixmatching(self.__theta, supptheta)
        elif supptheta is not None:
            ctheta = np.zeros(0)
            nctheta = np.array(range(0,supptheta.shape[0])).astype('int')
        
        if suppx is None:
            if nctheta.shape[0] < 0.5:
                print('Was not able to assign any new values because everything ' +
                      'was a replication of emu.__theta.')
                self.__supptheta = None
            else:
                if nctheta.shape[0] < supptheta.shape[0]:
                    print('Had to remove replications versus thetas.')
                    supptheta = supptheta[nctheta,:]
                self.__supptheta = copy.copy(supptheta)
        
        if self.__supptheta is not None and self.__suppx is None:
            return copy.copy(self.__supptheta), suppinfo
        else:
            raise ValueError('Something went wrong...')
        
    
    def update(self,x=None, theta=None, f=None,   args=None, options=None):
        r"""
        Chooses a new theta to be investigated.
        
        It can either come from the method or is automatted to use fit and
        predict from the method to complete the operation.

        Parameters
        ----------
        x : optional array of float
            xs you would like to append. Defaults to emu.__x.
            Will attempt to resolve if using all x and emu.__x.
        f : new f values
            A 2d array of responses at (theta, emu.__supptheta and/or emu.__theta)
            and (x, emu.__x).
        theta : optional array of float
            thetas you would like to append. Defaults to emu.__supptheta.
            Will attempt to resolve if using all theta and supptheta.
        args : optional dict
            A dictionary containing options you would like to pass to
            [method].update(f,theta,x,args). 
            Defaults to the one used to build emu.
        options : optional dict
            A dictionary containing options you would like to keep around
            to build the emulator.  Modify with update when you want to change
            it.

        Returns
        -------
        """
        if theta is not None:
            theta = copy.copy(theta)
        if f is not None:
            f = copy.copy(f)
        if x is not None:
            x = copy.copy(x)
        if options is not None:
            self.__optionsset(copy.copy(options))
        if args is not None:
            self._args = {**self._args, **copy.deepcopy(args)} #properly merge the arguments
        if f is not None and theta is None and x is None:
            if f.shape[0] == self.__f.shape[0]:# I think this part is weird. theta can be same theta, f can same f and this will still merge them?
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
                # in whihc cas this will be possible?
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
            if theta.shape[0] != self.__f.shape[1]:
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
            if self.__options['thetareps']:
                    self.__theta = np.vstack((self.__theta, theta))
                    self.__f = np.hstack((self.__f,f))
            else:
                nc, c, r = _matrixmatching(self.__theta, theta)
                self.__f[:, r] = f[:,c]
                if nc.shape[0] > 0.5 and self.__supptheta is not None:
                    f = f[:,nc]
                    theta = theta[nc,:]
                    nc, c, r = _matrixmatching(self.__supptheta, theta)
                    self.__f = np.hstack((self.__f, f[:,c]))
                    self.__theta = np.vstack((self.__theta, theta[c,:]))
                    self.__supptheta = np.delete(self.__supptheta, r, axis = 0)
                if nc.shape[0] > 0.5:
                    f = f[:, nc]
                    theta = theta[nc, :]
                    self.__f = np.hstack((self.__f,f))
                    self.__theta = np.vstack((self.__theta,theta))
        
        if (f is not None) and (theta is None) and (x is not None):
            #changing f.shape[1] to f.shape[0], ALSO there is sth wrong with the rest as well
            if x.shape[0] != f.shape[0]:
                raise ValueError('number of rows in f and rows in x does not align.')
            if x.shape[1] != self.__x.shape[1]:
                raise ValueError('x shape does not match old one,'
                                 + ' use emu.update(x = x) to update it first if' + 
                                 ' you changed your description of x.')
            if f.shape[1] != self.__f.shape[1]:
                raise ValueError('Rows of f are different than those provided originally,' +
                                 'please provide theta to allow for alignment')
            if options['xreps']:
                self.__x = np.vstack((self.__x, x))
                self.__f = np.vstack((self.__f,f))
            else:
                nc, c, r = _matrixmatching(self.__x, x)
                self.__f[r, :] = f[c, :]
                if nc.shape[0] > 0.5:
                    self.__f = np.vstack(self.__f, f[c,:])
        if (f is not None) and (theta is not None) and (x is not None):
                raise ValueError('Simultaneously adding new theta and x at once is currently' +
                                 ' not supported. Please supply either theta OR x.')
        if self.__options['autofit']:
            self.fit()
        return
    
    def remove(self, x=None, theta=None, cal=None, options=None):
        r"""
        Removes either an x or theta value, and the corresponding f values.
        
        It can either come from the method or is automatted to use fit and
        predict from the method to complete the operation.

        Parameters
        ----------
        x : optional array of float
            xs you would like to append. Defaults to emu.__x.
            Will attempt to resolve if using all x and emu.__x.
        theta : optional array of float
            thetas you would like to append. Defaults to emu.__supptheta.
            Will attempt to resolve if using all theta and supptheta.
        cal : optional calibrator object
            A calibrator object that contains information about removal. A user must provide 
            either x, theta or both or another object like cal.

        Returns
        -------
        """
        if cal is not None:
            totalseen = np.where(np.mean(np.logical_not(np.isfinite(self.__f)),0)
                                 < self.__options['thetarmnan'])[0]
            lpdfexisting = cal.theta.lpdf(self.__theta[totalseen,:])
            thetasort = np.argsort(lpdfexisting)
            numcutoff = np.minimum(-500, lpdfexisting[thetasort[max(lpdfexisting.shape[0]-
                                                          10*self.__theta.shape[1],0)]])
            if any(lpdfexisting < numcutoff):
                rmtheta = totalseen[np.where(lpdfexisting < numcutoff)[0]]
                theta = self.__theta[rmtheta,:]
                print('removing %d thetas' % rmtheta.shape[0])
        if (theta is not None):
            nc, c, r = _matrixmatching(theta, self.__theta)
            self.__theta = self.__theta[nc,:]
            self.__f = self.__f[:,nc]
            if self.__options['autofit']:
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
        
        # note: if thetarmnan and xrmnan take a string value other than the things below  np.isfinite(options['thetarmnan']) returns an error
        
        if 'thetarmnan' in options.keys():
            if type(options['thetarmnan']) is bool:
                if options['thetarmnan']:
                    self.__options['thetarmnan'] = 0
                else:
                    self.__options['thetarmnan'] =  1 + (10** (-12))
            elif type(options['thetarmnan']) is str:
                if isinstance(options['thetarmnan'],str) and options['thetarmnan']=='any':
                    self.__options['thetarmnan'] = 0
                elif isinstance(options['thetarmnan'],str) and options['thetarmnan']=='some':
                    self.__options['thetarmnan'] = 0.2
                elif isinstance(options['thetarmnan'],str) and options['thetarmnan']=='most':
                    self.__options['thetarmnan'] = 0.5
                elif isinstance(options['thetarmnan'],str) and options['thetarmnan']=='alot':
                    self.__options['thetarmnan'] = 0.8
                elif isinstance(options['thetarmnan'],str) and options['thetarmnan']=='all':
                    self.__options['thetarmnan'] = 1 - (10** (-8))
                elif isinstance(options['thetarmnan'],str)  and options['thetarmnan']=='never':
                    self.__options['thetarmnan'] = 1 + (10** (-8))
                else:
                    raise ValueError('option thetarmnan must be True, False, ''any'', ''some''' +
                                     ', ''most'', ''alot'', ''all'', ''never'' or an scaler bigger'+
                                     'than zero and less than one.')
            elif np.isfinite(options['thetarmnan']) and options['thetarmnan']>=0\
                and options['thetarmnan']<=1:
                self.__options['thetarmnan'] = options['thetarmnan']
            else:
                #print(options['thetarmnan'])
                raise ValueError('option thetarmnan must be True, False, ''any'', ''some''' +
                                 ', ''most'', ''alot'', ''all'', ''never'' or an scaler bigger'+
                                 'than zero and less than one.')
        if 'xrmnan' in options.keys():
            if type(options['xrmnan']) is bool:
                if options['xrmnan']:
                    self.__options['xrmnan'] = 0
                else:
                    self.__options['xrmnan'] =  1 + (10** (-12))
            elif type(options['xrmnan']) is str:
                if isinstance(options['xrmnan'],str) and options['xrmnan']=='any':
                    self.__options['xrmnan'] = 0
                elif isinstance(options['xrmnan'],str) and options['xrmnan']=='some':
                    self.__options['xrmnan'] = 0.2
                elif isinstance(options['xrmnan'],str) and options['xrmnan']=='most':
                    self.__options['xrmnan'] = 0.5
                elif isinstance(options['xrmnan'],str) and options['xrmnan']=='alot':
                    self.__options['xrmnan'] = 0.8
                elif isinstance(options['xrmnan'],str) and options['xrmnan']=='all':
                    self.__options['xrmnan'] = 1- (10** (-8))
                elif isinstance(options['xrmnan'],str) and  options['xrmnan']=='never':
                    self.__options['xrmnan'] = 1 + (10** (-8))
                else:
                    raise ValueError('option xrmnan must be True, False, ''any'', ''some'''+
                                     ', ''most'', ''alot'', ''all'', ''never'' or a scaler bigger '+
                                     'than zero and less than one.')
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
        
        # first, check missing thetas
        if self.__options['rmthetafirst']:
            j = np.where(np.mean(isnanf, 0) < self.__options['thetarmnan'])[0]
            f = f[:,j]
            if theta.ndim == 1:
                theta = theta[j]
            else:
                theta = theta[j,:]
        # then, check missing xs
        j = np.where(np.mean(isnanf, 1) < self.__options['xrmnan'])[0]
        f = f[j,:]
        if x is not None:
            if x.ndim == 1:
                x = x[j]
            else:
                x = x[j,:]

        if not self.__options['rmthetafirst']:
            j = np.where(np.mean(isnanf, 0) < self.__options['thetarmnan'])[0]
            f = f[:,j]
            if theta.ndim == 1:
                theta = theta[j]
            else:
                theta = theta[j,:]
        return x, theta, f

class prediction(object):
    r"""
    A class to represent an emulation prediction.  
    predict._info will give the dictionary from the method.
    """

    def __init__(self, _info, emu):
        self._info = _info
        self.emu = emu

    def __repr__(self):
        object_method = [method_name for method_name in dir(self)
                  if callable(getattr(self, method_name))]
        object_method = [x for x in object_method if not x.startswith('_')]
        object_method = [x for x in object_method if not x.startswith('emu')]
        strrepr = ('A emulation prediction object predict where the code in located in the file '
                   + ' emulation.  The main method are predict.' +
                   ', predict.'.join(object_method) + '.  Default of predict() is' +
                   ' predict.mean() and ' +
                   'predict(s) will run pred.rnd(s).  Run help(predict) for the document' +
                   ' string.')
        return strrepr

    def __call__(self, s=None, args=None):
        if s is None:
            return self.mean(args)
        else:
            return self.rnd(s, args)
        

    def __methodnotfoundstr(self, pfstr, opstr):
        print(pfstr + opstr + ' functionality not in method... \n' +
              ' Key labeled ' + opstr + ' not ' +
              'provided in ' + pfstr + '._info... \n' +
              ' Key labeled rnd not ' +
              'provided in ' + pfstr + '._info...')
        return 'Could not reconsile a good way to compute this value in current method.'

    def mean(self, args = None):
        r"""
        Returns the mean at theta and x in when building the prediction.
        """
        pfstr = 'predict' #prefix string
        opstr = 'mean' #operation string
        if (self.emu._emulator__ptf is None) and ((pfstr + opstr) in dir(self.emu.method)):
            if args is None:
                args = self.emu._args
            return copy.deepcopy(self.emu.method.predictmean(self._info, args))
        elif opstr in self._info.keys():
            return self._info[opstr]
        elif 'rnd' in self._info.keys():
            return copy.deepcopy(np.mean(self._info['rnd'], 0))
        else:
            raise ValueError(self.__methodnotfoundstr(pfstr, opstr))
    
    def mean_gradtheta(self, args = None):
        r"""
        Returns the gradient of the mean at theta and x  with respect to theta 
        when building the prediction.
        """
        pfstr = 'predict' #prefix string
        opstr = 'mean_gradtheta' #operation string
        if opstr in self._info.keys():
            return self._info[opstr]
        else:
            raise ValueError(self.__methodnotfoundstr(pfstr, opstr))

    def var(self, args = None):
        r"""
        Returns the pointwise variance at theta and x in when building the prediction.
        """
        pfstr = 'predict' #prefix string
        opstr = 'var' #operation string
        if (self.emu._emulator__ptf is None) and ((pfstr + opstr) in dir(self.emu.method)):
            if args is None:
                args = self.emu._args
            return copy.deepcopy(self.emu.method.predictvar(self._info, args))
        elif opstr in self._info.keys():
            return copy.deepcopy(self._info[opstr])
        elif 'rnd' in self._info.keys():
            return copy.deepcopy(np.var(self._info['rnd'], 0))
        else:
            raise ValueError(self.__methodnotfoundstr(pfstr, opstr))

    def covx(self, args = None):
        r"""
        Returns the covariance matrix at theta and x in when building the prediction.
        """
        pfstr = 'predict' #prefix string
        opstr = 'covx' #operation string
        if (self.emu._emulator__ptf is None) and ((pfstr + opstr) in dir(self.emu.method)):
            if args is None:
                args = self.emu._args
            return copy.deepcopy(self.emu.method.predictcov(self._info, args))
        elif opstr in self._info.keys():
            return copy.deepcopy(self._info[opstr])
        elif 'covxhalf' in self._info.keys():
            if self._info['covxhalf'].ndim == 2:
                return self._info['covxhalf'] @ self._info['covxhalf'].T
            else:
                am = self._info['covxhalf'].shape
                covx = np.ones((am[0],am[1],am[0]))
                for k in range(0, self._info['covxhalf'].shape[1]):
                    A = self._info['covxhalf'][:,k,:]
                    covx[:,k,:] = A @ A.T
            self._info['covx'] = covx
            return copy.deepcopy(self._info[opstr])
        else:
            raise ValueError(self.__methodnotfoundstr(pfstr, opstr))

    def covxhalf(self, args = None):
        r"""
        Returns the sqrt of the covariance matrix at theta and x in when building the prediction.
        That is, if this returns A = predict.covhalf(.)[k], than A.T @ A = predict.cov(.)[k]
        """
        pfstr = 'predict' #prefix string
        opstr = 'covxhalf' #operation string
        if (self.emu._emulator__ptf is None) and ((pfstr + opstr) in dir(self.emu.method)):
            if args is None:
                args = self.emu._args
            return copy.deepcopy(self.emu.method.predictcov(self._info, args))
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
            raise ValueError(self.__methodnotfoundstr(pfstr, opstr))
            
    
    def covxhalf_gradtheta(self, args = None):
        r"""
        Returns the sqrt of the covariance matrix at theta and x in when building the prediction.
        That is, if this returns A = predict.covhalf(.)[k], than A.T @ A = predict.cov(.)[k]
        """
        pfstr = 'predict' #prefix string
        opstr = 'covxhalf_gradtheta' #operation string
        if opstr in self._info.keys():
            return self._info[opstr]
        else:
            raise ValueError(self.__methodnotfoundstr(pfstr, opstr))

    def rnd(self, s=100, args=None):
        r"""
        Returns a rnd draws of size s at theta and x 
        """
        raise ValueError('rnd functionality not in method')

    def lpdf(self, f=None, args=None):
        r"""
        Returns a log pdf at theta and x 
        """
        raise ValueError('lpdf functionality not in method')

#### Below are some functions that I found useful.

def _matrixmatching(mat1, mat2):
    #This is an internal function to do matching between two vectors
    #it just came up alot
    #It returns the where each row of mat2 is first found in mat1
    #If a row of mat2 is never found in mat1, then 'nan' is in that location
    
    
    if (mat1.shape[0] > (10 ** (4))) or (mat2.shape[0] > (10 ** (4))):
        raise ValueError('too many matchings attempted.  Don''t make the method work so hard!')
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