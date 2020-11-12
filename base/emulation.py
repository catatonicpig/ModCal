# -*- coding: utf-8 -*-
"""Header here."""
import numpy as np
import importlib
import scipy.stats as sps
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
            to have.  This does not get passed to the software.  Some options are below:
                

        Returns
        -------
        emu : instance of emulation class
            An instance of the emulation class that can be used with the functions listed below.
        """
        self.__options = {}
        self.__optionsset(options)
        
        if f is None or theta is None:
            raise ValueError('You have no provided any theta and/or f.')
        
        if f.ndim < 0.5 or f.ndim > 2.5:
            raise ValueError('f must have either 1 or 2 demensions.')
        
        if f.shape[0] is not theta.shape[0]:
            raise ValueError('The rows in f must match' +
                             ' the rows in theta')
            
        if f.ndim == 1 and x is not None:
            raise ValueError('Cannot use x if f has a single column.')
        
        if f.ndim == 2 and x is not None and\
            f.shape[1] is not x.shape[0]:
            raise ValueError('The columns in f must match' +
                             ' the rows in x')
        
        if theta.ndim < 0.5 or theta.ndim > 2.5:
            raise ValueError('theta must have either 1 or 2 demensions.')
        
        if theta.shape[0] < self.__options['minsampsize']:
            raise ValueError('theta should have at least minsampsize' +
                             'rows.  change this in options if you do not like it.')
            
        self.__theta = copy.deepcopy(theta)
        self.__supptheta = None
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
    
    def __call__(self, theta=None, x=None):
        if theta is None:
            theta = self.__theta
        if x is None:
            x = self.__x
        return self.predict(theta, x)

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
        theta, f, x = self.__preprocess()
        self.software.fit(self._info, theta, f, x, args = args)


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
        info = {}
        self.software.predict(info, self._info, copy.deepcopy(theta),
                              copy.deepcopy(x), copy.deepcopy(args))
        return prediction(info, self)
    
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
        
        if 'reps' in self.__options.keys():
            allowreps = self.__options['reps']
        if args is None:
            args = self._args
        if n < 0.5:
            if n == 0:
                return copy.deepcopy(self.__supptheta)
            else:
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
        
        if 'supplement' not in dir(self.software):
            print('Using the default supplement function.')
            supptheta = copy.deepcopy(self.__defaultsupp(n, copy.deepcopy(thetadraw)))
        else:
            supptheta = copy.deepcopy(self.software.supplement(n, self._info, copy.deepcopy(thetadraw)))
        if append and self.__supptheta is not None:
            if not allowreps:
                nc, c, r = _matrixmatching(self.__supptheta, supptheta)
            else:
                nc = np.array(range(0,supptheta.shape[0])).astype('int')
            if nc.shape[0] < 0.5:
                print('Was not able to assign any new values because everything ' +
                      'was a replication of emu.__supptheta.')
                return self.supptheta
            elif nc.shape[0] < supptheta.shape[0]:
                print('Had to remove replications versus previous emu.__supptheta.')
                supptheta = supptheta[nc,:]
        if not allowreps:
            nc, c, r = _matrixmatching(self.__theta, supptheta)
        else:
            nc = np.array(range(0,supptheta.shape[0])).astype('int')
        if nc.shape[0] < 0.5:
            print('Was not able to assign any new values because everything ' +
                  'was a replication of emu.__theta.')
            if not append:
                self.__supptheta = None
        else:
            if nc.shape[0] < supptheta.shape[0]:
                print('Had to remove replications versus emu.__theta.')
                supptheta = supptheta[nc,:]
        if not append or self.__supptheta is None:
            self.__supptheta = supptheta
        else:
            self.__supptheta = np.append(self.__supptheta, supptheta, 0)
        
        return copy.deepcopy(self.__supptheta)
    
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
            if f.shape[1] == self.__f.shape[1]:
                if self.__supptheta is not None:
                    if f.shape[0] == self.__supptheta.shape[0]:
                        self.__theta = np.vstack((self.__theta, self.__supptheta))
                        self.__f = np.vstack((self.__f,f))
                        self.__supptheta = None
                    elif f.shape[0] == (self.__theta.shape[0] + self.__supptheta.shape[0]):
                        self.__theta = np.vstack((self.__theta, self.__supptheta))
                        self.__f = f
                        self.__supptheta = None
                    else:
                        raise ValueError('Could not resolve absense of theta,' +
                                     'please provide theta')
                elif f.shape[0] == self.__theta.shape[0]:
                    self.__f = f
                else:
                    raise ValueError('Could not resolve absense of theta,' +
                                     'please provide theta')
            else:
                raise ValueError('Could not resolve absense of x,' +
                                 'please provide x')
        if (x is not None) and (f is None):
            if x.shape[0] != self.__f.shape[1]:
                print('you have change the number of x, but not provided a new f...')
            else:
                self.__x = x
        if (theta is not None) and (f is None):
            if theta.shape[0] != self.__f.shape[0]:
                print('you have change the number of theta, but not provided a new f...')
            else:
                self.__theta = theta
        
        if (f is not None) and (theta is not None) and (x is None):
            if theta.shape[0] != f.shape[0]:
                raise ValueError('number of rows between theta and f does not align.')
            if theta.shape[1] != self.__theta.shape[1]:
                raise ValueError('theta shape does not match old one,'
                                 + ' use emu.update(theta = theta) to update it first if' + 
                                 ' you changed your parameterization.')
            if f.shape[1] != self.__f.shape[1]:
                raise ValueError('Columns of f are different than those provided originally,' +
                                 'please provide x to allow for alignment')
            if self.__options['reps']:
                    self.__theta = np.vstack((self.__theta, theta))
                    self.__f = np.vstack((self.__f,f))
            else:
                nc, c, r = _matrixmatching(self.__theta, theta)
                self.__f[r, :] = f[c,:]
                if nc.shape[0] > 0.5:
                    f = f[nc, :]
                    theta = theta[nc, :]
                    nc, c, r = _matrixmatching(self.__supptheta, theta)
                    self.__f = np.vstack((self.__f, f[c,:]))
                    self.__theta = np.vstack((self.__theta, theta[c,:]))
                    self.__supptheta = np.delete(self.__supptheta, r, axis = 0)
                if nc.shape[0] > 0.5:
                    f = f[nc, :]
                    theta = theta[nc, :]
                    self.__f = np.vstack(self.__f,f[c,:])
                    self.__theta = np.vstack(self.__f,theta[c,:])
        
        if (f is not None) and (theta is None) and (x is not None):
            if x.shape[0] != f.shape[1]:
                raise ValueError('number of columns in f and rows in x does not align.')
            if x.shape[1] != self.__x.shape[1]:
                raise ValueError('x shape does not match old one,'
                                 + ' use emu.update(x = x) to update it first if' + 
                                 ' you changed your description of x.')
            if f.shape[0] != self.__f.shape[0]:
                raise ValueError('Rows of f are different than those provided originally,' +
                                 'please provide theta to allow for alignment')
            if options['reps']:
                self.__x = np.vstack((self.__x, x))
                self.__f = np.hstack((self.__f,f))
            else:
                nc, c, r = _matrixmatching(self.__x, x)
                self.__f[:, r] = f[:, c]
                if nc.shape[0] > 0.5:
                    self.__f = np.hstack(self.__f, f[c,:])
        
        if (f is not None) and (theta is not None) and (x is not None):
                raise ValueError('Simultaneously adding new theta and x at once is currently'+
                                 'not supported.  Please supply either theta OR x.')
        
        self.fit()
        return
    
    
    def __optionsset(self, options=None):
        options = copy.deepcopy(options)
        options =  {k.lower(): v for k, v in options.items()} #options will always be lowercase
        
        if 'reps' in options.keys():
            if type(options['reps']) is bool:
                self.__options['reps'] = options['reps']
            else:
                raise ValueError('option reps must be true or false')
        
        if 'rmnan' in options.keys():
            if type(options['rmnan']) is bool:
                if options['rmnan']:
                    self.__options['rmnan'] = 'all'
                else:
                    self.__options['rmnan'] = 'never'
            elif options['rmnan'] is str and (options['rmnan']=='any' or 
                                              options['rmnan']=='all' or
                                              options['rmnan']=='never'):
                self.__options['rmnan'] = options['rmnan']
            else:
                raise ValueError('option rmnan must be True, False, ''all'', ''any'', ''never''')
        
        if 'minsampsize' in options.keys():
            if type(options['minsampsize']) is int and options['minsampsize'] > -0.5:
                self.__options['minsampsize'] = options['minsampsize']
            elif type(options['minsampsize']) is False:
                self.__options['rmnan'] = options['rmnan']
            else:
                raise ValueError('option minsampsize must be False or postitive integer.')
        
        if 'reps' not in self.__options.keys():
            self.__options['reps'] = False
        if 'rmnan' not in self.__options.keys():
            self.__options['rmnan'] = 'all'
        if 'minsampsize' not in self.__options.keys():
            self.__options['minsampsize'] = 10

    def __preprocess(self):
        theta = copy.copy(self.__theta)
        f = copy.copy(self.__f)
        x = copy.copy(self.__x)
        options = self.__options
        
        
        isinff = np.isinf(f)
        if np.any(isinff):
            print('All infs were converted to nans.')
            f[isinff] = float("NaN")
        
        isnanf = np.isnan(f)
        if options['rmnan'] == 'all':
            rownanf = np.all(isnanf,1)
        if options['rmnan'] == 'any':
            rownanf = np.any(isnanf,1)
        if options['rmnan'] != 'never' and np.any(rownanf):
            print('Row(s) %s removed due to nans.' % np.array2string(np.where(rownanf)[0]))
            j = np.where(np.logical_not(rownanf))[0]
            f = f[j,:]
            theta = theta[j,:]
        
        if options['minsampsize'] < 2 * theta.shape[1] and options['minsampsize'] == 10:
            options['minsampsize'] = 2 * theta.shape[1]
        
        colnumdone = np.sum(1-isnanf,0)
        notenoughvals = (colnumdone < options['minsampsize'])
        if np.any(notenoughvals):
            print('Column(s) %s removed due to not enough completed values.'
                  % np.array2string(np.where(notenoughvals)[0]))
            j = np.where(np.logical_not(notenoughvals))[0]
            f = f[:,j]
            x = x[j,:]
        return theta, f, x
    
    def __defaultsupp(self, n, supptheta):
        #if not self.__options['reps']:
        #    supptheta = np.unique(supptheta, axis=0)
        
        theta, f, x = self.__preprocess()
        pred = self.predict(supptheta, x)
        fstd = np.nanstd(f,0)
        fstd = np.maximum(fstd,10 ** (-10) * np.max(fstd))
        scvar = np.mean(pred.var(),1)
        mval = pred.mean()
        orderedlist = np.arange(np.minimum(supptheta.shape[0], 2*n))
        fulllist = np.arange(supptheta.shape[0])
        infotemp = copy.deepcopy(self._info)
        suppthetatemp = copy.copy(supptheta)
        val = np.zeros(2*n)
        for k in range(0, np.minimum(supptheta.shape[0], 2*n)):
            jstar = np.argmax(scvar)
            orderedlist[k] = fulllist[jstar]
            val[k] = scvar[jstar]
            predinfo = {}
            theta = np.vstack((theta,suppthetatemp[jstar,:]))
            f = np.vstack((f,mval[jstar,:]))
            self.software.fit(infotemp, theta, f, x, self._args)
            predinfo = {}
            self.software.predict(predinfo, infotemp, copy.deepcopy(suppthetatemp),
                                  copy.deepcopy(x), self._args)
            pred = prediction(predinfo, self)
            mval = pred.mean()
            H = pred.cov()[jstar]
            #mval += H @  sps.norm.rvs(0,1,H.shape[1])
            scvar = np.nanmean(pred.var() ,1)
            val[k] -= scvar[jstar]
            if k > 0.5:
                if val[k] > val[0] * 100:
                    orderedlist = orderedlist[:(k-1)]
                    break
            if not self.__options['reps']:
                suppthetatemp = np.delete(suppthetatemp, jstar, axis = 0)
                scvar = np.delete(scvar, jstar)
                mval = np.delete(mval, jstar, axis = 0)
                fulllist = np.delete(fulllist, jstar)
        supptheta = supptheta[orderedlist,:]
        return supptheta[:n,:]



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