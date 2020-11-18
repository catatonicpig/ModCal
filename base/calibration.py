"""Header here."""
import numpy as np, importlib, base.emulation, copy
from base.utilities import postsampler

#need to make 'info' defined at the class level

class calibrator(object):
    r"""
    A class to represent a calibrator.  cal.info will give the dictionary from the method.
    """

    def __init__(self, emu=None, y=None, x=None, thetaprior=None, yvar=None, method='BDM', args={}):
        r"""
        Intitalizes a calibration model.

        It directly calls "calibrationmethods.[method]" where [method] is
        the user option with default listed above. If you would like to change this method, just
        drop a new file in the "calibrationmethods" folder with the required formatting.

        Parameters
        ----------
        emu : class
            An emulator class instatance as defined in emulation
        y : array of float
            A one demensional array of observed values at x
        x : array of float
            An array of x values that match the definition of "emu.x".  Currently, it must be a
            subset of "emu.x".
        thetaprior : distribution class instatance with two built in functions
            thetaprior.rnd(n) :  produces n draws from the prior on theta, arranged in either a
            matrix or list.  It must align with the defination in "emu.theta"
            thetaprior.lpdf(theta) :  produces the log pdf from the prior on theta, arranged in a
            vector.
        yvar : array of float
            A one demensional array of the variance of observed values at x.  This is not required.
        method : str
            A string that points to the file located in "calibrationmethods" you would
            like to use.
        args : dict
            A dictionary containing options you would like to pass to fit and/or predict

        Returns
        -------
        cal : instance of calibration class
            An instance of the calibration class that can be used with the functions listed below.
        """
        self.args = args
        if y is None:
            raise ValueError('You have not provided any y.')
        if y.ndim > 1.5:
            y = np.squeeze(y)
        if y.shape[0] < 5:
            raise ValueError('5 is the minimum number of observations at this time.')
        self.y = y
        if emu is None:
            raise ValueError('You have not provided any emulator.')
        self.emu = emu
        if thetaprior is None:
            raise ValueError('You have not provided any prior function, stopping...')
        try:
            thetatestsamp = thetaprior.rnd(100)
            if thetatestsamp.shape[0] != 100:
                raise ValueError('thetaprior.rnd(100) failed to give 100 values.')
        except:
            raise ValueError('thetaprior.rnd(100) failed.')
        
        thetatestlpdf = thetaprior.lpdf(thetatestsamp)
        if thetatestlpdf.shape[0] != 100:
            raise ValueError('thetaprior.lpdf(thetaprior.rnd(100)) failed to give 100 values.')
        if thetatestlpdf.ndim != 1:
            raise ValueError('thetaprior.lpdf(thetaprior.rnd(100)) is demension higher than 1.')
       # except:
        #    print(thetaprior.lpdf(thetaprior.rnd(100)))
        #    raise ValueError('thetaprior.lpdf(thetaprior.rnd(100)) failed.')
        self.info = {}
        self.info['thetaprior'] = copy.deepcopy(thetaprior)
        
        if x is not None:
            if x.shape[0] != y.shape[0]:
                raise ValueError('If x is provided, shape[0] must align with the length of y.')
        self.x = copy.deepcopy(x)
        predtry = emu.predict(copy.copy(self.x), thetatestsamp)
        if y.shape[0] != predtry().shape[0]:
            if x is None:
                raise ValueError('y and emu.predict(theta) must have the same shape')
            else:
                raise ValueError('y and emu.predict(x,theta) must have the same shape')
        else:
            prednotfinite = np.logical_not(np.isfinite(predtry()))
            if np.any(prednotfinite):
                print('We have received some non-finite values from emulation.')
                fracfail = np.mean(prednotfinite, 1)
                if np.sum(fracfail <= 10**(-3)) < 5:
                    raise ValueError('Your emulator failed enough places to give up.')
                else: 
                    print('It looks like some locations are ok.')
                    print('Current protocol is to remove observations that have nonfinite values.')
                    whichrm = np.where(fracfail > 10**(-3))[0]
                    print('Removing values at %s.' % np.array2string(whichrm))
                    whichkeep = np.where(fracfail <= 10**(-3))[0]
                    if x is not None:
                        self.x = self.x[whichkeep,:]
                    self.y = self.y[whichkeep]
            else:
                whichkeep = None
        if yvar is not None:
            if yvar.shape[0] != y.shape[0] and yvar.shape[0] > 1.5:
                raise ValueError('yvar must be the same size as y or of size 1.')
            if np.min(yvar) < 0:
                raise ValueError('yvar has at least one negative value.')
            if np.min(yvar) < 10 ** (-6) or np.max(yvar) > 10 ** (6):
                raise ValueError('please rescale your problem so that the yvar is between 10 ^ -6 and 10 ^ 6.')
            self.info['yvar'] = copy.deepcopy(yvar)
            if whichkeep is not None:
                self.info['yvar'] = self.info['yvar'][whichkeep]
        self.method = importlib.import_module('base.calibrationmethods.' + method)
        try:
            self.method = importlib.import_module('base.calibrationmethods.' + method)
        except:
            raise ValueError('Module not found!')
        
        self.fit()

    def __repr__(self):
        object_method = [method_name for method_name in dir(self)
                  if callable(getattr(self, method_name))]
        object_method = [x for x in object_method if not x.startswith('__')]
        object_method = [x for x in object_method if not x.startswith('emu')]
        strrepr = ('A calibration object where the code in located in the file '
                   + ' calibration.  The main method are cal.' +
                   ', cal.'. join(object_method) + '.  Default of cal(x) is cal.predict(x).' +
                   '  Run help(cal) for the document string.')
        return strrepr
    
    
    def __call__(self, x=None):
        return self.predict(x)


    def fit(self, args=None):
        r"""
        Returns a draws from theta and phi given data.

        It directly calls  \"calibrationmethods.[method].fit\" where \"[method]\" is
        the user option.

        Parameters
        ----------
        args : dict
            A dictionary containing options you would like to pass
        """
        if args is None:
            args = self.args
        self.method.fit(self.info, self.emu, self.x, self.y, args)
        if hasattr(self, 'theta'):
            del self.theta
        newtheta = thetadist(self)
        self.theta = newtheta
        return


    def predict(self, x=None, args=None):
        r"""
        Returns a predictions at x.

        It directly calls  \"calibrationmethods.[method].predict\" where \"[method]\" is
        the user option.

        Parameters
        ----------
        x : array of objects
            An array of inputs to the model where you would like to predict.
        args : dict
            A dictionary containing options you would like to pass.

        Returns
        -------
        prediction : an instance of calibration class prediction
            prediction.predinfo : Gives the dictionary of what was produced by the method.
        """
        if args is None:
            args = self.args
        if args is x:
            x = self.x
        info = {}
        self.method.predict(info, self.info, self.emu, x, args)
        return prediction(info, self)


class prediction(object):
    r"""
    A class to represent a calibration prediction.  
    predict.info will give the dictionary from the method.
    """

    def __init__(self, info, cal):
        self.info = info
        self.cal = cal

    def __repr__(self):
        object_method = [method_name for method_name in dir(self)
                  if callable(getattr(self, method_name))]
        object_method = [x for x in object_method if not x.startswith('_')]
        object_method = [x for x in object_method if not x.startswith('cal')]
        strrepr = ('A calibration prediction object predict where the code in located in the file '
                   + ' calibration.  The main method are predict.' +
                   ', predict.'.join(object_method) + '.  Default of predict() is' +
                   ' predict.mean() and ' +
                   'predict(s) will run predict.rnd(s).  Run help(predict) for the document' +
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
              'provided in ' + pfstr + '.info... \n' +
              ' Key labeled rnd not ' +
              'provided in ' + pfstr + '.info...')
        return 'Could not reconsile a good way to compute this value in current method.'

    def mean(self, args = None):
        r"""
        Returns the mean at all x in when building the prediction.
        """
        pfstr = 'predict' #prefix string
        opstr = 'mean' #operation string
        if (pfstr + opstr) in dir(self.cal.method):
            if args is None:
                args = self.cal.args
            return copy.deepcopy(self.cal.method.predictmean(self.info, args))
        elif opstr in self.info.keys():
            return copy.deepcopy(self.info[opstr])
        elif 'rnd' in self.info.keys():
            self.info[opstr] = np.mean(self.info['rnd'], 0)
            return copy.deepcopy(self.info[opstr])
        else:
            raise ValueError(self.__methodnotfoundstr(pfstr, opstr))

    def var(self, args = None):
        r"""
        Returns the variance at all x in when building the prediction.
        """
        pfstr = 'predict' #prefix string
        opstr = 'var' #operation string
        if (pfstr + opstr) in dir(self.cal.method):
            if args is None:
                args = self.cal.args
            return copy.deepcopy(self.cal.method.predictvar(self.info, args))
        elif opstr in self.info.keys():
            return copy.deepcopy(self.info[opstr])
        elif 'rnd' in self.info.keys():
            self.info[opstr] = np.var(self.info['rnd'], 0)
            return copy.deepcopy(self.info[opstr])
        else:
            raise ValueError(self.__methodnotfoundstr(pfstr, opstr))
    
    def rnd(self, s=100, args=None):
        r"""
        Returns s random draws at all x in when building the prediction.
        """
        pfstr = 'predict' #prefix string
        opstr = 'rnd' #operation string
        if (pfstr + opstr) in dir(self.cal.method):
            if args is None:
                args = self.cal.args
            return copy.deepcopy(self.cal.method.predictrnd(self.info, args))
        elif 'rnd' in self.info.keys():
            return self.info['rnd'][np.random.choice(self.info['rnd'].shape[0], size=s), :]
        else:
            raise ValueError(self.__methodnotfoundstr(pfstr, opstr))

    def lpdf(self, y=None, args=None):
        r"""
        Returns a log pdf given theta.
        """
        raise ValueError('lpdf functionality not in method')

class thetadist(object):
    r"""
    A class to represent a theta predictive distribution.
    """
    
    def __init__(self, cal):
        self.cal = cal

    def __repr__(self):
        object_method = [method_name for method_name in dir(self)
                  if callable(getattr(self, method_name))]
        object_method = [x for x in object_method if not x.startswith('_')]
        object_method = [x for x in object_method if not x.startswith('cal')]
        strrepr = ('A theta distribution object where the code in located in the file '
                   + ' calibration.  The main method are cal.theta' +
                   ', cal.theta.'.join(object_method) + '.  Default of predict() is' +
                   ' cal.theta.mean() and ' +
                   'cal.theta(s) will cal.theta.rnd(s).  Run help(cal.theta) for the document' +
                   ' string.')
        return strrepr

    def __call__(self, s=None, args=None):
        if s is None:
            return self.mean(args)
        else:
            return self.rnd(s, args)
    
    def __methodnotfoundstr(self, pfstr, opstr):
        print(pfstr + opstr + 'functionality not in method... \n' +
              ' Key labeled ' + (pfstr+opstr) + ' not ' +
              'provided in cal.info... \n' +
              ' Key labeled ' + pfstr + 'rnd not ' +
              'provided in cal.info...')
        return 'Could not reconsile a good way to compute this value in current method.'
    
    def mean(self, args = None):
        r"""
        Returns mean of each element of theta found during calibration.
        """
        pfstr = 'theta' #prefix string
        opstr = 'mean' #operation string
        if (pfstr + opstr) in dir(self.cal.method):
            if args is None:
                args = self.cal.args
            return copy.deepcopy(self.cal.method.thetamean(self.cal.info, args))
        elif (pfstr+opstr) in self.cal.info.keys():
            return copy.deepcopy(self.cal.info[(pfstr+opstr)])
        elif (pfstr+'rnd') in self.cal.info.keys():
            return np.mean(self.cal.info[(pfstr+'rnd')], 0)
        else:
            raise ValueError(self.__methodnotfoundstr(pfstr, opstr))

    def var(self, args = None):
        r"""
        Returns predictive variance of each element of theta found during calibration.
        """
        pfstr = 'theta'  # prefix string
        opstr = 'var'  # operation string
        if (pfstr + opstr) in dir(self.cal.method):
            if args is None:
                args = self.cal.args
            return copy.deepcopy(self.cal.method.thetavar(self.cal.info, args))
        elif (pfstr+opstr) in self.cal.info.keys():
            return copy.deepcopy(self.cal.info[(pfstr+opstr)])
        elif (pfstr+'rnd') in self.cal.info.keys():
            return np.var(self.cal.info[(pfstr+'rnd')], 0)
        else:
            raise ValueError(self.__methodnotfoundstr(pfstr, opstr))
    
    def rnd(self, s=1000, args=None):
        r"""
        Returns s predictive draws for theta found during calibration.
        """
        pfstr = 'theta' #prefix string
        opstr = 'rnd' #operation string
        if (pfstr + opstr) in dir(self.cal.method):
            if args is None:
                args = self.cal.args
            return copy.deepcopy(self.cal.method.thetarnd(self.cal.info, s, args))
        elif (pfstr+opstr) in self.cal.info.keys():
            return self.cal.info['thetarnd'][
                        np.random.choice(self.cal.info['thetarnd'].shape[0], size=s), :]
        else:
            raise ValueError(self.__methodnotfoundstr(pfstr, opstr))
    
    def lpdf(self, theta=None, args=None):
        r"""
        Returns a log pdf given theta.
        """
        raise ValueError('lpdf functionality not in method')