"""Header here."""
import numpy as np, importlib, base.emulation, copy
from base.utilities import postsampler

#need to make 'info' defined at the class level

class calibrator(object):
    r"""
    A class to represent a calibrator.  cal.info will give the dictionary from the software.
    """

    def __init__(self, emu=None, y=None, x=None, thetaprior=None, software='BDM', args={}):
        r"""
        Intitalizes a calibration model.

        It directly calls "calibrationsubfuncs.[software]" where [software] is
        the user option with default listed above. If you would like to change this software, just
        drop a new file in the "calibrationsubfuncs" folder with the required formatting.

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
            thetaprior.rvs(n) :  produces n draws from the prior on theta, arranged in either a
            matrix or list.  It must align with the defination in "emu.theta"
            thetaprior.logpdf(theta) :  produces the log pdf from the prior on theta, arranged in a
            vector.
        software : str
            A string that points to the file located in "calibrationsubfuncs" you would
            like to use.
        args : dict
            A dictionary containing options you would like to pass to fit and/or predict

        Returns
        -------
        cal : instance of calibration class
            An instance of the calibration class that can be used with the functions listed below.
        """
        if y is None:
            raise ValueError('You have not provided any y.')
        else:
            if emu is None:
                raise ValueError('You have not provided any emulator.')
            else:
                if thetaprior is None:
                    raise ValueError('You have not provided any prior function, stopping...')
                elif type(emu) is tuple:
                    for k in range(0, len(emu)):
                        try:
                            ftry = emu[k].predict(emu[k].theta[0, :]).mean()
                        except:
                            raise ValueError('Your provided emulator failed to predict.')
                else:
                    try:
                        ftry = emu.predict(copy.deepcopy(emu.theta[0, :]), 
                                           x=copy.deepcopy(emu.x[range(0,10,2), :]))
                    except:
                        raise ValueError('Your provided emulator failed to predict.')
            self.emu = emu
            self.y = y
            
            if x is None:
                if y.shape[0] != ftry.shape[0]:
                    raise ValueError('If x is not provided, predictions must align with y and emu.predict()')
            if x is not None:
                if x.shape[0] != y.shape[0]:
                    raise ValueError('If x is provided, predictions must align with y and emu.predict()')
                else:
                    if x is not None:
                        self.x = x
                    else:
                        self.x = self.emu0.x
            try:
                self.software = importlib.import_module('base.calibrationsubfuncs.' + software)
            except:
                raise ValueError('Module not found!')
            
            self.info = {}
            self.args = args
            if thetaprior is None:
                raise ValueError('You must give a prior for theta.')
            else:
                self.info['thetaprior'] = thetaprior
            
            self.fit()
            self.theta = thetadist(self)

    def __repr__(self):
        object_methods = [method_name for method_name in dir(self)
                  if callable(getattr(self, method_name))]
        object_methods = [x for x in object_methods if not x.startswith('__')]
        object_methods = [x for x in object_methods if not x.startswith('emu')]
        strrepr = ('A calibration object where the code in located in the file '
                   + ' calibration.  The main methods are cal.' +
                   ', cal.'. join(object_methods) + '.  Default of cal(x) is cal.predict(x).' +
                   '  Run help(cal) for the document string.')
        return strrepr
    
    
    def __call__(self, x=None):
        return self.predict(x)


    def fit(self, args=None):
        r"""
        Returns a draws from theta and phi given data.

        It directly calls  \"calibrationsubfuncs.[software].fit\" where \"[software]\" is
        the user option.

        Parameters
        ----------
        args : dict
            A dictionary containing options you would like to pass
        """
        if args is None:
            args = self.args
        
        self.software.fit(self.info, self.emu, self.x, self.y, args)
        return None


    def predict(self, x=None, args=None):
        r"""
        Returns a predictions at x.

        It directly calls  \"calibrationsubfuncs.[software].predict\" where \"[software]\" is
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
            prediction.predinfo : Gives the dictionary of what was produced by the software.
        """
        if args is None:
            args = self.args
        if args is x:
            x = self.x
        info = {}
        self.software.predict(info, self.info, self.emu, x, args)
        return prediction(info, self)


class prediction(object):
    r"""
    A class to represent a calibration prediction.  
    predict.info will give the dictionary from the software.
    """

    def __init__(self, info, cal):
        self.info = info
        self.cal = cal

    def __repr__(self):
        object_methods = [method_name for method_name in dir(self)
                  if callable(getattr(self, method_name))]
        object_methods = [x for x in object_methods if not x.startswith('_')]
        object_methods = [x for x in object_methods if not x.startswith('cal')]
        strrepr = ('A calibration prediction object predict where the code in located in the file '
                   + ' calibration.  The main methods are predict.' +
                   ', predict.'.join(object_methods) + '.  Default of predict() is' +
                   ' predict.mean() and ' +
                   'predict(s) will run predict.rnd(s).  Run help(predict) for the document' +
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
        Returns the mean at all x in when building the prediction.
        """
        pfstr = 'predict' #prefix string
        opstr = 'mean' #operation string
        if (pfstr + opstr) in dir(self.cal.software):
            if args is None:
                args = self.cal.args
            return copy.deepcopy(self.cal.software.predictmean(self.info, args))
        elif opstr in self.info.keys():
            return copy.deepcopy(self.info[opstr])
        elif 'rnd' in self.info.keys():
            self.info[opstr] = np.mean(self.info['rnd'], 0)
            return copy.deepcopy(self.info[opstr])
        else:
            raise ValueError(self.__softwarenotfoundstr(pfstr, opstr))

    def var(self, args = None):
        r"""
        Returns the variance at all x in when building the prediction.
        """
        pfstr = 'predict' #prefix string
        opstr = 'var' #operation string
        if (pfstr + opstr) in dir(self.cal.software):
            if args is None:
                args = self.cal.args
            return copy.deepcopy(self.cal.software.predictvar(self.info, args))
        elif opstr in self.info.keys():
            return copy.deepcopy(self.info[opstr])
        elif 'rnd' in self.info.keys():
            self.info[opstr] = np.var(self.info['rnd'], 0)
            return copy.deepcopy(self.info[opstr])
        else:
            raise ValueError(self.__softwarenotfoundstr(pfstr, opstr))
    
    def rnd(self, s=100, args=None):
        r"""
        Returns s random draws at all x in when building the prediction.
        """
        pfstr = 'predict' #prefix string
        opstr = 'rnd' #operation string
        if (pfstr + opstr) in dir(self.cal.software):
            if args is None:
                args = self.cal.args
            return copy.deepcopy(self.cal.software.predictrnd(self.info, args))
        elif 'rnd' in self.info.keys():
            return self.info['rnd'][np.random.choice(self.info['rnd'].shape[0], size=s), :]
        else:
            raise ValueError(self.__softwarenotfoundstr(pfstr, opstr))

    def lpdf(self, y=None, args=None):
        r"""
        Returns a log pdf given theta.
        """
        raise ValueError('lpdf functionality not in software')

class thetadist(object):
    r"""
    A class to represent a theta predictive distribution.
    """
    
    def __init__(self, cal):
        self.cal = cal

    def __repr__(self):
        object_methods = [method_name for method_name in dir(self)
                  if callable(getattr(self, method_name))]
        object_methods = [x for x in object_methods if not x.startswith('_')]
        object_methods = [x for x in object_methods if not x.startswith('cal')]
        strrepr = ('A theta distribution object where the code in located in the file '
                   + ' calibration.  The main methods are cal.theta' +
                   ', cal.theta.'.join(object_methods) + '.  Default of predict() is' +
                   ' cal.theta.mean() and ' +
                   'cal.theta(s) will cal.theta.rnd(s).  Run help(cal.theta) for the document' +
                   ' string.')
        return strrepr

    def __call__(self, s=None, args=None):
        if s is None:
            return self.mean(args)
        else:
            return self.rnd(s, args)
    
    def __softwarenotfoundstr(self, pfstr, opstr):
        print(pfstr + opstr + 'functionality not in software... \n' +
              ' Key labeled ' + (pfstr+opstr) + ' not ' +
              'provided in cal.info... \n' +
              ' Key labeled ' + pfstr + 'rnd not ' +
              'provided in cal.info...')
        return 'Could not reconsile a good way to compute this value in current software.'
    
    def mean(self, args = None):
        r"""
        Returns mean of each element of theta found during calibration.
        """
        pfstr = 'theta' #prefix string
        opstr = 'mean' #operation string
        if (pfstr + opstr) in dir(self.cal.software):
            if args is None:
                args = self.cal.args
            return copy.deepcopy(self.cal.software.thetamean(self.cal.info, args))
        elif (pfstr+opstr) in self.cal.info.keys():
            return copy.deepcopy(self.cal.info[(pfstr+opstr)])
        elif (pfstr+'rnd') in self.cal.info.keys():
            self.cal.info[(pfstr+opstr)] = np.mean(self.cal.info[(pfstr+'rnd')], 0)
            return copy.deepcopy(self.cal.info[(pfstr+opstr)])
        else:
            raise ValueError(self.__softwarenotfoundstr(pfstr, opstr))

    def var(self, args = None):
        r"""
        Returns predictive variance of each element of theta found during calibration.
        """
        pfstr = 'theta'  # prefix string
        opstr = 'var'  # operation string
        if (pfstr + opstr) in dir(self.cal.software):
            if args is None:
                args = self.cal.args
            return copy.deepcopy(self.cal.software.thetavar(self.cal.info, args))
        elif (pfstr+opstr) in self.cal.info.keys():
            return copy.deepcopy(self.cal.info[(pfstr+opstr)])
        elif (pfstr+'rnd') in self.cal.info.keys():
            self.cal.info[(pfstr+opstr)] = np.var(self.cal.info[(pfstr+'rnd')], 0)
            return copy.deepcopy(self.cal.info[(pfstr+opstr)])
        else:
            raise ValueError(self.__softwarenotfoundstr(pfstr, opstr))
    
    def rnd(self, s=100, args=None):
        r"""
        Returns s predictive draws for theta found during calibration.
        """
        pfstr = 'theta' #prefix string
        opstr = 'rnd' #operation string
        if (pfstr + opstr) in dir(self.cal.software):
            if args is None:
                args = self.cal.args
            return copy.deepcopy(self.cal.software.thetarnd(self.cal.info, s, args))
        elif (pfstr+opstr) in self.cal.info.keys():
            return self.cal.info['thetarnd'][
                        np.random.choice(self.cal.info['thetarnd'].shape[0], size=s), :]
        else:
            raise ValueError(self.__softwarenotfoundstr(pfstr, opstr))
    
    def lpdf(self, theta=None, args=None):
        r"""
        Returns a log pdf given theta.
        """
        raise ValueError('lpdf functionality not in software')