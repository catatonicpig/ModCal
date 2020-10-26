"""Header here."""
import numpy as np, importlib, base.emulation, copy
from base.utilities import postsampler

#need to make 'info' defined at the class level

class calibrator(object):
    r"""
    A class used to represent a calibrator.
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
            A dictionary containing options you would like to pass to either
            calibrationsubfuncs.[software].loglik(theta, phi, args)
            or
            logprior(theta, phi, args)

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
                    print('You have not provided any prior function, stopping...')
                elif type(emu) is tuple:
                    print('You have provided a tuple of emulators')
                    for k in range(0, len(emu)):
                        try:
                            ftry = emu[k].predict(emu[k].theta[0, :])['mean']
                        except:
                            raise ValueError('Your provided emulator failed to predict.')

                        emu[k].infonum = k

                    self.emu0 = emu[0]
                    self.modelcount = len(emu)
                else:
                    try:
                        ftry = emu.predict(emu.theta[0, :])['mean']
                    except:
                        raise ValueError('Your provided emulator failed to predict.')
                    emu.infonum = 0
                    self.emu0 = emu
                    self.modelcount = 1
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
                        matchingvec = np.where(((x[:, None] > self.emu0.x - 1e-08) * (x[:, None] < self.emu0.x + 1e-08)).all(2))
                        xind = matchingvec[1][matchingvec[0]]
                        if xind.shape[0] < x.shape[0]:
                            raise ValueError('If x is provided, it must be a subset of emu.x')
                        self.xind = xind
                        self.x = x
                    else:
                        self.xind = range(0, x.shape[0])
                        self.x = self.emu0.x
            try:
                self.calsoftware = importlib.import_module('base.calibrationsubfuncs.' + software)
            except:
                raise ValueError('Module not found!')
            
            self.info = {}
            self.args = args
            if thetaprior is None:
                raise ValueError('You must give a prior for theta.')
            else:
                self.info['thetaprior'] = thetaprior
            
            self.fit()


    def fit(self, args=None):
        r"""
        Returns a draws from theta and phi given data.

        It directly calls  \"calibrationsubfuncs.[software].fit\" where \"[software]\" is
        the user option.

        Parameters
        ----------
        args : dict
            A dictionary containing options you would like to pass to either
        """
        if args is None:
            args = self.args
        
        self.calsoftware.fit(self.info, self.emu, self.y, self.x, args)
        return None


    def predict(self, x, args=None):
        r"""
        Returns a predictions at x.

        It directly calls  \"calibrationsubfuncs.[software].predict\" where \"[software]\" is
        the user option.

        Parameters
        ----------
        x : array of float
            An array of inputs to the model where you would like to predict.
        args : dict
            A dictionary containing options you would like to pass to either
            loglik(theta, phi, args)
            or
            logprior(theta, phi, args)

        Returns
        -------
        preddict : dict of prediction objects
            preddict['mean'] : Mean of prediction at each point x
            preddict['var'] : Pointwise variance of prediction at each point x
            preddict['draws'] : Draws of the prediction at each point x. The dependency will be 
                                preserved
            preddict['modeldraws'] : Draws of the model's prediction at each x. The dependency will be 
                                preserved
        """
        if args is None:
            args = self.args
            
        return self.calsoftware.predict(x, self.emu, self.info, args)


    def thetas(self):
        r"""
        Returns posterior draws of the parameters.

        Returns
        -------
        thetas : array of floats
            A matrix of posterior parameter values.
        """
        
        return self.info['theta']