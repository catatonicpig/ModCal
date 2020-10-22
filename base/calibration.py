"""Header here."""
import numpy as np, importlib, base.emulation, copy
from base.utilities import postsampler


class calibrator(object):
    r"""
    A class used to represent a calibrator.
    """

    def __init__(self, emu=None, y=None, x=None, thetaprior=None, phiprior=None,
                 software='BDM', args={}):
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
        thetaprior : distribution class instatance with two built in functions
            phiprior.rvs(n) :  produces n draws from the prior on phi, arranged in either a
            matrix or list.
            phiprior.logpdf(phi) :  produces the log pdf from the prior on phi, arranged in a
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

            self.args = args
            self.thetaprior = thetaprior
            if thetaprior is None:
                raise ValueError('You must give a prior for theta.')
            if phiprior is None:
                class phiprior:
                    def logpdf(phi):
                        return 0
                    def rvs(n):
                        return None
                self.phiprior = phiprior
            else:
                self.phiprior = phiprior
            
            self.thetadraw = None
            self.phidraw = None
            self.thetadraw, self.phidraw = self.fit()


    def fit(self, theta = None, phi = None, args=None):
        r"""
        Returns a draws from theta and phi given data.

        It directly calls  \"calibrationsubfuncs.[software].fit\" where \"[software]\" is
        the user option.

        Parameters
        ----------
        theta : array of float
            A list or matrix of intitial parameters. If not provided it will draw from prior.
        phi : array of float
            A list or matrix of statistical model parameters.  It must have the same number of rows
            or elements as \"theta\".  If not provided it will draw from prior.
        args : dict
            A dictionary containing options you would like to pass to either
            loglik(theta, phi, args)
            or
            logprior(theta, phi, args)

        Returns
        -------
        cal.info : dict
            cal.info['theta'] : array of float
                A list or matrix of parameters drawn from the posterior. If not provided it will
                draw from prior.
            cal.info['phi'] :
                A list or matrix of statistical model parameters drawn from the posterior.  It will
                have the same number of rows or elements as \"theta\".
        """
        if theta is None:
            if self.thetadraw is not None:
                theta = self.thetadraw
            else:
                self.thetadraw = self.thetaprior.rvs(1000)
                theta = self.thetadraw
        if phi is None:
            if self.phidraw is not None:
                phi = self.phidraw
            else:
                self.phidraw = self.phiprior.rvs(1000)
                phi = self.phidraw
        if args is None:
            args = self.args
        return self.calsoftware.fit(theta, phi, self.logpost)


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
        matchingvec = np.where(((x[:, None] > self.emu0.x - 1e-08) * (x[:, None] < self.emu0.x + 1e-08)).all(2))
        xind = matchingvec[1][matchingvec[0]]
        if args is None:
            args = self.args
            
        return self.emusoftware.predict(x, self.emu, self.info,
                              copy.deepcopy(theta),
                              args)


    def logprior(self, theta, phi, args=None):
        r"""
        Returns a log prior at the new parameters.

        Parameters
        ----------
        theta : array of float
            A list or matrix of parameters.
        phi : array of float
            A list or matrix of statistical model parameters.  It must have the same number of rows
            or elements as \"theta\".
        args : dict
            A dictionary containing options you would like to pass to either
            thetaprior.logpdf(theta, args)
            or
            phiprior.logpdf(theta, args)

        Returns
        -------
        array of float
            A vector of unnormlaized log prior.
        """
        return self.thetaprior.logpdf(theta) + self.phiprior.logpdf(phi)


    def loglik(self, theta, phi, args=None):
        r"""
        Returns a log likelihood at the new parameters.  It directly calls 
        \"calibrationsubfuncs.[software].loglik\" where \"[software]\" is the user option.

        Parameters
        ----------
        theta : array of float
            A list or matrix of parameters.
        phi : array of float
            A list or matrix of statistical model parameters.  It must have the same number of rows
            or elements as \"theta\".
        args : dict
            A dictionary containing options you would like to pass to either
            thetaprior.logpdf(theta, args)
            or
            phiprior.logpdf(theta, args)

        Returns
        -------
        array of float
            A vector of unnormlaized log likelihood.
        """
        if args is None:
            args = self.args
        return self.calsoftware.loglik(self.emu, theta, phi, self.y, self.xind, args)


    def logpost(self, theta, phi, args=None):
        r"""
        Returns a log posterior at the new parameters.  It is merely the summation of loglik and
        logprior

        Parameters
        ----------
        theta : array of float
            A list or matrix of parameters.
        phi : array of float
            A list or matrix of statistical model parameters.  It must have the same number of rows
            or elements as \"theta\".
        args : dict
            A dictionary containing options you would like to pass to either
            thetaprior.logpdf(theta, args)
            or
            phiprior.logpdf(theta, args)

        Returns
        -------
        array of float
            A vector of unnormlaized log posterior.
        """
        if args is None:
            args = self.args

        L0 = self.logprior(theta, phi)
        inds = np.where(np.isfinite(L0))[0]
        if phi is None:
            L0[inds] += self.loglik(theta[inds], None)
        else:
            L0[inds] += self.loglik(theta[inds], phi[inds])
        return L0