"""Header here."""
import numpy as np, importlib, base.emulation, copy
from base.utilities import postsampler


class calibrator(object):
    """
    A class used to represent a calibrator.
    """
    __doc__ = 'A class used to represent a calibrator.'

    def __init__(self, emu=None, y=None, x=None, thetaprior=None, phiprior=None, passoptions={}, options=None, software='BDM'):
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

                        emu[k].modelnum = k

                    self.emu0 = emu[0]
                    self.modelcount = len(emu)
                else:
                    try:
                        ftry = emu.predict(emu.theta[0, :])['mean']
                    except:
                        raise ValueError('Your provided emulator failed to predict.')
                    emu.modelnum = 0
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

            self.passoptions = passoptions
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

    def logprior(self, theta, phi, passoptions=None):
        return self.thetaprior.logpdf(copy.deepcopy(theta)) +\
            self.phiprior.logpdf(copy.deepcopy(phi))

    def logpost(self, theta, phi, passoptions=None):
        if passoptions is None:
            passoptions = self.passoptions
        
        L0 = self.logprior(theta, phi)
        inds = np.where(np.isfinite(L0))[0]
        if phi is None:
            L0[inds] += self.calsoftware.loglik(self.emu, theta[inds],
                                                None, self.y, self.xind, passoptions)
        else:
            L0[inds] += self.calsoftware.loglik(self.emu, theta[inds],
                                                phi[inds], self.y, self.xind, passoptions)
        return L0

    def fit(self, theta = None, phi = None, passoptions=None):
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
        if passoptions is None:
            passoptions = self.passoptions
        return self.calsoftware.fit(theta, phi, self.logpost)

    def predict(self, x, theta=None, phi=None, passoptions=None):
        matchingvec = np.where(((x[:, None] > self.emu0.x - 1e-08) * (x[:, None] < self.emu0.x + 1e-08)).all(2))
        xind = matchingvec[1][matchingvec[0]]
        if theta is None:
            theta = self.thetadraw
            phi = self.phidraw
        if passoptions is None:
            passoptions = self.passoptions
        return self.calsoftware.predict(xind, self.emu, theta, phi, self.y, self.xind, passoptions)