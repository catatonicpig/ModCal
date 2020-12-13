import numpy as np
import scipy.stats as sps
import sys
import os

current = os.path.abspath(os.getcwd())
sys.path.append(os.path.normpath(os.path.join(os.path.dirname(current), '..')))
from base.emulation import emulator
from base.calibration import calibrator

def balldropmodel_grav(x, theta):
    """Place description here."""
    f = np.zeros((theta.shape[0], x.shape[0]))
    for k in range(0, theta.shape[0]):
        t = x[:, 0]
        h0 = x[:, 1]
        g = theta[k]
        f[k, :] = h0 - (g / 2) * (t ** 2)
    return f.T

class priorphys_grav:
    """ This defines the class instance of priors provided to the method. """
    def lpdf(theta):
        if theta.ndim > 1.5:
            return np.squeeze(sps.gamma.logpdf(theta[:, 0], 2, 0, 5))  # gravity
        else:
            return np.squeeze(sps.gamma.logpdf(theta, 2, 0, 5))  # gravity

    def rnd(n):
        return np.reshape(sps.gamma.rvs(2, 0, 5, size=n), (-1,1))  # gravity

def balldroptrue(x):
    """Place description here."""
    def logcosh(x):
        # preventing crashing
        s = np.sign(x) * x
        p = np.exp(-2 * s)
        return s + np.log1p(p) - np.log(2)
    t = x[:, 0]
    h0 = x[:, 1]
    vter = 20
    g = 9.81
    y = h0 - (vter ** 2) / g * logcosh(g * t / vter)
    return y

x = np.array([[ 0.1, 25. ],
              [ 0.2, 25. ],
        [ 0.3, 25. ],
        [ 0.4, 25. ],
        [ 0.5, 25. ],
        [ 0.6, 25. ],
        [ 0.7, 25. ],
        [ 0.9, 25. ],
        [ 1.1, 25. ],
        [ 1.3, 25. ],
        [ 2.0, 25. ],
        [ 2.4, 25. ],
        [ 0.1, 50. ],
        [ 0.2, 50. ],
        [ 0.3, 50. ],
        [ 0.4, 50. ],
        [ 0.5, 50. ],
        [ 0.6, 50. ],
        [ 0.7, 50. ],
        [ 0.8, 50. ],
        [ 0.9, 50. ],
        [ 1.0, 50. ],
        [ 1.2, 50. ],
        [ 2.6, 50. ],
        [ 2.9, 50. ],
        [ 3.1, 50. ],
        [ 3.3, 50. ],
        [ 3.5, 50. ],
        [ 3.7, 50. ],
]).astype('object')
xv = x.astype('float')
x[x[:,1] == 25, 1] = 'lowdrop'
x[x[:,1] == 50, 1] = 'highdrop'
obsvar = 4*np.ones(x.shape[0])  # variance for the observations in 'y' below
y = balldroptrue(xv) + sps.norm.rvs(0, np.sqrt(obsvar)) #observations at each row of 'x'

# draw 50 random parameters from the prior
thetacompexp_grav = priorphys_grav.rnd(50)  

# create a computer experiment to build an emulator for the gravity simulation
grav_results = balldropmodel_grav(xv, thetacompexp_grav)  

# build an emulator for the gravity simulation
emu_grav = emulator(x, thetacompexp_grav, grav_results, method = 'PCGPwM')  

# build calibrators for the gravity simulation
cal_grav_1 = calibrator(emu = emu_grav, y = y, x = x, thetaprior = priorphys_grav, 
                     method = 'directbayes', 
                     yvar = obsvar)

#### NOTE: This is not a test basically. But, to check my temporary corrections to make it work with univariate theta.
#########  Please read my comments in directbayes.py between lines 263-307 to see if they are OK.