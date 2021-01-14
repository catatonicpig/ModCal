import numpy as np
import scipy.stats as sps
import sys
import os
import pytest
from contextlib import contextmanager

current = os.path.abspath(os.getcwd())
sys.path.append(os.path.normpath(os.path.join(os.path.dirname(current), '..')))
from base.emulation import emulator
from base.calibration import calibrator

##############################################
#            Simple scenarios                #
##############################################
def balldropmodel_linear(x, theta):
    f = np.zeros((theta.shape[0], x.shape[0]))
    for k in range(0, theta.shape[0]):
        t = x[:, 0]
        h0 = x[:, 1] + theta[k, 0]
        vter = theta[k, 1]
        f[k, :] = h0 - vter * t
    return f.T

tvec = np.concatenate((np.arange(0.1, 4.3, 0.1), np.arange(0.1, 4.3, 0.1))) 
h0vec = np.concatenate((25 * np.ones(42), 50 * np.ones(42)))  
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
              [ 3.7, 50. ],]).astype('object')
xv = x.astype('float')

class priorphys_lin:
    """ This defines the class instance of priors provided to the method. """
    def lpdf(theta):
        if theta.ndim > 1.5:
            return np.squeeze(sps.norm.logpdf(theta[:, 0], 0, 5) +  # initial height deviation
                              sps.gamma.logpdf(theta[:, 1], 2, 0, 10))   # terminal velocity
        else:
            return np.squeeze(sps.norm.logpdf(theta[0], 0, 5) +  # initial height deviation
                              sps.gamma.logpdf(theta[1], 2, 0, 10))   # terminal velocity

    def rnd(n):
        return np.vstack((sps.norm.rvs(0, 5, size=n),  # initial height deviation
                          sps.gamma.rvs(2, 0, 10, size=n))).T  # terminal velocity

theta = priorphys_lin.rnd(50) 
f = balldropmodel_linear(xv, theta) 
theta1 = theta[0:25,:]
def balldroptrue(x):
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

obsvar = 4*np.ones(x.shape[0])  
y = balldroptrue(xv)

#######################################################
# Unit tests for remove method of emulator class #
#######################################################

@contextmanager
def does_not_raise():
    yield

@pytest.mark.set1
class TestClass_1:
    '''
    Class of tests to check the remove()
    '''

    # test to check remove
    @pytest.mark.parametrize(
        "input1,expectation",
        [
            (theta1, does_not_raise()),
            ],
        )
    def test_remove(self, input1, expectation):
        emu = emulator(x = x, theta = theta, f = f, method = 'PCGPwM')
        with expectation:
            assert emu.remove(theta=input1) is None

    # test to check remove
    @pytest.mark.parametrize(
        "input1,expectation",
        [
            (theta1, does_not_raise()),
            ],
        )
    def test_remove_cal(self, input1, expectation):
        emu = emulator(x = x, theta = theta, f = f, method = 'PCGPwM')
        cal_bayes = calibrator(emu = emu, y = y, x = x, thetaprior = priorphys_lin, method = 'directbayes', yvar = obsvar)
        with expectation:
            assert emu.remove(theta=input1, cal = cal_bayes) is None