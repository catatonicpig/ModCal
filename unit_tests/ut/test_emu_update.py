import numpy as np
import scipy.stats as sps
import sys
import os
import pytest
from contextlib import contextmanager

current = os.path.abspath(os.getcwd())
sys.path.append(os.path.normpath(os.path.join(os.path.dirname(current), '..')))
from base.emulation import emulator

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
f1 = f[0:15,:]
x1 = x[0:15,:]

f1theta = f[:,0:15]
theta1 = theta[0:15,:]
theta_new = priorphys_lin.rnd(10)
#######################################################
# Unit tests for update method of emulator class #
#######################################################

@contextmanager
def does_not_raise():
    yield

@pytest.mark.set1
class TestClass_1:
    '''
    Class of tests to check the update()
    '''

    # test to check update
    @pytest.mark.parametrize(
        "input1,input2,input3,input4,expectation",
        [
            (x, theta, f, False, pytest.raises(ValueError)),
            (x, None, f, False, does_not_raise()), 
            (x, None, f, True, does_not_raise()), 
            (x1, None, f, True, pytest.raises(ValueError)), 
            (x, None, f1, True, pytest.raises(ValueError)), 
            (x1, None, f1, True, does_not_raise()), 
            ],
        )
    def test_update_x(self, input1, input2, input3, input4, expectation):
        emu = emulator(x = x, theta = theta, f = f, method = 'PCGPwM')
        with expectation:
            assert emu.update(x=input1, theta=input2, f=input3, options = {'xreps' : input4}) is None
    
    # test to check update
    @pytest.mark.parametrize(
        "input1,input2,input3,input4,expectation",
        [
            (x, theta, f, False, pytest.raises(ValueError)),
            (None, theta, f, False, does_not_raise()), 
            (None, theta, f, True, does_not_raise()), 
            (None, theta1, f, True, pytest.raises(ValueError)), 
            (None, theta, f1theta, True, pytest.raises(ValueError)), 
            (None, theta1, f1theta, True, does_not_raise()), 
            ],
        )
    def test_update_theta(self, input1, input2, input3, input4, expectation):
        emu = emulator(x = x, theta = theta, f = f, method = 'PCGPwM')
        with expectation:
            assert emu.update(x=input1, theta=input2, f=input3, options = {'thetareps' : input4}) is None
            
    # test to check update
    @pytest.mark.parametrize(
        "input1,input2,input3,expectation",
        [
            (None, None, f, pytest.raises(ValueError)),
            (x1, None, None, pytest.raises(ValueError)),
            (None, theta1, None, pytest.raises(ValueError)),
            (None, theta, None, does_not_raise()),
            (x, None, None, does_not_raise()),
            ],
        )
    def test_update_f(self, input1, input2, input3, expectation):
        emu = emulator(x = x, theta = theta, f = f, method = 'PCGPwM')
        with expectation:
            assert emu.update(x=input1, theta=input2, f=input3) is None
            
    # test to check update
    @pytest.mark.parametrize(
        "input1,input2,input3,expectation",
        [
            (None, theta, f, pytest.raises(ValueError)),
            ],
        )
    def test_update_supptheta(self, input1, input2, input3, expectation):
        emu = emulator(x = x, theta = theta, f = f, method = 'PCGPwM')
        emu.supplement(size = 10, x = x, theta = theta_new)
        with expectation:
            assert emu.update(x=input1, theta=input2, f=input3) is None    