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

#2-d x (30 x 2), 2-d theta (50 x 2), f (30 x 50)
x = np.vstack(( np.array(list(np.arange(0, 15))*2), np.repeat([1, 2], 15))).T
theta = np.vstack((sps.norm.rvs(0, 5, size=50), sps.gamma.rvs(2, 0, 10, size=50))).T
f = np.zeros((theta.shape[0], x.shape[0]))
for k in range(0, theta.shape[0]):
    f[k, :] = x[:, 0]*theta[k, 0] + x[:, 1]*theta[k, 1] 
f = f.T
#2-d x (30 x 2), 2-d theta (50 x 2), f1 (15 x 50)
f1 = f[0:15,:]
#2-d x (30 x 2), 2-d theta (50 x 2), f2 (30 x 25)
f2 = f[:,0:25]
#2-d x (30 x 2), 2-d theta1 (25 x 2), f (30 x 50)
theta1 = theta[0:25,:]
#2-d x1 (15 x 2), 2-d theta (50 x 2), f (30 x 50)
x1 = x[0:15,:]

#######################################################
# Unit tests for supplement method of emulator class #
#######################################################

@contextmanager
def does_not_raise():
    yield

@pytest.mark.set1
class TestClass_1:
    '''
    Class of tests to check the supplement()
    '''

    # test to check supplement_x
    @pytest.mark.parametrize(
        "input1,input2,input3,expectation",
        [
            (5, x, x1, pytest.raises(ValueError)),
            ],
        )
    def test_supplement_x(self, input1, input2, input3, expectation):
        emu = emulator(x = x, theta = theta, f = f, method = 'PCGPwM')
        with expectation:
            assert emu.supplement(size = input1, x = input2, xchoices = input3)  is not None
    
    # test to check supplement_theta
    @pytest.mark.parametrize(
        "input1,input2,input3,expectation",
        [
            (5, theta, theta1, does_not_raise()),
            ],
        )
    def test_supplement_theta(self, input1, input2, input3, expectation):
        emu = emulator(x = x, theta = theta, f = f, method = 'PCGPwM')
        with expectation:
            assert emu.supplement(size = input1, theta = input2, thetachoices = input3)  is not None
   