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

#2-d x (30 x 2), 2-d theta (50 x 2), f (30 x 50)
x = np.vstack(( np.array(list(np.arange(0, 15))*2), np.repeat([1, 2], 15))).T
theta = np.vstack((sps.norm.rvs(0, 5, size=50), sps.gamma.rvs(2, 0, 10, size=50))).T
f = np.zeros((theta.shape[0], x.shape[0]))
for k in range(0, theta.shape[0]):
    f[k, :] = x[:, 0]*theta[k, 0] + x[:, 1]*theta[k, 1] 
f = f.T
#
y = sps.norm.rvs(0, 5, size=30).reshape(30,1)
ysmall = sps.norm.rvs(0, 5, size=3).reshape(3,1)
# setting obsvar
obsvar = 0.001*sps.uniform.rvs(10, 20, size=30)
obsvar1 = 0.001*sps.uniform.rvs(10, 20, size=10)
obsvar2 = -sps.uniform.rvs(10, 20, size=30)
obsvar3 = 10**(10)*sps.uniform.rvs(10, 20, size=30)

# different prior examples
class prior_example:
    def lpdf(theta):
        if theta.ndim > 1.5:
            return np.squeeze(sps.norm.logpdf(theta[:, 0], 0, 5) + sps.gamma.logpdf(theta[:, 1], 2, 0, 10)) 
        else:
            return np.squeeze(sps.norm.logpdf(theta[0], 0, 5) + sps.gamma.logpdf(theta[1], 2, 0, 10)) 

    def rnd(n):
        return np.vstack((sps.norm.rvs(0, 5, size=n), sps.gamma.rvs(2, 0, 10, size=n))).T     
class fake_prior_example:
    def lpdf(theta):
        if theta.ndim > 1.5:
            return np.vstack((sps.norm.logpdf(theta[:, 0], 0, 5), sps.gamma.logpdf(theta[:, 1], 2, 0, 10))).T 
        else:
            return np.squeeze(sps.norm.logpdf(theta[0], 0, 5) + sps.gamma.logpdf(theta[1], 2, 0, 10)) 

    def rnd(n):
        return np.vstack((sps.norm.rvs(0, 5, size=n), sps.gamma.rvs(2, 0, 10, size=n))).T
class fake_prior_rnd:
    def lpdf(theta):
        return np.array([1,2,3])
    def rnd(n):
        return np.array([1,2,3])   
class fake_empty_prior_rnd:  
    def nothing():
        return None    
class fake_prior_lpdf:
    def lpdf(theta):
        return np.array([1,2,3])
    def rnd(n):
        return np.vstack((sps.norm.rvs(0, 5, size=n), sps.gamma.rvs(2, 0, 10, size=n))).T    
class fake_empty_prior_lpdf:  
    def rnd(n):
        return np.vstack((sps.norm.rvs(0, 5, size=n), sps.gamma.rvs(2, 0, 10, size=n))).T 
      
#2-d x (30 x 2), 2-d theta (50 x 2), f1 (15 x 50)
f1 = f[0:15,:]
#2-d x (30 x 2), 2-d theta (50 x 2), f2 (30 x 25)
f2 = f[:,0:25]
#2-d x (30 x 2), 2-d theta1 (25 x 2), f (30 x 50)
theta1 = theta[0:25,:]
#2-d x1 (15 x 2), 2-d theta (50 x 2), f (30 x 50)
x1 = x[0:15,:]
# 
f0d = np.array(1)
theta0d = np.array(1)
x0d = np.array(1)

emulator_test = emulator(x = x, theta = theta, f = f, method = 'PCGPwM')
##############################################
# Unit tests to initialize an emulator class #
##############################################

@contextmanager
def does_not_raise():
    yield

@pytest.mark.set1
class TestClass_1:
    '''
    Class of tests to check the input configurations
    '''

    # test to check none-type inputs
    @pytest.mark.parametrize(
        "input1,expectation",
        [
            ('normal', does_not_raise()),
            ('uniform', does_not_raise()),
            ],
        )
    def test_cal_input(self, input1, expectation):
        with expectation:
            assert calibrator(emu = emulator_test, y = y, x = x, thetaprior = prior_example, 
                   method = 'MLcal', yvar = obsvar, 
                   args = {'theta0': np.array([1, 1]), 
                           'numsamp' : 50, 
                           'stepType' : input1, 
                           'stepParam' : [0.3, 0.3]}) is not None
    
@pytest.mark.set2
class TestClass_2:
    '''
    Class of tests to check the input configurations
    '''

    # test to check none-type inputs
    @pytest.mark.parametrize(
        "input1,input2,input3,input4,input5,expectation",
        [
            (emulator_test, y, x, prior_example, obsvar, does_not_raise()),
            (emulator_test, y, x1, prior_example, obsvar, pytest.raises(ValueError)),
            (emulator_test, y, x, prior_example, obsvar1, pytest.raises(ValueError)),
            (emulator_test, y, x, prior_example, obsvar2, pytest.raises(ValueError)),
            (emulator_test, y, x, prior_example, obsvar3, pytest.raises(ValueError)),
            (emulator_test, y, x, fake_prior_rnd, obsvar, pytest.raises(ValueError)),
            (emulator_test, y, x, fake_empty_prior_rnd, obsvar, pytest.raises(ValueError)),
            (emulator_test, y, x, fake_prior_lpdf, obsvar, pytest.raises(ValueError)),
            (emulator_test, y, x, fake_empty_prior_lpdf, obsvar, pytest.raises(ValueError)),
            (emulator_test, y, x, fake_prior_example, obsvar, pytest.raises(ValueError)),
            (emulator_test, ysmall, x, prior_example, obsvar, pytest.raises(ValueError)),
            (emulator_test, None, x, prior_example, obsvar, pytest.raises(ValueError)),
            (None, y, x, prior_example, obsvar, pytest.raises(ValueError)),
            (emulator_test, y, x, None, obsvar, pytest.raises(ValueError)),
            ],
        )
    def test_cal_emu(self, input1, input2, input3, input4, input5, expectation):
        with expectation:
            assert calibrator(emu = input1, y = input2, x = input3, thetaprior = input4, 
                   method = 'MLcal', yvar = input5, 
                   args = {'theta0': np.array([1, 1]), 
                           'numsamp' : 50, 
                           'stepType' : 'normal', 
                           'stepParam' : [0.3, 0.3]}) is not None    

@pytest.mark.set3
class TestClass_3:
    '''
    Class of tests to check calibration methods
    '''

    # test to check none-type inputs
    @pytest.mark.parametrize(
        "input1,input2,input3,input4,input5,input6,expectation",
        [
            (emulator_test, y, x, prior_example, 'XXXX', obsvar, pytest.raises(ValueError)),
            ],
        )
    def test_cal_method1(self, input1, input2, input3, input4, input5, input6, expectation):
        with expectation:
            assert calibrator(emu = input1, y = input2, x = input3, thetaprior = input4, 
                   method = input5, yvar = input6) is not None    
            
@pytest.mark.set4
class TestClass_4:
    '''
    Class of tests to check the emulator repr()
    '''
    
    # test to check if an emulator module is imported
    @pytest.mark.parametrize(
        "expectation",
        [
            (does_not_raise()),
            ],
        )
    def test_repr(self, expectation):
        cal = calibrator(emu = emulator_test, y = y, x = x, thetaprior = prior_example, 
                   method = 'MLcal', yvar = obsvar, 
                   args = {'theta0': np.array([1, 1]), 
                           'numsamp' : 50, 
                           'stepType' : 'normal', 
                           'stepParam' : [0.3, 0.3]})
        with expectation:
            assert repr(cal) is not None

@pytest.mark.set5
class TestClass_5:
    '''
    Class of tests to check the emulator call()
    '''
    
    # test to check if an emulator module is imported
    @pytest.mark.parametrize(
        "expectation",
        [
            (does_not_raise()),
            ],
        )
    def test_call(self, expectation):
        cal = calibrator(emu = emulator_test, y = y, x = x, thetaprior = prior_example, 
                         method = 'MLcal', yvar = obsvar, 
                         args = {'theta0': np.array([1, 1]), 
                                   'numsamp' : 50, 
                                   'stepType' : 'normal', 
                                   'stepParam' : [0.3, 0.3]})
        with expectation:
            assert cal(x = x) is not None