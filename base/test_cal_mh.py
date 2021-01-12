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

theta_lin = priorphys_lin.rnd(50)  
f_lin = balldropmodel_linear(xv, theta_lin) 
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
emulator_test = emulator(x = x, theta = theta_lin, f = f_lin, method = 'PCGPwM')


# Additional examples
ysmall = y[0:3]
# setting obsvar
obsvar1 = obsvar[0:10]
obsvar2 = -obsvar
obsvar3 = 10**(10)*obsvar

#2-d x (30 x 2), 2-d theta (50 x 2), f1 (15 x 50)
f1 = f_lin[0:15,:]
#2-d x (30 x 2), 2-d theta (50 x 2), f2 (30 x 25)
f2 = f_lin[:,0:25]
#2-d x (30 x 2), 2-d theta1 (25 x 2), f (30 x 50)
theta1 = theta_lin[0:25,:]
#2-d x1 (15 x 2), 2-d theta (50 x 2), f (30 x 50)
x1 = x[0:15,:]
# 
f0d = np.array(1)
theta0d = np.array(1)
x0d = np.array(1)

#### #### #### different prior examples #### #### ####
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
#### #### #### #### #### #### #### #### #### #### #### ####

args1 = {'theta0': np.array([0, 9]), 'numsamp' : 50, 'stepType' : 'normal', 'stepParam' : [0.1, 1]}
args2 = {'theta0': np.array([0, 9]), 'numsamp' : 50, 'stepType' : 'uniform', 'stepParam' : [0.1, 1]}
args3 = {'theta0': np.array([0, 9]), 'stepParam' : [0.1, 1]}
args4 = {'stepParam' : [0.1, 1]}
args5 = {'theta0': np.array([0, 9])}
args6 = {'sampler' : 'plumlee'}
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
        "input1,input2,expectation",
        [
            (emulator_test,args1,does_not_raise()),
            (emulator_test,args2,does_not_raise()),
            (emulator_test,args3,does_not_raise()),
            (emulator_test,args4,pytest.raises(ValueError)),
            (emulator_test,args5,pytest.raises(ValueError)),
            (emulator_test,args6,does_not_raise()),
            ],
        )
    def test_cal_MLcal(self, input1, input2, expectation):
        with expectation:
            assert calibrator(emu = input1, y = y, x = x, thetaprior = priorphys_lin, method = 'MLcal', yvar = obsvar,
                              args = input2) is not None
            
    
@pytest.mark.set2
class TestClass_2:
    '''
    Class of tests to check the input configurations
    '''

    # test to check none-type inputs
    @pytest.mark.parametrize(
        "input1,input2,input3,input4,input5,expectation",
        [
            (emulator_test, y, x, priorphys_lin, obsvar, does_not_raise()),
            (emulator_test, y, x1, priorphys_lin, obsvar, pytest.raises(ValueError)),
            (emulator_test, y, x, priorphys_lin, obsvar1, pytest.raises(ValueError)),
            (emulator_test, y, x, priorphys_lin, obsvar2, pytest.raises(ValueError)),
            (emulator_test, y, x, priorphys_lin, obsvar3, pytest.raises(ValueError)),
            (emulator_test, y, x, fake_prior_rnd, obsvar, pytest.raises(ValueError)),
            (emulator_test, y, x, fake_empty_prior_rnd, obsvar, pytest.raises(ValueError)),
            (emulator_test, y, x, fake_prior_lpdf, obsvar, pytest.raises(ValueError)),
            (emulator_test, y, x, fake_empty_prior_lpdf, obsvar, pytest.raises(ValueError)),
            (emulator_test, y, x, fake_prior_example, obsvar, pytest.raises(ValueError)),
            (emulator_test, ysmall, x, priorphys_lin, obsvar, pytest.raises(ValueError)),
            (emulator_test, None, x, priorphys_lin, obsvar, pytest.raises(ValueError)),
            (None, y, x, priorphys_lin, obsvar, pytest.raises(ValueError)),
            (emulator_test, y, x, None, obsvar, pytest.raises(ValueError)),
            ],
        )
    def test_cal_emu(self, input1, input2, input3, input4, input5, expectation):
        with expectation:
            assert calibrator(emu = input1, y = input2, x = input3, thetaprior = input4, 
                   method = 'MLcal', yvar = input5, 
                   args = args1) is not None    

@pytest.mark.set3
class TestClass_3:
    '''
    Class of tests to check calibration methods
    '''

    # test to check none-type inputs
    @pytest.mark.parametrize(
        "input1,input2,input3,input4,input5,input6,expectation",
        [
            (emulator_test, y, x, priorphys_lin, 'XXXX', obsvar, pytest.raises(ValueError)),
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
        cal = calibrator(emu = emulator_test, y = y, x = x, thetaprior = priorphys_lin, 
                   method = 'MLcal', yvar = obsvar, 
                   args = args1)
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
        cal = calibrator(emu = emulator_test, y = y, x = x, thetaprior = priorphys_lin, 
                         method = 'MLcal', yvar = obsvar, 
                         args = args1)
        with expectation:
            assert cal(x = x) is not None