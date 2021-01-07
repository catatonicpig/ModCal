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
#
x3 = np.vstack(( np.array(list(np.arange(0, 10))*2), np.repeat([1, 2], 10), np.repeat([2, 3], 10))).T
#1-d theta
theta1d = sps.norm.rvs(0, 5, size=50)
#1-d theta 
theta1dx = sps.norm.rvs(0, 5, size=2)
##############################################
# Unit tests to initialize an emulator class #
##############################################

@contextmanager
def does_not_raise():
    yield

@pytest.mark.set1
class TestClass_1:
    '''
    Class of tests to check the predict method in emulator class
    '''

    # test to check the predict method with multivariate example
    @pytest.mark.parametrize(
        "input1,input2,expectation",
        [
            (x, theta, does_not_raise()),
            (x.T, theta, does_not_raise()),
            (None, theta, does_not_raise()),
            (x3, theta, pytest.raises(ValueError)),
            (x, None, does_not_raise()),
            (x, theta.T, does_not_raise()),
            (x, theta1d, pytest.raises(ValueError)),
            (x, theta1dx, does_not_raise()),
            ],
        )
    def test_predict_multi(self, input1, input2, expectation):
        emu = emulator(x = x, theta = theta, f = f, method = 'PCGPwM')
        with expectation:
            assert emu.predict(x = input1, theta = input2) is not None
            
@pytest.mark.set2
class TestClass_2:
    '''
    Class of tests to check the prediction class
    '''

    # test to check the prediction.mean()
    @pytest.mark.parametrize(
        "input1,expectation",
        [
            ('PCGPwM', does_not_raise()),
            ('PCGP_ozge', does_not_raise()),
            ],
        )
    def test_prediction_mean(self, input1, expectation):
        emu = emulator(x = x, theta = theta, f = f, method = input1)
        pred = emu.predict(x = x, theta = theta)
        with expectation:
            assert pred.mean() is not None
  
    # test to check the prediction.mean_gradtheta()
    @pytest.mark.parametrize(
        "input1,input2,expectation",
        [
            ('PCGPwM', False, pytest.raises(ValueError)),
            ('PCGPwM', True, does_not_raise()),
            ],
        )
    def test_prediction_mean_gradtheta(self, input1, input2, expectation):
        emu = emulator(x = x, theta = theta, f = f, method = input1)
        pred = emu.predict(x = x, theta = theta, args = {'return_grad' : input2})
        with expectation:
            assert pred.mean_gradtheta() is not None
            
    # test to check the prediction.var()
    @pytest.mark.parametrize(
        "input1,expectation",
        [
            ('PCGPwM', does_not_raise()),
            ('PCGP_ozge', does_not_raise()),
            ],
        )
    def test_prediction_var(self, input1, expectation):
        emu = emulator(x = x, theta = theta, f = f, method = input1)
        pred = emu.predict(x = x, theta = theta)
        with expectation:
            assert pred.var() is not None
        
    # test to check the prediction.covx()
    @pytest.mark.parametrize(
        "input1,expectation",
        [
            ('PCGPwM', does_not_raise()),
            ('PCGP_ozge', pytest.raises(ValueError)),
            ],
        )
    def test_prediction_covx(self, input1, expectation):
        emu = emulator(x = x, theta = theta, f = f, method = input1)
        pred = emu.predict(x = x, theta = theta)
        with expectation:
            assert pred.covx() is not None
            
    # test to check the prediction.covxhalf()
    @pytest.mark.parametrize(
        "input1,expectation",
        [
            ('PCGPwM', does_not_raise()),
            ('PCGP_ozge', pytest.raises(ValueError)),
            ],
        )
    def test_prediction_covxhalf(self, input1, expectation):
        emu = emulator(x = x, theta = theta, f = f, method = input1)
        pred = emu.predict(x = x, theta = theta)
        with expectation:
            assert pred.covxhalf() is not None
                      
    # test to check the prediction.covxhalf_gradtheta()
    @pytest.mark.parametrize(
        "input1,input2,expectation",
        [
            ('PCGPwM', False, pytest.raises(ValueError)),
            ('PCGPwM', True, does_not_raise()),
            ],
        )
    def test_prediction_covxhalf_gradtheta(self, input1, input2, expectation):
        emu = emulator(x = x, theta = theta, f = f, method = input1)
        pred = emu.predict(x = x, theta = theta, args = {'return_grad' : input2})
        with expectation:
            assert pred.covxhalf_gradtheta() is not None
            
    # test to check the prediction.rnd()
    @pytest.mark.parametrize(
        "input1,expectation",
        [
            ('PCGPwM', pytest.raises(ValueError)),
            ],
        )
    def test_prediction_rnd(self, input1, expectation):
        emu = emulator(x = x, theta = theta, f = f, method = input1)
        pred = emu.predict(x = x, theta = theta)
        with expectation:
            assert pred.rnd() is not None
            
    # test to check the prediction.lpdf()
    @pytest.mark.parametrize(
        "input1,expectation",
        [
            ('PCGPwM', pytest.raises(ValueError)),
            ],
        )
    def test_prediction_lpdf(self, input1, expectation):
        emu = emulator(x = x, theta = theta, f = f, method = input1)
        pred = emu.predict(x = x, theta = theta)
        with expectation:
            assert pred.lpdf() is not None