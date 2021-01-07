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
        "input1,input2,input3,expectation",
        [
            (x, theta, f, does_not_raise()),
            (x, None, f, does_not_raise()),
            (None, theta, f, does_not_raise()),
            (x, theta, None, pytest.raises(ValueError)),
            (x, None, None, pytest.raises(ValueError)),
            (None, theta, None, pytest.raises(ValueError)),
            (None, None, None, pytest.raises(ValueError)),
            ],
        )
    def test_none_input(self, input1, input2, input3, expectation):
        with expectation:
            assert emulator(x = input1, theta = input2, f = input3, method = 'PCGPwM') is not None
    
    # test to check the dimension of the inputs
    @pytest.mark.parametrize(
        "input1,input2,input3,expectation",
        [
            (x, theta, f, does_not_raise()),
            (x, theta, f.T, does_not_raise()),
            (x, theta.T, f, does_not_raise()),
            (x1, theta, f1, does_not_raise()),
            (x, theta, f1, pytest.raises(ValueError)),
            (x, theta, f2, pytest.raises(ValueError)),
            (x, theta1, f, pytest.raises(ValueError)),
            (x1, theta, f, pytest.raises(ValueError)),
            ],
        )        
    def test_size_input(self, input1, input2, input3, expectation):
        with expectation:
            assert emulator(x = input1, theta = input2, f = input3, method = 'PCGPwM') is not None

     # TO DO: Add tests for univariate data
     # TO DO: Add tests for data including NAs and infs
     
@pytest.mark.set2
class TestClass_2:
    '''
    Class of tests to check the emulator method configs
    '''
    
    # test to check if an emulator module is imported
    @pytest.mark.parametrize(
        "example_input,expectation",
        [
            ('PCGPwM', does_not_raise()),
            ('PCGP_ozge', does_not_raise()),
            ('XXXX', pytest.raises(ValueError)),
            ],
        )
    def test_method1(self, example_input, expectation):
        with expectation:
            assert emulator(x = x, theta = theta, f = f, method = example_input) is not None
    
    # test to check if an empty emulator module is imported
    @pytest.mark.parametrize(
        "example_input,expectation",
        [
            ('PCGPwM', does_not_raise()),
            ('fakeGP', pytest.raises(ValueError)),
            ],
        )
    def test_method2(self, example_input, expectation):
        with expectation:
            assert emulator(x = x, theta = theta, f = f, method = example_input) is not None

@pytest.mark.set3
class TestClass_3:
    '''
    Class of tests to check the options
    '''
    
    # test to check if 'thetareps' option is set correctly
    @pytest.mark.parametrize(
        "input1,expectation",
        [
            (True, does_not_raise()),
            (False, does_not_raise()),
            (0,  pytest.raises(ValueError)),
            (1,  pytest.raises(ValueError)),
            (0.5,  pytest.raises(ValueError)),
            ('XXXX', pytest.raises(ValueError)),
            ],
        )
    def test_options1(self, input1, expectation):
        with expectation:
            assert emulator(x = x, theta = theta, f = f, method = 'PCGPwM', options = {'thetareps': input1}) is not None
    
    # test to check if 'xreps' option is set correctly
    @pytest.mark.parametrize(
        "input1,expectation",
        [
            (True, does_not_raise()),
            (False, does_not_raise()),
            (0,  pytest.raises(ValueError)),
            (1,  pytest.raises(ValueError)),
            (0.5,  pytest.raises(ValueError)),
            ('XXXX', pytest.raises(ValueError)),
            ],
        )
    def test_options2(self, input1, expectation):
        with expectation:
            assert emulator(x = x, theta = theta, f = f, method = 'PCGPwM', options = {'xreps': input1}) is not None
    
    # test to check if 'thetarmnan' option is set correctly
    @pytest.mark.parametrize(
        "input1,expectation",
        [
            (True, does_not_raise()),
            (False, does_not_raise()),
            (0, does_not_raise()),
            (1, does_not_raise()),
            (0.5, does_not_raise()),
            (2, pytest.raises(ValueError)),
            ('any', does_not_raise()),
            ('some', does_not_raise()),
            ('most', does_not_raise()),
            ('alot', does_not_raise()),
            ('all', does_not_raise()),
            ('never', does_not_raise()),
            ('XXXX', pytest.raises(ValueError)),
            ],
        )
    def test_options3(self, input1, expectation):
        with expectation:
            assert emulator(x = x, theta = theta, f = f, method = 'PCGPwM', options = {'thetarmnan': input1}) is not None
    
    # test to check if 'xrmnan' option is set correctly    
    @pytest.mark.parametrize(
        "input1,expectation",
        [
            (True, does_not_raise()),
            (False, does_not_raise()),
            (0, does_not_raise()),
            (1, does_not_raise()),
            (0.5, does_not_raise()),
            (2, pytest.raises(ValueError)),
            ('any', does_not_raise()),
            ('some', does_not_raise()),
            ('most', does_not_raise()),
            ('alot', does_not_raise()),
            ('all', does_not_raise()),
            ('never', does_not_raise()),
            ('XXXX', pytest.raises(ValueError)),
            ],
        )
    def test_options4(self, input1, expectation):
        with expectation:
            assert emulator(x = x, theta = theta, f = f, method = 'PCGPwM', options = {'xrmnan': input1}) is not None
    
    # test to check if 'rmthetafirst' option is set correctly   
    @pytest.mark.parametrize(
        "input1,expectation",
        [
            (True, does_not_raise()),
            (False, does_not_raise()),
            (0,  pytest.raises(ValueError)),
            (1,  pytest.raises(ValueError)),
            (0.5,  pytest.raises(ValueError)),
            ('XXXX', pytest.raises(ValueError)),
            ],
        )
    def test_options5(self, input1, expectation):
        with expectation:
            assert emulator(x = x, theta = theta, f = f, method = 'PCGPwM', options = {'rmthetafirst': input1}) is not None
        
    # test to check if 'autofit' option is set correctly   
    @pytest.mark.parametrize(
        "input1,expectation",
        [
            (True, does_not_raise()),
            (False, does_not_raise()),
            (0,  pytest.raises(ValueError)),
            (1,  pytest.raises(ValueError)),
            (0.5,  pytest.raises(ValueError)),
            ('XXXX', pytest.raises(ValueError)),
            ],
        )
    def test_options6(self, input1, expectation):
        with expectation:
            assert emulator(x = x, theta = theta, f = f, method = 'PCGPwM', options = {'autofit': input1}) is not None
            
# pytest test_u1.py -m set3 --disable-warnings
# pytest --cov=ModCal/base/emulation test_u1.py