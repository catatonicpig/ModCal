import numpy as np
import scipy.stats as sps
import sys
import os
import pytest
from contextlib import contextmanager
from sklearn.ensemble import RandomForestClassifier
current = os.path.abspath(os.getcwd())
sys.path.append(os.path.normpath(os.path.join(os.path.dirname(current), '..')))
from base.emulation import emulator

##############################################
#            Simple scenarios                #
##############################################

#height
x = np.array([[0.178, 0.356, 0.534, 0.712, 0.89, 1.068, 1.246, 1.424, 1.602, 1.78, 1.958, 2.67, 2.848, 3.026, 3.204, 3.382, 3.56, 3.738, 3.916, 4.094, 4.272]]).T

#time
y = np.array([[0.27, 0.22, 0.27, 0.43, 0.41, 0.49, 0.46, 0.6, 0.65, 0.62, 0.7, 0.81, 0.69, 0.81, 0.89, 0.86, 0.89, 1.1, 1.05, 0.99, 1.05]]).T
obsvar = np.maximum(0.2*y, 0.1)

# Computer implementation of the mathematical model
def timedrop(x, theta, hr, gr):
    # Assume x and theta are within (0, 1)
    min_g = min(gr)
    range_g = max(gr) - min(gr)
    min_h = min(hr)
    range_h = max(hr) - min_h
    f = np.zeros((theta.shape[0], x.shape[0]))
    for k in range(0, theta.shape[0]):
        g = range_g*theta[k] + min_g
        h = range_h*x + min_h
        f[k, :] = np.sqrt(2*h/g).reshape(x.shape[0])
    return f.T

# Define prior
class prior_balldrop:
    """ This defines the class instance of priors provided to the method. """
    def lpdf(theta):
        if theta.ndim > 1.5:
            return np.squeeze(sps.uniform.logpdf(theta[:, 0], 0, 1))
        else:
            return np.squeeze(sps.uniform.logpdf(theta, 0, 1))

    def rnd(n):
        return np.vstack((sps.uniform.rvs(0, 1, size=n)))
    
    
# Draw 100 random parameters from uniform prior
n = 100
theta = prior_balldrop.rnd(n)
theta_range = np.array([1, 30])

# Standardize 
x_range = np.array([min(x), max(x)])
x_std = (x - min(x))/(max(x) - min(x))

# Obtain computer model output via filtered data
f = timedrop(x_std, theta, x_range, theta_range)
theta = theta.reshape((n,))
theta2d = theta.reshape((n,1))

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
            (x_std, theta, does_not_raise()),
            (x_std, theta2d, pytest.raises(ValueError)),
            ],
        )
    def test_predict_multi(self, input1, input2, expectation):
        emu = emulator(x = x_std, theta = theta, f = f, method = 'PCGP_ozge')
        with expectation:
            assert emu.predict(x = input1, theta = input2) is not None