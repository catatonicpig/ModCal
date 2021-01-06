import numpy as np
import scipy.stats as sps
import sys
import os
import pytest


current = os.path.abspath(os.getcwd())
sys.path.append(os.path.normpath(os.path.join(os.path.dirname(current), '..')))
from base.emulation import emulator
from base.calibration import calibrator




x = np.vstack(( np.array(list(np.arange(0, 15))*2), np.repeat([1, 2], 15))).T
theta = np.vstack((sps.norm.rvs(0, 5, size=50), sps.gamma.rvs(2, 0, 10, size=50))).T
f = np.zeros((theta.shape[0], x.shape[0]))
for k in range(0, theta.shape[0]):
    f[k, :] = x[:, 0]*theta[k, 0] + x[:, 1]*theta[k, 1] 
f = f.T

@pytest.mark.set1
class TestClass_1:
    
    def test_none1(self):
        with pytest.raises(ValueError) as excinfo:
            emulator(x = x, theta = theta, f = f, method = 'PCGPwM') 
        #assert 'You have not provided f' not in str(excinfo.value)
    
    def test_none2(self):
        with pytest.raises(ValueError) as excinfo:
            emulator(x = x, theta = theta, f = None, method = 'PCGPwM') 
        #assert 'You have not provided f' not in str(excinfo.value)
    
    def test_none3(self):
        with pytest.raises(ValueError) as excinfo:
            emulator(x = x, theta = None, f = None, method = 'PCGPwM') 
        #assert 'You have not provided f' not in str(excinfo.value)
    
    def test_none4(self):
        with pytest.raises(ValueError) as excinfo:
            emulator(x = None, theta = None, f = None, method = 'PCGPwM') 
        #assert 'You have not provided f' not in str(excinfo.value)
      
    def test_none5(self):
        with pytest.raises(ValueError) as excinfo:
            emulator(x = None, theta = theta, f = None, method = 'PCGPwM') 
        #assert 'You have not provided f' not in str(excinfo.value)
 
# WHICH ONE IS MORE SUITABLE?
@pytest.mark.set2
class TestClass_2:
    def test_method1(self):
        with pytest.raises(ValueError):
            emulator(x = x, theta = theta, f = f, method = 'XXXX') 
       
    def test_method2(self):
        with pytest.raises(ValueError) as excinfo:
            emulator(x = x, theta = theta, f = f, method = 'XXXX') 
        assert 'Module not loaded correctly' not in str(excinfo.value)
      
    def test_method3(self):
        emulator(x = x, theta = theta, f = f, method = 'XXXX') 
      
    def test_method4(self):
        with pytest.raises(ValueError) as excinfo:
            emulator(x = x, theta = theta, f = f, method = 'XXXX') 
        assert 'Module not loaded correctly' in str(excinfo.value)