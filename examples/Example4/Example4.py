import numpy as np
import sys
import os
from matplotlib import pyplot as plt
import scipy.stats as sps
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
current = os.path.abspath(os.getcwd())
sys.path.append(os.path.normpath(os.path.join(os.path.dirname(current), '..')))
from base.emulation import emulator
from base.calibration import calibrator


# Read the data
ball = np.loadtxt('ball.csv', delimiter=',')
n = len(ball)
#height
X = np.reshape(ball[:, 0], (n, 1))
X = X[0:21]
#time
Y = np.reshape(ball[:, 1], ((n, 1)))
# This is a stochastic one but we convert it deterministic by taking the average height
Ysplit = np.split(Y, 3)
ysum = 0
for y in  Ysplit:
    ysum += y 
obsvar = np.maximum(0.2*X, 0.5)
Y = ysum/3

# Observe the data
plt.scatter(X, Y, color = 'red')
plt.xlabel("height (meters)")
plt.ylabel("time (seconds)")
plt.show()

# Computer implementation of the mathematical model
def timedrop(x, theta, hr, gr):
    # Assume x and theta are within (0, 1)
    min_g = min(gr)
    range_g = max(gr) - min(gr)
    min_h = min(hr)
    range_h = max(hr) - min(hr)
    f = np.zeros((theta.shape[0], x.shape[0]))
    for k in range(0, theta.shape[0]):
        g = range_g*theta[k] + min_g
        h = range_h*x + min_h
        f[k, :] = np.sqrt(2*h/g).reshape(x.shape[0])
    return f.T

# Draw 100 random parameters from uniform prior
n2 = 100
theta = np.random.random(n2).reshape((n2, 1))
theta_range = np.array([1, 30])

# Standardize 
height_range = np.array([min(X), max(X)])
X_std = (X - min(X))/(max(X) - min(X))

# Obtain computer model output
Y_model = timedrop(X_std, theta, height_range, theta_range)

print(np.shape(theta))
print(np.shape(X_std))
print(np.shape(Y_model))

#import pdb
#pdb.set_trace()

# Fit emualtors using two different methods 

# Emulator 1
emulator_1 = emulator(x = X_std, theta = theta, f = Y_model, method = 'PCGPwM')

class prior_balldrop:
    """ This defines the class instance of priors provided to the method. """
    def lpdf(theta):
        return np.squeeze(sps.uniform.logpdf(theta, 0, 1))

    def rnd(n):
        return np.vstack((sps.uniform.rvs(0, 1, size=n)))
    
obsvar = np.maximum(0.2*Y, 0.1)    
cal_1 = calibrator(emu = emulator_1, y = Y, x = X_std, thetaprior = prior_balldrop, method = 'BDM', yvar = obsvar)

