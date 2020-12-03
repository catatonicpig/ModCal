import numpy as np
from scipy.stats import norm
#from pyDOE import *
import scipy.optimize as spo
import sys
import os
from matplotlib import pyplot as plt
import scipy.stats as sps


# Read the data
ball = np.loadtxt('ball.csv', delimiter=',')
n = len(ball)

# Note that this is a stochastic one but we convert it deterministic by taking the average height
X = np.reshape(ball[:, 0], (n, 1))
X = X[0:21]

# time
Y = np.reshape(ball[:, 1], ((n, 1)))
Ysplit = np.split(Y, 3)
ysum = 0
for y in  Ysplit:
    ysum += y 
obsvar = np.maximum(0.2*X, 0.01) #0.00001*np.ones(X.shape[0])
Y = ysum/3 #+ sps.norm.rvs(0, np.sqrt(obsvar)).reshape((21, 1)) 

# Computer implementation of the mathematical model
def timedrop(x, theta, hr, gr):
    min_g = min(gr)
    range_g = max(gr) - min(gr)
    min_h = min(hr)
    range_h = max(hr) - min(hr)
    f = np.zeros((theta.shape[0], x.shape[0]))
    for k in range(0, theta.shape[0]):
        g = range_g*theta[k] + min_g
        h = range_h*x + min(hr)
        f[k, :] = np.sqrt(2*h/g).reshape(x.shape[0])
    return f.T

# Draw 100 random parameters from uniform prior
n2 = 100
#theta = lhs(1, samples=n2)

theta = np.reshape(np.random.uniform(size=30),(-1,1))
theta_range = np.array([1, 30])

# Standardize 
height_range = np.array([min(X), max(X)])
X_std = (X - min(X))/(max(X) - min(X))

# Obtain computer model output
Y_model = timedrop(X_std, theta, height_range, theta_range)

current = os.path.abspath(os.getcwd())
sys.path.append(os.path.normpath(os.path.join(os.path.dirname(current), '..')))
from base.emulation import emulator

emulator_no_f = emulator(X_std, theta, Y_model, method = 'PCGPwM')
pred_model = emulator_no_f.predict(X_std, theta)
pred_mean = pred_model.mean()
print(np.shape(pred_mean))

# Filter out the data
ys = 1 - np.sum((Y_model - Y)**2, 0)/np.sum((Y - np.mean(Y))**2, 0)
theta_f = theta[ys > 0.5]

# Obtain computer model output via filtered data
Y_model = timedrop(X_std, theta_f, height_range, theta_range)
print(np.shape(Y_model))

# Build up an emulator
emulator_f = emulator(x = X_std, theta = theta_f, f = Y_model, method = 'PCGPwM')
pred_model = emulator_f.predict(X_std, theta_f)
pred_mean = pred_model.mean()
print(np.shape(pred_mean))

class prior_balldrop:
    """ This defines the class instance of priors provided to the method. """
    def lpdf(theta):
        return np.squeeze(sps.uniform.logpdf(theta, 0.1, 0.9))

    def rnd(n):
        return np.vstack((sps.uniform.rvs(0.1, 0.9, size=n)))


from base.calibration import calibrator  
#import pdb
#pdb.set_trace() 
cal_f_pl = calibrator(emulator_f, y = Y, x = X_std, thetaprior = prior_balldrop, method = 'MLcal', yvar = obsvar, 
                      args = {'method' : 'plumlee', 'theta0': np.array([0.4]), 'numsamp' : 1000, 'stepType' : 'normal', 'stepParam' : [0.8]})

# NOTE: plumleeMCMC_wgrad (Line 146) gives an error when we have theta of size 1 because of the size of covmat0.