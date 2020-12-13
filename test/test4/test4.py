import numpy as np
import sys
import os
from matplotlib import pyplot as plt
import scipy.stats as sps
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
Y = ysum/3

# Computer implementation of the mathematical model
def timedrop(x, theta, hr, gr):
    # Assume x and theta are within (0, 1)
    min_g = min(gr)
    range_g = max(gr) - min(gr)
    range_h = max(hr) - min(hr)
    f = np.zeros((theta.shape[0], x.shape[0]))
    for k in range(0, theta.shape[0]):
        g = range_g*theta[k] + min_g
        h = range_h*x + min(hr)
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

def plot_pred(X_std, Y, cal, theta_range):
    
    fig, axs = plt.subplots(1, 4, figsize=(14, 3))
    cal_theta = cal.theta.rnd(1000) 
    cal_theta = cal_theta*(theta_range[1] - theta_range[0]) + theta_range[0]  
    axs[0].plot(cal_theta)
    axs[1].boxplot(cal_theta)
    axs[2].hist(cal_theta)
    post = cal.predict(X_std)
    rndm_m = post.rnd(s = 1000)
    upper = np.percentile(rndm_m, 97.5, axis = 0)
    lower = np.percentile(rndm_m, 2.5, axis = 0)
    median = np.percentile(rndm_m, 50, axis = 0)
    axs[3].plot(median, color = 'black')
    axs[3].fill_between(range(0, 21), lower, upper, color = 'grey')
    axs[3].plot(range(0, 21), Y, 'ro', markersize = 5, color='red')
    plt.show()
    
# Draw 100 random parameters from uniform prior
n2 = 100
theta = np.random.random(n2).reshape((n2, 1))
theta_range = np.array([1, 30])

# Standardize 
height_range = np.array([min(X), max(X)])
X_std = (X - min(X))/(max(X) - min(X))

# Obtain computer model output
Y_model = timedrop(X_std, theta, height_range, theta_range) 

# Emulator 1
emulator_1 = emulator(X_std, theta, Y_model, method = 'PCGPwM')

#import pdb
#pdb.set_trace()
# Fit a calibrator with emulator 1 via method = 'directbayes' and 'sampler' = plumlee
obsvar = np.maximum(0.2*Y, 0.1)
cal_3 = calibrator(emulator_1, Y, X_std, thetaprior = prior_balldrop, method = 'directbayes', yvar = obsvar)  
plot_pred(X_std, Y, cal_3, theta_range)    

