import numpy as np
import sys
import os
from matplotlib import pyplot as plt
import scipy.stats as sps
current = os.path.abspath(os.getcwd())
sys.path.append(os.path.normpath(os.path.join(os.path.dirname(current), '..')))
from base.emulation import emulator
from base.calibration import calibrator

########################################################
# This example shows how to use GPy within our framework
########################################################

# Read the data
ball = np.loadtxt('ball.csv', delimiter=',')
m = len(ball)
#height
xrep = np.reshape(ball[:, 0], (m, 1))
x = xrep[0:21]
#time
y = np.reshape(ball[:, 1], ((m, 1)))

# Observe the data
plt.scatter(xrep, y, color = 'red')
plt.xlabel("height (meters)")
plt.ylabel("time (seconds)")
plt.show()

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
    
def plot_pred(x_std, xrep, y, cal, theta_range):
   
   fig, axs = plt.subplots(1, 4, figsize=(14, 3))

   cal_theta = cal.theta.rnd(1000) 
   cal_theta = cal_theta*(theta_range[1] - theta_range[0]) + theta_range[0]  
   axs[0].plot(cal_theta)
   axs[1].boxplot(cal_theta)
   axs[2].hist(cal_theta)
   
   post = cal.predict(x_std)
   rndm_m = post.rnd(s = 1000)
   upper = np.percentile(rndm_m, 97.5, axis = 0)
   lower = np.percentile(rndm_m, 2.5, axis = 0)
   median = np.percentile(rndm_m, 50, axis = 0)

   axs[3].plot(xrep[0:21].reshape(21), median, color = 'black')
   axs[3].fill_between(xrep[0:21].reshape(21), lower, upper, color = 'grey')
   axs[3].plot(xrep, y, 'ro', markersize = 5, color='red')
   
   plt.show()
    
# Draw 100 random parameters from uniform prior
n = 100
theta = prior_balldrop.rnd(n)
theta_range = np.array([1, 30])

# Standardize 
x_range = np.array([min(x), max(x)])
x_std = (x - min(x))/(max(x) - min(x))
xrep_std = (xrep - min(xrep))/(max(xrep) - min(xrep))

# Obtain computer model output
f = timedrop(x_std, theta, x_range, theta_range) 

# Emulator 1
emulator_GPy = emulator(x = x_std, theta = theta, f = f, method = 'GPy')
pred_emu = emulator_GPy.predict(x_std, theta)
pred_emu_mean = pred_emu.mean()

obsvar = np.maximum(0.2*y, 0.1)

# Fit a calibrator with emulator 1 via method = 'MLcal' and 'sampler' = metropolis-hastings 
cal_1 = calibrator(emu = emulator_GPy, y = y, x = xrep_std, thetaprior = prior_balldrop, 
                   method = 'MLcal', yvar = obsvar, 
                   args = {'theta0': np.array([0.4]), 
                           'numsamp' : 1000, 
                           'stepType' : 'normal', 
                           'stepParam' : [0.6]})

plot_pred(x_std, xrep, y, cal_1, theta_range)

# Using GPy itself
# ff = f.flatten('F').reshape(2100, 1)
# xtheta = np.array([(x, y) for y in theta for x in x_std ]).reshape(2100, 2)
# #Train GP on those realizations
# kernel = GPy.kern.RBF(input_dim=2, variance=1., lengthscale=1.)
# model_emulator = GPy.models.GPRegression(xtheta, ff, kernel)
# model_emulator.optimize()
# p = model_emulator.predict(xtheta)