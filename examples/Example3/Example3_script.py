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
m = len(ball)
# height
xrep = np.reshape(ball[:, 0], (m, 1))
x = xrep[0:21]
# time
y = np.reshape(ball[:, 1], ((m, 1)))

# Observe the data
plt.scatter(xrep, y, color = 'red')
plt.xlabel("height (meters)")
plt.ylabel("time (seconds)")
plt.show()

# Computer implementation of the mathematical model
def timedrop(x, theta, hr, gr):
    '''
    Parameters
    ----------
    x : m x 1 array
        Input settings.
    theta : n x 1 array 
        Parameters to be calibrated.
    hr : Array of size 2
        min and max value of height.
    gr : Array of size 2
        min and max value of gravity.

    Returns
    -------
    m x n array
        m x n computer model evaluations.

    '''
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
theta_range = np.array([6, 15])

# Standardize 
x_range = np.array([min(x), max(x)])
x_std = (x - min(x))/(max(x) - min(x))
xrep_std = (xrep - min(xrep))/(max(xrep) - min(xrep))

# Obtain computer model output
f = timedrop(x_std, theta, x_range, theta_range) 

print(np.shape(theta))
print(np.shape(x_std))
print(np.shape(f))

# Fit emualtors using two different methods 
# Emulator 1
emulator_1 = emulator(x = x_std, theta = theta, f = f, method = 'PCGPwM')
# Emulator 2
emulator_2 = emulator(x = x_std, theta = theta, f = f, method = 'PCGP_ozge')

# Compare emulators
# Generate random reasonable theta values
n_test = 1000
theta_test = prior_balldrop.rnd(n_test)
print(np.shape(theta_test))

# Obtain computer model output
f_test = timedrop(x_std, theta_test, x_range, theta_range)
print(np.shape(f_test))

#Predict
p_1 = emulator_1.predict(x_std, theta_test)
p_1_mean, p_1_var = p_1.mean(), p_1.var()
p_2 = emulator_2.predict(x_std, theta_test)
p_2_mean, p_2_var = p_2.mean(), p_2.var()

# compare emulators
def plot_residuals(f, pred_mean, pred_var):
    fig, axs = plt.subplots(1, figsize=(5, 5))
    t1 = (pred_mean - f)/np.sqrt(pred_var)
    p1_ub = np.percentile(t1, 97.5, axis = 1)
    p1_lb = np.percentile(t1, 2.5, axis = 1)
    plt.fill_between(range(21), p1_lb, p1_ub, color = 'grey', alpha=0.25)
    plt.hlines(0, 0, 21, linestyles = 'dashed', colors = 'black')
    plt.show()
    
plot_residuals(f_test, p_1_mean, p_1_var) 
plot_residuals(f_test, p_2_mean, p_2_var) 
     
print('SSE PCGPwM = ', np.sum((p_1_mean - f_test)**2))
print('SSE PCGP_ozge = ', np.sum((p_2_mean - f_test)**2))

print('Rsq PCGPwM = ', 1 - np.sum(np.square(p_1_mean - f_test))/np.sum(np.square(f_test.T - np.mean(f_test, axis = 1))))
print('Rsq PCGP_ozge = ', 1 - np.sum(np.square(p_2_mean - f_test))/np.sum(np.square(f_test.T - np.mean(f_test, axis = 1))))

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
    
obsvar = np.maximum(0.2*y, 0.1)

# Fit a calibrator with emulator 1 via via method = 'MLcal' and 'sampler' = metropolis-hastings 
cal_1 = calibrator(emu = emulator_1, y = y, x = xrep_std, thetaprior = prior_balldrop, 
                   method = 'MLcal', yvar = obsvar, 
                   args = {'theta0': np.array([0.4]), 
                           'numsamp' : 1000, 
                           'stepType' : 'normal', 
                           'stepParam' : [0.3]})

plot_pred(x_std, xrep, y, cal_1, theta_range)

# Fit a calibrator via method = 'MLcal' and 'sampler' : 'plumlee'
cal_2 = calibrator(emu = emulator_1, y = y, x = xrep_std, thetaprior = prior_balldrop, 
                   method = 'MLcal', yvar = obsvar, 
                   args = {'sampler' : 'plumlee'})

plot_pred(x_std, xrep, y, cal_2, theta_range)

# Fit a calibrator via method = 'directbayes' and 'sampler' : 'plumlee'
cal_3 = calibrator(emu = emulator_1, y = y, x = xrep_std, thetaprior = prior_balldrop, 
                   method = 'directbayes', yvar = obsvar)

plot_pred(x_std, xrep, y, cal_3, theta_range)
