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
# # # # # # # # # # # # # # # # # # # # 
# # # # ML adjustment approach # # # #  
# # # # # # # # # # # # # # # # # # # # 

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
    
    
# Draw 100 random parameters from uniform prior
n = 100
theta = prior_balldrop.rnd(n)
theta_range = np.array([1, 30])

# Standardize 
x_range = np.array([min(x), max(x)])
x_std = (x - min(x))/(max(x) - min(x))
xrep_std = (xrep - min(xrep))/(max(xrep) - min(xrep))

# Obtain computer model output
f = timedrop(xrep_std, theta, x_range, theta_range) 

print(np.shape(theta))
print(np.shape(x_std))
print(np.shape(f))


# Filter out the data
ys = 1 - np.sum((f - y)**2, 0)/np.sum((y - np.mean(y))**2, 0)
theta_f = theta[ys > 0.5]
print(theta_f.shape)

# Obtain computer model output via filtered data
f_f = timedrop(x_std, theta_f, x_range, theta_range)
print(np.shape(f_f))

# Fit an emulator via filtered data
emulator_f_1 = emulator(x = x_std, theta = theta_f, f = f_f, method = 'PCGPwM')

# Fit a classifier
y_cls = np.zeros(len(theta))
y_cls[ys > 0.5] = 1
clf = RandomForestClassifier(n_estimators = 100, random_state = 42)#
clf.fit(theta, y_cls)
print(clf.score(theta, y_cls))
print(confusion_matrix(y_cls, clf.predict(theta)))

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

# Fit a calibrator with emulator 1 (filtered & ML)
cal_1_f = calibrator(emu = emulator_f_1, y = y, x = xrep_std, thetaprior = prior_balldrop, method = 'MLcal', yvar = obsvar, 
                   args = {'clf_method': clf,
                           'theta0': np.array([0.4]), 
                           'numsamp' : 1000, 
                           'stepType' : 'normal', 
                           'stepParam' : [0.6]})

plot_pred(x_std, xrep, y, cal_1_f, theta_range)

# Fit a calibrator with emulator 1 (filtered & no ML)
cal_2_f = calibrator(emu = emulator_f_1, y = y, x = xrep_std, thetaprior = prior_balldrop, method = 'MLcal', yvar = obsvar, 
                   args = {'theta0': np.array([0.4]), 
                           'numsamp' : 1000, 
                           'stepType' : 'normal', 
                           'stepParam' : [0.6]})

plot_pred(x_std, xrep, y, cal_2_f, theta_range)

