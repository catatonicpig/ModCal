import numpy as np
from scipy.stats import norm
from pyDOE import *
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

# Observe the data
plt.scatter(X, Y)
plt.xlabel("height")
plt.ylabel("time")
plt.show()

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
theta = lhs(1, samples=n2)
theta_range = np.array([1, 30])

# Standardize 
height_range = np.array([min(X), max(X)])
X_std = (X - min(X))/(max(X) - min(X))

# Obtain computer model output
Y_model = timedrop(X_std, theta, height_range, theta_range)
plt.plot(X_std, Y_model)
plt.show()

print(np.shape(theta))
print(np.shape(X_std))
print(np.shape(Y_model))

current = os.path.abspath(os.getcwd())
sys.path.append(os.path.normpath(os.path.join(os.path.dirname(current), '..')))
from base.emulation import emulator

emulator_no_f = emulator(X_std, theta, Y_model, method = 'PCGPwM')
pred_model = emulator_no_f.predict(X_std, theta)
pred_mean = pred_model.mean()
print(np.shape(pred_mean))

plt.scatter(X, Y, color = 'grey')
for i in range(np.shape(pred_mean)[1]):
    plt.plot(X, pred_mean[:, i])
plt.xlabel("height")
plt.ylabel("time")
plt.title("Computer model surrogates for different theta")
plt.show()

# Filter out the data
ys = 1 - np.sum((Y_model - Y)**2, 0)/np.sum((Y - np.mean(Y))**2, 0)
theta_f = theta[ys > 0.5]

# Obtain computer model output via filtered data
Y_model = timedrop(X_std, theta_f, height_range, theta_range)
print(np.shape(Y_model))
plt.plot(X_std, Y_model)
plt.show()

emulator_f = emulator(X_std, theta_f, Y_model, method = 'PCGPwM')
pred_model = emulator_f.predict(X_std, theta_f)
pred_mean = pred_model.mean()
print(np.shape(pred_mean))

plt.scatter(X, Y, color = 'grey')
for i in range(np.shape(pred_mean)[1]):
    plt.plot(X, pred_mean[:, i])
plt.xlabel("height")
plt.ylabel("time")
plt.title("Computer model surrogates for different theta")
plt.show()

#Generate random reasonable theta values
theta_test = lhs(1, samples=1000)
theta_test = theta_test[(theta_test > 0.2) & (theta_test < 0.5)]
theta_test = theta_test.reshape(len(theta_test), 1)
theta_range = np.array([1, 30])
print(np.shape(theta_test))

# Obtain computer model output
Y_model_test = timedrop(X_std, theta_test, height_range, theta_range)
print(np.shape(Y_model_test))

#Predict
p_no_f = emulator_no_f.predict(X_std, theta_test)
pred_mean_no_f = p_no_f.mean()
p_f = emulator_f.predict(X_std, theta_test)
pred_mean_f = p_f.mean()
print(np.shape(pred_mean_no_f))
print(np.shape(pred_mean_f))

# Fit a classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

Y_cls = np.zeros(len(theta))
Y_cls[ys > 0.5] = 1
clf = RandomForestClassifier(n_estimators = 100, random_state = 42)#
clf.fit(theta, Y_cls)
print(clf.score(theta, Y_cls))
print(confusion_matrix(Y_cls, clf.predict(theta)))

class prior_balldrop:
    """ This defines the class instance of priors provided to the method. """
    def lpdf(theta):
        return np.squeeze(sps.uniform.logpdf(theta, 0.1, 0.9))

    def rnd(n):
        return np.vstack((sps.uniform.rvs(0.1, 0.9, size=n)))
    
#import pdb
#pdb.set_trace()
from base.calibration import calibrator  

# Fit a calibrator with filtered emulator without ML
cal_f = calibrator(emulator_f, Y, X_std, thetaprior = prior_balldrop, method = 'MLcal', yvar = obsvar, 
                   args = {'theta0': np.array([0.4]), 'numsamp' : 1000, 'stepType' : 'normal', 'stepParam' : [0.8]})

cal_f_theta = cal_f.theta.rnd(1000) 
cal_f_p = cal_f.predict(X_std)  
plt.plot(cal_f_theta)
plt.show()
plt.boxplot(cal_f_theta)
plt.show()

# Fit a calibrator with filtered emulator with ML
cal_f_ml = calibrator(emulator_f, Y, X_std, thetaprior = prior_balldrop, method = 'MLcal', yvar = obsvar, 
                      args = {'theta0': np.array([0.4]), 'clf_method': clf, 'numsamp' : 1000, 'stepType' : 'normal', 'stepParam' : [0.8]})

cal_f_ml_theta = cal_f_ml.theta.rnd(1000) 
cal_f_ml_p = cal_f_ml.predict(X_std)  
plt.plot(cal_f_ml_theta)
plt.show()
plt.boxplot(cal_f_ml_theta)
plt.show()

# Fit a calibrator with filtered emulator without ML (plumlee's sampler)
cal_f_pl = calibrator(emulator_f, Y, X_std, thetaprior = prior_balldrop, method = 'MLcal', yvar = obsvar, 
                      args = {'method' : 'plumlee', 'theta0': np.array([0.4]), 'numsamp' : 1000, 'stepType' : 'normal', 'stepParam' : [0.8]})

#cal_nf_ml = calibrator(emulator_no_f, Y, X_std, thetaprior = prior_balldrop, method = 'MLcal', yvar = obsvar, args = {'clf_method': clf}) 
#cal_nf = calibrator(emulator_no_f, Y, X_std, thetaprior = prior_balldrop, method = 'MLcal', yvar = obsvar, args = {'clf_method': None}) 
