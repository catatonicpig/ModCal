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

# Note that this is a stochastic one but we convert it deterministic by taking the average
# height
X = np.reshape(ball[:, 0], (n, 1))
X = X[0:21]

# time
Y = np.reshape(ball[:, 1], ((n, 1)))
Ysplit = np.split(Y, 3)
ysum = 0
for y in  Ysplit:
    ysum += y 
obsvar = 0.00001*np.ones(X.shape[0])
Y = ysum/3 + sps.norm.rvs(0, np.sqrt(obsvar)).reshape((21, 1)) 

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

print(np.shape(theta))
print(np.shape(X_std))
print(np.shape(Y_model))


plt.plot(X_std, Y_model)
plt.show()


current = os.path.abspath(os.getcwd())
sys.path.append(os.path.normpath(os.path.join(os.path.dirname(current), '..')))
from base.emulation import emulator
X_std2 = np.concatenate((X_std, X_std), axis = 1)
emulator_no_f = emulator(X_std2, theta, Y_model, method = 'PCGPwM')

#Predict
pred_model = emulator_no_f.predict(X_std2, theta)
pred_mean = pred_model.mean()
print(np.shape(pred_mean))

plt.scatter(X, Y, color = 'grey')
for i in range(np.shape(pred_mean)[1]):
    plt.plot(X, pred_mean[:, i])
plt.xlabel("height")
plt.ylabel("time")
plt.title("Computer model surrogates for different theta")
plt.show()

# emunf = emulator(X_std, theta_f, Y_model, method = 'PCGP_ozge', args = {'is_pca': False}) 
# prednf = emulator_f.predict(X_std, theta_f)
# pred_meannf = prednf.mean()
# plt.scatter(X, Y, color = 'grey')
# for i in range(np.shape(pred_meanf)[1]):
#     plt.plot(X, pred_meanf[:, i])
# plt.xlabel("height")
# plt.ylabel("time")
# plt.title("Computer model surrogates for different theta")
# plt.show()

ys = 1 - np.sum((Y_model - Y)**2, 0)/np.sum((Y - np.mean(Y))**2, 0)
theta_f = theta[ys > 0.5]
# Obtain computer model output
Y_model = timedrop(X_std, theta_f, height_range, theta_range)
print(np.shape(Y_model))

plt.plot(X_std, Y_model)
plt.show()

Y_cls = np.zeros(len(theta))
Y_cls[ys > 0.5] = 1

emulator_f = emulator(X_std2, theta_f, Y_model, method = 'PCGPwM')

#### TO check if the new emulator works with a single dim
# import pdb
# pdb.set_trace()
# emuf = emulator(X_std, theta_f, Y_model, method = 'PCGP_ozge', args = {'is_pca': False}) 
# predf = emuf.predict(X_std, theta_f)
# pred_meanf = predf.mean()
# plt.scatter(X, Y, color = 'grey')
# for i in range(np.shape(pred_meanf)[1]):
#     plt.plot(X, pred_meanf[:, i])
# plt.xlabel("height")
# plt.ylabel("time")
# plt.title("Computer model surrogates for different theta")
# plt.show()


pred_model = emulator_f.predict(X_std2, theta_f)
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
p_no_f = emulator_no_f.predict(X_std2, theta_test)
pred_mean_no_f = p_no_f.mean()
p_f = emulator_f.predict(X_std2, theta_test)
pred_mean_f = p_f.mean()
print(np.shape(pred_mean_no_f))
print(np.shape(pred_mean_f))


# Fit a classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
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
    

from base.calibration import calibrator   
# import pdb
# pdb.set_trace()
# cal_m_f = calibrator(emulator_f, Y.reshape(21), X_std2, thetaprior = prior_balldrop, yvar = obsvar, method = 'directbayes_wgrad')



import pdb
pdb.set_trace()
cal_nf = calibrator(emulator_f, Y, X_std2, thetaprior = prior_balldrop, 
                    method = 'MLcal', yvar = obsvar, 
                    args = {'clf_method': None}) 
cal_nf.theta.rnd(100)    
plt.plot(cal_nf.theta.rnd(100))
plt.boxplot(cal_nf.theta.rnd(1000))

cal_f = calibrator(emulator_f, Y, X_std2, thetaprior = prior_balldrop, 
                   method = 'MLcal', yvar = obsvar, 
                   args = {'clf_method': clf}) 
plt.boxplot(cal_f.theta.rnd(1000))