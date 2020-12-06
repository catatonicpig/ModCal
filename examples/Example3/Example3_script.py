import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import copy
import scipy.stats as sps

current = os.path.abspath(os.getcwd())
sys.path.append(os.path.normpath(os.path.join(os.path.dirname(current), '..')))
from base.emulation import emulator
from base.calibration import calibrator   

# Read data 
real_data = np.loadtxt('real_observations.csv', delimiter=',')
description = np.loadtxt('observation_description.csv', delimiter=',',dtype='object')
param_values = 1/np.loadtxt('param_values.csv', delimiter=',')
func_eval = np.loadtxt('func_eval.csv', delimiter=',')
param_values_test = 1/np.loadtxt('param_values_test.csv', delimiter=',')
func_eval_test = np.loadtxt('func_eval_test.csv', delimiter=',')

keepinds = np.squeeze(np.where(description[:,0].astype('float') > 30))
real_data = real_data[keepinds]
description = description[keepinds, :]
func_eval = func_eval[:,keepinds]
func_eval_test = func_eval_test[:, keepinds]

print('N:', func_eval.shape[0])
print('D:', param_values.shape[1])
print('M:', real_data.shape[0])
print('P:', description.shape[1])

from random import sample
rndsample = sample(range(0, 2000), 500)

func_eval_rnd = func_eval[rndsample, :]
param_values_rnd = param_values[rndsample, :]

def plot_observed_data(description, func_eval, real_data, param_values, title = None):
    '''
    Plots a list of profiles in the same figure. Each profile corresponds
    to a stochastic replica for the given instance.
    '''
    plt.rcParams["font.size"] = "8"
    N = len(param_values)
    D = description.shape[1]
    T = len(np.unique(description[:,0]))
    type_no = len(np.unique(description[:,1]))
    fig, axs = plt.subplots(type_no, figsize=(10, 12))
    if title is not None:
        fig.suptitle(title, fontsize=25)
    for j in range(type_no):
        for i in range(N):
            axs[j].plot(range(T), func_eval[i,(j*T):(j*T + T)], color='grey')
        axs[j].plot(range(T), real_data[(j*T):(j*T + T)], color='red')
    fig.tight_layout()
    fig.subplots_adjust(top=0.9) 
    plt.show()
    
plot_observed_data(description, func_eval_rnd, real_data, param_values_rnd, title='Computer model output (no filter)')

x = np.reshape(np.tile(range(134), 3), (402, 1))

# (No filter) Fit an emulator via 'PCGP_ozge'
emulator_nofilter = emulator(x, param_values_rnd, func_eval_rnd.T, method = 'PCGP_ozge', args = {'is_pca': True}) 
pred_model_nofilter = emulator_nofilter.predict(x, param_values_rnd)
pred_mean = pred_model_nofilter.mean()
plot_observed_data(description, pred_mean.T, real_data, param_values_rnd, title='PCGP Emulator mean (no filter)')

# (No filter) Fit an emulator via 'PCGPwM'
emulator_nofilter_2 = emulator(x, param_values_rnd, func_eval_rnd.T, method = 'PCGPwM') 
pred_model_nofilter_2 = emulator_nofilter_2.predict(x, param_values_rnd)
pred_mean_2 = pred_model_nofilter_2.mean()
plot_observed_data(description, pred_mean_2.T, real_data, param_values_rnd, title='PCGPwM Emulator mean (no filter)')

# Filter out the data and fit a new emulator with the filtered data 
par_out = param_values_rnd[np.logical_or.reduce((func_eval_rnd[:, 100] <= 200, func_eval_rnd[:, 20] >= 1000, func_eval_rnd[:, 100] >= 1000)),:]
par_in = param_values_rnd[np.logical_and.reduce((func_eval_rnd[:, 100] > 200, func_eval_rnd[:, 20] < 1000, func_eval_rnd[:, 100] < 1000)), :]
func_eval_in = func_eval_rnd[np.logical_and.reduce((func_eval_rnd[:, 100] > 200, func_eval_rnd[:, 20] < 1000, func_eval_rnd[:, 100] < 1000)), :]
par_in_test = param_values_test[np.logical_and.reduce((func_eval_test[:, 100] > 200, func_eval_test[:, 20] < 1000, func_eval_test[:, 100] < 1000)), :]
func_eval_in_test = func_eval_test[np.logical_and.reduce((func_eval_test[:, 100] > 200, func_eval_test[:, 20] < 1000, func_eval_test[:, 100] < 1000)), :]

# (Filter) Fit an emulator via 'PCGP_ozge'
emulator_filter = emulator(x, par_in, func_eval_in.T, method = 'PCGP_ozge', args = {'is_pca': True}) 
pred_model_filter = emulator_nofilter.predict(x, par_in)
pred_mean_f = pred_model_filter.mean()
plot_observed_data(description, pred_mean_f.T, real_data, par_in, title='PCGP Emulator mean (filtered)')

# (Filter) Fit an emulator via 'PCGPwM'
emulator_filter_2 = emulator(x, par_in, func_eval_in.T, method = 'PCGPwM') 
pred_model_filter_2 = emulator_filter_2.predict(x, par_in)
pred_mean_f_2 = pred_model_filter_2.mean()
plot_observed_data(description, pred_mean_f_2.T, real_data, par_in, title='PCGPwM Emulator mean (filtered)')

##### ##### ##### ##### #####
# Compare emulators
pred_model_nofilter_test = emulator_nofilter.predict(x, param_values_test)
pred_mean_test = pred_model_nofilter_test.mean()

print(np.mean(np.sum(np.square(pred_mean_test - func_eval_test.T), axis = 1)))

pred_model_nofilter_2_test = emulator_nofilter_2.predict(x, param_values_test)
pred_mean_2_test = pred_model_nofilter_2_test.mean()

print(np.mean(np.sum(np.square(pred_mean_2_test - func_eval_test.T), axis = 1)))

pred_model_filter_test = emulator_filter.predict(x, par_in_test)
pred_mean_test_f = pred_model_filter_test.mean()

print(np.mean(np.sum(np.square(pred_mean_test_f - func_eval_in_test.T), axis = 1)))

pred_model_filter_2_test = emulator_filter_2.predict(x, par_in_test)
pred_mean_2_test_f = pred_model_filter_2_test.mean()

print(np.mean(np.sum(np.square(pred_mean_2_test_f - func_eval_in_test.T), axis = 1)))
##### ##### ##### ##### #####

##### ##### ##### ##### #####
# Run a classification model
y = np.zeros(len(pred_mean.T))
y[np.logical_and.reduce((pred_mean.T[:, 130] > 200, pred_mean.T[:, 50] < 1000, pred_mean.T[:, 130] < 1000))] = 1
 
# Create the test data
pred_model_nofilter_test = emulator_nofilter.predict(x, param_values_test)
pred_mean_test = pred_model_nofilter_test.mean()
y_test = np.zeros(len(pred_mean_test.T))
y_test[np.logical_and.reduce((pred_mean_test.T[:, 130] > 200, pred_mean_test.T[:, 50] < 1000, pred_mean_test.T[:, 130] < 1000))] = 1

# Create a balanced data set
X_0 = param_values_rnd[y == 0]#[0:175]
y_0 = y[y == 0]#[0:175]
X_1 = param_values_rnd[y == 1]
y_1 = y[y == 1]
    
X = np.concatenate((X_0, X_1), axis=0)
y = np.concatenate((y_0, y_1), axis=0)

# Fit the classification model
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

model = RandomForestClassifier(n_estimators = 100, random_state = 42)
model.fit(X, y)

#Training accuracy
print(model.score(X, y))
print(confusion_matrix(y, model.predict(X)))

#Test accuracy
print(model.score(param_values_test, y_test))
print(confusion_matrix(y_test, model.predict(param_values_test)))
##### ##### ##### ##### #####

   
 
class prior_covid:
    """ This defines the class instance of priors provided to the method. """
    def lpdf(theta):
        return np.squeeze(sps.norm.logpdf(theta[:, 0], 2.5, 0.5) + 
                          sps.norm.logpdf(theta[:, 1], 4.0, 0.5) + 
                          sps.norm.logpdf(theta[:, 2], 4.0, 0.5) + 
                          sps.norm.logpdf(theta[:, 3], 1.875, 0.1) + 
                          sps.norm.logpdf(theta[:, 4], 14, 1.5) + 
                          sps.norm.logpdf(theta[:, 5], 18, 1.5) + 
                          sps.norm.logpdf(theta[:, 6], 20, 1.5) + 
                          sps.norm.logpdf(theta[:, 7], 14, 1.5) + 
                          sps.norm.logpdf(theta[:, 8], 13, 1.5) + 
                          sps.norm.logpdf(theta[:, 9], 12, 1.5))

    def rnd(n):
        return np.vstack((sps.norm.rvs(2.5, 0.5, size=n),
                          sps.norm.rvs(4.0, 0.5, size=n),
                          sps.norm.rvs(4.0, 0.5, size=n),
                          sps.norm.rvs(1.875, 0.1, size=n),
                          sps.norm.rvs(14, 1.5, size=n),
                          sps.norm.rvs(18, 1.5, size=n),
                          sps.norm.rvs(20, 1.5, size=n),
                          sps.norm.rvs(14, 1.5, size=n),
                          sps.norm.rvs(13, 1.5, size=n),
                          sps.norm.rvs(12, 1.5, size=n))).T


obsvar = np.maximum(0.2*real_data, 5)
import pdb
pdb.set_trace() 
cal_f = calibrator(emulator_filter, real_data, x, thetaprior = prior_covid, method = 'MLcal', yvar = obsvar, 
                   args = {'theta0': np.array([2, 4, 4, 1.875, 14, 18, 20, 14, 13, 12]), 
                           'numsamp' : 1000, 'stepType' : 'normal', 
                           'stepParam' : np.array([0.01, 0.01, 0.01, 0.01, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03])})


cal_f_theta = cal_f.theta.rnd(1000) 
pred_cal_f_theta = emulator_filter.predict(x, cal_f_theta)
pred_mean_cal_f_theta = pred_cal_f_theta.mean()
plot_observed_data(description, pred_mean_cal_f_theta.T, real_data, cal_f_theta, title='PCGP Posterior predictive mean (filtered)')
plt.rcParams["font.size"] = "18"
fig, axs = plt.subplots(5, 2, figsize=(15, 27))
paraind = 0
for i in range(5):
    for j in range(2):
        axs[i, j].boxplot(cal_f_theta[:, paraind])
        paraind += 1

# finally we invoke the legend (that you probably would like to customize...)
fig.tight_layout()
fig.subplots_adjust(bottom=0.05, top=0.95)
plt.show()



cal_f_2 = calibrator(emulator_filter_2, real_data, x, thetaprior = prior_covid, method = 'MLcal', yvar = obsvar, 
                   args = {'theta0': np.array([2, 4, 4, 1.875, 14, 18, 20, 14, 13, 12]), 
                           'numsamp' : 1000, 'stepType' : 'normal', 
                           'stepParam' : np.array([0.01, 0.01, 0.01, 0.01, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03])})


cal_f_2_theta = cal_f_2.theta.rnd(1000) 
pred_cal_f_2_theta = emulator_filter_2.predict(x, cal_f_2_theta)
pred_mean_cal_f_2_theta = pred_cal_f_2_theta.mean()
plot_observed_data(description, pred_mean_cal_f_2_theta.T, real_data, cal_f_theta, title='PCGPwM Posterior predictive mean (filtered)')


import pdb
pdb.set_trace() 
cal_f_pl = calibrator(emulator_filter, real_data, x, thetaprior = prior_covid, method = 'MLcal', yvar = obsvar, 
                      args = {'method' : 'plumlee'})

cal_f_pl_theta = cal_f_pl.theta.rnd(1000) 
pred_cal_f_pl_theta = emulator_filter.predict(x, cal_f_pl_theta)
pred_mean_cal_f_pl_theta = pred_cal_f_pl_theta.mean()
plot_observed_data(description, pred_mean_cal_f_pl_theta.T, real_data, cal_f_theta, title='PCGP Posterior predictive mean (filtered)')
plt.rcParams["font.size"] = "18"
fig, axs = plt.subplots(5, 2, figsize=(15, 27))
paraind = 0
for i in range(5):
    for j in range(2):
        axs[i, j].boxplot(cal_f_pl_theta[:, paraind])
        paraind += 1

# finally we invoke the legend (that you probably would like to customize...)
fig.tight_layout()
fig.subplots_adjust(bottom=0.05, top=0.95)
plt.show()


import pdb
pdb.set_trace() 
cal_f_2 = calibrator(emulator_filter_2, real_data, x, thetaprior = prior_covid, method = 'MLcal', yvar = obsvar, 
                   args = {'clf_method': model, 'theta0': np.array([2, 4, 4, 1.875, 14, 18, 20, 14, 13, 12]), 
                           'numsamp' : 1000, 'stepType' : 'normal', 
                           'stepParam' : np.array([0.01, 0.01, 0.01, 0.01, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03])})


cal_f_2_theta = cal_f_2.theta.rnd(1000) 
pred_cal_f_2_theta = emulator_filter_2.predict(x, cal_f_2_theta)
pred_mean_cal_f_2_theta = pred_cal_f_2_theta.mean()
plot_observed_data(description, pred_mean_cal_f_2_theta.T, real_data, cal_f_2_theta, title='PCGPwM Posterior predictive mean (filtered)')

plt.rcParams["font.size"] = "18"
fig, axs = plt.subplots(5, 2, figsize=(15, 27))
paraind = 0
for i in range(5):
    for j in range(2):
        axs[i, j].boxplot(cal_f_2_theta[:, paraind])
        paraind += 1

# finally we invoke the legend (that you probably would like to customize...)
fig.tight_layout()
fig.subplots_adjust(bottom=0.05, top=0.95)
plt.show()
