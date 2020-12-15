import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from random import sample
import scipy.stats as sps
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
current = os.path.abspath(os.getcwd())
sys.path.append(os.path.normpath(os.path.join(os.path.dirname(current), '..')))
from base.calibration import calibrator 
from base.emulation import emulator

# Read data 
real_data = np.loadtxt('real_observations.csv', delimiter=',')
description = np.loadtxt('observation_description.csv', delimiter=',',dtype='object')
param_values = 1/np.loadtxt('param_values.csv', delimiter=',')
func_eval = np.loadtxt('func_eval.csv', delimiter=',')
param_values_test = 1/np.loadtxt('param_values_test.csv', delimiter=',')
func_eval_test = np.loadtxt('func_eval_test.csv', delimiter=',')

# Remove the initial 30-days time period from the data
keepinds = np.squeeze(np.where(description[:,0].astype('float') > 30))
real_data = real_data[keepinds]
description = description[keepinds, :]
func_eval = func_eval[:,keepinds]
func_eval_test = func_eval_test[:, keepinds]

print('N:', func_eval.shape[0])
print('D:', param_values.shape[1])
print('M:', real_data.shape[0])
print('P:', description.shape[1])

# Get the random sample of 500 out of 2000
rndsample = sample(range(0, 2000), 500)
func_eval_rnd = func_eval[rndsample, :]
param_values_rnd = param_values[rndsample, :]

# Filter out the data and fit a new emulator with the filtered data 
par_out = param_values_rnd[np.logical_or.reduce((func_eval_rnd[:, 100] <= 200, func_eval_rnd[:, 20] >= 1000, func_eval_rnd[:, 100] >= 1000)),:]
par_in = param_values_rnd[np.logical_and.reduce((func_eval_rnd[:, 100] > 200, func_eval_rnd[:, 20] < 1000, func_eval_rnd[:, 100] < 1000)), :]
func_eval_in = func_eval_rnd[np.logical_and.reduce((func_eval_rnd[:, 100] > 200, func_eval_rnd[:, 20] < 1000, func_eval_rnd[:, 100] < 1000)), :]
par_in_test = param_values_test[np.logical_and.reduce((func_eval_test[:, 100] > 200, func_eval_test[:, 20] < 1000, func_eval_test[:, 100] < 1000)), :]
func_eval_in_test = func_eval_test[np.logical_and.reduce((func_eval_test[:, 100] > 200, func_eval_test[:, 20] < 1000, func_eval_test[:, 100] < 1000)), :]

# (Filter) Observe computer model outputs     
plot_model_data(description, func_eval_in, real_data, par_in, title='Computer model output')

# (Filter) Fit an emulator via 'PCGP_ozge'
emulator_f_1 = emulator(x = x, theta = par_in, f = func_eval_in.T, method = 'PCGP_ozge', args = {'is_pca': True}) 

# (Filter) Fit an emulator via 'PCGPwM'
emulator_f_2 = emulator(x = x, theta = par_in, f = func_eval_in.T, method = 'PCGPwM') 

##### ##### ##### ##### #####
# Compare emulators
pred_1_test = emulator_1.predict(x, par_in_test)
pred_mean_test_1 = pred_1_test.mean()
print("Rsq = ", 1 - np.sum(np.square(pred_mean_test_1 - func_eval_in_test.T))/np.sum(np.square(func_eval_in_test - np.mean(func_eval_in_test.T, axis = 1))))
print('MSE = ', np.mean(np.sum(np.square(pred_mean_test_1 - func_eval_in_test.T), axis = 1)))

pred_2_test = emulator_2.predict(x, par_in_test)
pred_mean_2_test = pred_2_test.mean()
print("Rsq = ", 1 - np.sum(np.square(pred_mean_2_test - func_eval_in_test.T))/np.sum(np.square(func_eval_in_test - np.mean(func_eval_in_test.T, axis = 1))))
print('MSE = ', np.mean(np.sum(np.square(pred_mean_2_test - func_eval_in_test.T), axis = 1)))

pred_f_1_test = emulator_f_1.predict(x, par_in_test)
pred_mean_test_f_1 = pred_f_1_test.mean()
print("Rsq = ", 1 - np.sum(np.square(pred_mean_test_f_1 - func_eval_in_test.T))/np.sum(np.square(func_eval_in_test - np.mean(func_eval_in_test.T, axis = 1))))
print('MSE = ', np.mean(np.sum(np.square(pred_mean_test_f_1 - func_eval_in_test.T), axis = 1)))

pred_f_2_test = emulator_f_2.predict(x, par_in_test)
pred_mean_test_f_2 = pred_f_2_test.mean()
print("Rsq = ", 1 - np.sum(np.square(pred_mean_test_f_2 - func_eval_in_test.T))/np.sum(np.square(func_eval_in_test - np.mean(func_eval_in_test.T, axis = 1))))
print('MSE = ', np.mean(np.sum(np.square(pred_mean_test_f_2 - func_eval_in_test.T), axis = 1)))
##### ##### ##### ##### #####

##### ##### ##### ##### #####
# Run a classification model
pred_1 = emulator_1.predict(x, param_values_rnd)
pred_mean_1 = pred_1.mean()
y = np.zeros(len(pred_mean_1.T))
y[np.logical_and.reduce((pred_mean_1.T[:, 100] > 200, pred_mean_1.T[:, 20] < 1000, pred_mean_1.T[:, 100] < 1000))] = 1
 
# Create the test data
pred_1_test = emulator_1.predict(x, param_values_test)
pred_mean_1_test = pred_1_test.mean()
y_test = np.zeros(len(pred_mean_1_test.T))
y_test[np.logical_and.reduce((pred_mean_1_test.T[:, 100] > 200, pred_mean_1_test.T[:, 20] < 1000, pred_mean_1_test.T[:, 100] < 1000))] = 1

# Create a balanced data set
X_0 = param_values_rnd[y == 0][0:130]
y_0 = y[y == 0][0:130]
X_1 = param_values_rnd[y == 1]
y_1 = y[y == 1]
    
X = np.concatenate((X_0, X_1), axis=0)
y = np.concatenate((y_0, y_1), axis=0)

# Fit the classification model
model = RandomForestClassifier(n_estimators = 100, random_state = 42)
model.fit(X, y)

#Training accuracy
print(model.score(X, y))
print(confusion_matrix(y, model.predict(X)))

#Test accuracy
print(model.score(param_values_test, y_test))
print(confusion_matrix(y_test, model.predict(param_values_test)))
##### ##### ##### ##### #####

#import pdb
#pdb.set_trace() 
def boxplot_param(theta):
    plt.rcParams["font.size"] = "16"
    fig, axs = plt.subplots(2, 5, figsize=(20, 8))
    paraind = 0
    for i in range(2):
        for j in range(5):
            axs[i, j].boxplot(theta[:, paraind])
            paraind += 1

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.05, top=0.95)
    plt.show()


import pdb
pdb.set_trace()     
cal_f_1 = calibrator(emu = emulator_f_1, y = real_data, x = x, thetaprior = prior_covid, method = 'MLcal', yvar = obsvar, 
                        args = {'theta0': np.array([2, 4, 4, 1.875, 14, 18, 20, 14, 13, 12]), 
                                'numsamp' : 1000, 
                                'stepType' : 'normal', 
                                'stepParam' : np.array([0.01, 0.01, 0.01, 0.01, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03])})

plot_pred_interval(cal_f_1)
cal_f_1_theta = cal_f_1.theta.rnd(1000) 
boxplot_param(cal_f_1_theta)

cal_f_ml_1 = calibrator(emu = emulator_f_1, y = real_data, x = x, thetaprior = prior_covid, method = 'MLcal', yvar = obsvar, 
                        args = {'clf_method': model, 
                                'theta0': np.array([2, 4, 4, 1.875, 14, 18, 20, 14, 13, 12]), 
                                'numsamp' : 1000, 
                                'stepType' : 'normal', 
                                'stepParam' : np.array([0.01, 0.01, 0.01, 0.01, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03])})

plot_pred_interval(cal_f_ml_1)
cal_f_ml_1_theta = cal_f_ml_1.theta.rnd(1000) 
boxplot_param(cal_f_ml_1_theta)

# cal_f_pl = calibrator(emu = emulator_f_1, y = real_data, x = x, thetaprior = prior_covid, method = 'MLcal', yvar = obsvar, 
#                       args = {'sampler' : 'plumlee'})

# plot_pred_interval(cal_f_pl)
# cal_f_pl_theta = cal_f_pl.theta.rnd(1000) 
# boxplot_param(cal_f_pl_theta)


cal_f_2 = calibrator(emu = emulator_f_2, y = real_data, x = x, thetaprior = prior_covid, method = 'MLcal', yvar = obsvar, 
                        args = {'theta0': np.array([2, 4, 4, 1.875, 14, 18, 20, 14, 13, 12]), 
                                'numsamp' : 1000, 
                                'stepType' : 'normal', 
                                'stepParam' : np.array([0.02, 0.02, 0.02, 0.02, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03])})

plot_pred_interval(cal_f_2)
cal_f_2_theta = cal_f_2.theta.rnd(1000) 
boxplot_param(cal_f_2_theta)


cal_f_ml_2 = calibrator(emu = emulator_f_2, y = real_data, x = x, thetaprior = prior_covid, method = 'MLcal', yvar = obsvar, 
                        args = {'clf_method': model,
                                'theta0': np.array([2, 4, 4, 1.875, 14, 18, 20, 14, 13, 12]), 
                                'numsamp' : 1000, 
                                'stepType' : 'normal', 
                                'stepParam' : np.array([0.02, 0.02, 0.02, 0.02, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03])})

plot_pred_interval(cal_f_ml_2)
cal_f_ml_2_theta = cal_f_ml_2.theta.rnd(1000) 
boxplot_param(cal_f_ml_2_theta)
