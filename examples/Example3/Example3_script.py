import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import copy

# Read data 
real_data = np.loadtxt('real_observations.csv', delimiter=',')
description = np.loadtxt('observation_description.csv', delimiter=',',dtype='object')
param_values = 1/np.loadtxt('param_values.csv', delimiter=',')
func_eval = np.loadtxt('func_eval.csv', delimiter=',')
param_values_test = 1/np.loadtxt('param_values_test.csv', delimiter=',')
func_eval_test = np.loadtxt('func_eval_test.csv', delimiter=',')

print('N:', func_eval.shape[0])
print('P:', param_values.shape[1])
print('M:', real_data.shape[0])
print('D:', description.shape[1])

def plot_observed_data(description, func_eval, real_data, param_values, title = None):
    '''
    Plots a list of profiles in the same figure. Each profile corresponds
    to a stochastic replica for the given instance.
    '''
    plt.rcParams["font.size"] = "6"
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
plot_observed_data(description, func_eval, real_data, param_values, title='Computer model output (filtered)')


current = os.path.abspath(os.getcwd())
sys.path.append(os.path.normpath(os.path.join(os.path.dirname(current), '..')))
from base.emulation import emulator

#import pdb
#pdb.set_trace()
x = np.reshape(np.tile(range(164), 3), (492, 1))
emulator_nofilter = emulator(x, param_values, func_eval.T, method = 'PCGP_ozge') 
#import pdb
#pdb.set_trace()
pred_model_nofilter = emulator_nofilter.predict(x, param_values)
pred_mean = pred_model_nofilter.mean()

plot_observed_data(description, pred_mean, real_data, param_values, title='Computer model output (filtered)')
