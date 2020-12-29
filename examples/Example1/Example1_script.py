import numpy as np
import scipy.stats as sps
import sys
import os
import matplotlib.pyplot as plt 

current = os.path.abspath(os.getcwd())
sys.path.append(os.path.normpath(os.path.join(os.path.dirname(current), '..')))
from base.emulation import emulator

def balldropmodel_linear(x, theta):
    """Place description here."""
    f = np.zeros((theta.shape[0], x.shape[0]))
    for k in range(0, theta.shape[0]):
        t = x[:, 0]
        h0 = x[:, 1] + theta[k, 0]
        vter = theta[k, 1]
        f[k, :] = h0 - vter * t 
    return f.T

def balldropmodel_grav(x, theta):
    """Place description here."""
    f = np.zeros((theta.shape[0], x.shape[0]))
    for k in range(0, theta.shape[0]):
        t = x[:, 0]
        h0 = x[:, 1]
        g = theta[k]
        f[k, :] = h0 - (g / 2) * (t ** 2)
    return f.T

class priorphys_lin:
    """ This defines the class instance of priors provided to the method. """
    def lpdf(theta):
        if theta.ndim > 1.5:
            return np.squeeze(sps.norm.logpdf(theta[:, 0], 0, 5) +  # initial height deviation
                              sps.gamma.logpdf(theta[:, 1], 2, 0, 10))   # terminal velocity
        else:
            return np.squeeze(sps.norm.logpdf(theta[0], 0, 5) +  # initial height deviation
                              sps.gamma.logpdf(theta[1], 2, 0, 10))   # terminal velocity

    def rnd(n):
        return np.vstack((sps.norm.rvs(0, 5, size=n),  # initial height deviation
                          sps.gamma.rvs(2, 0, 10, size=n))).T  # terminal velocity
     
class priorphys_grav:
    """ This defines the class instance of priors provided to the method. """
    def lpdf(theta):
        if theta.ndim > 1.5:
            return np.squeeze(sps.gamma.logpdf(theta[:, 0], 2, 0, 5))  # gravity
        else:
            return np.squeeze(sps.gamma.logpdf(theta, 2, 0, 5))  # gravity

    def rnd(n):
        return np.reshape(sps.gamma.rvs(2, 0, 5, size=n), (-1,1))  # gravity

 # the time vector of interest
tvec = np.concatenate((np.arange(0.1, 4.3, 0.1), np.arange(0.1, 4.3, 0.1))) 

# the drop heights vector of interest
h0vec = np.concatenate((25 * np.ones(42), 50 * np.ones(42)))  

# the input of interest
xtot = (np.vstack((tvec, h0vec)).T).astype('object')  
xtotv = xtot.astype('float')
xtot[xtot[:,1] == 25, 1] = 'lowdrop'
xtot[xtot[:,1] == 50, 1] = 'highdrop'

# draw 50 random parameters from the prior
theta_lin = priorphys_lin.rnd(50) 

# draw 50 random parameters from the prior
theta_grav = priorphys_grav.rnd(50)  

# create a computer experiment to build an emulator for the linear simulation
f_lin = balldropmodel_linear(xtotv, theta_lin)

# create a computer experiment to build an emulator for the gravity simulation
f_grav = balldropmodel_grav(xtotv, theta_grav)  

# build an emulator for the linear simulation
emu_lin_1 = emulator(x = xtot, theta = theta_lin, f = f_lin, method = 'PCGP_ozge') 

emu_lin_2 = emulator(x = xtot, theta = theta_lin, f = f_lin, method = 'PCGPwM') 

# build an emulator for the gravity simulation
emu_grav_1 = emulator(x = xtot, theta = theta_grav, f = f_grav, method = 'PCGP_ozge')

emu_grav_2 = emulator(x = xtot, theta = theta_grav, f = f_grav, method = 'PCGPwM')  

# (Test) draw 50 random parameters from the prior
theta_lin_test = priorphys_lin.rnd(50)  

# (Test) draw 50 random parameters from the prior
theta_grav_test = priorphys_grav.rnd(50) 

# (Test) the value of the linear simulation
f_lin_test = balldropmodel_linear(xtotv, theta_lin_test) 

# (Test) the value of the gravity simulation
f_grav_test = balldropmodel_grav(xtotv, theta_grav_test)  

# obtain predict() object
pred_lin_1 = emu_lin_1.predict(xtot, theta_lin_test)
pred_lin_2 = emu_lin_2.predict(xtot, theta_lin_test)

pred_grav_1 = emu_grav_1.predict(xtot, theta_grav_test)
pred_grav_2 = emu_grav_2.predict(xtot, theta_grav_test)

# get the prediction means and variances
pred_lin_1_m, pred_lin_1_var  = pred_lin_1.mean(), pred_lin_1.var()
pred_lin_2_m, pred_lin_2_var = pred_lin_2.mean(), pred_lin_2.var()
pred_grav_1_m, pred_grav_1_var = pred_grav_1.mean(), pred_grav_1.var()
pred_grav_2_m, pred_grav_2_var = pred_grav_2.mean(), pred_grav_2.var()

# compare emulators
def plot_residuals(f, pred_mean, pred_var):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    t1 = (pred_mean - f)/np.sqrt(pred_var)
    p1_ub = np.percentile(t1, 97.5, axis = 1)
    p1_lb = np.percentile(t1, 2.5, axis = 1)
    axs[0].fill_between(range(42), p1_lb[0:42], p1_ub[0:42], color = 'grey', alpha=0.25)
    axs[0].hlines(0, 0, 42, linestyles = 'dashed', colors = 'black')
    axs[1].fill_between(range(42), p1_lb[42:84], p1_ub[42:84], color = 'grey', alpha=0.25)
    axs[1].hlines(0, 0, 42, linestyles = 'dashed', colors = 'black')
    plt.show()
    
plot_residuals(f = f_lin_test, pred_mean = pred_lin_1_m, pred_var = pred_lin_1_var)
plot_residuals(f = f_lin_test, pred_mean = pred_lin_2_m, pred_var = pred_lin_2_var)
plot_residuals(f = f_grav_test, pred_mean = pred_grav_1_m, pred_var = pred_grav_1_var)
plot_residuals(f = f_grav_test, pred_mean = pred_grav_2_m, pred_var = pred_grav_2_var)

print('Rsq PCGP = ', 1 - np.sum(np.square(pred_lin_1_m - f_lin_test))/np.sum(np.square(f_lin_test.T - np.mean(f_lin_test, axis = 1))))
print('Rsq PCGPwM = ', 1 - np.sum(np.square(pred_lin_2_m - f_lin_test))/np.sum(np.square(f_lin_test.T - np.mean(f_lin_test, axis = 1))))

print('SSE PCGP = ', np.sum(np.square(pred_lin_1_m - f_lin_test)))
print('SSE PCGPwM = ', np.sum(np.square(pred_lin_2_m - f_lin_test)))

print('Rsq PCGP = ', 1 - np.sum(np.square(pred_grav_1_m - f_grav_test))/np.sum(np.square(f_grav_test.T - np.mean(f_grav_test, axis = 1))))
print('Rsq PCGPwM = ', 1 - np.sum(np.square(pred_grav_2_m - f_grav_test))/np.sum(np.square(f_grav_test.T - np.mean(f_grav_test, axis = 1))))

print('SSE PCGP = ', np.sum(np.square(pred_grav_1_m - f_grav_test)))
print('SSE PCGPwM = ', np.sum(np.square(pred_grav_2_m - f_grav_test)))

#import pdb
#pdb.set_trace()