import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import scipy.stats as sps
current = os.path.abspath(os.getcwd())
sys.path.append(os.path.normpath(os.path.join(os.path.dirname(current), '..')))
from base.emulation import emulator
from base.calibration import calibrator


########################################################
# This example shows how to use GPy within our framework
# via Derek's example at BANDcamp 
# (we might remove because it only has 1 observation--so it is not complete)
########################################################


np.random.seed(1)
def lin_hydro_model(theta, intercept = 0.12, slope = -0.25, noise = 0.1):
    '''
    A linear hydrodynamic model with an additional random noise error
    Parameters
    ----------
    theta : TYPE
        DESCRIPTION.
    intercept : TYPE, optional
        DESCRIPTION. The default is 0.12.
    slope : TYPE, optional
        DESCRIPTION. The default is -0.25.
    noise : TYPE, optional
        DESCRIPTION. The default is 0.1.

    Returns
    -------
    y : TYPE
        DESCRIPTION.
    dy : TYPE
        DESCRIPTION.

    '''
    y = intercept + slope * (theta) 
    dy = noise * y * np.random.normal() 
    y += dy 
    y = np.maximum(0, y)
    return y, dy

# number of design points
n_design_pts = 20 
# minimum value for the parameter (eta/s)
eta_over_s_min = 0. 
# maximum value for the parameter (eta/s)
eta_over_s_max = 4. / (4. * np.pi) 

theta_model = np.linspace(eta_over_s_min, eta_over_s_max, n_design_pts).reshape(-1,1)

f, f_epsilon = lin_hydro_model(theta = theta_model, noise = 0.1)

fig, axes = plt.subplots(1, 1, figsize=(8, 6))
plt.errorbar(theta_model.flatten(), f.flatten(), f_epsilon.flatten(), fmt='o', c='black')
plt.xlabel(r'$\eta/s$')
plt.ylabel(r'$v_2$')
plt.title('Model Design Predictions')
plt.tight_layout(True)
plt.show()

# Emulator 
emulator_GPy = emulator(theta = theta_model, f = f.T, method = 'GPy')
pred_emu = emulator_GPy.predict(theta = theta_model)
pred_emu_mean = pred_emu.mean()

# define two alternative priors
class prior_hydro_uniform:
    """ This defines the class instance of priors provided to the method. """
    def lpdf(theta):
        if theta.ndim > 1.5:
            return np.squeeze(sps.uniform.logpdf(theta[:, 0], 0, 4. / (4. * np.pi)))
        else:
            return np.squeeze(sps.uniform.logpdf(theta, 0, 4. / (4. * np.pi)))

    def rnd(n):
        return np.vstack((sps.uniform.rvs(0, 4. / (4. * np.pi), size=n)))

class prior_hydro_normal:
    """ This defines the class instance of priors provided to the method. """
    def lpdf(theta):
        if theta.ndim > 1.5:
            return np.squeeze(sps.truncnorm.logpdf(theta[:, 0], 0, 4. / (4. * np.pi), 2. / (4. * np.pi), (1. / (10. * np.pi))))
        else:
            return np.squeeze(sps.truncnorm.logpdf(theta, 0, 4. / (4. * np.pi), 2. / (4. * np.pi),(1. / (10. * np.pi))))

    def rnd(n):
        return np.vstack((sps.truncnorm.rvs(0, 4. / (4. * np.pi), 2. / (4. * np.pi), np.sqrt(1. / (10. * np.pi)), size=n)))

# observe the shape of the priors
#plt.hist(prior_hydro_uniform.rnd(100000), density = True)
#plt.hist(prior_hydro_normal.rnd(100000), density = True)

# experimental relative uncertainty
exp_rel_uncertainty = 0.1 

#v_2 experimental mean
y_exp = 0.09

#v_2 experimental uncertainty
dy_exp = y_exp * exp_rel_uncertainty 

#y = np.repeat(y_exp, theta_model.shape[0]).reshape((theta_model.shape[0], theta_model.shape[1]))

#obsvar = np.maximum(0.2*y, 0.1)
#obsvar = np.repeat(dy_exp, theta_model.shape[0]).reshape((theta_model.shape[0], theta_model.shape[1]))

import pdb
pdb.set_trace()
# Fit a calibrator with emulator_GPy via via method = 'MLcal' and 'sampler' = metropolis-hastings 
cal_1 = calibrator(emu = emulator_GPy, y = np.array([y_exp]), thetaprior = prior_hydro_uniform, 
                   method = 'MLcal', yvar = np.array([dy_exp]), 
                   args = {'theta0': np.array([0.15]), 
                           'numsamp' : 1000, 
                           'stepType' : 'normal', 
                           'stepParam' : [0.5]})
plt.hist(cal_1.theta.rnd(), density = True)
#plot_pred(x_std, xrep, y, cal_1, theta_range)

