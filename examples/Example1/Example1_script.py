import numpy as np
import scipy.stats as sps
import sys
import os
import copy

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
thetacompexp_lin = priorphys_lin.rnd(50) 

# draw 50 random parameters from the prior
thetacompexp_grav = priorphys_grav.rnd(50)  

# create a computer experiment to build an emulator for the linear simulation
lin_results = balldropmodel_linear(xtotv, thetacompexp_lin)

# create a computer experiment to build an emulator for the gravity simulation
grav_results = balldropmodel_grav(xtotv, thetacompexp_grav)  

# build an emulator for the linear simulation
emu_lin_1 = emulator(x = xtot, theta = thetacompexp_lin, f = lin_results, method = 'PCGP_ozge', args = {'is_pca': True}) 

emu_lin_2 = emulator(x = xtot, theta = thetacompexp_lin, f = lin_results, method = 'PCGPwM') 

# build an emulator for the gravity simulation
emu_grav_1 = emulator(x = xtot, theta = thetacompexp_grav, f = grav_results, method = 'PCGP_ozge', args = {'is_pca': True})

emu_grav_2 = emulator(x = xtot, theta = thetacompexp_grav, f = grav_results, method = 'PCGPwM')  

# (Test) draw 50 random parameters from the prior
thetacompexp_lin_test = priorphys_lin.rnd(50)  

# (Test) draw 50 random parameters from the prior
thetacompexp_grav_test = priorphys_grav.rnd(50) 

# (Test) the value of the linear simulation
lin_results_test = balldropmodel_linear(xtotv, thetacompexp_lin_test) 

# (Test) the value of the gravity simulation
grav_results_test = balldropmodel_grav(xtotv, thetacompexp_grav_test)  

pred_lin_1 = emu_lin_1.predict(xtot, thetacompexp_lin_test)
pred_lin_2 = emu_lin_2.predict(xtot, thetacompexp_lin_test)

pred_grav_1 = emu_grav_1.predict(xtot, thetacompexp_grav_test)
pred_grav_2 = emu_grav_2.predict(xtot, thetacompexp_grav_test)

pred_lin_1_m = pred_lin_1.mean()
pred_lin_2_m = pred_lin_2.mean()
pred_grav_1_m = pred_grav_1.mean()
pred_grav_2_m = pred_grav_2.mean()
print(np.shape(pred_lin_1_m))
print(np.shape(pred_lin_2_m))
print(np.shape(pred_grav_1_m))
print(np.shape(pred_grav_2_m))

print(np.sum((pred_lin_1_m - lin_results_test)**2))
print(np.sum((pred_lin_2_m - lin_results_test)**2))
print(np.sum((pred_grav_1_m - grav_results_test)**2))
print(np.sum((pred_grav_2_m - grav_results_test)**2))