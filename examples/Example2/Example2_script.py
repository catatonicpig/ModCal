import numpy as np
import scipy.stats as sps
import sys
import os
import copy
import matplotlib.pyplot as plt 
from scipy.stats import kde

current = os.path.abspath(os.getcwd())
sys.path.append(os.path.normpath(os.path.join(os.path.dirname(current), '..')))
from base.emulation import emulator
from base.calibration import calibrator

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

def balldroptrue(x):
    """Place description here."""
    def logcosh(x):
        # preventing crashing
        s = np.sign(x) * x
        p = np.exp(-2 * s)
        return s + np.log1p(p) - np.log(2)
    t = x[:, 0]
    h0 = x[:, 1]
    vter = 20
    g = 9.81
    y = h0 - (vter ** 2) / g * logcosh(g * t / vter)
    return y

x = np.array([[ 0.1, 25. ],
              [ 0.2, 25. ],
        [ 0.3, 25. ],
        [ 0.4, 25. ],
        [ 0.5, 25. ],
        [ 0.6, 25. ],
        [ 0.7, 25. ],
        [ 0.9, 25. ],
        [ 1.1, 25. ],
        [ 1.3, 25. ],
        [ 2.0, 25. ],
        [ 2.4, 25. ],
        [ 0.1, 50. ],
        [ 0.2, 50. ],
        [ 0.3, 50. ],
        [ 0.4, 50. ],
        [ 0.5, 50. ],
        [ 0.6, 50. ],
        [ 0.7, 50. ],
        [ 0.8, 50. ],
        [ 0.9, 50. ],
        [ 1.0, 50. ],
        [ 1.2, 50. ],
        [ 2.6, 50. ],
        [ 2.9, 50. ],
        [ 3.1, 50. ],
        [ 3.3, 50. ],
        [ 3.5, 50. ],
        [ 3.7, 50. ],
]).astype('object')
xv = x.astype('float')
x[x[:,1] == 25, 1] = 'lowdrop'
x[x[:,1] == 50, 1] = 'highdrop'
obsvar = 4*np.ones(x.shape[0])  # variance for the observations in 'y' below
y = balldroptrue(xv) + sps.norm.rvs(0, np.sqrt(obsvar)) #observations at each row of 'x'

# draw 50 random parameters from the prior
thetacompexp_lin = priorphys_lin.rnd(50) 

# draw 50 random parameters from the prior
thetacompexp_grav = priorphys_grav.rnd(50)  

# create a computer experiment to build an emulator for the linear simulation
lin_results = balldropmodel_linear(xv, thetacompexp_lin)

# create a computer experiment to build an emulator for the gravity simulation
grav_results = balldropmodel_grav(xv, thetacompexp_grav)  

# build an emulator for the linear simulation
emu_lin = emulator(x = x, theta = thetacompexp_lin, f = lin_results, method = 'PCGPwM') 

# build an emulator for the gravity simulation
emu_grav = emulator(x = x, theta = thetacompexp_grav, f = grav_results, method = 'PCGPwM')  

def plotpreds(cal, whichtheta):
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    cal_theta = cal.theta.rnd(1000) 
    axs[0].plot(cal_theta[:, whichtheta])
    axs[1].boxplot(cal_theta[:, whichtheta])
    axs[2].hist(cal_theta[:, whichtheta])
    # getting a prediction object   
    post = cal.predict(x)
    rndm_m = post.rnd(s = 1000)
        
    for k in (25,50):
        inds = np.where(xv[:,1] == k)[0]
        upper = np.percentile(rndm_m[:, inds], 97.5, axis = 0)
        lower = np.percentile(rndm_m[:, inds], 2.5, axis = 0)
        median = np.percentile(rndm_m[:, inds], 50, axis = 0)
        #axs[3].plot(xv[inds,0], median, 'k--', color = 'black')
        axs[3].plot(xv[inds,0], balldroptrue(xv[inds,:]), 'k--',linewidth=2)
        axs[3].fill_between(xv[inds,0], lower, upper, color = 'grey', alpha=0.25)
        axs[3].plot(xv[inds, 0], y[inds], 'ro', markersize = 5, color='red')
    
    plt.show()
    
# build calibrators for the linear simulation
cal_lin_1 = calibrator(emu = emu_lin, y = y, x = x, thetaprior = priorphys_lin, 
                     method = 'directbayes', 
                     yvar = obsvar)
                   
cal_lin_2 = calibrator(emu = emu_lin, y = y, x = x, thetaprior = priorphys_lin, 
                       method = 'MLcal', yvar = obsvar, 
                       args = {'theta0': np.array([0, 9]), 
                               'numsamp' : 1000, 
                               'stepType' : 'normal', 
                               'stepParam' : np.array([0.1, 1])})
                            
cal_lin_3 = calibrator(emu = emu_lin, y = y, x = x, thetaprior = priorphys_lin, 
                       method = 'MLcal', yvar = obsvar, 
                       args = {'sampler' : 'plumlee'})                            

# visualize posterior draws
plotpreds(cal_lin_1, 0)
plotpreds(cal_lin_2, 0)
plotpreds(cal_lin_3, 0)

plotpreds(cal_lin_1, 1)
plotpreds(cal_lin_2, 1)
plotpreds(cal_lin_3, 1)

# build calibrators for the gravity simulation
cal_grav_1 = calibrator(emu = emu_grav, y = y, x = x, thetaprior = priorphys_grav, 
                     method = 'directbayes', 
                     yvar = obsvar)
                   
cal_grav_2 = calibrator(emu = emu_grav, y = y, x = x, thetaprior = priorphys_grav, 
                        method = 'MLcal', yvar = obsvar, 
                        args = {'theta0': np.array([9]), 
                                'numsamp' : 1000, 
                                'stepType' : 'normal', 
                                'stepParam' : np.array([0.1])})
                            
cal_grav_3 = calibrator(emu = emu_grav, y = y, x = x, thetaprior = priorphys_grav, 
                       method = 'MLcal', yvar = obsvar, 
                       args = {'sampler' : 'plumlee'})                            

plotpreds(cal_grav_1, 0)
plotpreds(cal_grav_2, 0)
plotpreds(cal_grav_3, 0)



    