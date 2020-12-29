import numpy as np
import GPy

def fit(fitinfo, x, theta, f, args=None):

    if x is None:
        col_no = theta.shape[1]
        #Train GP on those realizations
        f = f.T
        kernel = GPy.kern.RBF(input_dim = col_no, variance = 1., lengthscale = 1.)
        white_kern = GPy.kern.White(1, variance=0.1)

        kernel = (kernel + white_kern)

        emulator = GPy.models.GPRegression(theta, f, kernel)
        emulator.optimize()
    
    else:
        # untensorized format
        row_flatten = x.shape[0]*theta.shape[0]
        col_flatten = x.shape[1] + theta.shape[1]
    
        f_flatten = f.flatten('F').reshape(row_flatten, 1)
        xtheta = np.array([(x_item, theta_item) for theta_item in theta for x_item in x]).reshape(row_flatten, col_flatten)
    
        #Train GP on those realizations
        kernel = GPy.kern.RBF(input_dim = col_flatten, variance = 1., lengthscale = 1.)
        emulator = GPy.models.GPRegression(xtheta, f_flatten, kernel)
        emulator.optimize()
    
    fitinfo['emu'] = emulator
    return

def predict(predinfo, fitinfo, x, theta, args=None):
    
    emulator = fitinfo['emu'] 
    
    if x is None:
        p = emulator.predict(theta)
        _mean = p[0].T
        _var = (p[1]**2).T
    else:
        shape_flatten = x.shape[0]*theta.shape[0]
        xtheta = np.array([(x_item, theta_item) for theta_item in theta for x_item in x]).reshape(shape_flatten, 2)
        p = emulator.predict(xtheta)
        
        _mean = p[0].reshape(-1, x.shape[0]).T
        _var = (p[1]**2).reshape(-1, x.shape[0]).T
    
    predinfo['mean'] = _mean
    predinfo['var'] = _var

    return 
