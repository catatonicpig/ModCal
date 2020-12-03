import numpy as np
from scipy.stats import norm
import scipy.stats as sps

def metropolis_hastings(logpostfunc, options):
    '''
    Parameters
    ----------
    logpostfunc : TYPE
        DESCRIPTION.
    options : TYPE
        DESCRIPTION.

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    theta : TYPE
        DESCRIPTION.

    '''
    
    # Initialize    
    if 'numsamp' in options.keys():
        n = options['numsamp']
    else:
        n = 2000
    
    if 'stepType' in options.keys():
        stepType = options['stepType']
    else:
        # default is normal
        stepType = 'normal' 
        
    if 'stepParam' in options.keys():
       stepParam = options['stepParam']
    else:
        raise ValueError('Unknown stepParam')
      
    if 'theta0' in options.keys():
        theta0 = options['theta0']
        thetastart = theta0.reshape(1, len(theta0))
    else:
        raise ValueError('Unknown theta0')
    
    p = thetastart.shape[1]
    lposterior = np.zeros(2*n)
    theta = np.zeros((2*n, thetastart.shape[1]))
    lposterior[0] = logpostfunc(thetastart)
    theta[0, :] = thetastart
    n_acc = 0  

    for i in range(1, 2*n):
        # Candidate theta
        if stepType == 'normal':
            theta_proposal = [theta[i-1, :][k] + stepParam[k] * sps.norm.rvs(0, 1, size = 1) for k in range(p)] 
        elif stepType == 'uniform':
            theta_proposal =[theta[i-1, :][k] + stepParam[k] * sps.uniform.rvs(-0.5, 0.5, size = 1) for k in range(p)] 
            
        theta_proposal = np.reshape(np.array(theta_proposal), (1, p))
        
        # Compute loglikelihood 
        logpost = logpostfunc(theta_proposal)

        if np.isfinite(logpost):
            p_accept = min(1, np.exp(logpost - lposterior[i-1]))
            accept = np.random.uniform() < p_accept
        else:
            accept = False
            
        # Accept proposal?
        if accept:
            # Update position
            theta[i, :] = theta_proposal
            lposterior[i] = logpost
            if i >= n:
                n_acc += 1
        else:
            theta[i, :] = theta[i-1, :]
            lposterior[i] = lposterior[i-1]

    theta = theta[(1*n):(2*n), :]

    print('n_acc rate=', n_acc/n)
    return theta