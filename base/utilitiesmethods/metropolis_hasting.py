import numpy as np
from scipy.stats import norm

def metropolis_hasting(thetastart, logpostfunc, numsamp):
    # Initialize
    n = numsamp
    p = thetastart.shape[1]
    lposterior = np.zeros(2*n)
    theta = np.zeros((2*n, thetastart.shape[1]))
    lposterior[0] = logpostfunc(thetastart)
    theta[0, :] = thetastart
    n_acc = 0  
    proposal_width = 0.8
    
    for i in range(1, 2*n):
        # Suggest new theta
        theta_proposal = norm(theta[i-1, :], proposal_width).rvs()  
        theta_proposal = np.reshape(theta_proposal, (1, p))
        
        # Compute loglikelihood 
        logpost = logpostfunc(theta_proposal)

        if np.isfinite(logpost):
            p_accept = min(1, np.exp(logpost - lposterior[i-1]))
            print(p_accept)
            accept = np.random.uniform() < p_accept
        else:
            accept = False
            
        # Accept proposal?
        if accept:
            #print(theta_proposal)
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