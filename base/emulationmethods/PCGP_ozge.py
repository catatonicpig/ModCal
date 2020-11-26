"""Header here."""

import numpy as np
import scipy.optimize as spo

"""
##############################################################################
##############################################################################
###################### THIS BEGINS THE REQUIRED PORTION ######################
######### THE NEXT FUNCTIONS REQUIRED TO BE CALLED BY CALIBRATION ############
##############################################################################
##############################################################################
"""

"""
##############################################################################
################################### fit ######################################
### The purpose of this is to take information and plug all of our fit
### information into fitinfo, which is a python dictionary. 
##############################################################################
##############################################################################
"""
def fit(fitinfo, x, theta, f, args=None):
    r"""
    Fits a calibration model.

    Parameters
    ----------
    fitinfo : dict
        An arbitary dictionary where you should place all of your fitting information once complete.
        This dictionary is pass by reference, so there is no reason to return anything. Keep
        only stuff that will be used by predict below. 
        theta : array of float
    theta :  An n-by-d matrix of parameters. n should be at least 2 times m. Each row in theta should
        correspond to a row in f.
    f : array of float
        An n-by-m matrix of responses with 'nan' representing responses not yet available. Each
        row in f should correspond to a row in theta. Each column should correspond to a row in
        x.
    x : array of objects
        An m-by-p matrix of inputs. Each column should correspond to a row in f.
    args : dict
        A dictionary containing options passed to you.
    """
    #Check this with Matt
    f = f.T
    fitinfo['offset'] = np.zeros(f.shape[1])
    fitinfo['scale'] = np.ones(f.shape[1])
    fitinfo['theta'] = theta
    fitinfo['x'] = x
    
    # Standardize the function evaluations f
    for k in range(0, f.shape[1]):
        fitinfo['offset'][k] = np.mean(f[:, k])
        fitinfo['scale'][k] = 0.9*np.std(f[:, k]) + 0.1*np.std(f)
        
    fstand = (f - fitinfo['offset']) / fitinfo['scale']
    
    # Do PCA to reduce the dimension of the function evaluations
    Vecs, Vals, _ = np.linalg.svd((fstand / np.sqrt(fstand.shape[0])).T)
    Vals = np.append(Vals, np.zeros(Vecs.shape[1] - Vals.shape[0]))
    Valssq = (fstand.shape[0]*(Vals ** 2) + 0.001) /(fstand.shape[0] + 0.001)
        
    # Find the best size of the reduced space
    numVals = 1 + np.sum(np.cumsum(Valssq) < 0.9995*np.sum(Valssq))
    numVals = np.maximum(np.minimum(2,fstand.shape[1]), numVals)
    
    # 
    fitinfo['Cs'] = Vecs * np.sqrt(Valssq)
    fitinfo['PCs'] = fitinfo['Cs'][:, :numVals]
    fitinfo['PCsi'] = Vecs[:,:numVals] * np.sqrt(1 / Valssq[:numVals])
    
    pcaval = fstand @ fitinfo['PCsi']
    fhat = pcaval @ fitinfo['PCs'].T
    fitinfo['extravar'] = np.mean((fstand-fhat) ** 2,0) * (fitinfo['scale'] ** 2)
        
    fitinfo['var0'] = np.ones(numVals)
    hypinds = np.zeros(numVals)
    emulist = [dict() for x in range(0, numVals)]
    
    # Fit an emulator for each pc
    for pcanum in range(0, numVals):
        if numVals > 1:
            #####  WILL BE MODIFIED LATER #####
            hypwhere = np.where(hypinds == np.array(range(0, numVals)))[0]
            hypstarts = None
            ##### ##### ##### ##### ##### #####
            
            emulist[pcanum] = emulation_fit(theta, pcaval[:, pcanum], hypstarts, hypwhere)
        else:
            emulist[pcanum] = emulation_fit(theta, pcaval[:, pcanum])
            hypstarts = np.zeros((numVals, emulist[pcanum]['hyp'].shape[0]))
            
        # hypstarts[pcanum, :] = emulist[pcanum]['hyp']
        # if emulist[pcanum]['hypind'] < -0.5:
        #     emulist[pcanum]['hypind'] = pcanum
        # hypinds[pcanum] = emulist[pcanum]['hypind']
    fitinfo['emulist'] = emulist
    return

 
def predict(predinfo, fitinfo, x, theta, args=None):
    r"""
    Finds prediction at theta and x given the dictionary fitinfo.

    Parameters
    ----------
    predinfo : dict
        An arbitary dictionary where you should place all of your prediction information once complete. 
        This dictionary is pass by reference, so there is no reason to return anything. Keep
        only stuff that will be used by predict.  Key elements
        predinfo['mean'] : predinfo['mean'][k] is mean of the prediction at all x at theta[k].
        predinfo['var'] : predinfo['var'][k] is variance of the prediction at all x at theta[k].
        predinfo['cov'] : predinfo['cov'][k] is mean of the prediction at all x at theta[k].
        predinfo['covhalf'] : if A = predinfo['covhalf'][k] then A.T @ A = predinfo['cov'][k]
        predinfo['rand'] : predinfo['rand'][l][k] lth draw of of x at theta[k].
    fitinfo : dict
        An arbitary dictionary where you placed all your important fitting information from the 
        fit function above.
    emu : instance of emulator class
        An emulator class instatance as defined in emulation.
    x : array of float
        An array of x values where you want to predict.
    args : dict
        A dictionary containing options passed to you.
    """
    infos = fitinfo['emulist']
    predvecs = np.zeros((theta.shape[0], len(infos)))
    predvars = np.zeros((theta.shape[0], len(infos)))
    
    #### MODIFY THIS PART LATER ####
    # if x is not None:
    #     matchingmatrix = np.ones((x.shape[0], fitinfo['x'].shape[0]))
    #     for k in range(0,x[0].shape[0]):
    #         try:
    #             matchingmatrix *= np.isclose(x[:,k][:,None].astype('float'),
    #                      fitinfo['x'][:,k].astype('float'))
    #         except:
    #             matchingmatrix *= np.equal(x[:,k][:,None],fitinfo['x'][:,k])
    #     xind = np.argwhere(matchingmatrix > 0.5)[:,1]
    # else:
    #### #### #### #### ####
    xind = range(0, fitinfo['x'].shape[0])

    for k in range(0, len(infos)):
        r = emulation_covmat(theta, fitinfo['theta'], infos[k]['hypcov'])
        predvecs[:, k] = infos[k]['muhat'] + r @ infos[k]['pw']
        Rinv = infos[k]['Rinv']
        predvars[:, k] = infos[k]['sigma2hat'] * (1 + np.exp(infos[k]['hypnug']) - np.sum(r.T * (Rinv @ r.T), 0))

    predmean = (predvecs @ fitinfo['PCs'][xind,:].T)*fitinfo['scale'][xind] + fitinfo['offset'][xind]
    predvar = fitinfo['extravar'][xind] + (predvars @ (fitinfo['PCs'][xind,:] ** 2).T) * (fitinfo['scale'][xind] ** 2)
    
    predinfo['mean'] = predmean
    predinfo['var'] = predvar

    return 


def emulation_covmat(x1, x2, gammav, returndir = False):
    '''
    Parameters
    ----------
    x1 : TYPE
        DESCRIPTION.
    x2 : TYPE
        DESCRIPTION.
    gammav : TYPE
        DESCRIPTION.
    returndir : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    d = gammav.shape[0]
    x1 = x1.reshape(1, d) if x1.ndim < 1.5 else x1
    x2 = x2.reshape(1, d) if x2.ndim < 1.5 else x2
    
    V = np.zeros([x1.shape[0], x2.shape[0]])
    R = np.ones([x1.shape[0], x2.shape[0]])
    if returndir:
        dR = np.zeros([x1.shape[0], x2.shape[0],d])
    for k in range(0, d):
        S = np.abs(np.subtract.outer(x1[:,k], x2[:,k])/np.exp(gammav[k]))
        R *= (1 + S)
        V -= S
        if returndir:
            dR[:,:, k] = (S * S) / (1 + S)
    R *= np.exp(V)
    if returndir:
        for k in range(0, d):
            dR[:,:,k] = R * dR[:,:,k]
        return R, dR
    else:
        return R
    
def emulation_negloglik(hyp, fitinfo):
    '''
    Parameters
    ----------
    hyp : TYPE
        DESCRIPTION.
    fitinfo : TYPE
        DESCRIPTION.

    Returns
    -------
    negloglik : TYPE
           Negative log-likelihood of single demensional GP model.

    '''
    
    # Obtain the hyperparameter values
    covhyp = hyp[0:fitinfo['p']]
    nughyp = hyp[fitinfo['p']]
    
    # Set the fitinfo values
    theta = fitinfo['theta']
    n = fitinfo['n']
    f = fitinfo['f']
    
    #
    R = emulation_covmat(theta, theta, covhyp)
    R = R + np.exp(nughyp)*np.diag(np.ones(n))
    W, V = np.linalg.eigh(R)
    
    #
    fspin = V.T @ f
    onespin = V.T @ np.ones(f.shape)
    muhat = np.sum(V @ (1/W * fspin)) / np.sum(V @ (1/W * onespin))
    fcenter = fspin - muhat * onespin
    sigma2hat = np.mean((fcenter) ** 2 / W)
    
    #
    negloglik = 1/2 * np.sum(np.log(W)) + n/2 * np.log(sigma2hat)
    negloglik += 1/2 * np.sum((hyp - fitinfo['hypregmean']) **2 /(fitinfo['hypregstd'] ** 2))
    return negloglik

def emulation_negloglikgrad(hyp, fitinfo):
    '''
    Parameters
    ----------
    hyp : TYPE
        DESCRIPTION.
    fitinfo : TYPE
        DESCRIPTION.

    Returns
    -------
    dnegloglik : TYPE
        DESCRIPTION.

    '''
    
    # Obtain the hyper-parameter values
    covhyp = hyp[0:fitinfo['p']]
    nughyp = hyp[fitinfo['p']]
    
    # Set the fitinfo values
    theta = fitinfo['theta']
    n = fitinfo['n']
    p = fitinfo['p']
    f = fitinfo['f']
    
    #
    R, dR = emulation_covmat(theta, theta, covhyp, True)
    R = R + np.exp(nughyp)*np.diag(np.ones(n))
    dRappend = np.exp(nughyp)*np.diag(np.ones(n)).reshape(R.shape[0], R.shape[1], 1)
    dR = np.append(dR, dRappend, axis=2)
    W, V = np.linalg.eigh(R)
    
    #
    fspin = V.T @ f
    onespin = V.T @ np.ones(f.shape)
    mudenom = np.sum(V @ (1/W * onespin))
    munum = np.sum(V @ (1/W * fspin))
    muhat = munum / mudenom
    fcenter = fspin - muhat * onespin
    sigma2hat = np.mean((fcenter) ** 2 / W)
    
    #
    dmuhat = np.zeros(p + 1)
    dsigma2hat = np.zeros(p + 1)
    dfcentercalc = (fcenter / W) @ V.T
    dfspincalc = (fspin / W) @ V.T
    donespincalc = (onespin / W) @ V.T
    Rinv = V @ np.diag(1/W) @ V.T
    dlogdet = np.zeros(p + 1)
    for k in range(0, dR.shape[2]):
        dRnorm = np.squeeze(dR[:,:,k])
        dmuhat[k] = -np.sum(donespincalc @ dRnorm @ dfspincalc) / mudenom + muhat * (np.sum(donespincalc @ dRnorm @ donespincalc)/mudenom)
        dsigma2hat[k] = -(1/n) * (dfcentercalc.T @ dRnorm @ dfcentercalc) + 2*dmuhat[k] * np.mean((fcenter * onespin) / W)
        dlogdet[k] = np.sum(Rinv * dRnorm)
        
    dnegloglik = 1/2 * dlogdet + n/2 * 1/sigma2hat * dsigma2hat
    dnegloglik += ((hyp - fitinfo['hypregmean'])  /(fitinfo['hypregstd'] ** 2))
    return dnegloglik

def emulation_fit(theta, pcaval, hypstarts=None, hypinds=None):
    '''    
    Parameters
    ----------
    theta : TYPE
        DESCRIPTION.
    pcaval : TYPE
        DESCRIPTION.
    hypstarts : TYPE, optional
        DESCRIPTION. The default is None.
    hypinds : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    subinfo : TYPE
        DESCRIPTION.
    '''
    subinfo = {}
    
    # why, why, why?
    covhyp0 = np.log(np.std(theta, 0)*3) + 1
    covhypLB = covhyp0 - 2
    covhypUB = covhyp0 + 3
    
    nughyp0 = -6
    nughypLB = -8
    nughypUB = 1
        
    # 
    nhyptrain = np.min((20*theta.shape[1], theta.shape[0]))
    thetac = np.random.choice(theta.shape[0], nhyptrain, replace=False)
    subinfo['theta'] = theta[thetac, :]
    subinfo['f'] = pcaval[thetac] # 1*obsmat[:nhyptrain, pcanum]
    subinfo['n'] = subinfo['f'].shape[0]
    subinfo['p'] = covhyp0.shape[0]
    
    subinfo['hypregmean'] =  np.append(covhyp0, nughyp0)
    subinfo['hypregstd'] =  np.append((covhypUB - covhypLB)/3, 1)
        
    # Run an optimizer to find the hyperparameters minimizing the negative likelihood
    bounds = spo.Bounds(np.append(covhypLB, nughypLB), np.append(covhypUB, nughypUB))
    opval = spo.minimize(emulation_negloglik, np.append(covhyp0, nughyp0), args=(subinfo), method='L-BFGS-B', options={'disp': False}, jac=emulation_negloglikgrad)
    
    # Obtain the optimized hyperparameter values
    hypcov = opval.x[:subinfo['p']]
    hypnug = opval.x[subinfo['p']]
        
    # 
    R = emulation_covmat(theta, theta, hypcov)
    R = R + np.exp(hypnug)*np.diag(np.ones(R.shape[0]))
    W, V = np.linalg.eigh(R)
    fspin = V.T @ pcaval
    onespin = V.T @ np.ones(pcaval.shape)    
    muhat = np.sum(V @ (1/W * fspin)) / np.sum(V @ (1/W * onespin))
    fcenter = fspin - muhat * onespin
    sigma2hat = np.mean((fcenter) ** 2 / W)
    Rinv = V @ np.diag(1/W) @ V.T
        
    #
    subinfo['hypcov'] = hypcov
    subinfo['hypnug'] = hypnug
    subinfo['R'] = R
    subinfo['Rinv'] = Rinv
    subinfo['pw'] = Rinv @ (pcaval - muhat)
    subinfo['muhat'] = muhat
    subinfo['sigma2hat'] = sigma2hat
    subinfo['theta'] = theta
   
    return subinfo

