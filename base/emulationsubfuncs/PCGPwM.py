"""Header here."""

import numpy as np
import scipy.optimize as spo
import copy

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
def fit(fitinfo, theta, f, x, args=None):
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
    if not np.all(np.isfinite(f)):
        fitinfo['mof'] = np.logical_not(np.isfinite(f))
        fitinfo['mofrows'] = np.where(np.any(fitinfo['mof'] > 0.5,1))[0]
    else:
        fitinfo['mof'] = None
    #Storing these values for future reference
    fitinfo['theta'] = theta
    fitinfo['x'] = x
    fitinfo['f'] = f
    #The double underline should be used to represent my local functions
    __standardizef(fitinfo)
    
    __PCs(fitinfo)
    
    
    for pcanum in range(0, numVals):
        if pcanum > 0.5:
            hypwhere = np.where(hypinds == np.array(range(0, numVals)))[0]
            emulist[pcanum] = emulation_fit(theta,
                                                  pcaval[:, pcanum],
                                                  hypstarts[hypwhere,:],
                                                  hypwhere)
        else:
            emulist[pcanum] = emulation_fit(theta,
                                                  pcaval[:, pcanum])
            hypstarts = np.zeros((numVals,
                                  emulist[pcanum]['hyp'].shape[0]))
        hypstarts[pcanum, :] = emulist[pcanum]['hyp']
        if emulist[pcanum]['hypind'] < -0.5:
            emulist[pcanum]['hypind'] = pcanum
        hypinds[pcanum] = emulist[pcanum]['hypind']
    fitinfo['emulist'] = emulist
    return


"""
##############################################################################
################################### predict ##################################
### The purpose of this is to take an emulator emu alongside fitinfo, and 
### predict at x. You shove all your information into the dictionary predinfo.
##############################################################################
##############################################################################
"""
def predict(predinfo, fitinfo, theta, x, args=None):
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
    if x is not None:
        matchingmatrix = np.ones((x.shape[0], fitinfo['x'].shape[0]))
        for k in range(0,x[0].shape[0]):
            try:
                matchingmatrix *= np.isclose(x[:,k][:,None].astype('float'),
                         fitinfo['x'][:,k].astype('float'))
            except:
                matchingmatrix *= np.equal(x[:,k][:,None],fitinfo['x'][:,k])
        xind = np.argwhere(matchingmatrix > 0.5)[:,1]
    else:
        xind = range(0,fitinfo['x'].shape[0])
    rsave = np.array(np.ones(len(infos)), dtype=object)
    for k in range(0, len(infos)):
        if infos[k]['hypind'] == k:
            rsave[k] = (1-infos[k]['nug']) *\
                emulation_covmat(theta, fitinfo['theta'], 
                                       infos[k]['hypcov'])
        r = np.squeeze(rsave[infos[k]['hypind']])
        Rinv = 1*infos[(infos[k]['hypind'])]['Rinv']
        predvecs[:, k] = r @ infos[k]['pw']
        predvars[:, k] = fitinfo['var0'][k] - np.sum(r.T * (Rinv @ r.T), 0)
    predmean = (predvecs @ fitinfo['PCs'][xind,:].T)*fitinfo['scale'][xind] +\
        fitinfo['offset'][xind]
    predvar = fitinfo['extravar'][xind] + (predvars @ (fitinfo['PCs'][xind,:] ** 2).T) *\
        (fitinfo['scale'][xind] ** 2)
    
    predinfo['mean'] = 1*predmean
    predinfo['var'] = 1*predvar
    CH = (np.sqrt(np.abs(predvars))[:,:,np.newaxis] *
                             (fitinfo['PCs'][xind,:].T)[np.newaxis,:,:])
    predinfo['covhalf'] = CH
    return 

"""
##############################################################################
##############################################################################
####################### THIS ENDS THE REQUIRED PORTION #######################
###### THE NEXT FUNCTIONS ARE OPTIONAL TO BE CALLED BY CALIBRATION ###########
## If this project works, there will be a list of useful calibration functions
## to provide as you want.
##############################################################################
##############################################################################
"""

"""
##############################################################################
##############################################################################
####################### THIS ENDS THE OPTIONAL PORTION #######################
######### USE SPACE BELOW FOR ANY SUPPORTING FUNCTIONS YOU DESIRE ############
##############################################################################
##############################################################################
"""

def __standardizef(fitinfo):
    "Standardizes f by creating offset, scale and fs."
    # Extracting from input dictionary
    f = fitinfo['f']
    mof = fitinfo['mof']
    mofrows = fitinfo['mofrows']
    
    # Initializing values
    offset = np.zeros(f.shape[1])
    scale = np.zeros(f.shape[1])
    fs = np.zeros(f.shape)
    if mof is None:
        for k in range(0, f.shape[1]):
            offset[k] = np.mean(f[:, k])
            scale[k] = np.std(f[:, k])
        scale = np.maximum(scale, 10 ** (-12) * np.max(scale))
        fs = (f - offset) / scale
    else:
        for k in range(0, f.shape[1]):
            offset[k] = np.nanmean(f[:, k])
            scale[k] = np.nanstd(f[:, k])
        scale = np.maximum(scale, 10 ** (-12) * np.max(scale))
        for k in range(0, f.shape[1]):
            fs[:,k] = (f[:,k] - offset[k]) / scale[k]
            fs[np.where(mof[:, k])[0], k] = (offset[k] / scale[k])
        
        for iters in range(0,20):
            U, S, _ = np.linalg.svd(fs.T, full_matrices=False)
            epsilon = np.minimum(10 ** (-6), 0.9 * (S[1] ** 2))
            Sp = S ** 2 - epsilon
            Up = U[:, Sp > 1.1*epsilon]
            Sp = Sp[Sp > 1.1*epsilon]
            for j in range(0,mofrows.shape[0]):
                rv = mofrows[j]
                wheremof = np.where(mof[rv,:] > 0.5)[0]
                wherenotmof = np.where(mof[rv,:] < 0.5)[0]
                H = Up[wherenotmof,:].T @ Up[wherenotmof,:]
                Amat = epsilon * np.diag(1 / (Sp ** 2)) + H
                J = Up[wherenotmof,:].T @ fs[rv,wherenotmof]
                fs[rv,wheremof]= (Up[wheremof,:] * ((Sp / np.sqrt(epsilon)) ** 2)) @ (J -\
                    H @ (np.linalg.solve(Amat, J)))
    
    # Assigning new values to the dictionary
    fitinfo['offset'] = 1
    fitinfo['scale'] = 1
    fitinfo['fs'] = np.zeros(f.shape)
    return


def __PCs(fitinfo):
    "Standardizes f by creating offset, scale and fs."
    # Extracting from input dictionary
    f = fitinfo['f']
    mof = fitinfo['mof']
    mofrows = fitinfo['mofrows']
    
    
    Vecs, Vals, _ = np.linalg.svd(fitinfo['fs'])
    
    numVals = 1 + np.sum(np.cumsum(Vals ** 2) < 0.9995*np.sum(Vals ** 2))
    numVals = np.maximum(np.minimum(2,fitinfo['fs'].shape[1]),numVals)
    fitinfo['CW'] = Vals[:numVals]
    fitinfo['PC'] = Vecs[:, :numVals]
    
    
    for k in range(0, feval.shape[0]):
        indsr = np.where(mofeval[k, :] < 0.5)[0]
        rhomatsave[:, :, k] = fitinfo['Ps'][indsr, :numVals].T @ \
            np.linalg.solve(fitinfo['Ps'][indsr, :] @ fitinfo['Cs'][indsr, :].T,
                            fitinfo['Cs'][indsr, :])
        pcaval[k, :] = fitinfo['Cs'][indsr, :numVals].T @ \
            np.linalg.solve(fitinfo['Cs'][indsr, :] @ fitinfo['Cs'][indsr, :].T,
                            fstand[k, indsr])
    for k in range(0, feval.shape[0]):
        for l in range(k, feval.shape[0]):
            rhoobs[k, l, :] = np.sum(rhomatsave[:, :, k] * rhomatsave[:, :, l], 1)
            rhoobs[l, k, :] = rhoobs[k, l, :]
    if options > 1.5:
        rhoobs = np.ones(rhoobs.shape)
        rhopred = np.ones(rhopred.shape)
    return


def __covmat(x1, x2, gammav, returndir = False):
    d = gammav.shape[0]
    x1 = x1.reshape(1,d) if x1.ndim < 1.5 else x1
    x2 = x2.reshape(1,d) if x2.ndim < 1.5 else x2
    
    V = np.zeros([x1.shape[0], x2.shape[0]])
    R = np.ones([x1.shape[0], x2.shape[0]])
    if returndir:
        dR = np.zeros([x1.shape[0], x2.shape[0],d])
    for k in range(0, d):
        S = np.abs(np.subtract.outer(x1[:,k],x2[:,k])/np.exp(gammav[k]))
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

def __fit(theta, pcaval, hypstarts=None, hypinds=None):
    """Return a fitted model from the emulator model using smart method."""
    subinfo = {}
    subinfo['hypregmean'] = np.append(0.5 + np.log(np.std(theta, 0)), (0, -10))
    subinfo['hypregLB'] = np.append(-1 + np.log(np.std(theta, 0)), (-10, -20))
    subinfo['hypregUB'] = np.append(3 + np.log(np.std(theta, 0)), (1, -4))
    subinfo['hypregstd'] = (subinfo['hypregUB'] - subinfo['hypregLB']) / 3
    subinfo['hypregstd'][-2] = 2
    subinfo['hypregstd'][-1] = 0.5
    subinfo['hyp'] = 1*subinfo['hypregmean']
    nhyptrain = np.min((20*theta.shape[1], theta.shape[0]))
    thetac = np.random.choice(theta.shape[0], nhyptrain, replace=False)
    subinfo['theta'] = theta[thetac, :]
    subinfo['f'] = pcaval[thetac]
    hypind0 = -1
    if hypstarts is not None:
        L0 = emulation_negloglik(subinfo['hyp'], subinfo)
        for k in range(0, hypstarts.shape[0]):
            L1 = emulation_negloglik(hypstarts[k, :], subinfo)
            if L1 < L0:
                subinfo['hyp'] = hypstarts[k, :]
                L0 = 1* L1
                hypind0 = hypinds[k]
    opval = spo.minimize(emulation_negloglik,
                         1*subinfo['hyp'], args=(subinfo), method='L-BFGS-B',
                         options={'gtol': 0.5 / (subinfo['hypregUB'] -
                                                  subinfo['hypregLB'])},
                         jac=emulation_negloglikgrad,
                         bounds=spo.Bounds(subinfo['hypregLB'],
                                           subinfo['hypregUB']))
    if hypind0 > -0.5 and 2 * (L0-opval.fun) < \
        (subinfo['hyp'].shape[0] + 3 * np.sqrt(subinfo['hyp'].shape[0])):
        subinfo['hypcov'] = subinfo['hyp'][:-1]
        subinfo['hypind'] = hypind0
        subinfo['nug'] = np.exp(subinfo['hyp'][-1])/(1+np.exp(subinfo['hyp'][-1]))
        R = emulation_covmat(theta, theta, subinfo['hypcov'])
        R =  (1-subinfo['nug'])*R + subinfo['nug'] * np.eye(R.shape[0])
        W, V = np.linalg.eigh(R)
        Rinv = V @ np.diag(1/W) @ V.T
    else:
        subinfo['hyp'] = opval.x[:]
        subinfo['hypind'] = -1
        subinfo['hypcov'] = subinfo['hyp'][:-1]
        subinfo['nug'] = np.exp(subinfo['hyp'][-1])/(1+np.exp(subinfo['hyp'][-1]))
        R = emulation_covmat(theta, theta, subinfo['hypcov'])
        subinfo['R'] =  (1-subinfo['nug'])*R + subinfo['nug'] * np.eye(R.shape[0])
        W, V = np.linalg.eigh(subinfo['R'])
        subinfo['Rinv'] = V @ np.diag(1/W) @ V.T
        Rinv = subinfo['Rinv']
    subinfo['pw'] = Rinv @ pcaval
    return subinfo


def __covmat(x1, x2, gammav, returndir=False):
    """Return the covariance between x1 and x2 given parameter gammav."""
    x1 = 1*x1.reshape(1, gammav.shape[0]-1) if x1.ndim < 1.5 else x1
    x2 = 1*x2.reshape(1, gammav.shape[0]-1) if x2.ndim < 1.5 else x2
    V = np.zeros([x1.shape[0], x2.shape[0]])
    R = np.ones([x1.shape[0], x2.shape[0]])
    x1 = x1/np.exp(gammav[:-1])
    x2 = x2/np.exp(gammav[:-1])
    if returndir:
        dR = np.zeros([x1.shape[0], x2.shape[0], gammav.shape[0]])
    for k in range(0, gammav.shape[0]-1):
        S = np.abs(np.subtract.outer(x1[:, k], x2[:, k]))
        R *= (1 + S)
        V -= S
        if returndir:
            dR[:, :, k] = (S ** 2) / (1 + S)
    R *= np.exp(V)
    RT = R * 1/(1+np.exp(gammav[-1])) + np.exp(gammav[-1])/(1+np.exp(gammav[-1]))
    if returndir:
        dR = R[:, :, None] * dR * 1/(1+np.exp(gammav[-1]))
        dR[:, :, -1] = np.exp(gammav[-1]) / ((1+np.exp(gammav[-1])) ** 2) *\
            (1-R)
    if returndir:
        return RT, dR
    else:
        return RT

def __negloglik(hyp, info):
    """Return penalized log likelihood of single demensional GP model."""
    R0 = emulation_covmat(info['theta'], info['theta'], hyp[:-1])
    nug = np.exp(hyp[-1])/(1+np.exp(hyp[-1]))
    R = (1-nug)* R0 + nug * np.eye(info['theta'].shape[0])
    W, V = np.linalg.eigh(R)
    Vh = V / np.sqrt(np.abs(W))
    fcenter = Vh.T @ info['f']
    negloglik = 1/2 * np.sum(np.log(np.abs(W))) +1/2 * np.sum(fcenter ** 2)
    negloglik += 0.5*np.sum(((hyp-info['hypregmean']) ** 2) /
                            (info['hypregstd'] ** 2))
    return negloglik


def __negloglikgrad(hyp, info):
    """Return gradient of the penalized log likelihood of single demensional GP model."""
    R0, dR = emulation_covmat(info['theta'], info['theta'], hyp[:-1], True)
    nug = np.exp(hyp[-1])/(1+np.exp(hyp[-1]))
    R = (1-nug)* R0 + nug * np.eye(info['theta'].shape[0])
    dR = (1-nug) * dR
    dRappend = nug/((1+np.exp(hyp[-1]))) *\
        (-R0+np.eye(info['theta'].shape[0]))
    dR = np.append(dR, dRappend[:,:,None], axis=2)
    W, V = np.linalg.eigh(R)
    Vh = V / np.sqrt(np.abs(W))
    fcenter = Vh.T @ info['f']
    dnegloglik = np.zeros(dR.shape[2])
    Rinv = Vh @ (np.eye(Vh.shape[0]) - np.multiply.outer(fcenter, fcenter)) @ Vh.T
    for k in range(0, dR.shape[2]):
        dnegloglik[k] = 0.5*np.sum(Rinv * dR[:, :, k])
    dnegloglik += (hyp-info['hypregmean'])/(info['hypregstd'] ** 2)
    return dnegloglik