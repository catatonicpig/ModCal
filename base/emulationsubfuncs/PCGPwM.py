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
    Fits a emulation model.

    Parameters
    ----------
    fitinfo : dict
        An arbitary dictionary where you should place all of your fitting information once complete.
        This dictionary is pass by reference, so there is no reason to return anything. Keep
        only stuff that will be used by predict below.  Note that if you want to leverage speedy,
        updates fitinfo will contain the previous fitinfo!  So use that information to accelerate 
        anything you wany, keeping in mind that the user might have changed theta, f, and x.
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
        fitinfo['mofrows'] = None
    #Storing these values for future reference
    fitinfo['theta'] = theta
    fitinfo['x'] = x
    fitinfo['f'] = f
    #The double underline should be used to represent my local functions
    skipstnd = False
    skipPC = True
    if ('offset' in fitinfo.keys()) and ('scale' in fitinfo.keys()):
        __standardizef(fitinfo, fitinfo['offset'], fitinfo['scale'])
    else:
        __standardizef(fitinfo)
        
    if ('pct' in fitinfo.keys()) and ('pcw' in fitinfo.keys()) and\
        ('extravar' in fitinfo.keys()):
        __PCs(fitinfo,fitinfo['pct'],fitinfo['pcw'],fitinfo['extravar'])
    else:
        __PCs(fitinfo)
    numpcs = fitinfo['pc'].shape[1]
    
    if fitinfo['PCAskip'] and np.sum(np.isfinite(f)) < 1.3 * fitinfo['lastup']:
        emulist = fitinfo['emulist']
    else:
        fitinfo['PCAskip'] = False
        fitinfo['lastup'] = np.sum(np.isfinite(f))
        emulist = [dict() for x in range(0, numpcs)]
    hypinds = np.zeros(numpcs)
    for pcanum in range(0, numpcs):
        hypskip = False
        if fitinfo['PCAskip']:
            whichhyp = emulist[pcanum]['hypind']
            __fitGP1d(theta, fitinfo['pc'][:, pcanum],
                                        fitinfo['pcstdvar'][:, pcanum], prevsubmodel = emulist[pcanum])
            print('skipped re-optimization!')
        else:
            if pcanum > 0.5:
                hypwhere = np.where(hypinds == np.array(range(0, numpcs)))[0]
                emulist[pcanum] = __fitGP1d(theta, fitinfo['pc'][:, pcanum],
                                            fitinfo['pcstdvar'][:, pcanum], hypstarts[hypwhere,:],
                                            hypwhere)
            else:
                emulist[pcanum] = __fitGP1d(theta,fitinfo['pc'][:, pcanum],fitinfo['pcstdvar'][:, pcanum])
                hypstarts = np.zeros((numpcs,
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
        xnewind = np.argwhere(matchingmatrix > 0.5)[:,0]
    else:
        xind = range(0,fitinfo['x'].shape[0])
    rsave = np.array(np.ones(len(infos)), dtype=object)
    for k in range(0, len(infos)):
        if infos[k]['hypind'] == k:
            rsave[k] = (1-infos[k]['nug']) *\
                __covmat(theta, fitinfo['theta'], 
                                       infos[k]['hypcov'])
        r = np.squeeze(rsave[infos[k]['hypind']])
        Rinv = 1*infos[(infos[k]['hypind'])]['Rinv']
        predvecs[:, k] = r @ infos[k]['pw']
        predvars[:, k] = infos[k]['sig2'] * (1 - np.sum(r.T * (Rinv @ r.T), 0))
    predinfo['mean'] = np.full((theta.shape[0], x.shape[0]),np.nan)
    predinfo['var'] = np.full((theta.shape[0], x.shape[0]),np.nan)
    predinfo['mean'][:,xnewind] = (predvecs @ fitinfo['pct'][xind,:].T)*fitinfo['scale'][xind] +\
        fitinfo['offset'][xind]
    predinfo['var'][:,xnewind] = (fitinfo['extravar'][xind] + predvars @ (fitinfo['pct'][xind,:] ** 2).T) *\
        (fitinfo['scale'][xind] ** 2)
    CH = (np.sqrt(np.abs(predvars))[:,:,np.newaxis] *
                             (fitinfo['pct'][xind,:].T)[np.newaxis,:,:])
    predinfo['covhalf'] = np.full((theta.shape[0], CH.shape[1], x.shape[0]), np.nan)
    predinfo['covhalf'][:,:,xnewind] = CH
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

def __standardizef(fitinfo, offset=None, scale=None):
    "Standardizes f by creating offset, scale and fs."
    # Extracting from input dictionary
    f = fitinfo['f']
    mof = fitinfo['mof']
    mofrows = fitinfo['mofrows']
    if (offset is not None) and (scale is not None):
        if offset.shape[0] == f.shape[1] and scale.shape[0] == f.shape[1]:
            if np.any(np.nanmean(np.abs(f-offset)/scale,1) > 4):
                print(np.nanmean(np.abs(f-offset)/scale))
                offset = None
                scale = None
            else:
                print('skipped standarization!')
        else:
            offset = None
            scale = None
    if offset is None or scale is None:
        offset = np.zeros(f.shape[1])
        scale = np.zeros(f.shape[1])
        for k in range(0, f.shape[1]):
            offset[k] = np.nanmean(f[:, k])
            scale[k] = np.nanstd(f[:, k])
        scale = np.maximum(scale, 10 ** (-12) * np.max(scale))
    # Initializing values
    fs = np.zeros(f.shape)
    if mof is None:
        fs = (f - offset) / scale
    else:
        for k in range(0, f.shape[1]):
            fs[:,k] = (f[:,k] - offset[k]) / scale[k]
            if np.sum(mof[:,k]) > 0:
                a = np.empty((np.sum(mof[:,k]),))
                a[::2] = 1
                a[1::2] = -1
                fs[np.where(mof[:, k])[0], k] = 0.25*a
        for iters in range(0,20):
            U, S, _ = np.linalg.svd(fs.T, full_matrices=False)
            epsilon = 10 ** (-8)
            Sp = S ** 2 - epsilon
            Up = U[:, Sp > 0]
            Sp = np.sqrt(Sp[Sp > 0])
            for j in range(0,mofrows.shape[0]):
                rv = mofrows[j]
                wheremof = np.where(mof[rv,:] > 0.5)[0]
                wherenotmof = np.where(mof[rv,:] < 0.5)[0]
                H = Up[wherenotmof,:].T @ Up[wherenotmof,:]
                Amat = epsilon * np.diag(1 / (Sp ** 2)) + H
                J = Up[wherenotmof,:].T @ fs[rv,wherenotmof]
                fs[rv,wheremof] = (Up[wheremof,:] * ((Sp / np.sqrt(epsilon)) ** 2)) @ (J -\
                    H @ (np.linalg.solve(Amat, J)))
    # Assigning new values to the dictionary
    fitinfo['offset'] = offset
    fitinfo['scale'] = scale
    fitinfo['fs'] = fs
    return


def __PCs(fitinfo, pct=None, pcw=None, extravar=None):
    "Creates BLANK."
    # Extracting from input dictionary
    epsilon = 10 ** (-4)
    f = fitinfo['f']
    fs = fitinfo['fs']
    mof = fitinfo['mof']
    mofrows = fitinfo['mofrows']
    theta = fitinfo['theta']
    PCAskip = False
    if (pct is not None) and (pcw is not None) and (extravar is not None):
        if pct.shape[0] == fs.shape[1] and extravar.shape[0] == fs.shape[1]:
            newextravar = np.mean((fs - (fs @ pct) @ pct.T) ** 2, 0)
            newpcw = np.sqrt(np.sum((fs @ pct) ** 2,0))
            if np.any(newpcw/pcw > 4) or np.any(np.sqrt(newextravar/extravar) > 4):
                pct = None
                pcw = None
                extravar = None
            else:
                print('Skipped PCA!')
                PCAskip = True
        else:
            pct = None
            pcw = None
            extravar = None
        
    if pct is None or pcw is None or extravar is None:
        U, S, _ = np.linalg.svd(fs.T, full_matrices=False)
        Sp = S ** 2 - epsilon
        pct = U[:, Sp > epsilon]
        pcw = np.sqrt(Sp[Sp > epsilon])
        extravar = np.mean((fs - (fs @ pct) @ pct.T) ** 2, 0)
    pc = fs @ pct
    pcstdvar = np.zeros((f.shape[0],pct.shape[1]))
    if mof is not None:
        for j in range(0,mofrows.shape[0]):
            rv = mofrows[j]
            wherenotmof = np.where(mof[rv,:] < 0.5)[0]
            H = pct[wherenotmof,:].T @ pct[wherenotmof,:]
            Amat =  np.diag(epsilon / (pcw ** 2)) + H
            J = pct[wherenotmof,:].T @ fs[rv,wherenotmof]
            pc[rv,:] = (pcw ** 2 /epsilon + 1) * (J - H @ np.linalg.solve(Amat, J))
            pcstdvar[rv, :] = 1-np.abs((np.diag(H) -  np.sum(H * np.linalg.solve(Amat, H.T),0)) *\
                (pcw ** 2 / epsilon + 1))
    fitinfo['pcw'] = pcw
    fitinfo['pct'] = pct
    fitinfo['extravar'] = extravar
    fitinfo['pc'] = pc
    fitinfo['pcstdvar'] = pcstdvar
    fitinfo['PCAskip'] = PCAskip
    return

def __fitGP1d(theta, g, gvar=None, hypstarts=None, hypinds=None,prevsubmodel=None):
    """Return a fitted model from the emulator model using smart method."""
    if prevsubmodel is None:
        subinfo = {}
        subinfo['hypregmean'] = np.append(0.5 + np.log(np.std(theta, 0)), (0, -10))
        subinfo['hypregLB'] = np.append(-1 + np.log(np.std(theta, 0)), (-10, -20))
        subinfo['hypregUB'] = np.append(3 + np.log(np.std(theta, 0)), (1, -1))
        subinfo['hypregstd'] = (subinfo['hypregUB'] - subinfo['hypregLB']) / 3
        subinfo['hypregstd'][-2] = 2
        subinfo['hypregstd'][-1] = 0.5
        subinfo['hyp'] = 1*subinfo['hypregmean']
        if theta.shape[0] > 100:
            nhyptrain = np.max(np.min((20*theta.shape[1], theta.shape[0])))
            thetac = np.random.choice(theta.shape[0], nhyptrain, replace=False)
        else:
            thetac = range(0,theta.shape[0])
        subinfo['theta'] = theta[thetac, :]
        subinfo['g'] = g[thetac]
        subinfo['gvar'] = gvar[thetac]
        hypind0 = -1
        # L0 = __negloglik(subinfo['hyp'], subinfo)
        # dL0 = __negloglikgrad(subinfo['hyp'], subinfo)
        # for k in range(0, subinfo['hyp'].shape[0]):
        #     hyp1 = 1 * subinfo['hyp']
        #     hyp1[k] += (10 ** (-6))
        #     L1 = __negloglik(hyp1, subinfo)
        #     print(dL0[k])
        #     print((10 ** 6) * (L1-L0))
        # raise
        if hypstarts is not None:
            L0 = __negloglik(subinfo['hyp'], subinfo)
            for k in range(0, hypstarts.shape[0]):
                L1 = __negloglik(hypstarts[k, :], subinfo)
                if L1 < L0:
                    subinfo['hyp'] = hypstarts[k, :]
                    L0 = 1* L1
                    hypind0 = hypinds[k]
        opval = spo.minimize(__negloglik,
                             1*subinfo['hyp'], args=(subinfo), method='L-BFGS-B',
                             options={'gtol': 0.5 / (subinfo['hypregUB'] -
                                                      subinfo['hypregLB'])},
                             jac=__negloglikgrad,
                             bounds=spo.Bounds(subinfo['hypregLB'],
                                               subinfo['hypregUB']))
        if hypind0 > -0.5 and 2 * (L0-opval.fun) < \
            (subinfo['hyp'].shape[0] + 3 * np.sqrt(subinfo['hyp'].shape[0])):
            subinfo['hypcov'] = subinfo['hyp'][:-1]
            subinfo['hypind'] = hypind0
            subinfo['nug'] = np.exp(subinfo['hyp'][-1])/(1+np.exp(subinfo['hyp'][-1]))
            R =  __covmat(theta, theta, subinfo['hypcov'])
            subinfo['R'] =  (1-subinfo['nug'])*R + subinfo['nug'] * np.eye(R.shape[0])
            if subinfo['gvar'] is not None:
                R += np.diag(gvar)
            W, V = np.linalg.eigh(R)
            Vh = V / np.sqrt(np.abs(W))
            fcenter = Vh.T @ g
            subinfo['sig2'] = np.mean(fcenter ** 2)
            subinfo['Rinv'] = V @ np.diag(1/W) @ V.T
        else:
            subinfo['hyp'] = opval.x[:]
            subinfo['hypind'] = -1
            subinfo['hypcov'] = subinfo['hyp'][:-1]
            subinfo['nug'] = np.exp(subinfo['hyp'][-1])/(1+np.exp(subinfo['hyp'][-1]))
            R =  __covmat(theta, theta, subinfo['hypcov'])
            subinfo['R'] =  (1-subinfo['nug'])*R + subinfo['nug'] * np.eye(R.shape[0])
            if subinfo['gvar'] is not None:
                subinfo['R'] += np.diag(gvar)
            W, V = np.linalg.eigh(subinfo['R'])
            Vh = V / np.sqrt(np.abs(W))
            fcenter = Vh.T @ g
            subinfo['sig2'] = np.mean(fcenter ** 2)
            subinfo['Rinv'] = V @ np.diag(1/W) @ V.T
        subinfo['pw'] = subinfo['Rinv'] @ g
    else:
        subinfo = prevsubmodel
        R =  __covmat(theta, theta, subinfo['hypcov'])
        subinfo['R'] =  (1-subinfo['nug'])*R + subinfo['nug'] * np.eye(R.shape[0])
        if subinfo['gvar'] is not None:
            subinfo['R'] += np.diag(gvar)
        W, V = np.linalg.eigh(subinfo['R'])
        Vh = V / np.sqrt(np.abs(W))
        fcenter = Vh.T @ g
        subinfo['sig2'] = np.mean(fcenter ** 2)
        subinfo['Rinv'] = V @ np.diag(1/W) @ V.T
        subinfo['pw'] = subinfo['Rinv'] @ g
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
    R0 =  __covmat(info['theta'], info['theta'], hyp[:-1])
    nug = np.exp(hyp[-1])/(1+np.exp(hyp[-1]))
    R = (1-nug)* R0 + nug * np.eye(info['theta'].shape[0]) 
    if info['gvar'] is not None:
        R += np.diag(info['gvar'])
    W, V = np.linalg.eigh(R)
    Vh = V / np.sqrt(np.abs(W))
    fcenter = Vh.T @ info['g']
    n = info['g'].shape[0]
    sig2hat = np.mean(fcenter ** 2)
    negloglik = 1/2 * np.sum(np.log(np.abs(W))) + 1/2 * n * np.log(sig2hat)
    negloglik += 0.5*np.sum(((hyp-info['hypregmean']) ** 2) /
                            (info['hypregstd'] ** 2))
    return negloglik


def __negloglikgrad(hyp, info):
    """Return gradient of the penalized log likelihood of single demensional GP model."""
    R0, dR = __covmat(info['theta'], info['theta'], hyp[:-1], True)
    nug = np.exp(hyp[-1])/(1+np.exp(hyp[-1]))
    R = (1-nug)* R0 + nug * np.eye(info['theta'].shape[0])
    if info['gvar'] is not None:
        R += np.diag(info['gvar'])
    dR = (1-nug) * dR
    dRappend = nug/((1+np.exp(hyp[-1]))) *\
        (-R0+np.eye(info['theta'].shape[0]))
    dR = np.append(dR, dRappend[:,:,None], axis=2)
    W, V = np.linalg.eigh(R)
    Vh = V / np.sqrt(np.abs(W))
    fcenter = Vh.T @ info['g']
    n = info['g'].shape[0]
    sig2hat = np.mean(fcenter ** 2)
    dnegloglik = np.zeros(dR.shape[2])
    Rinv = Vh @ Vh.T
    for k in range(0, dR.shape[2]):
        dsig2hat =  - np.sum((Vh @ np.multiply.outer(fcenter, fcenter) @ Vh.T) * dR[:, :, k]) / n
        dnegloglik[k] += 0.5* n * dsig2hat / sig2hat
        dnegloglik[k] += 0.5*np.sum(Rinv * dR[:, :, k])
    dnegloglik += (hyp-info['hypregmean'])/(info['hypregstd'] ** 2)
    return dnegloglik
