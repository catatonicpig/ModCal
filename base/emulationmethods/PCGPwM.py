"""Header here."""

import numpy as np
import scipy.optimize as spo
import scipy.linalg as spla
import copy

""" 
This [emulationfitinfo] automatically filled by docinfo.py when running updatedocs.py
The purpose of automatic documentation is to ease the burden for new methods.
##############################################################################
################################### fit ######################################
### The purpose of this is to take information and plug all of our fit
### information into fitinfo, which is a python dictionary. 
##############################################################################
##############################################################################
"""
def fit(fitinfo, x, theta, f, args):
    r"""
    Fits a emulation model.
    This [emulationfitdocstring] automatically filled by docinfo.py when running updatedocs.py
    
    Parameters
    ----------
    fitinfo : dict
        An arbitary dictionary where you should place all of your fitting information once complete.
        This dictionary is pass by reference, so there is no reason to return anything. Keep
        only stuff that will be used by predict below.  Note that if you want to leverage speedy,
        updates fitinfo will contain the previous fitinfo!  So use that information to accelerate 
        anything you wany, keeping in mind that the user might have changed theta, f, and x.
    x : array of objects
        An matrix (vector) of inputs. Each row should correspond to a row in f.
    theta :  array of objects
        An matrix (vector) of parameters. Each row should correspond to a row in f.
    f : array of float
        An matrix (vector) of responses.  Each row in f should correspond to a row in x. Each 
        column should correspond to a row in theta.
    args : dict
        A dictionary containing options passed to you.
    """
    f = f.T
    if not np.all(np.isfinite(f)):
        fitinfo['mof'] = np.logical_not(np.isfinite(f))
        fitinfo['mofrows'] = np.where(np.any(fitinfo['mof'] > 0.5,1))[0]
    else:
        fitinfo['mof'] = None
        fitinfo['mofrows'] = None
    #Storing these values for future reference
    fitinfo['x'] = x
    fitinfo['theta'] = theta
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
    
    if not fitinfo['PCAskip'] or not np.sum(np.isfinite(f)) < 2 * fitinfo['lastup']:
        fitinfo['PCAskip'] = False
        fitinfo['lastup'] = np.sum(np.isfinite(f))
        emulist = [dict() for x in range(0, numpcs)]
        hypinds = np.zeros(numpcs)
    if fitinfo['PCAskip']:
        for pcanum in range(0, numpcs):
            __fitGP1d(theta, fitinfo['pc'][:, pcanum],
                                        fitinfo['pcstdvar'][:, pcanum],
                                        prevsubmodel = fitinfo['emulist'][pcanum])
    else:
        for pcanum in range(0, numpcs):
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
This [emulationpredictinfo] automatically filled by docinfo.py when running updatedocs.py
##############################################################################
################################### predict ##################################
### The purpose of this is to take an emulator emu alongside fitinfo, and 
### predict at x and theta. You shove all your information into the dictionary predinfo.
##############################################################################
##############################################################################
"""
def predict(predinfo, fitinfo, x, theta,  args=None):
    r"""
    Finds prediction at theta and x given the dictionary fitinfo.
    This [emulationpredictdocstring] automatically filled by docinfo.py when running updatedocs.py

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
    x : array of objects
        An matrix (vector) of inputs for prediction.
    theta :  array of objects
        An matrix (vector) of parameters to prediction.
    args : dict
        A dictionary containing options passed to you.
    """
    infos = fitinfo['emulist']
    predvecs = np.zeros((theta.shape[0], len(infos)))
    predvars = np.zeros((theta.shape[0], len(infos)))
    if predvecs.ndim < 1.5:
        predvecs = predvecs.reshape((1,-1))
        predvars = predvars.reshape((1,-1))
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
            rsave[k] = __covmat(theta, fitinfo['theta'], infos[k]['hypcov'])
        r = (1-infos[k]['nug']) * np.squeeze(rsave[infos[k]['hypind']])
        rVh = r @ infos[k]['Vh']
        if rVh.ndim < 1.5:
            rVh = rVh.reshape((1,-1))
        predvecs[:, k] = r @ infos[k]['pw']
        predvars[:, k] = infos[k]['sig2'] * (1 - np.sum(rVh ** 2, 1))
    predinfo['mean'] = np.full((x.shape[0], theta.shape[0]),np.nan)
    predinfo['var'] = np.full((x.shape[0], theta.shape[0]),np.nan)
    predinfo['mean'][xnewind,:] = ((predvecs @ fitinfo['pct'][xind,:].T)*fitinfo['scale'][xind] +\
        fitinfo['offset'][xind]).T
    predinfo['var'][xnewind,:] = ((fitinfo['extravar'][xind] + predvars @ (fitinfo['pct'][xind,:] ** 2).T) *\
        (fitinfo['scale'][xind] ** 2)).T
    CH = (np.sqrt(np.abs(predvars))[:,:,np.newaxis] *
                             (fitinfo['pct'][xind,:].T)[np.newaxis,:,:])
    predinfo['covxhalf'] = np.full((theta.shape[0],CH.shape[1],x.shape[0]), np.nan)
    predinfo['covxhalf'][:,:,xnewind] = CH
    predinfo['covxhalf'] = predinfo['covxhalf'].transpose((1,0,2))
    return

"""
This [emulationadditionalfuncsinfo] automatically filled by docinfo.py when running updatedocs.py
##############################################################################
##############################################################################
####################### THIS ENDS THE REQUIRED PORTION #######################
###### THE NEXT FUNCTIONS ARE OPTIONAL TO BE CALLED BY CALIBRATION ###########
## If this project works, there will be a list of useful calibration functions
## to provide as you want.
##############################################################################
##############################################################################
"""
def supplementtheta(fitinfo, size, theta, cal, args):
    r"""
    Finds supplement theta given the dictionary fitinfo.
    This [emulationsupplementthetadocstring] is automatically filled by docinfo.py when running 
    updatedocs.py

    Parameters
    ----------
    fitinfo : dict
        An arbitary dictionary where you placed all your important fitting information from the 
        fit function above.
    size : integer
        The number of thetas the user wants.
    theta : array
        An array of theta values where you want to predict.  This will not always be provided.
    cal : instance of emulator class
        An emulator class instance as defined in calibration.  This will not always be provided.
    args : dict
        A dictionary containing options passed to you.
        
    Returns
    ----------
    Note that we should have theta.shape[0] * x.shape[0] < size
    theta : array
        An array of theta values that should be sampled should sample.
    info : array
        An an optional info dictionary that can pass back to the user.
    """
    if theta is None: 
        raise ValueError('this method is designed to take in the theta values.')
    
    infos = copy.copy(fitinfo['emulist'])
    thetaold = copy.copy(fitinfo['theta'])
    norig = thetaold.shape[0]
    varpca = copy.copy(fitinfo['pcstdvar'])
    
    thetaposs = copy.copy(theta)
    if thetaposs.shape[0] > 10 * size:
        thetac = np.random.choice(thetaposs.shape[0], 10 * size, replace=False)
        thetaposs = thetaposs[thetac,:]
    
    rsave = np.array(np.ones(len(infos)), dtype=object)
    rposssave = np.array(np.ones(len(infos)), dtype=object)
    rnewsave = np.array(np.ones(len(infos)), dtype=object)
    R = np.array(np.ones(len(infos)), dtype=object)
    crit = np.zeros(thetaposs.shape[0])
    weightma = np.mean(fitinfo['pct'] ** 2,0)
    for k in range(0, len(infos)):
        if infos[k]['hypind'] == k:
            rsave[k] = (1-infos[k]['nug']) * __covmat(theta, thetaold, infos[k]['hypcov'])
            rposssave[k] = (1-infos[k]['nug']) * __covmat(thetaposs, thetaold, infos[k]['hypcov'])
            rnewsave[k] = (1-infos[k]['nug']) * __covmat(thetaposs, theta, infos[k]['hypcov'])
            R[k] = __covmat(thetaold, thetaold, infos[k]['hypcov'])
            R[k] = (1-infos[k]['nug']) * R[k] + np.eye(R[k].shape[0]) * infos[k]['nug']
    critsave = np.zeros(thetaposs.shape[0])
    for j in range(0,4*size):
        crit = 0*crit
        if thetaposs.shape[0] < 1.5:
            thetaold = np.vstack((thetaold, thetaposs))
            break
        for k in range(0, len(infos)):
            Rh = R[infos[k]['hypind']] + np.diag(varpca[:, k])
            p = rnewsave[infos[k]['hypind']]
            q = rposssave[infos[k]['hypind']] @ np.linalg.solve(Rh,rsave[infos[k]['hypind']].T)
            r = rposssave[infos[k]['hypind']].T * np.linalg.solve(Rh,rposssave[infos[k]['hypind']].T)
            crit += weightma[k] * np.mean((p-q) ** 2,1) / (1 - np.sum(r,0))
        jstar = np.argmax(crit)
        critsave[j] = crit[jstar]
        thetaold = np.vstack((thetaold, thetaposs[jstar]))
        thetaposs = np.delete(thetaposs, jstar, 0)
        for k in range(0, len(infos)):
            if infos[k]['hypind'] == k:
                R[k] = np.vstack((R[k],rposssave[k][jstar,:]))
                R[k] = np.vstack((R[k].T,np.append(rposssave[k][jstar,:],1))).T
                newr = (1-infos[k]['nug']) * __covmat(thetaposs, thetaold[-1,:], infos[k]['hypcov'])
                rposssave[k] = np.delete(rposssave[k], jstar, 0)
                rposssave[k] = np.hstack((rposssave[k], newr))
                rsave[k] = np.hstack((rsave[k], rnewsave[k][jstar,:][:,None]))
                rnewsave[k] = np.delete(rnewsave[k], jstar, 0)
        crit = np.delete(crit, jstar)
        varpca = np.vstack((varpca,0*varpca[0, :]))
    
    for k in range(0, len(infos)):
        if infos[k]['hypind'] == k:
            rsave[k] = (1-infos[k]['nug']) * __covmat(theta, thetaold, infos[k]['hypcov'])
            R[k] = __covmat(thetaold, thetaold, infos[k]['hypcov'])
            R[k] = (1-infos[k]['nug']) * R[k] + np.eye(R[k].shape[0]) * infos[k]['nug']
    
    # eliminant some values and see what happens
    epsilon = fitinfo['epsilon']
    pcw = copy.copy(fitinfo['pcw'])
    pct = copy.copy(fitinfo['pct'] / pcw)
    
    G = np.array(np.ones(len(infos)), dtype=object)
    D = np.array(np.ones(len(infos)), dtype=object)
    for k in range(0, len(infos)):
        kind =infos[k]['hypind'] 
        R00 = R[kind][:norig, :][:, :norig]
        R0N = R[kind][:norig, :][:, norig:]
        RNN = R[kind][norig:, :][:, norig:]
        RA0 = rsave[kind][:, :norig]
        RAN = rsave[kind][:, norig:]
        Q = RA0 @ np.linalg.solve(R00, R0N) - RAN
        V = RNN - R0N.T @ np.linalg.solve(R00, R0N)
        D[k], H  = np.linalg.eigh(V)
        D[k] = np.minimum(D[k],0)
        G[k] = np.sum((Q @ H) ** 2,0)
    xorder = []
    wherenotmof = list(range(0,pct.shape[0]))
    for i  in range(0,pct.shape[0]):
        H = pct[wherenotmof,:].T @ pct[wherenotmof,:]
        pcstdvar = np.inf
        kstar = None
        numx = fitinfo['pct'].shape[0]
        
        pcstdvarprop = np.zeros((len(wherenotmof), pcw.shape[0]))
        totcrit = np.zeros(len(wherenotmof))
        for j in range(0,len(wherenotmof)):
            Hu = H - np.outer(pct[j,:], pct[j,:])
            Qmat = np.diag(epsilon / pcw ** 2) + Hu
            term3 = np.diag(Hu) - np.sum(Hu * spla.solve(Qmat,Hu),0)
            pcstdvarprop = np.abs(1 - (pcw ** 2 /epsilon +1) * term3)
            for k in range(0, len(infos)):
                totcrit[j] += weightma[k] * np.mean(G[k] / (D[k] + pcstdvarprop[k]))
        xindstar = wherenotmof[np.argmax(totcrit)]
        xorder.append(xindstar)
        wherenotmof.remove(xindstar)
    
    info ={}
    info['crit'] = critsave
    info['orderedx'] = fitinfo['x'][xorder,:]
    return thetaold[norig:(norig+size),:], info

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
    epsilon = 10 ** (-4)
    if (offset is not None) and (scale is not None):
        if offset.shape[0] == f.shape[1] and scale.shape[0] == f.shape[1]:
            if np.any(np.nanmean(np.abs(f-offset)/scale,1) > 4):
                offset = None
                scale = None
        else:
            offset = None
            scale = None
    if offset is None or scale is None:
        offset = np.zeros(f.shape[1])
        scale = np.zeros(f.shape[1])
        for k in range(0, f.shape[1]):
            offset[k] = np.nanmean(f[:, k])
            scale[k] = np.nanstd(f[:, k])
        scale = 0.99* scale + 0.01*np.mean(scale)
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
                fs[np.where(mof[:, k])[0], k] = a
        for iters in range(0,4):
            U, S, _ = np.linalg.svd(fs.T, full_matrices=False)
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
                    H @ (spla.solve(Amat, J,assume_a='pos')))
    fitinfo['epsilon'] = epsilon
    # Assigning new values to the dictionary
    fitinfo['offset'] = offset
    fitinfo['scale'] = scale
    fitinfo['fs'] = fs
    return


def __PCs(fitinfo, pct=None, pcw=None, extravar=None):
    "Creates BLANK."
    # Extracting from input dictionary
    f = fitinfo['f']
    fs = fitinfo['fs']
    mof = fitinfo['mof']
    mofrows = fitinfo['mofrows']
    theta = fitinfo['theta']
    epsilon = fitinfo['epsilon']
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
                PCAskip = True
        else:
            pct = None
            pcw = None
            extravar = None
    if pct is None or pcw is None or extravar is None:
        U, S, _ = np.linalg.svd(fs.T, full_matrices=False)
        Sp = S ** 2 - epsilon
        pct = U[:, Sp > 0]
        pcw = np.sqrt(Sp[Sp > 0])
        extravar = np.mean((fs - (fs @ pct) @ pct.T) ** 2, 0)
    pc = fs @ pct
    pcstdvar = np.zeros((f.shape[0],pct.shape[1]))
    if mof is not None:
        for j in range(0,mofrows.shape[0]):
            rv = mofrows[j]
            wherenotmof = np.where(mof[rv,:] < 0.5)[0]
            H = pct[wherenotmof,:].T @ pct[wherenotmof,:]
            Hp = np.diag(pcw ** 2 + epsilon) @ H
            Amat =  np.diag(epsilon / (pcw ** 2)) + H
            J = pct[wherenotmof,:].T @ fs[rv,wherenotmof]
            pc[rv,:] = (pcw ** 2 /epsilon + 1) * (J - H @ np.linalg.solve(Amat, J))
            Qmat = np.diag(epsilon / pcw ** 2) + H
            term3 = np.diag(H) - np.sum(H * spla.solve(Qmat,H,assume_a='pos'),0)
            pcstdvar[rv, :] = np.abs(1 - (pcw ** 2 /epsilon +1) * term3)
    fitinfo['pcw'] = pcw
    fitinfo['pct'] = pct * pcw
    fitinfo['extravar'] = extravar
    fitinfo['pc'] = pc / pcw
    fitinfo['pcstdvar'] = pcstdvar
    fitinfo['PCAskip'] = PCAskip
    return

def __fitGP1d(theta, g, gvar=None, hypstarts=None, hypinds=None,prevsubmodel=None):
    """Return a fitted model from the emulator model using smart method."""
    if prevsubmodel is None:
        subinfo = {}
        subinfo['hypregmean'] = np.append(-0.5 + np.log(theta.shape[1]) + np.log(np.std(theta, 0)), (0, -10))
        subinfo['hypregLB'] = np.append(-2 + np.log(np.std(theta, 0)), (-14, -30))
        subinfo['hypregUB'] = np.append(3 + np.log(np.std(theta, 0)), (2, -1))
        subinfo['hypregstd'] = (subinfo['hypregUB'] - subinfo['hypregLB']) / 2
        subinfo['hypregstd'][-2] = 4
        subinfo['hypregstd'][-1] = 4
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
        L0 = __negloglik(subinfo['hyp'], subinfo)
        dL0 = __negloglikgrad(subinfo['hyp'], subinfo)
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
            if gvar is not None:
                subinfo['R'] += np.diag(gvar)
            W, V = np.linalg.eigh(subinfo['R'])
            Vh = V / np.sqrt(np.abs(W))
            fcenter = Vh.T @ g
            subinfo['Vh'] = Vh
            n = subinfo['R'].shape[0]
            subinfo['sig2'] = (np.mean(fcenter ** 2)*n + 10)/(n+1)
            subinfo['Rinv'] = V @ np.diag(1/W) @ V.T
        else:
            subinfo['hyp'] = opval.x[:]
            subinfo['hypind'] = -1
            subinfo['hypcov'] = subinfo['hyp'][:-1]
            subinfo['nug'] = np.exp(subinfo['hyp'][-1])/(1+np.exp(subinfo['hyp'][-1]))
            R =  __covmat(theta, theta, subinfo['hypcov'])
            subinfo['R'] =  (1-subinfo['nug'])*R + subinfo['nug'] * np.eye(R.shape[0])
            if gvar is not None:
                subinfo['R'] += np.diag(gvar)
            n = subinfo['R'].shape[0]
            W, V = np.linalg.eigh(subinfo['R'])
            Vh = V / np.sqrt(np.abs(W))
            fcenter = Vh.T @ g
            subinfo['sig2'] = (np.mean(fcenter ** 2)*n + 10)/(n+1)
            subinfo['Rinv'] = Vh  @ Vh.T
            subinfo['Vh'] = Vh
        subinfo['pw'] = subinfo['Rinv'] @ g
    else:
        subinfo = prevsubmodel
        R =  __covmat(theta, theta, subinfo['hypcov'])
        subinfo['R'] =  (1-subinfo['nug'])*R + subinfo['nug'] * np.eye(R.shape[0])
        if gvar is not None:
            subinfo['R'] += np.diag(gvar)
        n = subinfo['R'].shape[0]
        W, V = np.linalg.eigh(subinfo['R'])
        Vh = V / np.sqrt(np.abs(W))
        fcenter = Vh.T @ g
        subinfo['sig2'] = (np.mean(fcenter ** 2)*n + 10)/(n+1)
        subinfo['Rinv'] = V @ np.diag(1/W) @ V.T
        subinfo['Vh'] = Vh
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
    sig2hat = (n* np.mean(fcenter ** 2) + 10) / (n + 1)
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
    sig2hat = (n* np.mean(fcenter ** 2) + 10) / (n + 1)
    dnegloglik = np.zeros(dR.shape[2])
    Rinv = Vh @ Vh.T
    for k in range(0, dR.shape[2]):
        dsig2hat =  - np.sum((Vh @ np.multiply.outer(fcenter, fcenter) @ Vh.T) * dR[:, :, k]) / (n+1)
        dnegloglik[k] += 0.5* n * dsig2hat / sig2hat
        dnegloglik[k] += 0.5*np.sum(Rinv * dR[:, :, k])
    dnegloglik += (hyp-info['hypregmean'])/(info['hypregstd'] ** 2)
    return dnegloglik
