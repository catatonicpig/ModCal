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
        
    fitinfo['epsilon'] = args['epsilon'] if 'epsilon' in args.keys() else 0.1 
    hyp1 = args['hypregmean'] if 'hypregmean' in args.keys() else -10
    hyp2 = args['hypregLB'] if 'hypregLB' in args.keys() else -20

    #Storing these values for future reference
    fitinfo['x'] = x
    fitinfo['theta'] = theta
    fitinfo['f'] = f
    #The double underline should be used to represent my local functions
    skipstnd = False
    skipPC = True
    __standardizef(fitinfo)
    __PCs(fitinfo)
    numpcs = fitinfo['pc'].shape[1]

    print(fitinfo['method'], 'considering ', numpcs, 'PCs')
    
    if 'emulist' in fitinfo.keys():
        hypstarts = np.zeros((numpcs, fitinfo['emulist'][0]['hyp'].shape[0]))
        hypinds = -1*np.ones(numpcs)
        for pcanum in range(0, min(numpcs, len(fitinfo['emulist']))):
            hypstarts[pcanum, :] = fitinfo['emulist'][pcanum]['hyp']
            hypinds[pcanum] = fitinfo['emulist'][pcanum]['hypind']
        emulist = [dict() for x in range(0, numpcs)]
    else:
        hypinds = -1 * np.ones(numpcs)
        emulist = [dict() for x in range(0, numpcs)]
        
    for iters in range(0,3):
        for pcanum in range(0, numpcs):
            if np.sum(hypinds == np.array(range(0, numpcs))) > 0.5:
                hypwhere = np.where(hypinds == np.array(range(0, numpcs)))[0]
                emulist[pcanum] = __fitGP1d(theta = theta, 
                                            g = fitinfo['pc'][:, pcanum],
                                            hyp1 = hyp1,
                                            hyp2 = hyp2, 
                                            gvar = fitinfo['pcstdvar'][:, pcanum], 
                                            hypstarts = hypstarts[hypwhere,:],
                                            hypinds = hypwhere)
            else:
                emulist[pcanum] = __fitGP1d(theta = theta, 
                                            g = fitinfo['pc'][:, pcanum],
                                            hyp1 = hyp1,
                                            hyp2 = hyp2, 
                                            gvar = fitinfo['pcstdvar'][:, pcanum])
                hypstarts = np.zeros((numpcs,
                                      emulist[pcanum]['hyp'].shape[0]))
            emulist[pcanum]['hypind'] = min(pcanum, emulist[pcanum]['hypind'])
            hypstarts[pcanum, :] = emulist[pcanum]['hyp']
            if emulist[pcanum]['hypind'] < -0.5:
                emulist[pcanum]['hypind'] = 1*pcanum
            hypinds[pcanum] = 1*emulist[pcanum]['hypind']
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
def predict(predinfo, fitinfo, x, theta, args={}):
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
    return_grad = False
    if args is not None and 'return_grad' in args.keys() and args['return_grad'] is True:
        return_grad = True
    infos = fitinfo['emulist']
    predvecs = np.zeros((theta.shape[0], len(infos)))
    predvars = np.zeros((theta.shape[0], len(infos)))
    if return_grad:
        predvecs_gradtheta = np.zeros((theta.shape[0], len(infos), theta.shape[1]))
        predvars_gradtheta = np.zeros((theta.shape[0], len(infos), theta.shape[1]))
        drsave = np.array(np.ones(len(infos)), dtype=object)
    if predvecs.ndim < 1.5:
        predvecs = predvecs.reshape((1,-1))
        predvars = predvars.reshape((1,-1))
    try:
        if x is None or np.all(np.equal(x,fitinfo['x'])) or np.allclose(x, fitinfo['x']):
            xind = np.arange(0,x.shape[0])
            xnewind = np.arange(0,x.shape[0])
        else:
            raise
    except:
        matchingmatrix = np.ones((x.shape[0], fitinfo['x'].shape[0]))
        for k in range(0,x[0].shape[0]):
            try:
                matchingmatrix *= np.isclose(x[:,k][:,None], fitinfo['x'][:,k]) #
            except:
                matchingmatrix *= np.equal(x[:,k][:,None],fitinfo['x'][:,k])
        xind = np.argwhere(matchingmatrix > 0.5)[:,1]
        xnewind = np.argwhere(matchingmatrix > 0.5)[:,0]
    rsave = np.array(np.ones(len(infos)), dtype=object)
    for k in range(0, len(infos)):
        if infos[k]['hypind'] == k:
            if return_grad:
                rsave[k], drsave[k] = __covmat(theta, fitinfo['theta'], infos[k]['hypcov'],
                                               return_gradx1=True)
            else:
                rsave[k] = __covmat(theta, fitinfo['theta'], infos[k]['hypcov'])
        r = (1-infos[k]['nug']) * np.squeeze(rsave[infos[k]['hypind']])
        if return_grad:
            dr = (1-infos[k]['nug']) * np.squeeze(drsave[infos[k]['hypind']])
        try:
            rVh = r @ infos[k]['Vh']
        except:
            for l in range(0, len(infos)):
                print((l,infos[l]['hypind']))
            raise ValueError('Something went wrong')
        if rVh.ndim < 1.5:
            rVh = rVh.reshape((1,-1))
        predvecs[:, k] = r @ infos[k]['pw']
        if return_grad:
            if dr.ndim == 2:
                drVh = dr.T @ infos[k]['Vh']
                predvecs_gradtheta[:, k, :] = dr.T @ infos[k]['pw']
                predvars_gradtheta[:, k, :] = -infos[k]['sig2'] *2*np.sum(rVh * drVh, 1)
            else:
                drVh = np.squeeze(dr.transpose(0,2,1) @ infos[k]['Vh'])
                predvecs_gradtheta[:, k,:] = np.squeeze(dr.transpose(0,2,1) @ infos[k]['pw'])
                predvars_gradtheta[:, k, :] = -infos[k]['sig2'] * 2 *np.sum(rVh * drVh.transpose(1,0,2),2).T
        predvars[:, k] = infos[k]['sig2'] * np.abs(1 - np.sum(rVh ** 2, 1))
    predinfo['mean'] = np.full((x.shape[0], theta.shape[0]),np.nan)
    predinfo['var'] = np.full((x.shape[0], theta.shape[0]),np.nan)
    pctscale =  (fitinfo['pct'].T * fitinfo['scale']).T
    predinfo['mean'][xnewind,:] = ((predvecs @ pctscale[xind,:].T) +\
        fitinfo['offset'][xind]).T
    predinfo['var'][xnewind,:] = ((fitinfo['extravar'][xind] + predvars @ (pctscale[xind,:] ** 2).T)).T
    CH = (np.sqrt(predvars)[:,:,None] *
                             (pctscale[xind,:].T)[None,:,:])
    predinfo['covxhalf'] = np.full((theta.shape[0], CH.shape[1], x.shape[0]), np.nan)
    predinfo['covxhalf'][:,:,xnewind] = CH
    predinfo['covxhalf'] = predinfo['covxhalf'].transpose((2,0,1))
    if return_grad:
        predinfo['mean_gradtheta'] = np.full((x.shape[0], theta.shape[0], theta.shape[1]),np.nan)
        predinfo['mean_gradtheta'][xnewind, : ,:] = ((predvecs_gradtheta.transpose(0,2,1) @
                                                     pctscale[xind,:].T)).transpose((2,0,1))
        dsqrtpredvars = 0.5 * (predvars_gradtheta.transpose(2, 0, 1) /np.sqrt(predvars)).transpose(1, 2, 0)
        if np.allclose(xnewind,xind):
            predinfo['covxhalf_gradtheta'] = (dsqrtpredvars.transpose(2,0,1)[:,:,:,None] *
                                  (pctscale[xind,:].T)[None,:,:]).transpose(3,1,2,0)
        else:
            predinfo['covxhalf_gradtheta'] = np.full((x.shape[0], theta.shape[0], 
                                                      CH.shape[1], theta.shape[1]), np.nan)
            predinfo['covxhalf_gradtheta'][xnewind] = (dsqrtpredvars.transpose(2,0,1)[:,:,:,None] *
                                      (pctscale[xind,:].T)[None,:,:]).transpose(3,1,2,0)
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
def supplementtheta(fitinfo, size, theta, thetachoices, choicecosts, cal, args):
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
        An array of theta values where you want to predict.
    thetachoices : array
        An array of to choice between.
    choicecosts : array
        The cost of each choice given to you.
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
    pending = None
    if ('pending' in args.keys()):
        pending = args['pending'].T
        pendvar = __getnewvar(fitinfo, pending)
    if ('includepending' in args.keys()) and (args['includepending'] is True):
        includepending = True
        if ('costpending' in args.keys()):
            costpending = args['costpending'] * np.ones(fitinfo['theta'].shape[0])
        else:
            costpending = np.mean(choicecosts) * np.ones(fitinfo['theta'].shape[0])
    else:
        includepending = False
    if theta is None: 
        raise ValueError('this method is designed to take in the theta values.')
    
    infos = copy.copy(fitinfo['emulist'])
    thetaold = copy.copy(fitinfo['theta'])
    norig = thetaold.shape[0]
    varpca = copy.copy(fitinfo['pcstdvar'])
    mof = copy.copy(fitinfo['mof'])
    thetaposs = thetachoices
    
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
    
    critcount = np.zeros((crit.shape[0],len(infos)))
    if pending is None:
        varpcause = 1*varpca
    else:
        varpcause = 1*pendvar
    
    thetachoicesave = np.zeros((size, fitinfo['theta'].shape[1]))
    for j in range(0,size):
        critcount = np.zeros((crit.shape[0],len(infos)))
        if thetaposs.shape[0] < 1.5:
            thetaold = np.vstack((thetaold, thetaposs))
            break
        for k in range(0, len(infos)):
            if infos[k]['hypind'] == k: #this is to speed things up a bit...
                Rh = R[infos[k]['hypind']] + np.diag(varpcause[:, k])
                p = rnewsave[infos[k]['hypind']]
                term1 = np.linalg.solve(Rh,rposssave[infos[k]['hypind']].T)
                q = rsave[infos[k]['hypind']] @ term1
                r = rposssave[infos[k]['hypind']].T * term1
                critcount[:,k] = weightma[k] * np.mean((p.T-q) ** 2,0) / np.abs(1 - np.sum(r,0))
            else:
                critcount[:,k] = weightma[k] / weightma[infos[k]['hypind']] *\
                   critcount[:, infos[k]['hypind']]
        crit = np.sum(critcount,1)
        jstar = np.argmax(crit / choicecosts)
        critsave[j] = crit[jstar] / choicecosts[jstar]
        thetaold = np.vstack((thetaold, thetaposs[jstar]))
        thetachoicesave[j] = thetaposs[jstar]
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
        critcount = np.delete(critcount, jstar, 0)
        choicecosts = np.delete(choicecosts, jstar)
        varpcause = np.vstack((varpcause,0*varpca[0, :]))
        varpca = np.vstack((varpca,0*varpca[0, :]))
        for k in range(0, len(infos)):
            if infos[k]['hypind'] == k:
                rsave[k] = (1-infos[k]['nug']) * __covmat(theta, thetaold, infos[k]['hypcov'])
                R[k] = __covmat(thetaold, thetaold, infos[k]['hypcov'])
                R[k] = (1-infos[k]['nug']) * R[k] + np.eye(R[k].shape[0]) * infos[k]['nug']
    
    # eliminant some values and see what happens
    info ={}
    info['crit'] = critsave
    if includepending:
        critpend = np.zeros((fitinfo['theta'].shape[0],len(infos)))
        for k in range(0, len(infos)):
            if infos[k]['hypind'] == k: #this is to speed things up a bit...
                Rh = R[infos[k]['hypind']] + np.diag(varpca[:, k])
                term1 = np.linalg.solve(Rh,rsave[infos[k]['hypind']].T)
                delta = (pendvar[:,k] - varpca[:fitinfo['theta'].shape[0], k])
                term3 = np.diag(np.linalg.inv(Rh))[:fitinfo['theta'].shape[0]]
                critpend[:,k] = -weightma[k] *  delta * np.mean((term1[:fitinfo['theta'].shape[0],:] ** 2),1) \
                    / (1 + delta * term3)
            else:
                critpend[:,k] = weightma[k] / weightma[infos[k]['hypind']] * critpend[:, infos[k]['hypind']]
        critpend = np.sum(critpend,1)
        info['obviatesugg'] = np.where(np.any(pending,1) *(np.mean(critsave[:size]) > critpend / costpending) > 0.5)[0]
    return thetachoicesave, info


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
    epsilon = fitinfo['epsilon']
        
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
            if scale[k] == 0:
                scale[k] = 0.0001
        #scale = 0.999* scale + 0.001*np.mean(scale)
    # Initializing values
    fs = np.zeros(f.shape)
    if mof is None:
        fs = (f - offset) / scale
    else:
        for k in range(0, f.shape[1]):
            fs[:,k] = (f[:,k] - offset[k]) / scale[k]
            if np.sum(mof[:,k]) > 0:
                a = np.empty((np.sum(mof[:,k]),))
                a[::2] = 2
                a[1::2] = -2
                fs[np.where(mof[:, k])[0], k] = (a - np.mean(a))
        for iters in range(0,40):
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
    #fitinfo['epsilon'] = epsilon
    # Assigning new values to the dictionary
    fitinfo['offset'] = offset
    fitinfo['scale'] = scale
    fitinfo['fs'] = fs
    return


def __PCs(fitinfo):
    "Creates BLANK."
    # Extracting from input dictionary
    f = fitinfo['f']
    fs = fitinfo['fs']
    mof = fitinfo['mof']
    mofrows = fitinfo['mofrows']
    theta = fitinfo['theta']
    epsilon = fitinfo['epsilon']
    pct = None
    pcw = None
    extravar = None
    U, S, _ = np.linalg.svd(fs.T, full_matrices=False)
    Sp = S ** 2 - epsilon
    pct = U[:, Sp > 0]
    pcw = np.sqrt(Sp[Sp > 0])
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
            pcstdvar[rv, :] = 1 - (pcw ** 2 /epsilon +1) * term3
    fitinfo['pcw'] = pcw
    fitinfo['pcto'] = 1*pct
    fitinfo['pct'] = pct * pcw / np.sqrt(pc.shape[0])
    fitinfo['pcti'] = pct  * (np.sqrt(pc.shape[0])/ pcw)
    fitinfo['pc'] = pc  * (np.sqrt(pc.shape[0]) / pcw)
    fitinfo['extravar'] = np.mean((fs - fitinfo['pc'] @ fitinfo['pct'].T) ** 2, 0) *\
        (fitinfo['scale'] ** 2)
    fitinfo['pcstdvar'] = 10*pcstdvar
    return


def __getnewvar(fitinfo, pending):
    "Creates BLANK."
    # Extracting from input dictionary
    pct = copy.copy(fitinfo['pcto'])
    pcw = copy.copy(fitinfo['pcw'])
    epsilon = fitinfo['epsilon']
    
    realfail = np.logical_and(np.logical_not(pending), fitinfo['mof'])
    
    failrows = np.where(np.any(realfail,1))[0]
    pcstdvar = np.zeros((pending.shape[0],pct.shape[1]))
    for j in range(0, failrows.shape[0]):
        rv = failrows[j]
        wherenotmof = np.where(realfail[rv,:] < 0.5)[0]
        H = pct[wherenotmof,:].T @ pct[wherenotmof,:]
        Hp = np.diag(pcw ** 2 + epsilon) @ H
        Qmat = np.diag(epsilon / pcw ** 2) + H
        term3 = np.diag(H) - np.sum(H * spla.solve(Qmat,H,assume_a='pos'),0)
        pcstdvar[rv, :] = 1 - (pcw ** 2 /epsilon +1) * term3
    return (10*pcstdvar)

def __fitGP1d(theta, g, hyp1, hyp2, gvar=None, hypstarts=None, hypinds=None, prevsubmodel=None):
    """Return a fitted model from the emulator model using smart method."""
    
    # subinfo['hypregmean'] = np.append(0 + 0.5*np.log(theta.shape[1]) + np.log(np.std(theta, 0)), (0, -25))
    # subinfo['hypregLB'] = np.append(-4 + 0.5*np.log(theta.shape[1]) + np.log(np.std(theta, 0)), (-12, -35))
    
    subinfo = {}
    subinfo['hypregmean'] = np.append(0 + 0.5*np.log(theta.shape[1]) + np.log(np.std(theta, 0)), (0, hyp1))
    subinfo['hypregLB'] = np.append(-4 + 0.5*np.log(theta.shape[1]) + np.log(np.std(theta, 0)), (-12, hyp2))
    
    subinfo['hypregUB'] = np.append(4 + 0.5*np.log(theta.shape[1]) + np.log(np.std(theta, 0)), (2, 0))
    subinfo['hypregstd'] = (subinfo['hypregUB'] - subinfo['hypregLB']) / 8
    subinfo['hypregstd'][-2] = 2
    subinfo['hypregstd'][-1] = 4
    subinfo['hyp'] = 1*subinfo['hypregmean']
    nhyptrain = np.max(np.min((20*theta.shape[1], theta.shape[0])))
    if theta.shape[0] > nhyptrain:
        thetac = np.random.choice(theta.shape[0], nhyptrain, replace=False)
    else:
        thetac = range(0,theta.shape[0])
    subinfo['theta'] = theta[thetac, :]
    subinfo['g'] = g[thetac]
    subinfo['gvar'] = gvar[thetac]
    hypind0 = -1
    L0 = __negloglik(subinfo['hyp'], subinfo)
    dL0 = __negloglikgrad(subinfo['hyp'], subinfo)
    if hypstarts is not None:
        L0 = __negloglik(subinfo['hyp'], subinfo)
        for k in range(0, hypstarts.shape[0]):
            L1 = __negloglik(hypstarts[k, :], subinfo)
            if L1 < L0:
                subinfo['hyp'] = hypstarts[k, :]
                L0 = 1* L1
                hypind0 = hypinds[k]
                
        
    if hypind0 > -0.5 and hypstarts.ndim > 1:
        dL = __negloglikgrad(subinfo['hyp'], subinfo)
        scalL = np.std(hypstarts,0) * hypstarts.shape[0] / (1+hypstarts.shape[0]) +\
            1 / (1+hypstarts.shape[0]) * subinfo['hypregstd']
        if np.sum((dL * scalL) ** 2) < 1.2 * (subinfo['hyp'].shape[0] +
                                        4*np.sqrt(subinfo['hyp'].shape[0])):
            skipop = True
        else:
            skipop = False
    else:
        skipop = False
    if (not skipop):
        
        def scaledlik(hypv):
            hyprs = subinfo['hypregmean'] + hypv * subinfo['hypregstd']
            return __negloglik(hyprs,subinfo)
        def scaledlikgrad(hypv):
            hyprs = subinfo['hypregmean'] + hypv * subinfo['hypregstd']
            return __negloglikgrad(hyprs,subinfo) * subinfo['hypregstd']
        newLB = (subinfo['hypregLB'] - subinfo['hypregmean']) / subinfo['hypregstd']
        newUB = (subinfo['hypregUB'] - subinfo['hypregmean']) / subinfo['hypregstd']
        newhyp0 = (subinfo['hyp'] - subinfo['hypregmean']) / subinfo['hypregstd']
        opval = spo.minimize(scaledlik, newhyp0,  method='L-BFGS-B',
                             options={'gtol': 0.1},
                             jac=scaledlikgrad,
                             bounds=spo.Bounds(newLB,newUB))
        hypn = subinfo['hypregmean'] + opval.x * subinfo['hypregstd']
        likdiff = (L0- __negloglik(hypn,subinfo))
    else:
        likdiff = 0
    if hypind0 > -0.5 and (2 * likdiff) < \
        1.2 * (subinfo['hyp'].shape[0] + 4 * np.sqrt(subinfo['hyp'].shape[0])):
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
        subinfo['sig2'] = (np.mean(fcenter ** 2)*n + 1)/(n+1)
        subinfo['Rinv'] = V @ np.diag(1/W) @ V.T
    else:
        subinfo['hyp'] = hypn
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
        subinfo['sig2'] = (np.mean(fcenter ** 2)*n + 1)/(n+1)
        subinfo['Rinv'] = Vh  @ Vh.T
        subinfo['Vh'] = Vh
    subinfo['pw'] = subinfo['Rinv'] @ g
    return subinfo

def __covmat(x1, x2, gammav, return_gradhyp=False, return_gradx1=False):
    """Return the covariance between x1 and x2 given parameter gammav."""
    x1 = x1.reshape(1, gammav.shape[0]-1)/np.exp(gammav[:-1]) if x1.ndim < 1.5 else \
        x1/np.exp(gammav[:-1])
    x2 = x2.reshape(1, gammav.shape[0]-1)/np.exp(gammav[:-1]) if x2.ndim < 1.5 else \
        x2/np.exp(gammav[:-1])
    V = np.zeros([x1.shape[0], x2.shape[0]])
    R = np.full((x1.shape[0], x2.shape[0]),1/(1+np.exp(gammav[-1])))
    if return_gradhyp:
        dR = np.zeros([x1.shape[0], x2.shape[0], gammav.shape[0]])
    elif return_gradx1:
        dR = np.zeros([x1.shape[0], x2.shape[0], x1.shape[1]])
    for k in range(0, gammav.shape[0]-1):
        if return_gradx1:
            S = np.subtract.outer(x1[:, k], x2[:, k])
            Sign = np.sign(S)
            S = np.abs(S)
        else:
            S = np.abs(np.subtract.outer(x1[:, k], x2[:, k]))
        R *= (1 + S)
        V -= S
        if return_gradhyp:
            dR[:, :, k] = (S ** 2) / (1 + S)
        if return_gradx1:
            dR[:, :, k] = -(S * Sign) / (1 + S) / np.exp(gammav[k])
    R *= np.exp(V)
    if return_gradhyp:
        dR *= R[:, :, None]
        dR[:, :, -1] = np.exp(gammav[-1]) / ((1+np.exp(gammav[-1]))) *\
            (1/ (1+np.exp(gammav[-1])) - R)
    elif return_gradx1:
        dR *= R[:, :, None]
    R += np.exp(gammav[-1])/(1+np.exp(gammav[-1]))
    if return_gradhyp or return_gradx1:
        return R, dR
    else:
        return R

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
    sig2hat = (n* np.mean(fcenter ** 2) + 10) / (n + 10)
    negloglik = 1/2 * np.sum(np.log(np.abs(W))) + 1/2 * n * np.log(sig2hat)
    negloglik += 0.5*np.sum((((10**(-8) + hyp-info['hypregmean'])) /
                            (info['hypregstd'])) ** 2)
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
    sig2hat = (n* np.mean(fcenter ** 2) + 10) / (n + 10)
    dnegloglik = np.zeros(dR.shape[2])
    Rinv = Vh @ Vh.T
    for k in range(0, dR.shape[2]):
        dsig2hat =  - np.sum((Vh @ np.multiply.outer(fcenter, fcenter) @ Vh.T) * dR[:, :, k]) / (n+10)
        dnegloglik[k] += 0.5* n * dsig2hat / sig2hat
        dnegloglik[k] += 0.5*np.sum(Rinv * dR[:, :, k])
    dnegloglik += (10**(-8) + hyp-info['hypregmean'])/((info['hypregstd']) ** 2)
    return dnegloglik
