"""Includes main functions for default Gaussian Process grid emulation."""
import numpy as np
import scipy.optimize as spo
from line_profiler import LineProfiler
profile = LineProfiler()


def emulation_smart_builder(thetao, fevalo, mofevalo, options=1):
    """Return a Gaussian Process grid emulator model using smart method."""
    wherevals = np.where(np.sum(mofevalo, 1) < 0.99 * mofevalo.shape[1])[0]
    mofeval = mofevalo[wherevals, :]
    theta = thetao[wherevals, :]
    feval = fevalo[wherevals, :]
    fitinfo = {}
    fitinfo['offset'] = np.zeros(feval.shape[1])
    fitinfo['scale'] = np.ones(feval.shape[1])
    fitinfo['theta'] = 1*theta
    fstand = 1*feval
    for k in range(0, feval.shape[1]):
        inds = np.where(mofeval[:, k] < 0.5)[0]
        fitinfo['offset'][k] = np.mean(feval[inds, k])
        fitinfo['scale'][k] = np.std(feval[inds, k])
        fstand[np.where(mofeval[:, k] > 0.5)[0], k] = fitinfo['offset'][k]
    fstand = (fstand - fitinfo['offset']) / fitinfo['scale']
    for iters in range(0,20):
        Sig0 = fstand.T  @ fstand
        for k in range(0, feval.shape[0]):
            Sig = (Sig0 - ((fstand[k,:].reshape((1,-1))).T @ (fstand[k,:].reshape((1,-1))))) / (fstand.shape[0]-1)
            Sig = fstand.shape[0]/(fstand.shape[0] + 0.1) * Sig + 0.1/(fstand.shape[0] + 0.1) *np.diag(np.diag(Sig))
            indsr = np.where(mofeval[k, :] < 0.5)[0]
            nindsr = np.where(mofeval[k, :] > 0.5)[0]
            fstand[k, nindsr] = Sig[nindsr,:][:,indsr] @ np.linalg.solve(Sig[indsr,:][:,indsr],fstand[k, indsr])

    Vecs, Vals, _ = np.linalg.svd((fstand / np.sqrt(fstand.shape[0])).T)
    Vals = np.append(Vals, np.zeros(Vecs.shape[1] - Vals.shape[0]))
    Valssq = (fstand.shape[0]*(Vals ** 2) + 0.1) / (fstand.shape[0] + 0.1)
    numVals = 1 + np.sum(np.cumsum(Valssq) < 0.9999*np.sum(Valssq))
    fitinfo['Cs'] = Vecs * np.sqrt(Valssq)
    fitinfo['PCs'] = fitinfo['Cs'][:, :numVals]

    pcaval = np.zeros((fstand.shape[0], numVals))
    fitinfo['pcavar'] = np.zeros((fstand.shape[0], numVals))
    rhoobs = np.zeros((theta.shape[0], theta.shape[0], numVals))
    rhopred = np.zeros((theta.shape[0], numVals))
    rhomatsave = np.zeros((numVals, fitinfo['Cs'].shape[1], feval.shape[0]))
    for k in range(0, feval.shape[0]):
        indsr = np.where(mofeval[k, :] < 0.5)[0]
        rhomatsave[:, :, k] = fitinfo['Cs'][indsr, :numVals].T @ \
            np.linalg.solve(fitinfo['Cs'][indsr, :] @ fitinfo['Cs'][indsr, :].T,
                            fitinfo['Cs'][indsr, :])
        pcaval[k, :] = fitinfo['Cs'][indsr, :numVals].T @ \
            np.linalg.solve(fitinfo['Cs'][indsr, :] @ fitinfo['Cs'][indsr, :].T,
                            fstand[k, indsr])
        rhopred[k, :] = np.sum(rhomatsave[:, :, k] * rhomatsave[:, :, k], 1)
    for k in range(0, feval.shape[0]):
        for l in range(k, feval.shape[0]):
            rhoobs[k, l, :] = np.sum(rhomatsave[:, :, k] * rhomatsave[:, :, l], 1)
            rhoobs[l, k, :] = rhoobs[k, l, :]
    if options > 1.5:
        rhoobs = np.ones(rhoobs.shape)
        rhopred = np.ones(rhopred.shape)
    fhat= fstand @ np.linalg.solve(fitinfo['Cs'] @ fitinfo['Cs'].T, fitinfo['PCs'] @ fitinfo['PCs'].T)
    hypinds = np.zeros(numVals)
    emulist = [dict() for x in range(0, numVals)]
    fitinfo['extravar'] = np.mean((fstand-fhat) ** 2,0) * (fitinfo['scale'] ** 2)
    fitinfo['var0'] = np.ones(rhoobs.shape[2])
    fitinfo['rhomatsave'] = rhomatsave
    fitinfo['rhopred'] = rhopred
    fitinfo['rhoobs'] = rhoobs
    for pcanum in range(0, numVals):
        if pcanum > 0.5:
            hypwhere = np.where(hypinds == np.array(range(0, numVals)))[0]
            emulist[pcanum] = emulation_smart_fit(theta,
                                                  pcaval[:, pcanum],
                                                  np.squeeze(rhoobs[:, :, pcanum]),
                                                  np.squeeze(rhopred[:, pcanum]),
                                                  hypstarts[hypwhere,:],
                                                  hypwhere)
        else:
            emulist[pcanum] = emulation_smart_fit(theta,
                                                  pcaval[:, pcanum],
                                                  np.squeeze(rhoobs[:, :, pcanum]),
                                                  np.squeeze(rhopred[:, pcanum]))
            hypstarts = np.zeros((numVals,
                                  emulist[pcanum]['hyp'].shape[0]))
        hypstarts[pcanum, :] = emulist[pcanum]['hyp']
        if emulist[pcanum]['hypind'] < -0.5:
            emulist[pcanum]['hypind'] = pcanum
        hypinds[pcanum] = emulist[pcanum]['hypind']
    fitinfo['emulist'] = emulist
    return fitinfo


def emulation_smart_select(fitinfo, theta, thetaposs,
                           numselect=1, mofposs=None, options=None):
    """Return a prediction from the emulator model using smart method with supplemental data."""
    numVals = len(fitinfo['emulist'])


    selectind = np.zeros(numselect)
    fullind = np.array(range(0, thetaposs.shape[0]))
    rsave1 = np.array(np.ones(len(fitinfo['emulist'])), dtype=object)
    rsave2 = np.array(np.ones(len(fitinfo['emulist'])), dtype=object)
    rsave3 = np.array(np.ones(len(fitinfo['emulist'])), dtype=object)
    rsave4 = np.array(np.ones(len(fitinfo['emulist'])), dtype=object)
    mofvar = np.array(np.ones(len(fitinfo['emulist'])), dtype=object)
    rhomatsave = fitinfo['rhomatsave']
    rhopred = fitinfo['rhopred']
    rhoobs = fitinfo['rhoobs']
    thetaO = 1*fitinfo['theta']
    rhoposssave = np.zeros((numVals, fitinfo['Cs'].shape[1], thetaposs.shape[0]))
    rhoposs = np.zeros((thetaposs.shape[0], numVals))
    rhopossposs = np.zeros((thetaposs.shape[0], numVals))
    rhopossold = np.zeros((thetaposs.shape[0], thetaO.shape[0], numVals))
    for k in range(0, thetaposs.shape[0]):
        indsr = np.where(mofposs[k, :] < 0.5)[0]
        rhoposssave[:, :, k] = fitinfo['Cs'][indsr, :numVals].T @ \
            np.linalg.solve(fitinfo['Cs'][indsr, :] @ fitinfo['Cs'][indsr, :].T,
                            fitinfo['Cs'][indsr, :])
        rhoposs[k, :] = np.diag(rhoposssave[:, :, k])
        for l in range(0, thetaO.shape[0]):
            rhopossold[k , l, :] = np.sum(rhoposssave[:, :, k] *
                                          fitinfo['rhomatsave'][:, :, l], 1)
        rhopossposs[k, :] = np.sum(rhoposssave[:, :, k] *
                                          rhoposssave[:, :, k], 1)
    for k in range(0, len(fitinfo['emulist'])):
        emumodel = fitinfo['emulist'][k]
        if emumodel['hypind'] == k:
            rsave1[k] = (1 - emumodel['nug']) *\
                emulation_smart_covmat(theta, thetaO, emumodel['hypcov'])
            rsave2[k] = (1 - emumodel['nug']) *\
                emulation_smart_covmat(thetaposs, thetaO, emumodel['hypcov'])
            rsave3[k] = (1 - emumodel['nug']) *\
                emulation_smart_covmat(theta, thetaposs, emumodel['hypcov'])
            rsave4[k] = (1 - emumodel['nug']) *\
                emulation_smart_covmat(thetaO, thetaO, emumodel['hypcov'])
    w = (fitinfo['PCs'] .T * fitinfo['scale']) ** 2
    critsave = np.zeros(numselect)
    for sampval in range(0, numselect):
        crit3 = np.zeros((thetaposs.shape[0],len(fitinfo['emulist'])))
        for k in range(0, len(fitinfo['emulist'])):
            emumodel = fitinfo['emulist'][k]
            R = 1*rsave4[emumodel['hypind']] * rhoobs[:, :, k] +\
                emumodel['nug'] * np.eye(thetaO.shape[0])
            W, V = np.linalg.eigh(R)
            Rinv = V @ np.diag(1/W) @ V.T
            ralt1 = rsave1[emumodel['hypind']] * rhopred[:, k]
            ralt2 = rsave2[emumodel['hypind']] * rhopossold[:, :, k]
            ralt3 = rsave3[emumodel['hypind']] * rhoposs[:, k]
            Qmat = Rinv @ ralt2.T
            predvarm = rhopossposs[:, k] - np.sum(ralt2 * Qmat.T, 1)
            crit3[:,k] = np.mean(((ralt1 @ Qmat - ralt3) ** 2), 0) / predvarm
        criteria = np.mean((crit3 @ (fitinfo['PCs'].T ** 2)) * (fitinfo['scale'] ** 2),1)
        kstar = np.argmax(criteria)
        critsave[sampval] = criteria[kstar]
        extrarho = np.zeros((thetaposs.shape[0], numVals))
        for l in range(0, thetaposs.shape[0]):
            extrarho[l, :] = np.sum(rhoposssave[:, :, l] *
                                          rhoposssave[:, :, kstar], 1)
        extrarho2 = np.sum(rhoposssave[:, :, kstar] *
                                          rhoposssave[:, :, kstar], 1)
        rhopred = np.append(rhopred,rhoposs[kstar,:].reshape((1, -1)),0)
        rhoobs = np.append(rhoobs, rhopossold[kstar,:, :].reshape((1, -1, rhopossold.shape[2])), 0)
        rhoobs = np.append(rhoobs,
                              np.append(rhopossold[kstar, :, :], rhopossposs[kstar, :]).reshape((-1, 1, rhopossold.shape[2])),
                              1)
        rhopossold = np.append(rhopossold, extrarho.reshape((-1, 1, rhopossold.shape[2])), 1)
        rhopossold = np.delete(rhopossold, kstar, 0)
        rhoposssave = np.delete(rhoposssave, kstar, 2)
        rhoposs = np.delete(rhoposs, kstar, 0)
        rhopossposs = np.delete(rhopossposs, kstar, 0)
        thetaO = np.append(thetaO, thetaposs[kstar, :].reshape(1, -1), 0)
        thetaposs = np.delete(thetaposs, kstar, 0)
        for k in range(0, len(fitinfo['emulist'])):
            emumodel = fitinfo['emulist'][k]
            if emumodel['hypind'] == k:
                extracov = (1 - emumodel['nug']) *\
                    emulation_smart_covmat(thetaO[-1, :],
                                            thetaposs,
                                            emumodel['hypcov'])
                rsave4[k] = np.append(rsave4[k],
                                      rsave2[k][kstar, :].reshape((1, -1)), 0)
                rsave4[k] = np.append(rsave4[k].T,
                                      np.append(rsave2[k][kstar, :], (1 - emumodel['nug'])).reshape((1, -1)),
                                      0)
                rsave1[k] = np.append(rsave1[k].T,
                                      rsave3[k][:, kstar].reshape(1, -1), 0).T
                rsave2[k] = np.delete(rsave2[k], kstar, 0)
                rsave3[k] = np.delete(rsave3[k], kstar, 1)
                rsave2[k] = np.append(rsave2[k].T, extracov.reshape(1, -1), 0).T
        selectind[sampval] = fullind[kstar]
        fullind = np.delete(fullind, kstar)
    return selectind.astype('int'), critsave


@profile
def emulation_smart_loglik(fitinfo, theta, options=None):
    """Return -0.5 log(I+cov(Var)) - 0.5 predmean^T (I+cov(Var)) predmean."""
    predvec = np.zeros((theta.shape[0], len(fitinfo['emulist'])))
    predvar = np.zeros((theta.shape[0], len(fitinfo['emulist'])))
    rsave = np.array(np.ones(len(fitinfo['emulist'])), dtype=object)
    for k in range(0, len(fitinfo['emulist'])):
        if fitinfo['emulist'][k]['hypind'] == k:
            rsave[k] = (1-fitinfo['emulist'][k]['nug']) *\
                emulation_smart_covmat(theta,
                                       fitinfo['theta'],
                                       fitinfo['emulist'][k]['hypcov'])
        r = rsave[fitinfo['emulist'][k]['hypind']]
        predvec[:, k] = r @ fitinfo['emulist'][k]['pw']
        predvar[:, k] = 1 - np.sum(r.T * (fitinfo['emulist'][k]['Rinv'] @ r.T), 0)
    hAiV = (fitinfo['PCs'].T * (fitinfo['scale']/np.sqrt(1+fitinfo['extravar'])))
    normv = np.zeros(predvec.shape[0])
    detv = np.sum(np.log((1 + fitinfo['extravar']))) * np.ones(predvec.shape[0])
    for k in range(0, predvec.shape[0]):
        U, W, _ = np.linalg.svd(hAiV.T * predvar[k, :], full_matrices=False)
        predmeanstd = predvec[k, :] @ hAiV + \
            fitinfo['offset'] / np.sqrt(1 + fitinfo['extravar'])
        predmeanstd2 = (predmeanstd.T @ U) * (W / np.sqrt(1 + W ** 2))
        normv[k] = np.sum(predmeanstd ** 2) - np.sum(predmeanstd2 ** 2)
        detv[k] += np.sum(np.log(1 + W ** 2))
    loglik = -1/2*detv-1/2*normv
    return loglik


def emulation_smart_prediction(fitinfo, theta, options=None):
    """Return a prediction from the emulator model using smart method."""
    emumodel = fitinfo['emulist']
    predvec = np.zeros((theta.shape[0], len(emumodel)))
    predvar = np.zeros((theta.shape[0], len(emumodel)))
    rsave = np.array(np.ones(len(emumodel)), dtype=object)
    for k in range(0, len(emumodel)):
        if emumodel[k]['hypind'] == k:
            rsave[k] = (1-emumodel[k]['nug']) *\
                emulation_smart_covmat(theta, fitinfo['theta'], emumodel[k]['hypcov'])
        r = np.squeeze(rsave[emumodel[k]['hypind']])
        predvec[:, k] = r @ emumodel[k]['pw']
        predvar[:, k] = 1 - np.sum(r.T * (emumodel[k]['Rinv'] @ r.T), 0)

    predmean = (predvec @ fitinfo['PCs'].T)*fitinfo['scale'] + fitinfo['offset']
    predvar = 0*fitinfo['extravar'] + (predvar @ (fitinfo['PCs'] ** 2).T) *\
        (fitinfo['scale'] ** 2)
    return predmean, predvar


def emulation_smart_draws(fitinfo, theta, options=None):
    """Return a draw from the emulator model using smart method."""
    if options is None:
        numsamples = 500
    else:
        numsamples = 500 if 'numsamples' not in options else options['numsamples']
    emumodel = fitinfo['emulist']
    predvec = np.zeros((theta.shape[0], len(emumodel)))
    predvar = np.zeros((theta.shape[0], len(emumodel)))
    rsave = np.zeros((theta.shape[0], fitinfo['theta'].shape[0], len(emumodel)))
    for k in range(0, len(emumodel)):
        if emumodel[k]['hypind'] == k:
            rsave[:, :, k] = (1-emumodel[k]['nug']) *\
                emulation_smart_covmat(theta, fitinfo['theta'], emumodel[k]['hypcov'])
        r = rsave[:, :, emumodel[k]['hypind']]
        predvec[:, k] = r @ emumodel[k]['pw']
        predvar[:, k] = 1 - np.sum(r.T * (emumodel[k]['Rinv'] @ r.T), 0)
    fdraws = np.ones((theta.shape[0], fitinfo['offset'].shape[0], numsamples))
    for l2 in range(0, numsamples):
        randomval = predvec + np.sqrt(predvar) * np.random.normal(0, 1, predvar.shape)
        fdraws[:, :, l2] = (randomval @ fitinfo['PCs'].T) * fitinfo['scale'] +\
            fitinfo['offset'] +\
            np.random.normal(0, 1, fdraws.shape[:2]) * np.sqrt(fitinfo['extravar'])
    return fdraws


def emulation_smart_fit(theta, pcaval, rhoobs, rhopred, hypstarts=None, hypinds=None):
    """Return a fitted model from the emulator model using smart method."""
    subfitinfo = {}
    subfitinfo['hypregmean'] = np.append(0.5 + np.log(np.std(theta, 0)), (0, -5))
    subfitinfo['hypregLB'] = np.append(-1 + np.log(np.std(theta, 0)), (-10, -20))
    subfitinfo['hypregUB'] = np.append(3 + np.log(np.std(theta, 0)), (1, -4))
    subfitinfo['hypregstd'] = (subfitinfo['hypregUB'] - subfitinfo['hypregLB']) / 3
    subfitinfo['hypregstd'][-2] = 2
    subfitinfo['hypregstd'][-1] = 2
    subfitinfo['hyp'] = 1*subfitinfo['hypregmean']
    nhyptrain = np.min((20*theta.shape[1], theta.shape[0]))
    thetac = np.random.choice(theta.shape[0], nhyptrain, replace=False)
    subfitinfo['theta'] = theta[thetac, :]
    subfitinfo['f'] = pcaval[thetac]
    subfitinfo['rhoobs'] = rhoobs[thetac, :][:, thetac]
    hypind0 = -1


    # L0 = emulation_smart_negloglik(1*subfitinfo['hyp'], subfitinfo)
    # dL0 = emulation_smart_negloglikgrad(1*subfitinfo['hyp'], subfitinfo)
    # for k in range(0, subfitinfo['hyp'].shape[0]):
    #     hyp0p = 1*subfitinfo['hyp']
    #     hyp0p[k] += 10 ** (-4)
    #     L1 = emulation_smart_negloglik(hyp0p, subfitinfo)
    #     print((L1-L0) * (10 ** 4))
    #     print(dL0[k])


    if hypstarts is not None:
        L0 = emulation_smart_negloglik(subfitinfo['hyp'], subfitinfo)
        for k in range(0, hypstarts.shape[0]):
            L1 = emulation_smart_negloglik(hypstarts[k, :], subfitinfo)
            if L1 < L0:
                subfitinfo['hyp'] = hypstarts[k, :]
                L0 = 1* L1
                hypind0 = hypinds[k]


    opval = spo.minimize(emulation_smart_negloglik,
                         1*subfitinfo['hyp'], args=(subfitinfo), method='L-BFGS-B',
                         options={'gtol': 0.5 / (subfitinfo['hypregUB'] -
                                                  subfitinfo['hypregLB'])},
                         jac=emulation_smart_negloglikgrad,
                         bounds=spo.Bounds(subfitinfo['hypregLB'],
                                           subfitinfo['hypregUB']))
    if hypind0 > -0.5 and 2 * (L0-opval.fun) < \
        (subfitinfo['hyp'].shape[0] + 3 * np.sqrt(subfitinfo['hyp'].shape[0])):
        subfitinfo['hypind'] = 1*hypind0
    else:
        subfitinfo['hyp'] = opval.x[:]
        subfitinfo['hypind'] = -1
    subfitinfo['hypcov'] = subfitinfo['hyp'][:-1]
    subfitinfo['nug'] = np.exp(subfitinfo['hyp'][-1])/(1+np.exp(subfitinfo['hyp'][-1]))
    R = emulation_smart_covmat(theta, theta, subfitinfo['hypcov'])
    R = R * rhoobs
    subfitinfo['R'] =  (1-subfitinfo['nug'])*R + subfitinfo['nug'] * np.eye(R.shape[0])
    W, V = np.linalg.eigh(subfitinfo['R'])
    subfitinfo['Rinv'] = V @ np.diag(1/W) @ V.T
    subfitinfo['Rinv'] = (subfitinfo['Rinv'].T * rhopred).T
    subfitinfo['pw'] =subfitinfo['Rinv'] @ pcaval
    subfitinfo['Rinv'] =  (subfitinfo['Rinv'] * rhopred)
    return subfitinfo


def emulation_smart_covmat(x1, x2, gammav, returndir=False):
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

def emulation_smart_negloglik(hyp, fitinfo):
    """Return penalized log likelihood of single demensional GP model."""
    R0 = emulation_smart_covmat(fitinfo['theta'], fitinfo['theta'], hyp[:-1])
    nug = np.exp(hyp[-1])/(1+np.exp(hyp[-1]))
    R0 = R0 *fitinfo['rhoobs']
    R = (1-nug)* R0 + nug * np.eye(fitinfo['theta'].shape[0])
    W, V = np.linalg.eigh(R)
    Vh = V / np.sqrt(np.abs(W))
    fcenter = Vh.T @ fitinfo['f']
    negloglik = 1/2 * np.sum(np.log(np.abs(W))) +1/2 * np.sum(fcenter ** 2)
    negloglik += 0.5*np.sum(((hyp-fitinfo['hypregmean']) ** 2) /
                            (fitinfo['hypregstd'] ** 2))
    return 1*negloglik


def emulation_smart_negloglikgrad(hyp, fitinfo):
    """Return gradient of the penalized log likelihood of single demensional GP model."""
    R0, dR = emulation_smart_covmat(fitinfo['theta'], fitinfo['theta'], hyp[:-1], True)
    nug = np.exp(hyp[-1])/(1+np.exp(hyp[-1]))
    R0 = R0 *fitinfo['rhoobs']
    R = (1-nug)* R0 + nug * np.eye(fitinfo['theta'].shape[0])
    for k in range(0, dR.shape[2]):
        dRn = (1-nug) * dR[:,:,k]
        dRn = dRn *fitinfo['rhoobs']
        dR[:,:,k] = dRn
    dRappend = nug/((1+np.exp(hyp[-1]))) *\
        (-R0+np.eye(fitinfo['theta'].shape[0]))
    dRappend = dRappend
    dR = np.append(dR, dRappend[:,:,None], axis=2)
    W, V = np.linalg.eigh(R)
    Vh = V / np.sqrt(np.abs(W))
    fcenter = Vh.T @ fitinfo['f']
    dnegloglik = np.zeros(dR.shape[2])
    Rinv = Vh @ (np.eye(Vh.shape[0]) - np.multiply.outer(fcenter, fcenter)) @ Vh.T
    for k in range(0, dR.shape[2]):
        dnegloglik[k] = 0.5*np.sum(Rinv * dR[:, :, k])
    dnegloglik += (hyp-fitinfo['hypregmean'])/(fitinfo['hypregstd'] ** 2)
    return 1*dnegloglik

