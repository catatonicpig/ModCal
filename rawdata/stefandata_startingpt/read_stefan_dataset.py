# -*- coding: utf-8 -*-
"""
Created on Mon May 25 14:39:54 2020

@author: moses
"""

import scipy.io as spio
import numpy as np
import copy

import os
from pathlib import Path
fpath = Path(os.path.abspath(os.path.join(os.path.dirname( __file__ ))))

def gen_stefan_startingpt_dataset(fname,seed=0):

    if seed != None:
        np.random.seed(seed)

    mat = spio.loadmat(fpath / 'starting_points_test_info.mat')
    bigmap = np.loadtxt(fpath / 'bigmapout.txt',delimiter=',',dtype=int)

    fvals = mat['Fhist']
    errvals = mat['Errorhist']
    thetavals = mat['X0mat'].T
    obsvals = mat['fvals'].T
    inputs = np.loadtxt(fpath / 'inputdata.csv', delimiter=',', dtype=object)

    textcol = inputs[:,-1]
    textcol = [ti.strip().replace(' ','') for ti in textcol]

    inputs[:,-1] = textcol

    toterr = errvals @ bigmap
    errvalssimple = toterr > 0.5

    fvals_wfail = copy.deepcopy(fvals)



    N = fvals.shape[0]
    Nind = np.random.permutation(N)
    trainN = int(0.2*N)
    # testN = N - trainN

    trainind = Nind[:trainN]
    testind = Nind[trainN:]

    train_dp = {
        'fname': ''.join(('Emu_', fname, '_Train')),
        'x': inputs,
        'functionevals': fvals_wfail[trainind,:],
        'theta': thetavals[trainind,:],
        'obs':obsvals[trainind],
        'failval': errvalssimple[trainind,:]
        }

    test_dp = {
        'fname': ''.join(('Emu_', fname, '_Test')),
        'x': inputs,
        'functionevals': fvals_wfail[testind,:],
        'theta': thetavals[testind,:],
        'obs':obsvals[testind],
        'failval': errvalssimple[testind,:]
        }

    return train_dp, test_dp