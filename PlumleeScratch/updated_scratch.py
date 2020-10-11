# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 10:56:34 2020

@author: Plumlee
"""

import os
import numpy as np
import scipy.optimize as spo
import scipy.stats as sps
from plumleemodel import *
import matplotlib.pyplot as plt

os.chdir(os.path.abspath(os.path.join(os.path.dirname( __file__ ))))
obs = np.loadtxt('real_observations.csv',delimiter=',')
inputeval = np.loadtxt('observation_description.csv',delimiter=',',dtype='object')

thetaeval = np.loadtxt('param_values_LowAcc.csv',delimiter=',')
feval = np.loadtxt('func_eval_LowAcc.csv',delimiter=',')

#get rid of early datapoints
keepinds = np.squeeze(np.where(inputeval[:,0].astype('float') > 30))
obs = obs[keepinds]
inputeval = inputeval[keepinds,:]
feval = feval[:,keepinds]

#NOT EMULATOR STUFF
stdobs = np.maximum(0.2*obs,10)

#filter some training data
ftest = feval[600:,:]
thetatest = thetaeval[600:,:]
feval = feval[:600,:]
thetaeval = thetaeval[:600,:]

#do not transform
obs = obs
feval = feval

#scale early on, this type of scalling is not needed always
pcaoffset = obs
pcascale = stdobs
fstand = (feval/pcascale - pcaoffset/pcascale)
obsstand = (obs/pcascale - pcaoffset/pcascale)

#do PCA
U, S, _ = np.linalg.svd(fstand.T, full_matrices = False)

#keeping 100 PCs (just because)
pcamat =  U[:,0:100]
pcaweight =  S[0:100]
pcamodel = fstand @ pcamat
print('Percentage of variance kept: ' + str(np.sum(S[0:100] ** 2)/np.sum(S ** 2)))

# build up emulator
plumleeemu = emubuild(thetaeval, pcamodel)

#transform to pca to make easier.  
#Meaning if Y \sim N(f(\theta),np.diag(stdobs))
#Meaning if A Y \sim N( A f(\theta),A np.diag(stdobs) A.T)

pcaobs = obsstand @ pcamat
varpcaobs = pcamat.T @ np.diag((stdobs **2) / (pcascale **2)) @ pcamat

logpostsave = np.zeros(400)
for k in range(0,400):
    predmean, predvar = emupred(thetatest[k:(k+1),:], plumleeemu, returnvar = True)
    CovMat = np.diag(np.squeeze(predvar)) + varpcaobs
    
    
    CovMatEigS, CovMatEigW = np.linalg.eigh(CovMat)
    resid = predmean - pcaobs
    CovMatEigInv = CovMatEigW @ np.diag(1/CovMatEigS) @ CovMatEigW.T
    logpostsave[k] = - 1/2 * resid @ CovMatEigInv @ resid.T - 1/2 * np.sum(np.log(CovMatEigS)) 
    #print(logpostsave[k])

M = np.max(logpostsave)
postsave = np.exp(logpostsave-M)/ np.sum(np.exp(logpostsave-M))

plt.figure()
paraind = 3
theta_smooth = np.linspace(thetatest[:,paraind].min(), thetatest[:,paraind].max(), 200)

Wmat = np.exp(-(np.subtract.outer(theta_smooth,thetatest[:,paraind])/(np.max(theta_smooth)-np.min(theta_smooth))*30) ** 2)

postproj = Wmat @ postsave / np.sum(Wmat,1)
plt.plot(theta_smooth,postproj, 'o', color='black')
plt.ylim(0,np.max(postproj))