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
thetaeval = np.loadtxt('param_values.csv',delimiter=',')
obs = np.loadtxt('real_observations.csv',delimiter=',')
feval = np.loadtxt('func_eval.csv',delimiter=',')
inputeval = np.loadtxt('observation_description.csv',delimiter=',',dtype='object')

#get rid of early datapoints
keepinds = np.squeeze(np.where(inputeval[:,0].astype('float') > 30))
obs = obs[keepinds]
inputeval = inputeval[keepinds,:]
feval = feval[:,keepinds]


#filter some training data
ftest = feval[600:,:]
thetatest = thetaeval[600:,:]
feval = feval[:600,:]
thetaeval = thetaeval[:600,:]

#transform
obs = np.sqrt(obs)
feval = np.sqrt(feval)

#scale early on
pcaoffset = np.mean(feval,0)
pcascale = np.maximum(0.1*obs,np.std(feval,0))
fstand = (feval/pcascale - pcaoffset/pcascale)

#do PCA
U, S, _ = np.linalg.svd(fstand.T, full_matrices = False)

#keeping 30 PCs (just because)
pcamat =  U[:,0:120]
pcaweight =  S[0:120]
pcaobs = fstand @ pcamat
print('Percentage of variance kept: ' + str(np.sum(S[0:120] ** 2)/np.sum(S ** 2)))

# print(pcaobs.T @ pcaobs)  #check that it is diagonal

#choose correlation function

plumleeemu = emubuild(thetaeval, pcaobs)
predmean, predvar = emupred(thetatest, plumleeemu, returnvar = True)

varobs =  predvar[0,:] @ pcamat.T

#transform back
fpredstand = (predmean @ pcamat.T) * pcascale + pcaoffset
fpred = (fpredstand) ** 2 #this a median prediction

varpredstand = (predvar @ (pcamat **2).T) * (pcascale **2)
varpred = 2*(varpredstand ** 2) + 4*varpredstand * fpred #FIX LATER
#https://math.stackexchange.com/questions/2809008/what-is-the-variance-of-the-square-of-a-normal-dataset-with-non-zero-mean

whereicu = np.where(inputeval[:,1] == 'icu_admission')
plt.plot(inputeval[whereicu,0].astype('float'), fpred[40,whereicu], 'o', color='black')
plt.plot(inputeval[whereicu,0].astype('float'), ftest[40,whereicu], 'o', color='black')

whereadm = np.where(inputeval[:,1] == 'daily_admission')
plt.plot(inputeval[whereadm,0].astype('float'), fpred[40,whereadm], 'o', color='black')
plt.plot(inputeval[whereicu,0].astype('float'), ftest[40,whereadm], 'o', color='black')


wherehosp = np.where(inputeval[:,1] == 'total_hosp')
plt.plot(inputeval[wherehosp,0].astype('float'), fpred[40,wherehosp], 'o', color='black')
plt.plot(inputeval[wherehosp,0].astype('float'), ftest[40,wherehosp], 'o', color='black')

plt.figure()
plt.plot(ftest[:,280], fpred[:,280], 'o', color='black')

corrv = np.mean((fpred-np.mean(fpred,0))*(ftest-np.mean(ftest,0))/(np.std(ftest,0)*np.std(fpred,0)),0)
plt.plot(inputeval[:,0], corrv, 'o', color='black')


#NOT EMULATOR STUFF
obsstd = pcascale

covapprox =  pcamat @ pcamat.T