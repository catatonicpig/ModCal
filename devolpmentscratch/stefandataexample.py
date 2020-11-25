# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 10:56:34 2020

@author: Plumlee
"""
import numpy as np
import scipy.stats as sps
import sys
import os
import copy
import time
from line_profiler import LineProfiler
from boreholetestfunctions import borehole_model, borehole_true
SCRIPT_DIR = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, '..')))
from base.emulation import emulator
from base.calibration import calibrator

thetaeval = np.loadtxt('stefandata/thetavals.csv',delimiter=',')
feval = np.loadtxt('stefandata/functionevals.csv',delimiter=',')
inputeval = np.loadtxt('stefandata/inputdata.csv',delimiter=',',dtype='object')
failmat = np.genfromtxt('stefandata/failval.csv', delimiter=',')

feval[failmat > 0.5] = np.nan

x = inputeval
f = feval[:100,:].T
theta = thetaeval[:100,:]
emu = emulator(x = x, theta=theta, f=f, method = 'PCGPwM')  # this builds an emulator 

thetatest = thetaeval[400:,:]
fpred = emu.predict(theta = thetatest)
ftest = feval[400:,:]
print(np.nanmean(np.abs(fpred.mean()- ftest.T)))
f = feval[100:200,:].T
theta = thetaeval[100:200,:]
emu.update(theta=theta, f=f)  # this builds an emulator 
thetatest = thetaeval[400:,:]
fpred = emu.predict(theta = thetatest)
ftest = feval[400:,:]
print(np.nanmean(np.abs(fpred.mean()- ftest.T)))