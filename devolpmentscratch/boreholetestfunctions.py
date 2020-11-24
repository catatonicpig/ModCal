# -*- coding: utf-8 -*-
"""Header here."""
import numpy as np


def borehole_model(x, theta):
    """Given x and theta, return matrix of [row x] times [row theta] of values."""

    theta = tstd2theta(theta)
    x = xstd2x(x)
    p = x.shape[0]
    n = theta.shape[0]

    theta_stacked = np.repeat(theta, repeats=p, axis=0)
    x_stacked = np.tile(x.astype(float), (n, 1))

    f = borehole_vec(x_stacked, theta_stacked).reshape((n, p))
    return f.T


def borehole_true(x):
    """Given x, return matrix of [row x] times 1 of values."""
    # assume true theta is [0.5]^d
    theta0 = np.atleast_2d(np.array([0.5] * 4))
    f0 = borehole_model(x, theta0)


    return f0


def borehole_vec(x, theta):
    """Given x and theta, return vector of values."""
    (Hu, Ld_Kw, Treff, powparam) = np.split(theta, theta.shape[1], axis=1)
    (rw,  Hl) = np.split(x[:, :-1], 2, axis=1)
    numer = 2 * np.pi *  (Hu - Hl)
    denom1 = 2 * Ld_Kw / rw ** 2
    denom2 = Treff
    f = np.log(((numer / ((denom1 + denom2))) * np.exp(powparam * rw)).reshape(-1))
    return f


def tstd2theta(tstd, hard=True):
    """Given standardized theta in [0, 1]^d, return non-standardized theta."""
    if tstd.ndim < 1.5:
        tstd = tstd[:,None].T
    (Treffs, Hus, LdKw, powparams) = np.split(tstd, tstd.shape[1], axis=1)
    
    Treff = np.exp((np.log(0.5)-np.log(0.05)) * Treffs + np.log(0.05))
    Hu = Hus * (1110 - 990) + 990
    if hard:
        Ld_Kw = np.exp(LdKw* (np.log(1680/1500) - np.log(1120/15000)) + np.log(1120/15000))
    else:
        Ld_Kw = np.exp(LdKw*(np.log(1680/9855)- np.log(1120/12045)) + np.log(1120/12045))
    
    powparam = powparams * (0.5 - (- 0.5)) + (-0.5)
    
    theta = np.hstack((Hu, Ld_Kw, Treff, powparam))
    return theta


def xstd2x(xstd):
    """Given standardized x in [0, 1]^2 x {0, 1}, return non-standardized x."""
    assert xstd.ndim == 2
    (rws, Hls, labels) = np.split(xstd, xstd.shape[1], axis=1)

    rw = rws * (np.log(0.5) - np.log(0.05)) + np.log(0.05)
    rw = np.exp(rw)
    Hl = Hls * (820 - 400) + 400

    x = np.hstack((rw, Hl, labels))
    return x
