# -*- coding: utf-8 -*-
"""Header here."""
import numpy as np


def borehole_model(x, theta):
    """Given x and theta, return matrix of [row x] times [row theta] of values."""
    assert theta.ndim == 2
    assert x.ndim == 2

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
    theta0 = np.atleast_2d(np.array([0.5] * 6))
    f0 = borehole_model(x, theta0)


    return f0


def borehole_vec(x, theta):
    """Given x and theta, return vector of values."""
    (Tu, Tl, Hu, Hl, r, Kw) = np.split(theta, theta.shape[1], axis=1)
    (rw, L) = np.split(x[:, :-1], 2, axis=1)

    numer = 2 * np.pi * Tu * (Hu - Hl)
    denom1 = 2 * L * Tu / (np.log(r/rw) * rw**2 * Kw)
    denom2 = Tu / Tl

    f = (numer / (np.log(r/rw) * (1 + denom1 + denom2))).reshape(-1)

    f[x[:, -1] == 1] = f[x[:, -1].astype(bool)] ** (1.5)
    return f.T


def tstd2theta(tstd, hard=True):
    """Given standardized theta in [0, 1]^d, return non-standardized theta."""
    assert tstd.ndim == 2
    (Tus, Tls, Hus, Hls, rs, Kws) = np.split(tstd, tstd.shape[1], axis=1)

    Tu = Tus * (115600 - 63070) + 63070
    Tl = Tls * (116 - 63.1) + 63.1
    Hu = Hus * (1110 - 990) + 990
    Hl = Hls * (820 - 700) + 700
    r = rs * (50000 - 100) + 100
    if hard:
        Kw = Kws * (15000 - 1500) + 1500
    else:
        Kw = Kws * (12045 - 9855) + 9855

    theta = np.hstack((Tu, Tl, Hu, Hl, r, Kw))
    return theta


def xstd2x(xstd):
    """Given standardized x in [0, 1]^2 x {0, 1}, return non-standardized x."""
    assert xstd.ndim == 2
    (rws, Ls, labels) = np.split(xstd, xstd.shape[1], axis=1)

    rw = rws * (0.15 - 0.5) + 0.5
    L = Ls * (1680 - 1120) + 1120

    x = np.hstack((rw, L, labels))
    return x
