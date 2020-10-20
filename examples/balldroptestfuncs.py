# -*- coding: utf-8 -*-
"""Header here."""
import numpy as np


def balldropmodel_linear(theta, x):
    """Place description here."""
    f = np.zeros((theta.shape[0], x.shape[0]))
    for k in range(0, theta.shape[0]):
        t = x[:, 0]
        h0 = x[:, 1] + theta[k, 0]
        vter = theta[k, 1]
        # g = theta[k, 2]
        f[k, :] = h0 - vter * t
    return f


def balldropmodel_grav(theta, x):
    """Place description here."""
    f = np.zeros((theta.shape[0], x.shape[0]))
    for k in range(0, theta.shape[0]):
        t = x[:, 0]
        h0 = x[:, 1]
        # vter = theta[k, 1]
        g = theta[k, 2]
        f[k, :] = h0 - (g / 2) * (t ** 2)
    return f

def balldroptrue(x):
    """Place description here."""
    def logcosh(x):
        # preventing crashing
        s = np.sign(x) * x
        p = np.exp(-2 * s)
        return s + np.log1p(p) - np.log(2)
    t = x[:, 0]
    h0 = x[:, 1]
    vter = 16.5
    g = 9.81
    y = h0 - (vter ** 2) / g * logcosh(g * t / vter)
    return y

# def balldropmodel_drag(theta, x):
#     """Place description here."""
#     f = np.zeros((theta.shape[0], x.shape[0]))
#     def logcosh(x):
#         # preventing crashing
#         s = np.sign(x) * x
#         p = np.exp(-2 * s)
#         return s + np.log1p(p) - np.log(2)
#     for k in range(0, theta.shape[0]):
#         t = x[:, 0]
#         h0 = x[:, 1]
#         vter = theta[k, 1]
#         g = theta[k, 0]
#         f[k, :] = h0 - (vter ** 2) / g * logcosh(g * t / vter)
#     return f

# def balldroptruealt(x):
#     """Place description here."""
#     vter = (((XXX)))
#     g = (((XXX)))
#     tau = vter/g
#     t = x[:, 0]
#     h0 = x[:, 1]
#     y = h0 - vter*(t - tau*(1-np.exp(-t/tau)))
#     return y
