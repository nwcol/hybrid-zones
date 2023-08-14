import numpy as np

import scipy

from scipy import stats

from scipy.special import erfc

import scipy.optimize as opt

import math


def compute_pd(x, s):
    """Compute a vector of probability density values for a normal distribution
    with standard deviation s, mean 0.
    """
    return 1 / (s * np.sqrt(2 * np.pi)) * np.exp(-0.5 * np.square(x / s))


def logistic_fxn(x, k, x_0):
    return 1 / (1.0 + np.exp(-k * (x - x_0)))


def optimize_logistic(data, x):
    popt, pcov = opt.curve_fit(logistic_fxn, x, data)
    return popt, pcov


def get_time_string():
    return str(time.strftime("%H:%M:%S", time.localtime()))


