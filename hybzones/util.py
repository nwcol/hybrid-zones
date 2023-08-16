import numpy as np

import scipy.optimize as opt

import time


def get_ranges(n_ranges):
    """
    Return a 2d array of x ranges
    """
    ranges = np.zeros((n_ranges, 2))
    ranges[:, :] = np.arange(0, 1, 1 / n_ranges)[:, None]
    ranges[:, 1] += 1 / n_ranges
    return ranges


def get_bins(bin_size):
    """
    Helper function to get spatial bins
    """
    n_bins = int(1 / bin_size)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    return bin_edges, n_bins


def get_n_bins(bin_size):
    """
    Return the number of bins at a given bin size
    """
    return int(1 / bin_size)


def get_bin_mids(bin_size):
    """
    Return the centers of the spatial bins specified by bin_size
    """
    n_bins = get_n_bins(bin_size)
    h = bin_size / 2
    return np.linspace(h, 1 - h, n_bins)


def setup_space_plot(sub, ymax, ylabel, title):
    sub.set_xticks(np.arange(0, 1.1, 0.1))
    sub.set_xlabel("x coordinate")
    sub.set_ylabel(ylabel)
    sub.set_ylim(-0.01, ymax)
    sub.set_xlim(-0.01, 1.01)
    sub.set_title(title)
    return sub


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
