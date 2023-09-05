import matplotlib.pyplot as plt

import numpy as np

import scipy.optimize as opt

import time


def get_snaps(g, n_snaps):
    """
    Get an array of n_snaps evenly spaced integers between g and 0 for
    """
    return np.linspace(g, 0, n_snaps).astype(np.int32)


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


def plot_spaced_histogram(x, y, y_label, n_bins=100):
    """
    Take a vector of x positions and a statistic y; bin the statistic using
    x positions in n_bins bins and plot the bin means
    """
    bin_edges, dump = get_bins(1/n_bins)
    bin_mids = get_bin_mids(1/n_bins)
    mean_y = []
    std_y = []
    for i in np.arange(n_bins):
        index = np.nonzero((x > bin_edges[i]) & (x < bin_edges[i + 1]))[0]
        mean_y.append(np.mean(y[index]))
        std_y.append(np.std(y[index]))
    figure = plt.figure(figsize=(8, 6))
    sub = figure.add_subplot(111)
    sub.errorbar(bin_mids, mean_y, yerr=std_y, color="black", marker="x")
    sub.set_xlim(-0.01, 1.01)
    sub.set_ylabel(y_label)
    sub.set_title(y_label)


def get_figure(n, length):
    """
    :param n: number of desired post-founder subplots.
    """
    n += 1
    snaps = np.linspace(length, 0, n).astype(np.int32)

    ### fix
    n_rows = 2
    n_cols = (snaps + 1) // 2
    plot_shape = (n_rows, n_cols)
    size = (n_cols * 4, n_rows * 3)
    figure, axs = plt.subplots(n_rows, n_cols, figsize=size, sharex='all')
