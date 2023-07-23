import numpy as np


class Const:
    bin_size = 0.01
    n_bins = int(1 / bin_size)
    bins = np.arange(0, 1 + bin_size, bin_size)
    bins_left = np.arange(0, 1, bin_size)
    bins_right = np.arange(bin_size, 1 + bin_size, bin_size)
    bins_mid = np.arange(bin_size / 2, 1 + bin_size / 2, bin_size)
    plot_size = (6, 4.5)

    allele_colors = ["red",
                     "blue",
                     "lightcoral",
                     "royalblue"]

    subpop_colors = ["red",
                     "orange",
                     "palevioletred",
                     "chartreuse",
                     "green",
                     "lightseagreen",
                     "purple",
                     "deepskyblue",
                     "blue"]

    allele_legend = ["$A^1$",
                     "$A^2$",
                     "$B^1$",
                     "$B^2$"]

    subpop_legend = ["A = 1 B = 1",
                     "A = 1 B = H",
                     "A = 1 B = 2",
                     "A = H B = 1",
                     "A = H B = H",
                     "A = H B = 2",
                     "A = 2 B = 1",
                     "A = 2 B = H",
                     "A = 2 B = 2"]

    n_subpops = 9
    allele_sums = np.array([[2, 2],
                            [2, 3],
                            [2, 4],
                            [3, 2],
                            [3, 3],
                            [3, 4],
                            [4, 2],
                            [4, 3],
                            [4, 4]])

    # there are more possible arrangements of alleles
    genotypes = np.array([[1, 1, 1, 1],
                          [1, 1, 1, 2],
                          [1, 1, 2, 2],
                          [1, 2, 1, 1],
                          [1, 2, 1, 2],
                          [1, 2, 2, 2],
                          [2, 2, 1, 1],
                          [2, 2, 1, 2],
                          [2, 2, 2, 2]])

    allele_manifold = np.array([[[2, 0], [2, 0]],
                                [[2, 0], [1, 1]],
                                [[2, 0], [0, 2]],
                                [[1, 1], [2, 0]],
                                [[1, 1], [1, 1]],
                                [[1, 1], [0, 2]],
                                [[0, 2], [2, 0]],
                                [[0, 2], [1, 1]],
                                [[0, 2], [0, 2]]])
