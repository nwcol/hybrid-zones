import numpy as np


class Struc:
    dtype = np.float32
    n_cols = 11
    sex = 0
    i = 1
    x = 2
    t = 3
    mat_id = 4
    pat_id = 5
    A_loc0 = 6
    A_loc1 = 7
    B_loc0 = 8
    B_loc1 = 9
    flag = 10
    coords = [2, 3]
    parents = [4, 5]
    alleles = [6, 7, 8, 9]
    A_loci = [6, 7]
    B_loci = [8, 9]
    n_alleles = 4
    mat_allele_positions = [0, 2]
    pat_allele_positions = [1, 3]
    adjust_fac = 1.01


class Const:
    n_subpops = 9
    allele_sums = np.array([[2, 2], [2, 3], [2, 4], [3, 2], [3, 3], [3, 4],
        [4, 2], [4, 3], [4, 4]])
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
                     "royalblue"
                     ]
    subpop_colors = ["red",
                     "orange",
                     "palevioletred",
                     "chartreuse",
                     "green",
                     "lightseagreen",
                     "purple",
                     "deepskyblue",
                     "blue"
                     ]
    allele_legend = ["$A^1$",
                     "$A^2$",
                     "$B^1$",
                     "$B^2$"
                     ]
    subpop_legend = ["A = 1 B = 1",
                     "A = 1 B = H",
                     "A = 1 B = 2",
                     "A = H B = 1",
                     "A = H B = H",
                     "A = H B = 2",
                     "A = 2 B = 1",
                     "A = 2 B = H",
                     "A = 2 B = 2"
                     ]
    allele_manifold = np.array([[[2, 0], [2, 0]], [[2, 0], [1, 1]], [[2, 0],
            [0, 2]], [[1, 1], [2, 0]], [[1, 1], [1, 1]], [[1, 1], [0, 2]],
            [[0, 2], [2, 0]], [[0, 2], [1, 1]], [[0, 2], [0, 2]]])
