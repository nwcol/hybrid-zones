import numpy as np


class Constants:
    """
    Defines some useful objects and conventions. Importantly, all objects
    pertaining to genotypes are presented in the order
        0 : A 1, B 1
        1 : A 1, B H
        2 : A 1, B 2
        3 : A H, B 1
        4 : A H, B H
        5 : A H, B 2
        6 : A 2, B 1
        7 : A 2, B H
        8 : A 2, B 2
    """
    plot_size = (8, 6)

    n_loci = 2
    n_A_alelles = 2
    n_B_alleles = 2
    n_genotypes = 9

    # the colors associated with alleles, in the order A^1, A^2, B^1, B^2
    allele_colors = ["red",
                     "blue",
                     "lightcoral",
                     "royalblue"]

    # the colors associated with genotypes
    genotype_colors = ["red",
                       "orange",
                       "palevioletred",
                       "chartreuse",
                       "green",
                       "lightseagreen",
                       "purple",
                       "deepskyblue",
                       "blue"]

    # names of the alleles
    allele_legend = ["$A^1$",
                     "$A^2$",
                     "$B^1$",
                     "$B^2$"]

    # names of the 9 genotypes
    subpop_legend = ["A = 1 B = 1",
                     "A = 1 B = H",
                     "A = 1 B = 2",
                     "A = H B = 1",
                     "A = H B = H",
                     "A = H B = 2",
                     "A = 2 B = 1",
                     "A = 2 B = H",
                     "A = 2 B = 2"]

    # because there are 16 possible arrangements of alleles and only 9
    # genotypes, it is convenient to sum the allele values at each locus up to
    # classify organisms by genotype. This array classifies those sums
    allele_sums = np.array([[2, 2],
                            [2, 3],
                            [2, 4],
                            [3, 2],
                            [3, 3],
                            [3, 4],
                            [4, 2],
                            [4, 3],
                            [4, 4]], dtype=np.uint8)

    # there are more possible arrangements of alleles
    # these are the ones used when creating founding generations
    genotypes = np.array([[1, 1, 1, 1],
                          [1, 1, 1, 2],
                          [1, 1, 2, 2],
                          [1, 2, 1, 1],
                          [1, 2, 1, 2],
                          [1, 2, 2, 2],
                          [2, 2, 1, 1],
                          [2, 2, 1, 2],
                          [2, 2, 2, 2]], dtype=np.uint8)

    # this is an object used to convert genotype counts into allele counts.
    # each row sums to 4, as every organism has 4 alleles
    # counts: [[A^1, A^2], [B^1, B^2]]
    allele_manifold = np.array([[[2, 0], [2, 0]],
                                [[2, 0], [1, 1]],
                                [[2, 0], [0, 2]],
                                [[1, 1], [2, 0]],
                                [[1, 1], [1, 1]],
                                [[1, 1], [0, 2]],
                                [[0, 2], [2, 0]],
                                [[0, 2], [1, 1]],
                                [[0, 2], [0, 2]]], dtype=np.uint8)

    # subplot dimensions for history plots
    shape_dict = {1: (1, 1),
                  2: (1, 2),
                  3: (1, 3),
                  4: (2, 2),
                  5: (2, 3),
                  6: (2, 3),
                  8: (2, 4),
                  9: (3, 3),
                  10: (2, 5),
                  11: (3, 4),
                  12: (3, 4),
                  16: (4, 4),
                  21: (3, 7)}