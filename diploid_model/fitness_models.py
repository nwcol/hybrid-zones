import numpy as np

import math_fxns

from constants import Struc


def intrinsic_fitness(generation, params):
    """Exert the effects of intrinsic fitness on a generation.

    Intrinsic fitness corresponds to the post-zygotic fitness effects which
    sometimes affect hybrid organisms. Currently only signal heterozygotes,
    eg organisms with the A1A2 genotype, are affected by intrinsic fitness.
    """
    signal_sums = generation.get_signal_sums()
    H_idx = np.where(signal_sums == 3)[0]
    U_vec = np.random.uniform(0, 1, size=len(n_H))
    idx = H_idx[U_vec > params.H_fitness]
    generation.set_flags(idx, -2)


def extrinsic_fitness(generation, params):
    """exert the effects of extrinsic (environmental) fitness on a generation.

    This function uses logarithmic functions to compute the additive reductions
    in fitness for each allele, and sums them to get fitnesses for each
    organism. Organisms which are killed by fitness effects are flagged with
    flag = -1, preventing them from mating.
    """
    x = generation.get_x()
    N = generation.get_N()
    idx11, idx12, idx22 = generation.get_signal_indices()
    P = np.full(N, 1, dtype=np.float32)
    P[idx11] = 1 - 2 * s_1(x[idx11], params)
    P[idx12] = 1 - s_1(x[idx12], params) - s_2(x[idx12], params)
    P[idx22] = 1 - 2 * s_2(x[idx22], params)
    U = np.random.uniform(0, 1, N)
    idx = np.where(U > P)[0]
    generation.set_flags(idx, -1)


def s_1(x, params):
    """the function for the A1 logarithmic fitness curve"""
    s1 = params.mu - params.mu / (1 + np.exp(-params.k_1 * (x - params.mid_1)))
    return s1


def s_2(x, params):
    """the function for the A2 logarithmic fitness curve"""
    s2 = params.mu - params.mu / (1 + np.exp(-params.k_2 * (x - params.mid_2)))
    return s2
