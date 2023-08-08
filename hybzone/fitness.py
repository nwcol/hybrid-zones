import numpy as np


def main(generation_table):
    """
    Apply fitness functions to a generation_table in-place

    :param generation_table:
    :return: None
    """
    if generation_table.params.intrinsic_fitness:
        intrinsic_fitness(generation_table)
    if generation_table.params.extrinsic_fitness:
        intrinsic_fitness(generation_table)


def intrinsic_fitness(generation_table):
    """
    Exert the effects of intrinsic fitness on a generation.

    Intrinsic fitness is the same as post-zygotic fitness effects. In our
    model it affects only signal heterozygotes
    """
    params = generation_table.params
    if params.female_fitness:
        hyb_index = generation_table.cols.get_subpop_index(signal=1)
    else:
        hyb_index = generation_table.cols.get_subpop_index(signal=1, sex=1)
    u = np.random.uniform(0, 1, size=len(hyb_index))
    index = hyb_index[u > params.hyb_fitness]
    generation_table.set_flags(index, -2)


def extrinsic_fitness(generation_table):
    """
    Exert the effects of extrinsic (environmental) fitness on a generation.

    This function uses logarithmic functions to compute the additive reductions
    in fitness for each allele, and sums them to get fitnesses for each
    organism. Organisms which are killed by fitness effects are flagged with
    flag = -1, preventing them from mating.
    """
    params = generation_table.params
    mask = generation_table.living_mask # mask dead organisms so they
    # aren't killed twice
    living_cols = generation_table.cols[mask]
    x = living_cols.x[mask]
    n = len(living_cols)
    if params.female_fitness:
        index_11 = living_cols.get_subpop_index(signal=0)
        index_12 = living_cols.get_subpop_index(signal=1)
        index_22 = living_cols.get_subpop_index(signal=2)
    else:
        index_11 = living_cols.get_subpop_index(signal=0, sex=1)
        index_12 = living_cols.get_subpop_index(signal=1, sex=1)
        index_22 = living_cols.get_subpop_index(signal=2, sex=1)
    p = np.full(n, 1, dtype=np.float32)
    p[index_11] = 1 - 2 * s_1(x[index_11], params)
    p[index_12] = 1 - s_1(x[index_12], params) - s_2(x[index_12], params)
    p[index_22] = 1 - 2 * s_2(x[index_22], params)
    u = np.random.uniform(0, 1, n)
    index = np.nonzero(u > p)[0]
    real_index = mask[index]
    generation_table.set_flags(real_index, -1)


def s_1(x, params):
    """the function for the A1 logarithmic fitness curve"""
    s1 = params.mu - params.mu / (1 + np.exp(-params.k_1 * (x - params.mid_1)))
    return s1


def s_2(x, params):
    """the function for the A2 logarithmic fitness curve"""
    s2 = params.mu - params.mu / (1 + np.exp(-params.k_2 * (x - params.mid_2)))
    return s2
