
# flux_edge was never fully re-implemented. 

import numpy as np

import scipy

from scipy import stats

from scipy.special import erfc

import math

from hybzones.bounds import Bounds


def random_dispersal(generation_table):
    """
    Draw a set of displacements for the generation from a normal distribution
    with mean 0 and standard deviation delta.

    :param generation_table:
    :return: delta; vector of random displacements
    """
    n = len(generation_table.cols)
    s = generation_table.params.delta
    delta = np.random.normal(loc=0, scale=s, size=n)
    return delta


def scale_dispersal(generation_table):
    """
    Get a vector of dispersal distances using the scale model. Under this
    model, female dispersal is sampled from a normal distribution with
    scale params.delta, but male dispersal has a variable scale. Scale is
    computed for each male using scale_func and is a function of the proportion
    of adjacent same-signal males.

    :param generation_table:
    :return: delta; vector of displacements
    """
    params = generation_table.params
    female_index = generation_table.cols.get_subpop_index(sex=0)
    male_index = generation_table.cols.get_subpop_index(sex=1)
    n = len(generation_table.cols)
    delta = np.zeros(n, dtype=np.float32)
    delta[female_index] = np.random.normal(loc=0.0, scale=params.delta,
                                           size=len(female_index))
    limits = [-params.mating_bound, params.mating_bound]
    signal_prop, ah_index = get_signal_props(generation_table, limits)
    scale = scale_func(signal_prop, params)
    scale[ah_index] = params.delta
    delta[male_index] = np.random.normal(loc=0, scale=scale,
                                         size=len(male_index))
    return delta


def scale_func(prop, params):
    """
    Compute standard deviations for male dispersal in the scale dispersal
    model. SD are a linear function of adjacent signal proportion; at prop = 0,
    the SD equals params.d_scale * params.delta. At prop = 1, SD equals
    params.delta.

    :param prop: vector of signal proportions
    :param params:
    :return: vector of standard deviations
    """
    fac = params.scale_factor
    return ((1 - fac) * prop + fac) * params.delta


def shift_dispersal(generation_table):
    """
    Get a vector of dispersal distances using the shift model. Under this
    model, male dispersal distributions have an offset applied to them based
    on the proportions of same-signal males within a left and right bound,
    biasing males to move towards males with the same signal phenotype.

    :param generation_table:
    :return: delta; vector of displacements
    """
    params = generation_table.params
    female_index = generation_table.cols.get_subpop_index(sex=0)
    male_index = generation_table.cols.get_subpop_index(sex=1)
    n = len(generation_table.cols)
    delta = np.zeros(n, dtype=np.float32)
    delta[female_index] = np.random.normal(loc=0.0, scale=params.delta,
                                           size=len(female_index))
    left = [-params.mating_bound, 0]
    left_props, hyb_index = get_signal_props(generation_table, left)
    right = [0, params.mating_bound]
    right_props, hyb_index = get_signal_props(generation_table, right)
    loc = loc_func(left_props, right_props, params)
    loc[hyb_index] = 0
    delta[male_index] = np.random.normal(loc=loc, scale=params.delta,
                                         size=len(male_index))
    return delta


def loc_func(left_props, right_props, params):
    """
    Compute shifts for male dispersal. Used by the shift_dispersal model

    :param left_props: vector of signal proportions in the left bounds
    :param right_props: vector of signal proportions in the left bounds
    :param params:
    :return: vector of standard deviations
    """
    return (right_props - left_props) * (params.shift_factor * params.delta)


def get_signal_props(generation_table, limits):
    """
    Compute the proportion of same-signal males within a spatial limit
    for each male in a generation. Returns a vector of floats

    :param generation_table:
    :param limits: tuple or list of length 2 defining spatial limits
    """
    male_table = generation_table.get_subpop(sex=1)
    bounds = Bounds(male_table, 1, 1, limits)
    if limits[0] != 0 and limits[1] != 0:
        self_counting = True
    else:
        self_counting = False
    counts = bounds.get_bound_pops()
    if self_counting:
        counts -= 1
    a1_index = male_table.cols.get_subpop_index(signal=0)
    a2_index = male_table.cols.get_subpop_index(signal=2)
    hyb_index = male_table.cols.get_subpop_index(signal=1)
    n_males = len(male_table)
    signal_props = np.zeros(n_males, dtype=np.float32)
    signal_props[a1_index] = (
        np.searchsorted(a1_index, bounds.bounds[a1_index, 1])
        - np.searchsorted(a1_index, bounds.bounds[a1_index, 0]))
    signal_props[a2_index] = (
        np.searchsorted(a2_index, bounds.bounds[a2_index, 1])
        - np.searchsorted(a2_index, bounds.bounds[a2_index, 0]))
    if self_counting:
        signal_props -= 1  # returns negative numbers for hybrids; this is ok
    # mask nonzeros
    # check self counting
    signal_props /= counts
    signal_props[np.isnan(signal_props)] = 1
    return signal_props, hyb_index


def closed_edge(generation_table, delta):
    """
    Set the displacements of those organisms whose displacements would carry
    them outside the space to zero, freezing them inside the edges

    :param generation_table:
    :param delta: the vector of displacements to be applied with this model
    """
    positions = generation_table.cols.x + delta
    delta[positions < 0] = 0
    delta[positions > 1] = 0
    generation_table.cols.x += delta


def flux_edge(generation_table, delta):
    """

    :param generation_table:
    :param delta: the vector of displacements to be applied with this model
    """
    params = generation_table.params
    generation_table.cols.x += delta
    left_exits = np.nonzero(generation_table.cols.x < 0)[0]
    right_exits = np.nonzero(generation_table.cols.x > 1)[0]
    exits = np.concatenate((left_exits, right_exits))
    generation_table.set_flag(exits, -3)
    left_x = draw_migrants(params)
    right_x = 1 - draw_migrants(params)
    left_migrants = Generation.get_migrants(left_x, 1, t, params)
    right_migrants = Generation.get_migrants(right_x, 2, t, params)
    migrants = Generation.merge(left_migrants, right_migrants)


def draw_migrants(params):
    """
    Get an array of incoming migrants and their positions, given the shape
    (sigma) of the distribution and the carrying capacity K.
    """
    E_total_num = params.K * params.delta / math.sqrt(2 * math.pi)
    total_num = stats.poisson.rvs(E_total_num)
    positions = DispersalDist().rvs(params.delta, size=total_num)
    return positions


class DispersalDist(stats.rv_continuous):

    def _pdf(self, x, shape):
        return math.sqrt(math.pi / 2 / shape ** 2) * scipy.special.erfc(
            x / math.sqrt(2 * shape ** 2))

    def _cdf(self, x, shape):
        return (1 - math.exp(-(x ** 2) / 2 / shape ** 2)
                + math.sqrt(math.pi / 2 / shape ** 2) * x
                * erfc(x / math.sqrt(2 * shape ** 2)))


def ring_edge(generation_table, delta):
    """
    For all organisms exiting the space, wrap positions to the other side
    of space by adding or subtracting 1, simulating a circular environment -
    possibly analogous to a circumpolar range. Experimental.

    :param generation_table:
    :param delta: the vector of displacements to be applied with this model
    """
    positions = generation_table.cols.x + delta
    delta[positions < 0] += 1
    delta[positions > 1] -= 1
    generation_table.cols.x += delta


dispersal_models = {"random": random_dispersal,
                    "scale": scale_dispersal,
                    "shift": shift_dispersal}

edge_models = {"closed": closed_edge,
               "flux": flux_edge,
               "ring": ring_edge}


def main(generation_table):
    """
    Apply the dispersal and edge models given in generation_table.params to the
    generation table.

    :param generation_table:
    """
    params = generation_table.params
    model = params.dispersal_model
    delta = dispersal_models[model](generation_table)
    edge_model = params.edge_model
    edge_models[edge_model](generation_table, delta)
