import numpy as np

import scipy

from scipy import stats

from scipy.special import erfc

import math


def random_dispersal(generation_table):
    """
    Draw a set of displacements for the generation from a normal
    distribution with mean 0 and standard deviation delta.
    """
    n = generation_table.cols.filled_rows
    s = generation_table.params.delta
    delta = np.random.normal(loc=0, scale=s, size=n)
    return delta


def scale_dispersal(generation, params):
    """Get a vector of dispersal distances using the scale model.

    Females sample dispersals from normal distributions with std params.delta.
    Males check the signal
    """
    n_females = generation.get_n_females()
    n_males = generation.get_n_males()
    n = n_females + n_males
    f_idx, m_idx = generation.get_sex_indices()
    delta = np.zeros(n, dtype=np.float32)
    delta[f_idx] = np.random.normal(loc=0.0, scale=params.delta,
                                    size=n_females)
    prop, hyb_idx = generation.get_signal_props([-params.bound, params.bound])
    scale = scale_func(prop, params)
    scale[hyb_idx] = params.delta
    delta[m_idx] = np.random.normal(loc=0, scale=scale, size=n_males)
    return delta


def scale_func(prop, params):
    """Compute standard deviations for male dispersal. Used by the
    scale_dispersal dispersal model.
    """
    max_scale = params.d_scale
    scale = (1 - max_scale) * prop + max_scale
    scale *= params.delta
    return scale


def shift_dispersal(generation, params):
    """Get a vector of dispersal distances using the shift model"""
    n_females = generation.get_n_females()
    n_males = generation.get_n_males()
    n = n_females + n_males
    f_idx, m_idx = generation.get_sex_indices()
    delta = np.zeros(n, dtype=np.float32)
    delta[f_idx] = np.random.normal(loc=0.0, scale=params.delta,
                                    size=n_females)
    l_prop, hyb_idx = generation.get_signal_props([-params.bound, 0])
    r_prop, hyb_idx = generation.get_signal_props([0, params.bound])
    loc = loc_func(l_prop, r_prop, params)
    loc[hyb_idx] = 0
    delta[m_idx] = np.random.normal(loc=loc, scale=params.delta, size=n_males)
    return delta


def loc_func(l_prop, r_prop, params):
    """Compute shifts for male dispersal. Used by the shift_dispersal model
    """
    # nan errors
    m = params.d_scale  # slope
    dif = r_prop - l_prop
    loc = dif * m
    loc *= params.delta
    return loc


def draw_migrants(params):
    """Get an array of incoming migrants and their positions, given the shape
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


def closed(generation_table, delta):
    """
    Set the displacement of any individual who would exit the space when
    displacements are applied to 0, freezing them in place
    """
    positions = generation_table.cols.x + delta
    delta[positions < 0] = 0
    delta[positions > 1] = 0
    generation_table.cols.x += delta


dispersal_models = {"random": random_dispersal,
                    "scale" : scale_dispersal,
                    "shift" : shift_dispersal}

edge_models = {"closed": closed,
               "flux": None, #flux,
               "ring": None, #self.ring
 }


def disperse(generation_table):
    delta = dispersal_models[generation_table.params.dispersal_model](generation_table)
    edge_models[generation_table.params.edge_model](generation_table, delta)

