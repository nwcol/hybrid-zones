import numpy as np

from scipy import stats

from scipy.special import erfc

import math


def random_dispersal(generation, params):
    """Draw a set of displacements for the generation from a normal
    distribution with mean 0 and standard deviation delta.
    """
    N = generation.get_N()
    delta = np.random.normal(loc=0, scale=params.delta, size=N)
    return delta


def scale_dispersal(generation, params):
    """Get a vector of dispersal distances using the scale model.

    Females sample dispersals from normal distributions with std params.delta.
    Males check the signal
    """
    N = generation.get_N()
    F = generation.get_F()
    M = N - F
    f_idx, m_idx = generation.get_sex_indices()
    delta = np.zeros(N, dtype=np.float32)
    delta[f_idx] = np.random.normal(loc=0.0, scale=params.delta, size=F)
    prop, hyb_idx = generation.get_signal_props([-params.bound, params.bound])
    scale = scale_func(prop, params)
    scale[hyb_idx] = params.delta
    delta[m_idx] = np.random.normal(loc=0, scale=scale, size=M)
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
    N = generation.get_N()
    F = generation.get_F()
    M = N - F
    f_idx, m_idx = generation.get_sex_indices()
    delta = np.zeros(N, dtype=np.float32)
    delta[f_idx] = np.random.normal(loc=0.0, scale=params.delta, size=F)
    l_prop, hyb_idx = generation.get_signal_props([-params.bound, 0])
    r_prop, hyb_idx = generation.get_signal_props([0, params.bound])
    loc = loc_func(l_prop, r_prop, params)
    loc[hyb_idx] = 0
    delta[m_idx] = np.random.normal(loc=loc, scale=params.delta, size=M)
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


dispersal_model_dict = {"random": random_dispersal,
                        "scale" : scale_dispersal,
                        "shift" : shift_dispersal}
