import numpy as np

import math_fxns


def gaussian(x, m_x_vec, pref_vec, bound, params):
    """A mating model where each female's chance to mate with a given male is
    weighted normally by distance and by the female's signal preference.
    """
    d_vec = x - m_x_vec[bound[0]:bound[1]]
    p_vec = math_fxns.compute_pd(d_vec, params.beta)
    p_vec *= pref_vec[bound[0]:bound[1]]
    S = np.sum(p_vec)
    if S > 0:
        p_vec /= S
        cd = np.cumsum(p_vec)
        X = np.random.uniform()
        m_id = np.searchsorted(cd, X) + bound[0]
    else:
        m_id = -1
    return m_id


def uniform(x, m_x_vec, pref_vec, bound, params):
    """A mating function where females pick a mate with assortation within a
    bound but without weighting by pairing distance
    """
    if bound[1] - bound[0] > 0:
        p_vec = np.cumsum(pref_vec[bound[0]:bound[1]])
        S = np.sum(p_vec)
        p_vec /= S
        cd = np.cumsum(p_vec)
        X = np.random.uniform()
        m_id = np.searchsorted(cd, X) + bound[0]
    else:
        m_id = -1
    return m_id


def unbounded(x, m_x_vec, pref_vec, bound, params):
    """Identical to the gaussian mating function except that it does not impose
    spatial bounds on mate choice, making long-distance matings possible
    but very improbable
    """
    d_vec = x - m_x_vec
    p_vec = math_fxns.compute_pd(d_vec, params.beta)
    p_vec *= pref_vec
    S = np.sum(p_vec)
    if S > 0:
        p_vec /= S
        cd = np.cumsum(p_vec)
        X = np.random.uniform()
        m_id = np.searchsorted(cd, X)
    else:
        m_id = -1
    return m_id
