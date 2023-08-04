import numpy as np

from diploid import math_fxns





class IDmap:

    def __init__(self, generation_table):
        """
        Definitions.

        absolute. the absolute genealogical IDs of generation_table
        relative. the indices of all individuals within generation_table
        female_index. the indices of all females within generation_table
        male_index. the indices of all males within generation_table

        NOTE. currently np.int32 is the type used for internal IDs in the
        column, but np.int64 is okay for indexing

        :param generation_table:
        """
        self.absolute = generation_table.cols.ID
        self.relative = np.arange(generation_table.cols.filled_rows)
        self.female_index = generation_table.cols.get_sex_index(target_sex=0)
        self.male_index = generation_table.cols.get_sex_index(target_sex=1)

    def relative_to_absolute(self, relative_IDs):
        """
        Map relative IDs to absolute IDs

        :param relative_IDs:
        :return:
        """
        return self.absolute[relative_IDs]

    def female_to_relative(self, relative_female_IDs):
        """
        Map relative female IDs to relative generation IDs.
        see :male_to_relative:

        :param relative_female_IDs:
        :return:
        """
        return self.female_index[relative_female_IDs]

    def male_to_relative(self, relative_male_IDs):
        """
        Map relative male IDs to relative generation IDs

        >>>IDmap.male_to_relative(10)
        Out: 17

        >>>IDmap.male_to_relative([10, 20])
        Out: array([17, 34], dtype=int64)

        >>>IDmap.male_to_relative(np.array([10, 20, 21]))
        Out: array([17, 34, 35], dtype=int64)

        :return: relative IDs
        """
        return self.male_index[relative_male_IDs]


class Bounds:

    def __init__(self, generation_table, seeking_sex, target_sex, limits):
        """
        Compute bounds lol

        :param seeking_sex:
        :param target_sex: if -1, target the entire generation. else target
            sex 0 (females) or 1 (males)
        :param limits:
        """
        if seeking_sex == -1:
            self.seeking_index = np.arange(generation_table.cols.filled_rows)
        else:
            self.seeking_index = generation_table.cols.get_sex_index(
                seeking_sex)
        if target_sex == -1:
            self.target_index = np.arange(generation_table.cols.filled_rows)
        else:
            self.target_index = generation_table.cols.get_sex_index(target_sex)
        seeking_x = generation_table.cols.x[self.seeking_index]
        target_x = generation_table.cols.x[self.target_index]
        x_limits = seeking_x[:, np.newaxis] + limits
        self.bounds = np.searchsorted(target_x, x_limits)

    def get_bound_pops(self):
        """
        Compute the number of organisms captured by each bound

        :return:
        """
        return self.bounds[:, 1] - self.bounds[:, 0]


class LongX:

    def __init__(self, generation_table, id_map, mating_bounds):
        male_x = generation_table.cols.x[id_map.male_index]
        female_x = generation_table.cols.x[id_map.female_index]
        pops = mating_bounds.get_bound_pops()
        total = np.sum(pops) # about 1,000,000 long; roughly 200 * 5000
        cum = np.concatenate((np.array([0]), np.cumsum(pops)))
        long_x = np.zeros(total, dtype=np.float32)
        f_x = np.zeros(total, dtype=np.float32)
        for i in np.arange(len(female_x)):
            long_x[cum[i]:cum[i+1]] = male_x[mating_bounds.bounds[i,0]:
                                             mating_bounds.bounds[i,1]]
            f_x[cum[i]:cum[i+1]] = female_x[i]
        d = f_x - long_x
        self.p = compute_pd(d, self.params.beta)


def compute_pd(x, s):
    """Compute a vector of probability density values for a normal distribution
    with standard deviation s, mean 0.
    """
    return 1 / (s * np.sqrt(2 * np.pi)) * np.exp(-0.5 * np.square(x / s))


class Matings:

    """
    It is useful to define two types of 'relative ID' for the purposes of
    simulating mating and mate choice. Note that x-positions are unchangeable
    after dispersal and that we require that after dispersal, generations be
    sorted by increasing x-position

    1. relative ID. the index of an individual within the x-sorted generation
        with K = 10,000 these should range from 0 to approx. 10,000.

    2. relative in-sex ID. the index of an individual within the x-sorted
        sex subset of the generation. These should range from 0 to about 5,000
        with K = 10,000.

    We separate the generation into sexes. The IDs retrieved by mating are
    sex-relative. These are then mapped to relative IDs to easily access
    parent character (eg x-position and genotypes), and then to absolute IDs
    for the assignment of maternal and paternal IDs.

    ABSOLUTE ID vs relative id. vs relative sex id
        relative id: within the sorted generation
        (positions must be immutable after dispersal)
        relative sex id: within the sorted sex subset of a generation
        absolute id: the assigned id.

    steps to mating
    -compute pop densities -> get expected n offspring
     -> get actual n offspring
    -
    -compute idx bounds for female mate access
    -compute mating pdfs (cdfs too) for mate choice
    -mate choice. add lower idx bound
    (this all done with relative ids)



    """

    def __init__(self, parent_generation_table):
        self.params = parent_generation_table.params
        self.id_map = IDmap(parent_generation_table)
        d_limits = [-self.params.density_bound, self.params.density_bound]
        self.density_bounds = Bounds(parent_generation_table, 0, -1, d_limits)
        m_limits = [-self.params.bound, self.params.bound]
        self.mating_bounds = Bounds(parent_generation_table, 0, 1, m_limits)
        self.n_offspring = self.compute_n_offspring()
        self.n = np.sum(self.n_offspring)
        maternal_ids, paternal_ids = self.go(parent_generation_table)
        self.maternal_ids = self.id_map.female_to_relative(maternal_ids)
        self.paternal_ids = self.id_map.male_to_relative(paternal_ids)
        self.abs_maternal_ids = self.id_map.relative_to_absolute(self.maternal_ids)
        self.abs_paternal_ids = self.id_map.relative_to_absolute(self.paternal_ids)

    def compute_n_offspring(self):
        """
        Compute the numbers of offspring produced by mating.

        :return:
        """
        k = self.params.K * 2 * self.params.density_bound
        density = self.density_bounds.get_bound_pops()
        density += self.edge_density_adjustment()
        expectation = 2 * np.exp(self.params.r * (1 - (density / k)))
        n_offspring = np.random.poisson(expectation)
        return n_offspring

    def edge_density_adjustment(self):
        """
        Compute a vector of density adjustments for the edges of space

        """
        # rebuild
        return 0

    def compute_pref_index(self, generation_table):
        """
        Compute indices to select the correct preference vector index for each
        female's preference: B1B1 0, B1B2 1, B2B2 2
        """
        pref_sums = np.sum(generation_table.cols.B_alleles[
            self.id_map.female_index], axis=1)
        pref_idx = pref_sums - 2
        return pref_idx

    def compute_pref_matrix(self, generation_table):
        """
        Compute 3 vectors of preferences targeting 'trait' in 'sex'.
        Preference vecs are combined in a 2d array in the order B1B1, B1B2,
        B2B2
        """
        signals = np.sum(generation_table.cols.A_alleles[
            self.id_map.male_index], axis=1) - 2
        n = len(self.id_map.male_index)
        pref_matrix = np.full((n, 3), 1, dtype=np.float32)
        c_matrix = self.params.get_c_matrix()
        for i in [0, 1, 2]:
            pref_matrix[:, i] = c_matrix[signals, i]
        return pref_matrix

    def go(self, generation_table):
        female_x = generation_table.x[self.id_map.female_index]
        male_x = generation_table.x[self.id_map.male_index]
        pref_matrix = self.compute_pref_matrix(generation_table)
        pref_index = self.compute_pref_index(generation_table)
        maternal_ids = np.zeros(self.n, dtype=np.int32)
        paternal_ids = np.zeros(self.n, dtype=np.int32)
        i = 0
        for f_id in np.arange(len(female_x)):
            if self.n_offspring[f_id] > 0:
                x = female_x[f_id]
                m_id = self.gaussian(x, male_x,
                                     pref_matrix[:, pref_index[f_id]],
                                     self.mating_bounds.bounds[f_id])
                maternal_ids[i:i + self.n_offspring[f_id]] = f_id
                paternal_ids[i:i + self.n_offspring[f_id]] = m_id
                i += self.n_offspring[f_id]
            else:
                pass
        return maternal_ids, paternal_ids

    def gaussian(self, x, male_x, pref_vec, bound):
        """
        A mating model where each female's chance to mate with a given male is
        weighted normally by distance and by the female's signal preference.
        """
        d_vec = x - male_x[bound[0]:bound[1]]
        p_vec = compute_pd(d_vec, self.params.beta)
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

    def get_zygotes(self, parent_table):
        """
        Use the self.get_gametes to randomly draw gametes from parents and
        combine them in a pre-initialized array to get an array of alleles.
        This array is used to designate alleles for the child generation
        """
        zygotes = np.zeros((self.n, 4), dtype=np.uint8)
        zygotes[:, [0, 2]] = self.get_gametes(0, parent_table)
        zygotes[:, [1, 3]] = self.get_gametes(1, parent_table)
        # get rid of magic constants
        return zygotes

    def get_gametes(self, sex, parent_table):
        if sex == 0:
            row_index = self.maternal_ids
        elif sex == 1:
            row_index = self.paternal_ids
        A_index = np.random.randint(0, 2, size=self.n)
        B_index = np.random.randint(0, 2, size=self.n) + 2
        gametes = np.zeros((self.n, 2), dtype=np.uint8)
        gametes[:, 0] = parent_table.cols.alleles[row_index, A_index]
        gametes[:, 1] = parent_table.cols.alleles[row_index, B_index]
        return gametes























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
