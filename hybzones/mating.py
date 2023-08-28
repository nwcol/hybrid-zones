import numpy as np

from hybzones import util

from hybzones.bounds import Bounds


class IdMap:

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
        self.abs = generation_table.cols.id
        self.rel = np.arange(generation_table.cols.filled_rows)
        self.female_index = generation_table.cols.get_sex_index(target_sex=0)
        self.male_index = generation_table.cols.get_sex_index(target_sex=1)

    def rel_to_abs(self, rel_ids):
        """
        Map relative IDs to absolute IDs

        :param rel_ids:
        :return:
        """
        return self.abs[rel_ids]

    def female_to_abs(self, female_ids):
        """
        Map relative female IDs to absolute IDs.

        :param female_ids:
        :return:
        """
        return self.abs[self.female_to_rel(female_ids)]

    def female_to_rel(self, female_ids):
        """
        Map relative female IDs to relative generation IDs.
        see :male_to_relative:

        :param female_ids:
        :return:
        """
        return self.female_index[female_ids]

    def male_to_abs(self, male_ids):
        """
        Map relative male IDs to absolute IDs.

        :param male_ids:
        :return:
        """
        return self.abs[self.male_to_rel(male_ids)]

    def male_to_rel(self, male_ids):
        """
        Map relative male IDs to relative generation IDs

        >>>IdMap.male_to_rel(10)
        Out: 17

        >>>IdMap.male_to_rel([10, 20])
        Out: array([17, 34], dtype=int64)

        >>>IdMap.male_to_rel(np.array([10, 20, 21]))
        Out: array([17, 34, 35], dtype=int64)

        :return: relative IDs
        """
        return self.male_index[male_ids]


class Matings:
    """

    """

    def __init__(self, parent_table):
        self.params = parent_table.params
        living_index = parent_table.living_index
        living_mask = parent_table.living_mask
        # mask the parent generation table so it contains only living organisms
        living_table = parent_table[living_mask]
        self.id_map = IdMap(living_table)
        self.n_offspring = self.compute_n_offspring(living_table)
        self.n_ = np.sum(self.n_offspring)
        self.n = None
        self.mating_bounds = Bounds(living_table, 0, 1, limits=[
            -self.params.bound, self.params.bound])
        rel_mat_ids, rel_pat_ids = self.run(living_table)
        # to relative id in the living table
        mat_ids = self.id_map.female_to_rel(rel_mat_ids)
        pat_ids = self.id_map.male_to_rel(rel_pat_ids)
        # to relative id in the parent table
        self.maternal_ids = living_index[mat_ids]
        self.paternal_ids = living_index[pat_ids]
        self.abs_maternal_ids = parent_table.cols.id[self.maternal_ids]
        self.abs_paternal_ids = parent_table.cols.id[self.paternal_ids]

    def compute_n_offspring(self, parent_table):
        """
        Compute the numbers of offspring produced by mating.

        :return:
        """
        k = self.params.K * 2 * self.params.density_bound
        density_bounds = Bounds(parent_table, 0, -1,
                                limits=[-self.params.density_bound,
                                        self.params.density_bound])
        density = density_bounds.get_bound_pops()
        density = density + self.edge_density_adjustment(parent_table)
        expectation = 2 * np.exp(self.params.r * (1 - (density / k)))
        n_offspring = np.random.poisson(expectation)
        return n_offspring

    def edge_density_adjustment(self, parent_table):
        """
        Compute a vector of density adjustments for the edges of space

        """
        b = self.params.density_bound
        k = self.params.K * b
        female_x = parent_table.cols.x[parent_table.cols.get_sex_index(0)]
        adjustment = np.zeros(len(female_x), dtype=np.float32)
        adjustment[female_x < b] = (b - female_x[female_x < b]) / b * k
        adjustment[female_x > 1 - b] = ((female_x[female_x > 1 - b] - 1 + b)
                                        / b * k)
        return adjustment

    def compute_pref_index(self, generation_table):
        """
        Compute indices to select the correct preference vector index for each
        female's preference: B1B1 0, B1B2 1, B2B2 2
        """
        return generation_table.cols.preference[self.id_map.female_index]

    def compute_pref_matrix(self, generation_table):
        """
        Compute 3 vectors of preferences targeting 'trait' in 'sex'.
        Preference vecs are combined in a 2d array in the order B1B1, B1B2,
        B2B2
        """
        signals = generation_table.cols.signal[self.id_map.male_index]
        n = len(self.id_map.male_index)
        pref_matrix = np.full((n, 3), 1, dtype=np.float32)
        c_matrix = self.params.get_c_matrix()
        for i in [0, 1, 2]:
            pref_matrix[:, i] = c_matrix[signals, i]
        return pref_matrix

    def run(self, generation_table):
        female_x = generation_table.cols.x[self.id_map.female_index]
        male_x = generation_table.cols.x[self.id_map.male_index]
        pref_matrix = self.compute_pref_matrix(generation_table)
        pref_index = self.compute_pref_index(generation_table)
        maternal_ids = np.full(self.n_, -1, dtype=np.int32)
        paternal_ids = np.full(self.n_, -1, dtype=np.int32)
        i = 0
        for f_id in np.arange(len(female_x)):
            n_children = self.n_offspring[f_id]
            if n_children > 0:
                x = female_x[f_id]
                prefs = pref_matrix[:, pref_index[f_id]]
                bounds = self.mating_bounds.bounds[f_id]
                m_id = self.gaussian(x, male_x, prefs, bounds)
                upper = i + n_children
                maternal_ids[i:upper] = f_id
                paternal_ids[i:upper] = m_id
                i = upper
            else:
                pass
        n_males = len(male_x)
        mask = np.nonzero((paternal_ids > -1) & (paternal_ids < n_males))
        maternal_ids = maternal_ids[mask]
        paternal_ids = paternal_ids[mask]
        self.n = len(maternal_ids)
        return maternal_ids, paternal_ids

    def gaussian(self, x, male_x, prefs, bound):
        """
        A mating model where each female's chance to mate with a given male is
        weighted normally by distance and by the female's signal preference.
        """
        d_vec = x - male_x[bound[0]:bound[1]]
        p_vec = util.compute_pd(d_vec, self.params.beta)
        p_vec *= prefs[bound[0]:bound[1]]
        s = np.sum(p_vec)
        if s > 0:
            p_vec /= s
            cd = np.cumsum(p_vec)
            u = np.random.uniform()
            m_id = np.searchsorted(cd, u) + bound[0]
        else:
            m_id = -1
        return m_id

    def uniform(self, x, male_x, prefs, bound):
        """
        A mating function where females pick a mate with assortation within a
        bound but without weighting by pairing distance
        """
        if bound[1] - bound[0] > 0:
            p_vec = np.cumsum(prefs[bound[0]:bound[1]])
            s = np.sum(p_vec)
            p_vec /= s
            cd = np.cumsum(p_vec)
            u = np.random.uniform()
            m_id = np.searchsorted(cd, u) + bound[0]
        else:
            m_id = -1
        return m_id

    def unbounded(self, x, male_x, prefs, bound):
        """
        Identical to the gaussian mating function except that it does not impose
        spatial bounds on mate choice, making long-distance matings possible
        but very improbable
        """
        d_vec = x - male_x
        p_vec = util.compute_pd(d_vec, self.params.beta)
        p_vec *= prefs
        s = np.sum(p_vec)
        if s > 0:
            p_vec /= s
            cd = np.cumsum(p_vec)
            u = np.random.uniform()
            m_id = np.searchsorted(cd, u)
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
        else:
            raise ValueError("invalid sex code!")
        a_index = np.random.randint(0, 2, size=self.n)
        b_index = np.random.randint(0, 2, size=self.n) + 2
        gametes = np.zeros((self.n, 2), dtype=np.uint8)
        gametes[:, 0] = parent_table.cols.alleles[row_index, a_index]
        gametes[:, 1] = parent_table.cols.alleles[row_index, b_index]
        return gametes
