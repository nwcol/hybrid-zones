import numpy as np

import time

import matplotlib.pyplot as plt

import parameters

import mating_models

import dispersal_models

import fitness_models

from constants import Struc, Const

plt.rcParams['figure.dpi'] = 100


class Generation:
    organism_axis = 0
    character_axis = 1

    def __init__(self, N, x, t, parent_ids, alleles, params):
        """n corresponds to n_filled in the Pedigree"""
        self.params = params
        self.t = t
        self.N = N
        self.K = params.K
        arr = np.zeros((N, Struc.n_cols), dtype=Struc.dtype)
        arr[:, Struc.sex] = np.random.randint(0, 2, N)
        arr[:, Struc.x] = x
        arr[:, Struc.t] = t
        arr[:, Struc.mat_id] = parent_ids[:, 0]
        arr[:, Struc.pat_id] = parent_ids[:, 1]
        arr[:, Struc.alleles] = alleles
        arr[:, Struc.flag] = 1
        self.arr = arr
        self.flag = 1
        self.sort()

    @classmethod
    def get_founding(cls, params):
        """create the founding generation"""
        Ns = np.array([params.N_A1B1, params.N_A1B2, params.N_A2B1,
                       params.N_A2B2], dtype=np.int32)
        limits = [params.spawnlimit_A1B1, params.spawnlimit_A1B2,
                  params.spawnlimit_A2B1, params.spawnlimit_A2B2]
        genotypes = np.array([[1, 1, 1, 1], [1, 1, 2, 2], [2, 2, 1, 1],
                              [2, 2, 2, 2]], dtype=Struc.dtype)
        a = []
        xx = []
        for i in np.arange(4):
            if Ns[i] > 0:
                a.append(np.repeat(genotypes[None, i], Ns[i], axis=0))
                xx.append(np.random.uniform(limits[i][0], limits[i][1], Ns[i]))
        alleles = np.vstack(a)
        x = np.concatenate(xx)
        N = np.sum(Ns)
        parent_ids = np.full((N, 2), -1, dtype=Struc.dtype)
        t = params.g
        return cls(N, x, t, parent_ids, alleles, params)

    @classmethod
    def mate(cls, old_generation, params):
        t = old_generation.t - 1
        mating_pairs = MatingPairs(old_generation, params)
        N = mating_pairs.get_N()
        x = mating_pairs.get_maternal_x()
        parent_ids = mating_pairs.get_parental_ids()
        alleles = mating_pairs.get_zygotes()
        return cls(N, x, t, parent_ids, alleles, params)

    @classmethod
    def merge(cls, gen1, gen2):
        """merge two generations with the same t value. useful for combining
        migrants into the primary generation
        """
        params = gen1.params
        t1 = gen1.t
        t2 = gen2.t
        N = gen1.get_N() + gen2.get_N()
        t = t1
        x = np.concatenate((gen1.get_x(), gen2.get_x()))
        parent_ids = np.vstack((gen1.get_parents(), gen2.get_parents()))
        alleles = np.vstack((gen1.get_alleles(), gen2.get_alleles()))
        return cls(N, x, t, parent_ids, alleles, params)

    @classmethod
    def get_migrants(cls, x, species, t, params):
        N = len(x)
        parent_ids = np.full((N, 2), -1)
        alleles = np.full((N, Struc.n_alleles), species)
        return cls(N, x, t, parent_ids, alleles, params)

    def precompute(self):
        """Compute a set of useful vectors, arrays etc once a generation
        has been completely processed eg dispersed, affected by fitness, and
        indexed+sorted.
        """
        self.f_idx = self.get_f_idx()
        self.m_idx = self.get_m_idx()

    def compute_n_offspring(self, params):
        """Get a vector of numbers of offspring"""
        local_K = params.K * 2 * params.density_bound
        density = self.compute_local_densities(params)
        expectation = 2 * np.exp(params.r * (1 - (density / local_K)))
        n_offspring_vec = np.random.poisson(expectation)
        return n_offspring_vec

    def compute_local_densities(self, params):
        """Compute the population densities about each female in the
        generation
        """
        b = params.density_bound
        females = self.get_females()
        bound = self.compute_bounds(females, self.arr, [-b, b]).astype(np.float32)
        densities = bound[:, 1]
        densities -= bound[:, 0]
        densities += self.edge_density_adjustment(params)
        return densities

    def edge_density_adjustment(self, params):
        """Compute a vector of density adjustments for the edges of space"""
        b = params.density_bound
        k = params.K * b
        females = self.get_females()
        f_x = females[:, Struc.x]
        adjustment = np.zeros(len(f_x))
        adjustment[f_x < b] = (b - f_x[f_x < b]) / b * k
        adjustment[f_x > 1 - b] = (f_x[f_x > 1 - b] - 1 + b) / b * k
        return adjustment

    def compute_pref_idx(self):
        """Compute indices to select the correct preference vector index for each
        female's preference: B1B1 0, B1B2 1, B2B2 2
        """
        f_idx = self.get_f_idx()
        B_sums = np.sum(self.arr[np.ix_(f_idx, Struc.B_loci)], axis = 1)
        pref_idx = (B_sums - 2).astype(np.int32)
        return pref_idx

    def compute_pref_matrix(self, params):
        """Compute 3 vectors of preferences targeting 'trait' in 'sex'.
        Preference vecs are combined in a 2d array in the order B1B1, B1B2,
        B2B2
        """
        m_idx = self.get_m_idx()
        A_sums = np.sum(self.arr[np.ix_(m_idx, Struc.A_loci)], axis = 1)
        trait_vec = (A_sums - 2).astype(np.int32)
        pref_matrix = np.full((self.get_M(), 3), 1, dtype=np.float32)
        for i in [0, 1, 2]:
            pref_matrix[:, i] = params.c_matrix[trait_vec, i]
        return pref_matrix

    @staticmethod
    def interpret_dispersal_model(params):
        if params.dispersal_model == "random":
            fxn = dispersal_models.random_dispersal
        elif params.dispersal_model == "scale":
            fxn = dispersal_models.scale_dispersal
        elif params.dispersal_model == "shift":
            fxn = dispersal_models.shift_dispersal
        else:
            print("invalid dispersal model")
            fxn = None
        return fxn

    def disperse(self, params):
        fxn = self.interpret_dispersal_model(params)
        delta = fxn(self, params)
        if params.edge_model == "closed":
            self.closed(delta)
        elif params.edge_model == "flux":
            self.flux(delta, params)
        elif params.edge_model == "ring":
            self.ring(delta)
        else:
            print("invalid edge model")
        self.arr[:, Struc.x] += delta

    def closed(self, delta):
        """Set the displacement of any individual who would exit the space when
        displacements are applied to 0, freezing them in place
        """
        positions = delta + self.get_x()
        delta[positions < 0] = 0
        delta[positions > 1] = 0
        self.arr[:, Struc.x] += delta

    def flux(self, delta, params):
        self.arr[:, Struc.x] += delta
        t = self.t
        exits = self.detect_exits()
        self.set_flags(exits, -3)
        left_x = dispersal_models.draw_migrants(params)
        left_migrants, runlog = Generation.get_migrants(left_x, 1, t, params)
        right_x = 1 - dispersal_models.draw_migrants(params)
        right_migrants, runlog = Generation.get_migrants(right_x, 2, t, params)
        migrants = Generation.merge(left_migrants, right_migrants)
        self = Generation.merge(self, migrants)
        # no idea if this works

    def ring(self, delta):
        """An experimental edge function where individuals who exit the space
        are deposited onto the other side of space, creating a closed loop
        """
        self.arr[:, Struc.x] += delta
        self.arr[self.detect_l_exits(), Struc.x] += 1
        self.arr[self.detect_r_exits(), Struc.x] -= 1

    def detect_l_exits(self):
        """Return an index of left exits"""
        return np.where(self.arr[:, Struc.x] < 0)[0]

    def detect_r_exits(self):
        """Return an index of right exits"""
        return np.where(self.arr[:, Struc.x] > 1)[0]

    def detect_exits(self):
        """Return the indices of individuals with invalid x coordinates outside
        [0, 1]
        """
        return np.concatenate((self.detect_l_exits(), self.detect_l_exits))

    def fitness(self, params):
        if params.intrinsic_fitness:
            fitness_models.intrinsic_fitness(self, params)
        if params.extrinsic_fitness:
            fitness_models.extrinsic_fitness(self, params)

    def set_flags(self, idx, x):
        """Flag individuals at index idx with flag x"""
        self.arr[idx, Struc.flag] = x

    @staticmethod
    def compute_bounds(seeking_sex, target_sex, limits):
        """Compute bounds, given two arrays as arguments"""
        x_0 = seeking_sex[:, Struc.x]
        x_1 = target_sex[:, Struc.x]
        l_bound = np.searchsorted(a = x_1, v = x_0 + limits[0])
        r_bound = np.searchsorted(a = x_1, v = x_0 + limits[1])
        bounds = np.column_stack((l_bound, r_bound))
        return bounds

    def compute_sex_bounds(self, seeking_i, target_i, limits):
        """Compute bounds, given two integers representing sexes as args"""
        idx_0 = self.get_sex_idx(seeking_i)
        idx_1 = self.get_sex_idx(target_i)
        x_0 = self.get_x()[idx_0]
        x_1 = self.get_x()[idx_1]
        l_bound = np.searchsorted(a = x_1, v = x_0 + limits[0])
        r_bound = np.searchsorted(a = x_1, v = x_0 + limits[1])
        bounds = np.column_stack((l_bound, r_bound))
        return bounds

    def get_signal_props(self, limits):
        """Compute the proportion of same-signal males within a spatial limit
        for each male in a generation.

        Used to adjust the direction or scale of male dispersal under nonrandom
        dispersal models.
        """
        bounds = self.compute_sex_bounds(1, 1, limits)
        if limits[0] != 0 and limits[1] != 0:
            self_counting = True
        else:
            self_counting = False
        n = bounds[:, 1] - bounds[:, 0]
        if self_counting:
            n -= 1
        idx11, idx12, idx22 = self.get_male_signal_indices()
        signal_props = np.zeros(self.get_M(), dtype = np.float32)
        signal_props[idx11] = (
                np.searchsorted(idx11, bounds[idx11, 1])
              - np.searchsorted(idx11, bounds[idx11, 0]))
        signal_props[idx22] = (
                np.searchsorted(idx22, bounds[idx22, 1])
              - np.searchsorted(idx22, bounds[idx22, 0]))
        if self_counting:
            signal_props -= 1
        signal_props /= n
        signal_props[np.isnan(signal_props)] = 1
        return signal_props, idx12

    def get_male_signal_sums(self):
        """Return the sum of signal alleles for males only"""
        males = self.get_males()
        return np.sum(males[:, Struc.A_loci], axis = 1)

    def get_male_signal_indices(self):
        signal_sums = self.get_male_signal_sums()
        idx11 = np.where(signal_sums == 2)[0]
        idx12 = np.where(signal_sums == 3)[0]
        idx22 = np.where(signal_sums == 4)[0]
        return idx11, idx12, idx22

    def get_signal_sums(self):
        """Return the sum of signal alleles for all individuals"""
        return np.sum(self.arr[:, Struc.A_loci], axis = 1)

    def get_signal_indices(self):
        signal_sums = self.get_signal_sums()
        idx11 = np.where(signal_sums == 2)[0]
        idx12 = np.where(signal_sums == 3)[0]
        idx22 = np.where(signal_sums == 4)[0]
        return idx11, idx12, idx22

    def get_sex_indices(self):
        """Return the indices of females and of males in a generation
        """
        f_idx = np.where(self.arr[:, Struc.sex] == 0)[0]
        m_idx = np.where(self.arr[:, Struc.sex] == 1)[0]
        return f_idx, m_idx

    def get_sex_idx(self, sex):
        """Given an integer sex = 0 or sex = 1, return the appropriate sex"""
        return np.where(self.arr[:, Struc.sex] == sex)

    def get_f_idx(self):
        """Return the indices of the females in a generation"""
        return np.where(self.arr[:, Struc.sex] == 0)[0]

    def get_m_idx(self):
        """return the indices of the males in the generation"""
        return np.where(self.arr[:, Struc.sex] == 1)[0]

    def get_f_x(self):
        """Return a vector of male x positions"""
        return self.arr[self.get_f_idx(), Struc.x]

    def get_m_x(self):
        """Return a vector of male x positions"""
        return self.arr[self.get_m_idx(), Struc.x]

    def update_N(self):
        """Update N"""
        self.N = self.get_N()

    def get_N(self):
        """Return the number of individuals in the generation"""
        return np.shape(self.arr)[0]

    def get_F(self):
        """Return the number of females in a generation"""
        return np.size(self.get_f_idx())

    def get_M(self):
        """Return the number of males in a generation"""
        return np.size(self.get_m_idx())

    def get_males(self):
        """Return the males in the generation"""
        return self.arr[self.get_m_idx()]

    def get_females(self):
        """Return the females in the generation"""
        return self.arr[self.get_f_idx()]

    def get_alleles(self):
        return self.arr[:, Struc.alleles]

    def get_parents(self):
        return self.arr[:, Struc.parents].astype(np.int32)

    def get_t(self):
        return self.t

    def get_x(self):
        return self.arr[:, Struc.x]

    def split_gen(self):
        """Get arrays containing only the female and male individuals of a
        generation
        """
        females = self.arr[self.get_f_idx()]
        males = self.arr[self.get_m_idx()]
        return females, males

    def sort(self):
        """Sort the generation by x coordinate. Sorting is vital to the
        correct operation of many Generation functions
        """
        x = self.get_x()
        self.arr = self.arr[x.argsort()]

    def sort_and_id(self, i0):
        """Sort the generation by x coordinate and enter the id column, where
        each element matches the index of its row. This may seem redundant
        but it is convenient for sampling sub-pedigrees. i1 is return to the
        Trial object to keep track of the index
        """
        self.sort()
        N = self.get_N()
        i1 = i0 + N
        ids = np.arange(i0, i1)
        self.arr[:, Struc.i] = ids
        return i1

    def get_allele_sums(self):
        allele_sums = np.zeros((self.N, 2))
        allele_sums[:, 0] = np.sum(self.arr[:, Struc.A_loci], axis=1)
        allele_sums[:, 1] = np.sum(self.arr[:, Struc.B_loci], axis=1)
        return allele_sums

    def get_subpop_idx(self):
        """Return a subpop index for each organism in the generation"""
        allele_sums = self.get_allele_sums()
        subpop_idx = np.zeros(self.N, dtype=np.int32)
        for i in np.arange(9):
            idx = np.where((allele_sums[:, 0] == Const.allele_sums[i, 0])
                           & (allele_sums[:, 1] == Const.allele_sums[i, 1]))[0]
            subpop_idx[idx] = i
        return subpop_idx

    def get_living(self):
        """Return the index of living organisms with flag = 1"""
        return np.where(self.arr[:, Struc.flag] == 1)[0]

    def senescence(self):
        self.flag = 0
        self.arr[self.get_living(), Struc.flag] = 0

    def get_dead_idx(self):
        """Get the index of individuals with flag < 0"""
        return np.where(self.arr[:, Struc.flag] < 0)[0]

    def remove_dead(self):
        """Delete individuals with flag != 1 or 0. Counterintuitively,
        individuals with flag = 0 are retained; this is because of the order
        in which functions ar currently executed in the main loop
        """
        self.arr = np.delete(self.arr, self.get_dead_idx(), axis=0)

    def plot_subpops(self):
        gen_subpop_arr = GenSubpopArr(self)
        fig = gen_subpop_arr.density_plot()
        return fig


class MatingPairs:

    i_axis = 0
    sex_axis = 1
    character_axis = 2

    def __init__(self, generation, params):
        self.N = None
        self.arr = None
        self.pair_ids = self.compute_pair_ids(generation, params)
        self.get_mating_pairs(generation)

    def initialize_pair_ids(self, n_offspring):
        self.N = np.sum(n_offspring)
        mating_pair_ids = np.zeros((self.N, 2), dtype = np.int32)
        return mating_pair_ids

    @staticmethod
    def interpret_mating_fxn(params):
        if params.mating_model == "gaussian":
            fxn = mating_models.gaussian
        elif params.mating_model == "uniform":
            fxn = mating_models.uniform
        elif params.mating_model == "unbounded":
            fxn = mating_models.unbounded
        else:
            print("invalid mating function")
        return fxn

    def compute_pair_ids(self, generation, params):
        fxn = self.interpret_mating_fxn(params)
        m_x_vec = generation.get_m_x()
        f_x_vec = generation.get_f_x()
        pref_matrix = generation.compute_pref_matrix(params)
        pref_vec_idx = generation.compute_pref_idx()
        n_offspring = generation.compute_n_offspring(params)
        b = params.bound
        bounds = generation.compute_sex_bounds(0, 1, [-b, b])
        mating_pair_ids = self.initialize_pair_ids(n_offspring)
        F = generation.get_F()
        i = 0
        for f_id in np.arange(F):
            if n_offspring[f_id] > 0:
                x = f_x_vec[f_id]
                m_id = fxn(x, m_x_vec, pref_matrix[:, pref_vec_idx[f_id]],
                    bounds[f_id], params)
                mating_pair_ids[i:i + n_offspring[f_id]] = [f_id, m_id]
                i += n_offspring[f_id]
            else:
                pass
        return mating_pair_ids

    def get_mating_pairs(self, generation):
        """Uses the mating_id_arr to organize a 3d array containing each
        parent's pedigree entry explicitly.
        """
        self.arr = np.zeros((len(self.pair_ids), 2, Struc.n_cols))
        females, males = generation.split_gen()
        self.arr[:, 0, :] = females[self.pair_ids[:, 0]]
        self.arr[:, 1, :] = males[self.pair_ids[:, 1]]

    def get_zygotes(self):
        zygotes = np.zeros((self.N, Struc.n_alleles), dtype = Struc.dtype)
        zygotes[:, Struc.mat_allele_positions] = self.get_gametes(0)
        zygotes[:, Struc.pat_allele_positions] = self.get_gametes(1)
        return zygotes

    def get_gametes(self, sex):
        idx0 = np.arange(self.N)
        A_idx = np.random.randint(0, 2, size = self.N) + Struc.A_loc0
        B_idx = np.random.randint(0, 2, size = self.N) + Struc.B_loc0
        gametes = np.zeros((self.N, 2), dtype = Struc.dtype)
        gametes[:, 0] = self.arr[idx0, sex, A_idx]
        gametes[:, 1] = self.arr[idx0, sex, B_idx]
        return gametes

    def get_parental_ids(self):
        return self.arr[:, :, Struc.i]

    def get_maternal_x(self):
        return self.arr[:, 0, Struc.x]

    def get_N(self):
        return self.N


class Pedigree:

    organism_axis = 0
    character_axis = 1

    def __init__(self, arr, params, max = None):
        self.arr = arr
        self.params = params
        self.g = params.g
        self.t = params.g
        self.K = params.K
        if max: self.max = max

    @classmethod
    def new(cls, params):
        max = int(params.K * (params.g + 1) * Struc.adjust_fac)
        arr = np.zeros((max, Struc.n_cols), dtype=Struc.dtype)
        return cls(arr, params, max)

    def load(self, filename):
        pass

    def enter_gen(self, generation, i0, i1):
        if i1 > self.max:
            self.expand()
        self.insert_gen(generation, i0, i1)
        self.t = generation.t

    def insert_gen(self, generation, i0, i1):
        self.arr[i0:i1, :] = generation.arr

    def expand(self):
        """expand the pedigree to accommodate more organisms"""
        g_elapsed = self.g - self.t
        avg_util = int(self.max / g_elapsed)
        new_max = int(avg_util * self.t * Struc.adjust_fac)
        aux_arr = np.zeros((new_max, Struc.n_cols), dtype=Struc.dtype)
        self.arr = np.vstack((self.arr, aux_arr))
        self.max += new_max
        print("pedigree arr expanded " + str(new_max)
              + " rows at est. utilization " + str(avg_util))

    def trim(self, i1):
        """trim the pop array of unfilled rows"""
        self.arr = self.arr[:i1]
        excess = self.max - i1
        frac = np.round((1 - excess / self.max) * 100, 2)
        print("pedigree trimmed of " + str(excess) + " excess rows, "
              + str(frac) + "% utilization")

    def get_gen_arr(self, t):
        return self.arr[np.where(self.arr[:, Struc.t] == t)[0]]

    def get_generation(self, t):
        """Return an array of organisms in a given gen"""
        gen = self.get_gen_arr(t)
        N = len(gen)
        generation = Generation(N, gen[:, Struc.x], t, gen[:, Struc.parents],
                                gen[:, Struc.alleles], self.params)
        generation.arr[:, Struc.i] = gen[:, Struc.i]
        return generation

    def get_total_N(self):
        return np.shape(self.arr)[0]

    def get_gen_N(self, t):
        return np.size(np.where(self.arr[:, Struc.t] == t)[0])

    def get_gen_idx(self, t):
        return np.where(self.arr[:, Struc.t] == t)[0]

    def compute_ancestry(self, t = 0):
        N = self.get_total_N()
        anc = np.zeros((N, 3), dtype=np.float32)
        parents = self.arr[:, Struc.parents].astype(np.int32)
        founders = self.get_gen_idx(self.g)
        anc[founders, :2] = parents[founders]
        anc[founders, 2] = self.arr[founders, Struc.A_loc0] - 1
        for i in np.arange(self.g - 1, t - 1, -1):
            gen_idx = self.get_gen_idx(i)
            gen_parents = parents[gen_idx]
            anc[gen_idx, :2] = gen_parents
            anc[gen_idx, 2] = np.mean([anc[gen_parents[:, 0], 2],
                                       anc[gen_parents[:, 1], 2]], axis=0)
        ancs = anc[gen_idx, 2]
        return ancs


class AbbrevPedigree:
    """Abbreviated to only the bare minimum information to run coalescence
    simulations. Id matches row index and is therefore strictly unnecessary
    """
    organism_axis = 0
    character_axis = 1
    dtype = np.int32
    n_cols = 4
    t = 0
    mat_id = 1
    pat_id = 2
    subpop_code = 4
    map = np.array([3, 4, 5])

    def __init__(self, arr, params, max = None):
        self.arr = arr
        self.params = params
        self.g = params.g
        self.t = params.g
        self.K = params.K
        self.founding_gen = None
        self.last_gen = None
        if max:
            self.max = max

    @classmethod
    def new(cls, params):
        max = int(params.K * (params.g + 1) * Struc.adjust_fac)
        arr = np.zeros((max, cls.n_cols), dtype=cls.dtype)
        return cls(arr, params, max)

    @classmethod
    def load(cls, filename):
        pass

    def save_first_gen(self, generation):
        """Save the founding generation so that the initial state is known"""
        self.founding_gen = generation

    def enter_gen(self, generation, i0, i1):
        if i1 > self.max:
            self.expand()
        self.insert_gen(generation, i0, i1)
        self.t = generation.t

    def insert_gen(self, generation, i0, i1):
        self.arr[i0:i1, :3] = generation.arr[:, self.map]
        self.arr[i0:i1, 3] = generation.get_subpop_idx()

    def expand(self):
        """expand the pedigree to accommodate more organisms"""
        g_elapsed = self.g - self.t
        avg_util = int(self.max / g_elapsed)
        new_max = int(avg_util * self.t * Struc.adjust_fac)
        aux_arr = np.zeros((new_max, self.n_cols), dtype=Struc.dtype)
        self.arr = np.vstack((self.arr, aux_arr))
        self.max += new_max
        print("pedigree arr expanded " + str(new_max)
              + " rows at est. utilization " + str(avg_util))

    def save_last_gen(self, generation):
        """Save the final gen, so that spatial sampling etc. is possible"""
        self.last_gen = generation

    def trim(self, i1):
        """trim the pop array of unfilled rows"""
        self.arr = self.arr[:i1]
        excess = self.max - i1
        frac = np.round((1 - excess / self.max) * 100, 2)
        print("pedigree trimmed of " + str(excess) + " excess rows, "
              + str(frac) + "% utilization")


class Trial:

    def __init__(self, params, plot_int=None):
        self.time0 = time.time()
        self.t = params.g
        self.i0 = 0
        self.i1 = 0
        self.complete = False
        self.params = params
        self.time_vec = np.zeros(params.g + 1)
        self.report_int = max(min(100, self.params.g // 10), 1)
        self.plot_int = plot_int
        self.figs = []
        self.type = params.history_type
        if self.type == "Pedigree":
            self.pedigree = Pedigree.new(params)
            self.get_pedigree()
        elif self.type == "AbbrevPedigree":
            self.abbrev_pedigree = AbbrevPedigree.new(params)
            self.get_abbrev_pedigree()
        elif self.type == "SubpopArr":
            self.subpop_arr = SubpopArr.new()
            self.get_subpop_arr()

    def get_pedigree(self):
        print("simulation initiated @ " + self.get_time_string())
        generation = Generation.get_founding(params)
        i1 = generation.sort_and_id(self.i0)
        self.update_i(i1)
        generation.senescence()
        self.pedigree.enter_gen(generation, self.i0, self.i1)
        self.time_vec[self.params.g] = 0
        if self.plot_int:
            self.figs.append(generation.plot_subpops())
        while self.t > 0:
            generation.senescence()
            generation = self.cycle(generation)
            self.pedigree.enter_gen(generation, self.i0, self.i1)
        self.pedigree.trim(self.i1)
        if self.plot_int:
            for fig in self.figs:
                fig.show()
        print("simulation complete")

    def get_abbrev_pedigree(self):
        print("simulation initiated @ " + self.get_time_string())
        generation = Generation.get_founding(params)
        i1 = generation.sort_and_id(self.i0)
        self.update_i(i1)
        self.abbrev_pedigree.enter_gen(generation, self.i0, self.i1)
        self.abbrev_pedigree.save_first_gen(generation)
        self.time_vec[self.params.g] = 0
        if self.plot_int:
            self.figs.append(generation.plot_subpops())
        while self.t > 0:
            generation = self.cycle(generation)
            self.abbrev_pedigree.enter_gen(generation, self.i0, self.i1)
        self.abbrev_pedigree.save_last_gen(generation)
        self.abbrev_pedigree.trim(self.i1)
        if self.plot_int:
            for fig in self.figs:
                fig.show()
        print("simulation complete")

    def get_subpop_arr(self):
        print("simulation initiated @ " + self.get_time_string())
        generation = Generation.get_founding(params)
        generation.sort_and_id(self.i0)
        self.time_vec[self.params.g] = 0
        if self.plot_int:
            self.figs.append(generation.plot_subpops())
        while self.t > 0:
            self.subpop_arr.enter_generation(generation)
            generation = self.cycle(generation)
            if self.t == 0:
                self.complete = True
        self.subpop_arr.enter_generation(generation)
        if self.plot_int:
            for fig in self.figs:
                fig.show()
        print("simulation complete")

    def cycle(self, generation):
        self.update_t()
        old_generation = generation
        old_generation.remove_dead()
        generation = old_generation.mate(old_generation, self.params)
        generation.disperse(self.params)
        generation.fitness(self.params)
        i1 = generation.sort_and_id(self.i1)
        self.update_i(i1)
        self.report()
        if self.plot_int:
            if self.t % self.plot_int == 0:
                self.figs.append(generation.plot_subpops())
        return generation

    def update_i(self, i1):
        """Update the upper and lower indices to appropriate new values"""
        self.i0 = self.i1
        self.i1 = i1

    def update_t(self):
        self.t -= 1
        if self.t == 0:
            self.complete = True

    def report(self):
        self.time_vec[self.t] = time.time() - self.time0
        if self.t % self.report_int == 0:
            t = self.time_vec[self.t]
            t_last = self.time_vec[self.t + self.report_int]
            mean_t = str(np.round((t - t_last) / self.report_int, 3))
            run_t = str(np.round(self.time_vec[self.t], 2))
            time_string = self.get_time_string()
            print(f"g{self.t : > 6} complete, runtime = {run_t : >8}"
                  + f" s, averaging {mean_t : >8} s/gen, @ {time_string :>8}")

    @staticmethod
    def get_time_string():
        return str(time.strftime("%H:%M:%S", time.localtime()))


class GenSubpopArr:
    space_axis = 0
    subpop_axis = 1

    def __init__(self, generation):
        self.arr = np.zeros((Const.n_bins, Const.n_subpops), dtype=np.int32)
        x = generation.get_x()
        subpop_idx = generation.get_subpop_idx()
        for i in np.arange(9):
            self.arr[:, i] = np.histogram(x[subpop_idx == i], bins=Const.bins)[0]
        self.params = generation.params
        self.t = generation.t
        self.N = generation.get_N()

    def get_N_vec(self):
        """Return a vector of whole-population bin densities"""
        return np.sum(self.arr, axis=1)

    def get_subpop_N_vec(self):
        return np.sum(self.arr, axis=0)

    def get_N_hyb_vec(self):
        return np.sum(self.arr[:, 1:8], axis=1)

    def density_plot(self):
        fig = plt.figure(figsize=Const.plot_size)
        sub = fig.add_subplot(111)
        b = Const.bins_mid
        N_vec = self.get_N_vec()
        sub.plot(b, N_vec, color="black", linestyle='dashed')
        sub.plot(b, self.get_N_hyb_vec(), color='green', linestyle='dashed')
        c = Const.subpop_colors
        for i in np.arange(9):
            sub.plot(b, self.arr[:, i], color=c[i], linewidth=2)
        ymax = self.params.K * 1.3 * Const.bin_size
        title = "t = " + str(self.t) + " N = " + str(self.N)
        sub = setup_space_plot(sub, ymax, "subpop density", title)
        plt.legend(["N", "Hyb"] + Const.subpop_legend, fontsize=8,
                   bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
        fig.show()
        return fig


class SubpopArr:
    time_axis = 0
    space_axis = 1
    subpop_axis = 2

    def __init__(self, arr, params):
        self.arr = arr
        self.params = params
        self.t_vec = np.arange(np.shape(arr)[0])
        self.g = params.g

    @classmethod
    def new(cls, params):
        length = params.g + 1
        arr = np.zeros((length, Const.n_bins, Const.n_subpops), dtype=np.int32)
        return cls(arr, params)

    @classmethod
    def from_pedigree(cls, pedigree):
        t_vec = np.arange(pedigree.g + 1)
        t_len = pedigree.g + 1
        arr = np.zeros((t_len, Const.n_bins, Const.n_subpops), dtype=np.int32)
        for t in t_vec:
            generation = pedigree.get_generation(t)
            arr[t, :, :] = GenSubpopArr(generation).arr
        return cls(arr, pedigree.params)

    @classmethod
    def load(cls, filename):
        pass

    def enter_generation(self, generation):
        t = generation.get_t()
        self.arr[t, :, :] = GenSubpopArr(generation).arr


class GenAlleleArr:
    space_axis = 0
    locus_axis = 1
    allele_axis = 2

    def __init__(self, arr):
        self.arr = arr

    @classmethod
    def from_generation(cls, generation):
        x = generation.get_x()
        b = Const.bins
        alleles = generation.get_alleles()
        loci = np.array([[0, 1], [0, 1], [2, 3], [2, 3]])
        arr = np.zeros((Const.n_bins, 2, 2))
        for i in np.arange(4):
            j, k = np.unravel_index(i, (2, 2))
            a = i % 2 + 1
            arr[:, j, k] = (np.histogram(x[alleles[:, loci[i, 0]] == a], b)[0]
                          + np.histogram(x[alleles[:, loci[i, 1]] == a], b)[0])
        return cls(arr)

    @classmethod
    def from_subpop_arr(cls, subpoparr):
        factor = Const.allele_manifold
        arr = np.sum(subpoparr.arr[:, :, None, None] * factor, axis = 1)
        return cls(arr)

    def get_freq(self):
        N = np.sum(np.sum(self.arr, axis=2), axis=1) / 2
        return self.arr / N[:, None, None]

    def plot_freq(self):
        pass


class AlleleArr:
    time_axis = 0
    space_axis = 1
    locus_axis = 2
    allele_axis = 3

    def __init__(self, arr):
        self.arr = arr

    @classmethod
    def from_pedigree(cls, pedigree):
        t_len = pedigree.g + 1
        t_vec = np.arange(t_len)
        arr = np.zeros((t_len, Const.n_bins, 2, 2), dtype=np.int32)
        for t in t_vec:
            generation = pedigree.get_generation(t)
            arr[t, :, :, :] = GenAlleleArr.from_generation(generation).arr
        return cls(arr)

    @classmethod
    def from_subpoparr(cls, pedigree):
        t_len = pedigree.g + 1
        t_vec = np.arange(t_len)
        arr = np.zeros((t_len, Const.n_bins, 2, 2), dtype=np.int32)
        for t in t_vec:
            generation = pedigree.get_generation(t)
            arr[t, :, :, :] = GenAlleleArr.from_generation(generation).arr
        return cls(arr)


def setup_space_plot(sub, ymax, ylabel, title):
    sub.set_xticks(np.arange(0, 1.1, 0.1))
    sub.set_xlabel("x coordinate")
    sub.set_ylabel(ylabel)
    sub.set_ylim(-0.01, ymax)
    sub.set_xlim(-0.01, 1.01)
    sub.set_title(title)
    return sub


def plot_ancestry(pedigree):
    fig = plt.figure(figsize=Const.plot_size)
    sub = fig.add_subplot(111)
    x = pedigree[thisgen, PedStruc.Col.x]
    sub.plot(x, ancs, color="black", linewidth=2)
    sub.set_xlim(-0.01, 1.01), sub.set_ylim(-0.01, 1.01)
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.title("Ancestry coefficients after " + str(int(G -g)) + " generations")


params = parameters.Params(10000, 10, 0.1)
gen = Generation.get_founding(params)
