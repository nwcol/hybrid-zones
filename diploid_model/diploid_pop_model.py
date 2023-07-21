import numpy as np

import time

import matplotlib.pyplot as plt

import parameters

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
    def get_migrants(cls):
        # todo
        pass

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

    def disperse(self, params):
        dispersal_fxn = eval(params.dispersal_fxn)
        delta = dispersal_fxn(self, params)
        edge_fxn = eval(params.edge_fxn)
        delta = edge_fxn(self, delta, params)
        self.arr[:, Struc.x] += delta

    def compute_signal_prop(self, limits):
        """Compute the proportion of same-signal males within a spatial limit for
        each male in a generation.
        """
        M = self.get_M()
        m_idx = self.get_m_idx()
        bounds = self.compute_sex_bounds(1, 1, limits)
        if limits[0] != 0 and limits[1] != 0:
            self_counting = True
        else:
            self_counting = False
        N_vec = bounds[:, 1] - bounds[:, 0]
        if self_counting == True:
            N_vec -= 1
        signal_sums = np.sum(self.arr[np.ix_(m_idx, Struc.A_loci)], axis=1)
        A1A1_idx = np.where(signal_sums == 2)[0]
        A1A2_idx = np.where(signal_sums == 3)[0]
        A2A2_idx = np.where(signal_sums == 4)[0]
        signal_prop = np.zeros(M, dtype=np.float32)
        signal_prop[A1A1_idx] = (
                    np.searchsorted(A1A1_idx, bounds[A1A1_idx, 1]) -
                    np.searchsorted(A1A1_idx, bounds[A1A1_idx, 0]))
        signal_prop[A2A2_idx] = (
                    np.searchsorted(A2A2_idx, bounds[A2A2_idx, 1]) -
                    np.searchsorted(A2A2_idx, bounds[A2A2_idx, 0]))
        if self_counting == True:
            signal_prop -= 1
        signal_prop /= N_vec
        signal_prop[np.isnan(signal_prop)] = 1
        return signal_prop, A1A2_idx

    def fitness(self, params):
        pass

    def remove_dead(self):
        pass

    def compute_bounds(self, seekingsex, targetsex, limits):
        """Compute bounds, given two arrays as arguments"""
        x_0 = seekingsex[:, Struc.x]
        x_1 = targetsex[:, Struc.x]
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
        x = self.get_x()
        # more efficient to create new arr?
        self.arr = self.arr[x.argsort()]

    def sort_and_index(self, i0):
        N = self.get_N()
        self.sort()
        ids = np.arange(N) + i0
        self.arr[:, Struc.i] = ids

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
        return np.where(self.arr[:, Struc.flag] == 1)[0].astype(np.int32)

    def senescence(self):
        self.flag = 0
        self.arr[self.get_living(), Struc.flag] = 0

    def plot_subpops(self):
        gen_subpop_arr = GenSubpopArr(self)
        fig = gen_subpop_arr.density_plot()
        return fig


class MatingPairs:

    i_axis = 0
    sex_axis = 1
    character_axis = 2

    def __init__(self, generation, params):
        self.pair_ids = self.compute_pair_ids(generation, params)
        self.get_mating_pairs(generation)

    def initialize_pair_ids(self, n_offspring):
        self.N = np.sum(n_offspring)
        mating_pair_ids = np.zeros((self.N, 2), dtype = np.int32)
        return mating_pair_ids

    def compute_pair_ids(self, generation, params):
        func = eval(params.mating_fxn)
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
                m_id = func(x, m_x_vec, pref_matrix[:, pref_vec_idx[f_id]],
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
        self.i0 = 0
        self.i1 = 0
        if max: self.max = max

    @classmethod
    def new(cls, params):
        max = int(params.K * (params.g + 1) * Struc.adjust_fac)
        arr = np.zeros((max, Struc.n_cols), dtype=Struc.dtype)
        return cls(arr, params, max)

    def load(self, filename):
        pass

    def enter_gen(self, generation):
        self.i1 += generation.get_N()
        if self.i1 > self.max:
            self.expand()
        self.insert_gen(generation)
        self.t = generation.t

    def insert_gen(self, generation):
        self.arr[self.i0:self.i1, :] = generation.arr
        self.i0 = self.i1

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

    def trim(self):
        """trim the pop array of unfilled rows"""
        self.arr = self.arr[:self.i1]
        excess = self.max - self.i1
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
    n_cols = 3
    t = 0
    mat_id = 1
    pat_id = 2
    map = np.array([3, 4, 5])

    def __init__(self, arr, params, max = None):
        self.arr = arr
        self.params = params
        self.g = params.g
        self.t = params.g
        self.K = params.K
        self.i0 = 0
        self.i1 = 0
        self.founding_gen = None
        self.last_gen = None
        if max: self.max = max

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

    def enter_gen(self, generation):
        self.i1 += generation.get_N()
        if self.i1 > self.max:
            self.expand()
        self.insert_gen(generation)
        self.t = generation.t

    def enter_gen(self, generation):
        self.i1 += generation.get_N()
        if self.i1 > self.max:
            self.expand()
        self.insert_gen(generation)
        self.t = generation.t

    def insert_gen(self, generation):
        self.arr[self.i0:self.i1, :] = generation.arr[:, map]
        self.i0 = self.i1

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

    def trim(self):
        """trim the pop array of unfilled rows"""
        self.arr = self.arr[:self.i1]
        excess = self.max - self.i1
        frac = np.round((1 - excess / self.max) * 100, 2)
        print("pedigree trimmed of " + str(excess) + " excess rows, "
              + str(frac) + "% utilization")


class Trial:

    def __init__(self, params, plot_int=None):
        self.time0 = time.time()
        self.t = params.g
        self.complete = False
        self.params = params
        self.time_vec = np.zeros(params.g + 1)
        self.report_int = max(min(100, self.params.g // 10), 1)
        self.plot_int = plot_int
        self.figs = []
        if params.history_type == "Pedigree":
            self.pedigree = Pedigree.new(params)
            self.get_pedigree()
        elif params.history_type == "AbbrevPedigree":
            self.abbrev_pedigree = AbbrevPedigree.new(params)
            self.get_abbrev_pedigree()
        elif params.history_type == "SubpopArr":
            self.subpop_arr = SubpopArr.new()
            self.get_subpop_arr()

    def get_pedigree(self):
        print("simulation initiated @ " + get_time_string())
        generation = Generation.get_founding(params)
        generation.sort_and_index(self.pedigree.i0)
        self.time_vec[self.params.g] = 0
        if self.plot_int:
            self.figs.append(generation.plot_subpops())
        while self.t > 0:
            generation.senescence()
            self.pedigree.enter_gen(generation)
            generation = self.cycle(generation)
            if self.t == 0:
                self.complete = True
        self.pedigree.enter_gen(generation)
        self.pedigree.trim()
        if self.plot_int:
            for fig in figs:
                fig.show()
        print("simulation complete")

    def get_abbrev_pedigree(self):
        print("simulation initiated @ " + get_time_string())
        generation = Generation.get_founding(params)
        generation.sort_and_index(self.abbrev_pedigree.i0)
        self.abbrev_pedigree.save_first_gen(generation)
        self.time_vec[self.params.g] = 0
        if self.plot_int:
            self.figs.append(generation.plot_subpops())
        while self.t > 0:
            self.abbrev_pedigree.enter_gen(generation)
            generation = self.cycle(generation)
            if self.t == 0:
                self.complete = True
        self.abbrev_pedigree.save_last_gen(generation)
        self.abbrev_pedigree.enter_gen(generation)
        self.abbrev_pedigree.trim()
        if self.plot_int:
            for fig in self.figs:
                fig.show()
        print("simulation complete")

    def get_subpop_arr(self):
        print("simulation initiated @ " + get_time_string())
        generation = Generation.get_founding(params)
        generation.sort_and_index(self.pedigree.i0)
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
        generation.remove_dead()
        old_generation = generation
        generation = old_generation.mate(old_generation, self.params)
        generation.disperse(self.params)
        generation.fitness(self.params)
        generation.sort_and_index(self.pedigree.i0)
        self.t -= 1
        self.report()
        if self.plot_int:
            if self.t % self.plot_int == 0:
                self.figs.append(generation.plot_subpops())
        return generation

    def report(self):
        self.time_vec[self.t] = time.time() - self.time0
        if self.t % self.report_int == 0:
            t = self.time_vec[self.t]
            t_last = self.time_vec[self.t + self.report_int]
            mean_t = str(np.round((t - t_last) / self.report_int, 3))
            run_t = str(np.round(self.time_vec[self.t], 2))
            time_string = get_time_string()
            print(f"g{self.t : > 6} complete, runtime = {run_t : >8}"
                  + f" s, averaging {mean_t : >8} s/gen, @ {time_string :>8}")


def plot_ancestry():
    fig = plt.figure(figsize=space_plotsize)
    sub = fig.add_subplot(111)
    x = pedigree[thisgen, PedStruc.Col.x]
    sub.plot(x, ancs, color="black", linewidth=2)
    sub.set_xlim(-0.01, 1.01), sub.set_ylim(-0.01, 1.01)
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.title("Ancestry coefficients after " + str(int(G -g)) + " generations")

def setup_space_plot(sub, ymax, ylabel, title):
    sub.set_xticks(np.arange(0, 1.1, 0.1))
    sub.set_xlabel("x coordinate")
    sub.set_ylabel(ylabel)
    sub.set_ylim(-0.01, ymax)
    sub.set_xlim(-0.01, 1.01)
    sub.set_title(title)
    return (sub)


def gaussian_mating(x, m_x_vec, pref_vec, bound, params):
    """A mating model where each female's chance to mate with a given male is
    weighted normally by distance and by the female's signal preference.
    """
    d_vec = x - m_x_vec[bound[0]:bound[1]]
    p_vec = compute_pd(d_vec, params.beta)
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


def uniform_mating(x, m_x_vec, pref_vec, bound, params):
    """A mating function where females pick a mate assortatively"""
    if bound[1] - bound[0] > 0:
        p_vec = np.cumsum(pref_vec[bound[0]:bound[1]])
        S = np.sum(p_vec)
        p_vec /= S
        cd = np.cumsum(p_vec)
        X = np.random.uniform()
        m_id = np.searchsorted(cd, X) + bound[0]
    else:
        m_id = -1
    return (m_id)


def unbounded_mating(x, m_x_vec, pref_vec, bound, params):
    """A mating function where females pick a mate randomly with
    assortation
    """
    d_vec = x - m_x_vec
    p_vec = compute_pd(d_vec, params.beta)
    p_vec *= pref_vec
    S = np.sum(p_vec)
    if S > 0:
        p_vec /= S
        cd = np.cumsum(p_vec)
        X = np.random.uniform()
        m_id = np.searchsorted(cd, X)
    else:
        m_id = -1
    return (m_id)


def compute_pd(x, s):
    """Compute a vector of probability density values for a normal distribution
    with standard deviation s, mean 0.
    """
    return 1 / (s * np.sqrt(2 * np.pi)) * np.exp(-0.5 * np.square(x / s))


def random_dispersal(generation, params):
    """Draw a set of displacements for the generation from a normal
    distribution with mean 0 and standard deviation delta.
    """
    N = generation.get_N()
    return np.random.normal(loc=0, scale=params.delta, size=N)


def scale_dispersal(gen, params, runlog):
    """Get a vector of dispersals using the scale model.

    Females sample dispersals from normal distributions with std params.delta.
    Males check the signal
    """
    N = get_N(gen)
    F = get_F(gen)
    M = get_M(gen)
    f_idx, m_idx = get_sex_indices(gen)
    males = get_males(gen)
    delta_x = np.zeros(N, dtype=np.float32)
    delta_x[f_idx] = np.random.normal(
        loc=0.0, scale=params.delta, size=F)
    prop, A1A2_idx = compute_signal_prop(males, [-params.bound, params.bound])
    std = scale_func(prop, params)
    std[A1A2_idx] = params.delta
    delta_x[m_idx] = np.random.normal(loc=0, scale=std, size=M)
    return (delta_x)


def scale_func(prop, params):
    """Compute standard deviations for male dispersals. Used by the
    scale_dispersal dispersal model.

    Arguments
    ------------
    prop : np array
        the vector of signal proportions. scale is a function of this vector

    params : Params class instance

    Returns
    ------------
    scale : np array
        the vector of male dispersal standard deviations
    """
    max_scale = params.d_scale
    scale = (1 - max_scale) * prop + max_scale
    scale *= params.delta
    return (scale)


def shift_dispersal(gen, params, runlog):
    """Get a vector of dispersals using the shift model.

    Females sample dispersals from normal distributions with std params.delta.
    Males check the signal
    """
    N = get_N(gen)
    F = get_F(gen)
    M = get_M(gen)
    f_idx, m_idx = get_sex_indices(gen)
    males = get_males(gen)
    delta_x = np.zeros(N, dtype=np.float32)
    delta_x[f_idx] = np.random.normal(
        loc=0.0, scale=params.delta, size=F
    )
    l_prop, A1A2_idx = compute_signal_prop(males, [-params.bound, 0])
    r_prop, A1A2_idx = compute_signal_prop(males, [0, params.bound])
    locs = loc_func(l_prop, r_prop, params)
    locs[A1A2_idx] = 0
    delta_x[m_idx] = np.random.normal(loc=locs, scale=params.delta, size=M)
    return (delta_x)


def loc_func(l_prop, r_prop, params):
    """

    Arguments
    ------------
    l_prop : TYPE
        DESCRIPTION.
    r_prop : TYPE
        DESCRIPTION.
    params : TYPE
        DESCRIPTION.

    Returns
    ------------
    None.

    """
    # nan errors
    m = params.d_scale  # slope
    dif = r_prop - l_prop
    loc = dif * m
    loc *= params.delta
    return (loc)


def static_reflect(generation, delta, params):
    """Set the displacement of any individual who would exit the space when
    displacements are applied to 0, freezing them in place
    """
    positions = delta + generation.get_x()
    delta[positions < 0] = 0
    delta[positions > 1] = 0
    return delta


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

    @classmethod
    def new(cls, params):
        arr = np.zeros((t_len, Const.n_bins, Const.n_subpops), dtype=np.int32)
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


params = parameters.Params(10000, 10, 0.1)
gen = Generation.get_founding(params)

"""
import cProfile


pr = cProfile.Profile()
pr.enable()
x = Trial(params)
pr.disable()
# after your program ends
pr.print_stats(sort="calls")
"""
