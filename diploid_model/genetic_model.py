import numpy as np

# import networkx as nx

import matplotlib.pyplot as plt

import msprime

import tskit

from diploid import pop_model


def get_sample_ranges(params):
    n_bins = params.n_sample_bins
    sample_ranges = np.zeros((n_bins, 2))
    sample_ranges[:, :] = np.arange(0, 1, 1 / n_bins)[:, None]
    sample_ranges[:, 1] += 1 / n_bins
    return sample_ranges


class SamplePedigree(pop_model.PedigreeLike):
    factor = 0.7

    def __init__(self, pedigree, params):
        self.max = int(pedigree.get_total_N() * self.factor)
        arr = np.zeros((self.max, self.n_cols))
        super().__init__(arr, params)
        self.g0 = 0
        self.g = params.g
        self.t = self.g
        self.k0 = 0
        self.k1 = 0
        self.sample_n = params.sample_n
        self.n_bins = params.n_sample_bins
        self.sample_bins = np.linspace(0, 1, self.n_bins + 1)
        self.sample_ranges = get_sample_ranges(params)
        self.get_sample(pedigree)
        self.n_organisms = self.get_n_organisms()

    @classmethod
    def from_trial(cls, trial):
        return cls(trial.pedigree, trial.params)

    def get_sample(self, pedigree):
        """Set up n_bins even spatial bins and sample n organisms from each
        of them. Then sample their entire lineages.

        The pedigree is pre-initialized to improve performance at very large
        population sizes or long times.
        """
        gen_sample = self.sample_gen_0(pedigree)
        self.k1 = len(gen_sample)
        self.arr[self.k0:self.k1] = gen_sample
        t_vec = np.arange(self.g0, self.g)
        for t in t_vec:
            self.t = t
            old_sample = gen_sample
            parent_idx = np.unique(old_sample[:, self._parents]).astype(
                np.int32)
            parent_idx = np.delete(parent_idx, np.where(parent_idx[:] == -1))
            gen_sample = pedigree.arr[parent_idx]
            self.k0 = self.k1
            self.k1 += len(gen_sample)
            self.enter_sample(gen_sample)
        self.trim_pedigree()
        self.sort_by_id()
        old_ids = self.arr[:, self._i].astype(np.int32)
        new_ids = np.arange(self.get_n_organisms(), dtype=np.int32)
        self.arr[:, self._i] = new_ids
        for i in [4, 5]:
            self.arr[self.arr[:, i] != -1, i] = self.remap_ids(
                self.arr[:, i], old_ids, new_ids)
        print("Pedigree sampling complete")

    def sample_gen_0(self, pedigree):
        """Sample n organisms in n_sample_bins bins from the most recent
        generation of a Pedigree instance
        """
        last_g = np.min(pedigree.arr[:, self._t])
        generation_0 = pedigree.get_generation(last_g)
        x = generation_0.get_x()
        sample_ids = []
        for range in self.sample_ranges:
            idx = np.where((x > range[0]) & (x < range[1]))[0]
            sample_idx = np.random.choice(idx, self.sample_n, replace=False)
            sample_ids.append(sample_idx)
        sample_ids = np.concatenate(sample_ids)
        gen_0_sample = generation_0.arr[sample_ids]
        return gen_0_sample

    def enter_sample(self, gen_sample):
        """Enter a generation's sample into the sample array"""
        if self.k1 < self.max:
            self.arr[self.k0:self.k1] = gen_sample
        else:
            self.extend_pedigree()
            self.arr[self.k0:self.k1] = gen_sample

    def extend_pedigree(self):
        """Extend a pedigree array to accommodate more individuals"""

        ### check!!! implementation may be broken
        factor = 0.8
        g_complete = self.g - self.t
        utilization = int(self.max / self.t)
        new_max = int(utilization * (self.g + 1) * factor)
        aux = np.zeros((new_max, self.n_cols))
        self.arr = np.vstack((self.arr, aux))
        self.max = new_max
        print(f"pedigree expanded {new_max} rows at est. utilization "
              f"utilization")

    def trim_pedigree(self):
        """Eliminate excess rows from a pedigree array"""
        self.arr = self.arr[0:self.k1]
        excess = self.max - self.k1
        frac = np.round((1 - excess / self.max) * 100, 2)
        print("pedigree trimmed of " + str(excess) + " excess rows, "
              + str(frac) + "% utilization")

    def sort_by_id(self):
        """Sort the sample pedigree array by id"""
        self.arr = self.arr[self.arr[:, self._i].argsort()]

    @staticmethod
    def remap_ids(vec, old_ids, new_ids):
        """Remap the ids of individuals in the pedigree from an increasing
        integer vector with gaps to an increasing integer vector without
        gaps
        """
        idx = np.searchsorted(old_ids, vec[vec != -1])
        return new_ids[idx]

    def get_tc(self, get_metadata=True):
        """Build a pedigree table collection from the sample pedigree array
        """
        if self.params.demographic_model == "onepop":
            demog = make_onepop_demog(self.params)
        elif self.params.demographic_model == "threepop":
            demog = make_threepop_demog(self.params)
        ped_tc = msprime.PedigreeBuilder(demography=demog,
                                         individuals_metadata_schema=
                                         tskit.MetadataSchema.permissive_json()
                                         )
        # for the individuals table
        ind_flags = np.zeros(self.n_organisms, dtype=np.uint32)
        n = self.get_n_organisms()
        x = self.get_x()
        x_offsets = np.arange(n + 1, dtype=np.uint32)
        parents = np.ravel(self.get_parents()).astype(np.int32)
        parents_offset = np.arange(n + 1, dtype=np.uint32) * 2
        # sex = self.get_sex().astype(int)
        # genotype_idx = self.get_subpop_idx().astype(int)
        # METADATA
        if get_metadata:
            # getting dicts in a vectorized way?
            # metadata_column = [{"sex": sex[i], "genotype": genotype_idx[i]}
            #                   for i in np.arange(n)]
            metadata_column = [
                {"sex": ind[0], "genotype": (ind[6], ind[7], ind[8], ind[9])}
                for ind in self.arr]
            encoded_metadata = [
                ped_tc.individuals.metadata_schema.validate_and_encode_row(row)
                for row in metadata_column]
            metadata, metadata_offset = tskit.pack_bytes(encoded_metadata)
            ped_tc.individuals.append_columns(
                flags=ind_flags,
                location=x,
                location_offset=x_offsets,
                parents=parents,
                parents_offset=parents_offset,
                metadata=metadata,
                metadata_offset=metadata_offset)
        else:
            ped_tc.individuals.append_columns(
                flags=ind_flags,
                location=x,
                location_offset=x_offsets,
                parents=parents,
                parents_offset=parents_offset)
        # for the nodes table
        ind_flags[self.arr[:, self._t] == 0] = tskit.NODE_IS_SAMPLE
        node_flags = np.repeat(ind_flags, 2)
        node_times = np.repeat(self.get_t(), 2).astype(np.float64)
        ind_pops = np.zeros(n, dtype=np.int32)
        if self.params.demographic_model == "onepop":
            pass
        elif self.params.demographic_model == "threepop":
            idx1 = np.where((self.arr[:, self._mat_id] == -1)
                            & (self.arr[:, self._A_loc1] == 1))
            ind_pops[idx1] = 1
            idx2 = np.where((self.arr[:, self._mat_id] == -1)
                            & (self.arr[:, self._A_loc1] == 2))
            ind_pops[idx2] = 2
        node_pops = np.repeat(ind_pops, 2)
        node_inds = np.repeat(np.arange(n, dtype=np.int32), 2)
        ped_tc.nodes.append_columns(
            flags=node_flags,
            time=node_times,
            population=node_pops,
            individual=node_inds)
        ped_tc = ped_tc.finalise(sequence_length=self.params.seq_length)
        return ped_tc


class AbbrevSamplePedigree(pop_model.AbbrevPedigree):
    factor = 0.7

    def __init__(self, abbrev_pedigree, params):
        self.max = int(abbrev_pedigree.get_n_organisms() * self.factor)
        arr = np.zeros((self.max, self.n_cols))
        super().__init__(arr, params)
        self.params = params
        self.g0 = 0
        self.g = params.g
        self.t = self.g
        self.k0 = 0
        self.k1 = 0
        self.sample_ranges = get_sample_ranges(params)
        self.get_sample(abbrev_pedigree)

    @classmethod
    def from_trial(cls, trial):
        if not trial.abbrev_pedigree:
            raise Exception("Trial instance has no abbreviated pedigree!")
        return cls(trial.abbrev_pedigree, params)

    def get_sample(self, abbrev_pedigree):
        """Set up n_bins even spatial bins and sample n organisms from each
        of them. Then sample their entire lineages.

        The pedigree is pre-initialized to improve performance at very large
        population sizes or long times.
        """
        gen_sample = self.sample_gen_0(abbrev_pedigree)
        self.k1 = len(gen_sample)
        self.arr[self.k0:self.k1] = gen_sample
        t_vec = np.arange(self.g0, self.g)
        for t in t_vec:
            self.t = t
            old_sample = gen_sample
            parent_idx = np.unique(old_sample[:, self._parents]).astype(
                np.int32)
            parent_idx = np.delete(parent_idx, np.where(parent_idx[:] == -1))
            gen_sample = abbrev_pedigree.arr[parent_idx]
            self.k0 = self.k1
            self.k1 += len(gen_sample)
            self.enter_sample(gen_sample)
        self.trim_pedigree()
        self.sort_by_id()
        ### needs work
        old_ids = np.arange(self.k1, dtype=np.int32)
        new_ids = np.arange(len(self), dtype=np.int32)
        # id'd by position
        # self.arr[:, self._i] = new_ids
        for i in [4, 5]:
            self.arr[self.arr[:, i] != -1, i] = self.remap_ids(
                self.arr[:, i], old_ids, new_ids)
        print("Pedigree sampling complete")

    def sample_gen_0(self, abbrev_pedigree):
        """Sample n organisms in n_sample_bins bins from the most recent
        generation of a Pedigree instance
        """
        generation_0 = abbrev_pedigree.last_gen
        x = generation_0.get_x()
        gen_sample_ids = []
        for zone in self.sample_ranges:
            idx = np.where((x > zone[0]) & (x < zone[1]))[0]
            sample_idx = np.random.choice(idx, self.params.sample_n,
                                          replace=False)
            gen_sample_ids.append(sample_idx)
        gen_sample_ids = np.concatenate(gen_sample_ids)
        # sample_ids are indices in generation_0. translate to the whole arr.
        sample_ids = gen_sample_ids + abbrev_pedigree.get_min_gen_0_id()
        gen_0_sample = abbrev_pedigree.arr[sample_ids]
        return gen_0_sample

    def enter_sample(self, gen_sample):
        """Enter a generation's sample into the sample array"""
        if self.k1 < self.max:
            self.arr[self.k0:self.k1] = gen_sample
        else:
            self.extend_pedigree()
            self.arr[self.k0:self.k1] = gen_sample

    def extend_pedigree(self):
        """Extend a pedigree array to accommodate more individuals"""

        ### check!!! implementation may be broken
        factor = 0.8
        g_complete = self.g - self.t
        utilization = int(self.max / self.t)
        new_max = int(utilization * (self.g + 1) * factor)
        aux = np.zeros((new_max, self.n_cols))
        self.arr = np.vstack((self.arr, aux))
        self.max = new_max
        print(f"pedigree expanded {new_max} rows at est. utilization "
              f"utilization")

    def trim_pedigree(self):
        """Eliminate excess rows from a pedigree array"""
        self.arr = self.arr[0:self.k1]
        excess = self.max - self.k1
        frac = np.round((1 - excess / self.max) * 100, 2)
        print("pedigree trimmed of " + str(excess) + " excess rows, "
              + str(frac) + "% utilization")

    def sort_by_id(self):
        """Sort the sample pedigree array by id"""
        self.arr = self.arr[self.arr[:, self._i].argsort()]

    @staticmethod
    def remap_ids(vec, old_ids, new_ids):
        """Remap the ids of individuals in the pedigree from an increasing
        integer vector with gaps to an increasing integer vector without
        gaps
        """
        idx = np.searchsorted(old_ids, vec[vec != -1])
        return new_ids[idx]

    def get_tc(self, get_metadata=True):
        """Build a pedigree table collection from the sample pedigree array
        """
        if self.params.demographic_model == "onepop":
            demog = make_onepop_demog(self.params)
        elif self.params.demographic_model == "threepop":
            demog = make_threepop_demog(self.params)
        ped_tc = msprime.PedigreeBuilder(demography=demog,
                                         individuals_metadata_schema=
                                         tskit.MetadataSchema.permissive_json()
                                         )
        # for the individuals table
        ind_flags = np.zeros(self.n_organisms, dtype=np.uint32)
        n = self.get_n_organisms()
        x = self.get_x()
        x_offsets = np.arange(n + 1, dtype=np.uint32)
        parents = np.ravel(self.get_parents()).astype(np.int32)
        parents_offset = np.arange(n + 1, dtype=np.uint32) * 2
        # sex = self.get_sex().astype(int)
        # genotype_idx = self.get_subpop_idx().astype(int)
        # METADATA
        if get_metadata:
            # getting dicts in a vectorized way?
            # metadata_column = [{"sex": sex[i], "genotype": genotype_idx[i]}
            #                   for i in np.arange(n)]
            metadata_column = [
                {"sex": ind[0], "genotype": (ind[6], ind[7], ind[8], ind[9])}
                for ind in self.arr]
            encoded_metadata = [
                ped_tc.individuals.metadata_schema.validate_and_encode_row(row)
                for row in metadata_column]
            metadata, metadata_offset = tskit.pack_bytes(encoded_metadata)
            ped_tc.individuals.append_columns(
                flags=ind_flags,
                location=x,
                location_offset=x_offsets,
                parents=parents,
                parents_offset=parents_offset,
                metadata=metadata,
                metadata_offset=metadata_offset)
        else:
            ped_tc.individuals.append_columns(
                flags=ind_flags,
                location=x,
                location_offset=x_offsets,
                parents=parents,
                parents_offset=parents_offset)
        # for the nodes table
        ind_flags[self.arr[:, self._t] == 0] = tskit.NODE_IS_SAMPLE
        node_flags = np.repeat(ind_flags, 2)
        node_times = np.repeat(self.get_t(), 2).astype(np.float64)
        ind_pops = np.zeros(n, dtype=np.int32)
        if self.params.demographic_model == "onepop":
            pass
        elif self.params.demographic_model == "threepop":
            idx1 = np.where((self.arr[:, self._mat_id] == -1)
                            & (self.arr[:, self._A_loc1] == 1))
            ind_pops[idx1] = 1
            idx2 = np.where((self.arr[:, self._mat_id] == -1)
                            & (self.arr[:, self._A_loc1] == 2))
            ind_pops[idx2] = 2
        node_pops = np.repeat(ind_pops, 2)
        node_inds = np.repeat(np.arange(n, dtype=np.int32), 2)
        ped_tc.nodes.append_columns(
            flags=node_flags,
            time=node_times,
            population=node_pops,
            individual=node_inds)
        ped_tc = ped_tc.finalise(sequence_length=self.params.seq_length)
        return ped_tc



def make_onepop_demog(params):
    """Construct a basic demography with a single population"""
    demography = msprime.Demography()
    demography.add_population(name="pop0", initial_size=params.K)
    return demography


def make_threepop_demog(params):
    """Construct a basic demography with three populations

    All explicitly simulated individuals excluding the founding generation,
    including all samples, are members of population 'pop0'. The founding
    generation is partitioned into ancestral populations 'pop1' and 'pop2'
    for individuals of genotype [1, 1, 1, 1] and [2, 2, 2, 2] respectively.

    A symmetric migration rate exists between the ancestral populations pop1
    and pop2 to allow coalescence to occur
    """
    demography = msprime.Demography()
    demography.add_population(name="pop0", initial_size=params.K)
    demography.add_population(name="pop1", initial_size=params.K // 2)
    demography.add_population(name="pop2", initial_size=params.K // 2)
    demography.set_symmetric_migration_rate(["pop1", "pop2"], params.mig_rate)
    return demography


def explicit_coalescent(tc, params):
    """Execute a coalescence simulation over a pedigree table collection using
    the "fixed pedigree" msprime sim_ancestry model"""
    ts = msprime.sim_ancestry(initial_state=tc, model="fixed_pedigree",
                              recombination_rate=params.recombination_rate)
    return ts


def reconstructive_coalescent(ts0, params):
    """Continue an explicit coalescence simulation"""
    if params.demographic_model == "onepop":
        demography = make_onepop_demog(params)
    elif params.demographic_model == "threepop":
        demography = make_threepop_demog(params)
    ts = msprime.sim_ancestry(initial_state=ts0, demography=demography,
                              model="dtwf", recombination_rate=
                              params.recombination_rate)
    return ts


class SampleGen(pop_model.PedigreeLike):
    """Class of sample generations drawn from sample pedigrees"""

    def __init__(self, arr, params, t):
        super().__init__(arr, params)
        self.t = t

    @classmethod
    def from_sample_pedigree(cls, sample_pedigree, t):
        arr = sample_pedigree.arr[sample_pedigree.arr[:, cls._t] == t]
        return cls(arr, sample_pedigree.params, t)


class SampleSet:
    """Class to keep track of node/individual ids in sample pedigrees and in
    tskit tree structures.
    """

    def __init__(self, ind_ids, node_ids):
        self.ind_ids = ind_ids
        self.node_ids = node_ids
        self.n_ind_ids = np.size(ind_ids)

    def __repr__(self):
        # abuse of __repr__ but useful
        return f"SampleSet of {self.n_ind_ids} ind ids"

    def __len__(self):
        return self.n_ind_ids

    @classmethod
    def from_node_ids(cls, node_ids):
        ind_ids = cls.map_node_to_ind(node_ids)
        return cls(ind_ids, node_ids)

    @classmethod
    def from_ind_ids(cls, ind_ids):
        node_ids = cls.map_ind_to_node(ind_ids)
        return cls(ind_ids, node_ids)

    @staticmethod
    def map_node_to_ind(node_ids):
        """map a vector of table collection 'node' indices to table collection
        'individual' indices
        """
        ind_ids = np.array(list(set(node_ids // 2)))
        return ind_ids

    @staticmethod
    def map_ind_to_node(ind_ids):
        """transform a vector of indices of a table collection 'individuals'
        table into indices of a table collection 'nodes' table.
        """
        node_ids = np.repeat(ind_ids, 2, axis=0) * 2
        increment = np.ravel(np.repeat([[0, 1]], np.size(ind_ids), axis=0))
        node_ids += increment
        return node_ids


class MultiWindow:
    """The class of multi-window diversity and divergence analyses"""

    def __init__(self, params):
        self.params = params
        self.sample_ranges = get_sample_ranges(params)
        self.sample_sets = []
        self.subpop_sample_sets = []
        n_bins = params.n_sample_bins
        self.pi = np.zeros((params.n_windows, n_bins))
        self.pi_XY = np.zeros((params.n_windows, n_bins, n_bins))
        window_kb = self.params.seq_length / 1000
        self.Mb = self.params.seq_length * self.params.n_windows / 1e6
        print(f"sampling {self.params.n_windows} x {window_kb} " 
              f"kb regions for {self.Mb} Mb total")

    # write str, repr etc

    def get_sample_sets(self, sample_gen):
        x = sample_gen.get_x()
        ids = (sample_gen.get_ids()).astype(np.int32)
        for bin_ in self.sample_ranges:
            ind_ids = ids[np.where((x > bin_[0]) & (x <= bin_[1]))[0]]
            self.sample_sets.append(SampleSet.from_ind_ids(ind_ids))

    def get_subpop_sample_sets(self, sample_gen):
        """Return a list of sample sets"""
        x = sample_gen.get_x()
        ids = sample_gen.get_ids().astype(np.int32)
        subpop_idx = sample_gen.get_subpop_idx()
        for subpop in np.arange(9):
            subpop_sets = []
            for lims in self.sample_ranges:
                inds = ids[np.where((x > lims[0]) & (x <= lims[1])
                                    & (subpop_idx == subpop))[0]]
                subpop_sets.append(SampleSet.from_ind_ids(inds))
            self.subpop_sample_sets.append(subpop_sets)

    def get_diversities(self, ts, sample_sets):
        n_sets = len(sample_sets)
        pi = np.zeros(n_sets, dtype=np.float32)
        for i in np.arange(n_sets):
            pi[i] = ts.diversity(sample_sets=sample_sets[i].node_ids,
                                 mode="branch")
        pi *= self.params.u
        return pi

    def get_divergences(self, ts, sample_sets):
        n_sets = len(sample_sets)
        pi_XY = np.zeros((n_sets, n_sets), dtype=np.float32)
        for linear_idx in np.arange(np.square(n_sets)):
            idx = np.unravel_index(linear_idx, (n_sets, n_sets))
            pi_XY[idx] = ts.divergence(
                sample_sets=[sample_sets[idx[0]].node_ids,
                             sample_sets[idx[1]].node_ids,],
                mode="branch")
        pi_XY *= self.params.u
        return pi_XY


class RootedMultiWindow(MultiWindow):

    def __init__(self, sample_pedigree):
        """Perform a series of coalescence simulations over a single sample
        pedigree and return mean pi and piXY vectors

        Useful for simulating larger regions of the genome than would be practical
        using a single coalescence simulation with recombination.
        """
        super().__init__(params)
        tc = sample_pedigree.get_tc(get_metadata=False)
        gen_0_sample = SampleGen.from_sample_pedigree(sample_pedigree, t=0)
        self.get_sample_sets(gen_0_sample)
        for i in np.arange(self.params.n_windows):
            pi, pi_XY = self.get_window(tc)
            self.pi[i, :] = pi
            self.pi_XY[i, :, :] = pi_XY
            print(f"window {i} complete")
        self.mean_pi = np.mean(self.pi, axis=0)
        self.mean_pi_XY = np.mean(self.pi_XY, axis=0)

    def get_window(self, tc):
        ts = explicit_coalescent(tc, self.params)
        ts = reconstructive_coalescent(ts, self.params)
        pi = self.get_diversities(ts, self.sample_sets)
        pi_XY = self.get_divergences(ts, self.sample_sets)
        return pi, pi_XY


class SubpopMultiWindow(MultiWindow):
    """Performs coalescence simulations and returns diversity and divergence
    values within/between spatial bin/subpopulation sample sets
    """

    def __init__(self, sample_pedigree):
        super().__init__(sample_pedigree.params)
        tc = sample_pedigree.get_tc(get_metadata=False)
        gen_0_sample = SampleGen.from_sample_pedigree(sample_pedigree, t=0)
        self.get_subpop_sample_sets(gen_0_sample)
        # overwrites
        n_bins = params.n_sample_bins
        n_subpops = pop_model.PedigreeLike.n_subpops
        self.pi = np.zeros((params.n_windows, n_subpops, n_bins))
        for i in np.arange(self.params.n_windows):
            pi = self.get_window(tc)
            self.pi[i, :, :] = pi
            print(f"window {i} complete")
        self.mean_pi = np.mean(self.pi, axis=0)

    def get_window(self, tc):
        ts = explicit_coalescent(tc, self.params)
        ts = reconstructive_coalescent(ts, self.params)
        n_subpops = pop_model.PedigreeLike.n_subpops
        pi = np.zeros((n_subpops, params.n_sample_bins))
        for subpop in np.arange(n_subpops):
            subpop_set = self.subpop_sample_sets[subpop]
            lengths = np.array([len(s) for s in subpop_set])
            idx = np.where(lengths > 0)[0]
            nonzero_subpop_sets = []
            for i in idx:
                nonzero_subpop_sets.append(subpop_set[i])
            pi[subpop, idx] = self.get_diversities(ts, nonzero_subpop_sets)
        return pi


class UnrootedMultiWindow(MultiWindow):

    def __init__(self, sample_pedigree):
        super().__init__(sample_pedigree.params)
        tc = sample_pedigree.get_tc(get_metadata=False)
        gen_0_sample = SampleGen.from_sample_pedigree(sample_pedigree, t=0)
        self.get_sample_sets(gen_0_sample)
        for i in np.arange(self.params.n_windows):
            pi, pi_XY = self.get_window(tc)
            self.pi[i, :] = pi
            self.pi_XY[i, :, :] = pi_XY
            print(f"window {i} complete")
        self.mean_pi = np.mean(self.pi, axis=0)
        self.mean_pi_XY = np.mean(self.pi_XY, axis=0)

    def get_window(self, tc):
        ts = explicit_coalescent(tc, self.params)
        pi = self.get_diversities(ts, self.sample_sets)
        pi_XY = self.get_divergences(ts, self.sample_sets)
        return pi, pi_XY


params = pop_model.Params(10_000, 30, 0.1)
trial = pop_model.Trial(params)
sample = SamplePedigree.from_trial(trial)
multi_window = RootedMultiWindow(sample)

"""
params = pop_model.Params(10_000, 30, 0.1)
params.history_type = "AbbrevPedigree"
trial = pop_model.Trial(params)
abbrev_pedigree = trial.abbrev_pedigree
sample = AbbrevSamplePedigree.from_trial(trial)
"""
"""
multi_window = MultiWindow(params)

sample_gen = SampleGen.from_sample_pedigree(sample, t=0)
multi_window.get_sample_sets(sample_gen)
sample_sets = multi_window.sample_sets
tc = sample.get_tc()
ts = explicit_coalescent(tc, params)
ts = reconstructive_coalescent(ts, params)
"""