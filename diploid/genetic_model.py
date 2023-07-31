import numpy as np

# import networkx as nx

import matplotlib.pyplot as plt

import msprime

import tskit

import time

from diploid import pop_model


def get_sample_zones(params):
    """Get a 2d vector where each row specifies the bounds of a sample zone as
    defined in params
    """
    n_bins = params.n_sample_bins
    sample_zones = np.zeros((n_bins, 2))
    sample_zones[:, :] = np.arange(0, 1, 1 / n_bins)[:, None]
    sample_zones[:, 1] += 1 / n_bins
    return sample_zones


class SamplePedigree(pop_model.FullPedigree):
    factor = 0.62

    def __init__(self, pedigree, params):
        """Initialize pedigree sampling"""
        self.max = int(len(pedigree) * self.factor)
        arr = np.zeros((self.max, self.n_cols))
        super().__init__(arr, params)
        self.t_slice = self.get_t_slice()
        self.t = self.t_slice[0]
        self.i_slice = np.array([self.max, self.max], dtype=np.int32)
        self.sample_n = params.sample_n
        self.n_bins = params.n_sample_bins
        self.sample_bins = np.linspace(0, 1, self.n_bins + 1)
        self.sample_zones = get_sample_zones(params)
        self.get_sample(pedigree)

    @classmethod
    def from_trial(cls, trial):
        return cls(trial.pedigree, trial.params)

    def get_t_slice(self):
        """Return the continuous block of time defined by the time cutoffs
        parameter, or by 0 and the g parameter if cutoffs are not defined
        """
        lower = self.params.lower_t_cutoff
        upper = self.params.upper_t_cutoff
        if not lower:
            lower = 0
        if not upper:
            upper = self.params.g
        t_slice = np.arange(lower, upper + 1)
        return t_slice

    def get_sample(self, pedigree):
        """Set up n_bins even spatial bins and sample n organisms from each
        of them. Then sample their entire lineages.

        The pedigree is pre-initialized to improve performance at very large
        population sizes or long times, and is filled bottom-up to place the
        most ancient generation at the top.
        """
        sample = self.sample_final_gen(pedigree)
        self.i_slice[0] -= len(sample)
        self.arr[self.i_slice[0]:self.i_slice[1]] = sample
        for t in self.t_slice[1:]:
            self.t = t
            last_sample = sample
            parent_ids = np.unique(last_sample[:, self._parents]) \
                         .astype(np.int32)
            parent_ids = np.delete(parent_ids, np.where(parent_ids[:] == -1))
            sample = pedigree.arr[parent_ids]
            self.i_slice[1] = self.i_slice[0]
            self.i_slice[0] -= len(sample)
            self.enter_sample(sample)
        self.trim_pedigree()
        old_ids = self.arr[:, self._id].astype(np.int32)
        new_ids = np.arange(self.get_n_organisms(), dtype=np.int32)
        self.arr[:, self._id] = new_ids
        for i in [self._mat_id, self._pat_id]:
            self.arr[self.arr[:, i] != -1, i] = self.remap_ids(
                self.arr[:, i], old_ids, new_ids)
        print("Pedigree sampling complete")

    def sample_final_gen(self, pedigree):
        """Sample n organisms in n_sample_bins bins from the lowest (first)
        time defined in self.t_slice and sort them by id
        """
        final_gen = pedigree.get_generation(self.t_slice[0])
        x = final_gen.get_x()
        sample_ids = []
        for zone in self.sample_zones:
            idx = np.where((x > zone[0]) & (x < zone[1]))[0]
            sample_idx = np.random.choice(idx, self.sample_n, replace=False)
            sample_ids.append(sample_idx)
        sample_ids = np.concatenate(sample_ids)
        sample = final_gen.arr[sample_ids]
        sample = sample[sample[:, self._id].argsort()]
        return sample

    def enter_sample(self, gen_sample):
        """Enter a generation's sample into the sample array"""
        while self.i_slice[0] < 0:
            self.extend_pedigree()
        self.arr[self.i_slice[0]:self.i_slice[1]] = gen_sample

    def extend_pedigree(self):
        """Extend a pedigree array to accommodate more individuals"""
        length = self.t_slice[-1] - self.t_slice[0]
        t_left = self.t_slice[-1] - self.t + 1
        frac_left = t_left / length
        extra = int(self.max * frac_left)
        aux = np.zeros((extra, self.n_cols), dtype=np.int32)
        self.arr = np.vstack((aux, self.arr))
        self.i_slice += extra
        self.max += extra
        print(f"pedigree expanded {extra} rows at t = {self.t}")

    def trim_pedigree(self):
        """Eliminate excess rows from a pedigree array"""
        self.arr = self.arr[self.i_slice[0]:]
        excess = self.i_slice[0]
        frac = np.round((1 - excess / self.max) * 100, 2)
        print("pedigree trimmed of " + str(excess) + " excess rows, "
              + str(frac) + "% utilization")

    def sort_by_id(self):
        """Sort the sample pedigree array by id"""
        self.arr = self.arr[self.arr[:, self._id].argsort()]

    @staticmethod
    def remap_ids(vec, old_ids, new_ids):
        """Remap the ids of individuals in the pedigree from an increasing
        integer vector with gaps to an increasing integer vector without
        gaps
        """
        idx = np.searchsorted(old_ids, vec[vec != -1])
        return new_ids[idx]

    def get_generation_sizes(self):
        """Return the number of individuals in each generation of the sample
        """
        length = len(self.t_slice)
        sizes = np.zeros(length)
        for i, t in zip(np.arange(length), self.t_slice):
            sizes[i] = len(np.where(self.arr[:, self._t] == t)[0])
        return sizes

    def get_tc(self, get_metadata=False):
        """Build a pedigree table collection from the sample pedigree array
        """
        demography = demographic_models[self.params.demographic_model](
            self.params)
        ped_tc = msprime.PedigreeBuilder(demography=demography,
            individuals_metadata_schema=tskit.MetadataSchema.permissive_json())
        # INDS
        ind_flags = np.zeros(self.get_n_organisms(), dtype=np.uint32)
        n = self.get_n_organisms()
        x = self.get_x()
        x_offsets = np.arange(n + 1, dtype=np.uint32)
        parents = np.ravel(self.get_parents()).astype(np.int32)
        parents_offset = np.arange(n + 1, dtype=np.uint32) * 2
        if get_metadata:
            subpops = self.get_subpop_idx().astype(float)
            sex = self.get_sex()
            metadata_column = [{"sex": sex[i], "subpop": subpops[i]}
                               for i in np.arange(n)]
            encoded_metadata = [
                ped_tc.individuals.metadata_schema.validate_and_encode_row(row)
                for row in metadata_column]
            metadata, metadata_offset = tskit.pack_bytes(encoded_metadata)
            ped_tc.individuals.append_columns(flags=ind_flags, location=x,
                                              location_offset=x_offsets,
                                              parents=parents,
                                              parents_offset=parents_offset,
                                              metadata=metadata,
                                              metadata_offset=metadata_offset)
        else:
            ped_tc.individuals.append_columns(flags=ind_flags, location=x,
                                              location_offset=x_offsets,
                                              parents=parents,
                                              parents_offset=parents_offset)
        # NODES
        ind_flags[self.arr[:, self._t] == 0] = tskit.NODE_IS_SAMPLE
        node_flags = np.repeat(ind_flags, 2)
        node_times = np.repeat(self.get_t(), 2).astype(np.float64)
        pop_methods = {"three_pop": self.get_three_pop_inds,
                       "one_pop": self.get_one_pop_inds}
        ind_pops = pop_methods[self.params.demographic_model]()
        node_pops = np.repeat(ind_pops, 2)
        node_inds = np.repeat(np.arange(n, dtype=np.int32), 2)
        ped_tc.nodes.append_columns(
            flags=node_flags,
            time=node_times,
            population=node_pops,
            individual=node_inds)
        ped_tc = ped_tc.finalise(sequence_length=self.params.seq_length)
        return ped_tc

    def get_three_pop_inds(self):
        """Get populations under the 3-population demography regime"""
        n = len(self)
        ind_pops = np.zeros(n, dtype=np.int32)
        subpops = self.get_subpops()
        ind_pops[subpops == 1] = 1
        ind_pops[subpops == 8] = 2
        return ind_pops

    def get_one_pop_inds(self):
        n = len(self)
        ind_pops = np.zeros(n, dtype=np.int32)
        return ind_pops


class AbbrevSamplePedigree(pop_model.AbbrevPedigree):
    factor = 0.62

    def __init__(self, abbrev_pedigree, params):
        self.max = int(abbrev_pedigree.get_n_organisms() * self.factor)
        arr = np.zeros((self.max, self.n_cols), dtype=np.int32)
        super().__init__(arr, params)
        self.t_slice = self.get_t_slice()
        self.t = self.t_slice[0]
        self.i_slice = np.array([self.max, self.max])
        self.sample_zones = get_sample_zones(params)
        self.last_gen = abbrev_pedigree.last_gen
        self.founding_gen = abbrev_pedigree.founding_gen
        self.last_gen_x = None
        self.founding_gen_x = None
        self.get_sample(abbrev_pedigree)

    @classmethod
    def from_trial(cls, trial):
        if not trial.abbrev_pedigree:
            raise Exception("Trial instance has no abbreviated pedigree!")
        params = trial.params
        return cls(trial.abbrev_pedigree, params)

    def __len__(self):
        return len(self.arr)

    def get_t_slice(self):
        """Return the continuous block of time defined by the time cutoffs
        parameter, or by 0 and the g parameter if cutoffs are not defined

        abbreviated pedigrees do not support lower bounds as presently only the
        final generation is saved in its entirety
        """
        lower = 0
        upper = self.params.upper_t_cutoff
        if not upper:
            upper = self.params.g
        t_slice = np.arange(lower, upper + 1)
        return t_slice

    def get_sample(self, abbrev_pedigree):
        """Set up n_bins even spatial bins and sample n organisms from each
        of them. Then sample their entire lineages.

        The pedigree is pre-initialized to improve performance at very large
        population sizes or long times.
        """
        sample = self.sample_last_gen(abbrev_pedigree)
        self.i_slice[0] -= len(sample)
        self.arr[self.i_slice[0]:self.i_slice[1]] = sample
        for t in self.t_slice[1:]:
            self.t = t
            last_sample = sample
            parent_ids = np.unique(last_sample[:, self._parents])
            parent_ids = np.delete(parent_ids, np.where(parent_ids[:] == -1))
            sample = abbrev_pedigree.arr[parent_ids]
            self.i_slice[1] = self.i_slice[0]
            self.i_slice[0] -= len(sample)
            self.enter_sample(sample)
        self.founding_gen_x = self.founding_gen.get_x()[parent_ids]
        self.trim_pedigree()
        old_ids = self.arr[:, self._id].astype(np.int32)
        new_ids = np.arange(self.get_n_organisms(), dtype=np.int32)
        self.arr[:, self._id] = new_ids
        for i in [self._mat_id, self._pat_id]:
            self.arr[self.arr[:, i] != -1, i] = self.remap_ids(
                self.arr[:, i], old_ids, new_ids)
        print("Pedigree sampling complete")

    def sample_last_gen(self, abbrev_pedigree):
        """Sample n organisms in n_sample_bins bins from the most recent
        generation of a Pedigree instance. sample_ids are relative to the last
        generation and not the entire pedigree
        """
        last_gen = abbrev_pedigree.last_gen
        x = last_gen.get_x()
        sample_ids = []
        for zone in self.sample_zones:
            idx = np.where((x > zone[0]) & (x < zone[1]))[0]
            sample_idx = np.random.choice(idx, self.params.sample_n,
                                          replace=False)
            sample_ids.append(sample_idx)
        sample_ids = np.sort(np.concatenate(sample_ids))
        self.last_gen_ids = sample_ids
        self.last_gen_x = last_gen.get_x()[sample_ids]
        sample = abbrev_pedigree.get_gen_arr(0)[sample_ids]
        return sample

    def enter_sample(self, gen_sample):
        """Enter a generation's sample into the sample array"""
        while self.i_slice[0] < 0:
            self.extend_pedigree()
        self.arr[self.i_slice[0]:self.i_slice[1]] = gen_sample

    def extend_pedigree(self):
        """Extend a pedigree array to accommodate more individuals"""
        length = self.t_slice[-1] - self.t_slice[0]
        t_left = self.t_slice[-1] - self.t + 1
        frac_left = t_left / length
        extra = int(self.max * frac_left)
        aux = np.zeros((extra, self.n_cols), dtype=np.int32)
        self.arr = np.vstack((aux, self.arr))
        self.i_slice += extra
        self.max += extra
        print(f"pedigree expanded {extra} rows at t = {self.t}")

    def trim_pedigree(self):
        """Eliminate excess rows from a pedigree array"""
        self.arr = self.arr[self.i_slice[0]:]
        excess = self.i_slice[0]
        frac = np.round((1 - excess / self.max) * 100, 2)
        print("pedigree trimmed of " + str(excess) + " excess rows, "
              + str(frac) + "% utilization")

    def sort_by_id(self):
        """Sort the sample pedigree array by id"""
        self.arr = self.arr[self.arr[:, self._id].argsort()]

    @staticmethod
    def remap_ids(vec, old_ids, new_ids):
        """Remap the ids of individuals in the pedigree from an increasing
        integer vector with gaps to an increasing integer vector without
        gaps
        """
        idx = np.searchsorted(old_ids, vec[vec != -1])
        return new_ids[idx]

    def get_founding_ids(self):
        """Return the indexes of sample organisms in the founding generation
        """
        return np.where(self.arr[:, self._t] == self.t_slice[-1])

    def get_final_ids(self):
        """Return the indices of sample organisms in the final generation"""
        return np.where(self.arr[:, self._t] == self.t_slice[0])

    def get_generation_sizes(self):
        """Return the number of individuals in each generation of the sample
        """
        length = len(self.t_slice)
        sizes = np.zeros(length)
        for i, t in zip(np.arange(length), self.t_slice):
            sizes[i] = len(np.where(self.arr[:, self._t] == t)[0])
        return sizes

    def get_tc(self):
        """Build a pedigree table collection from the sample pedigree array
        """
        n = self.get_n_organisms()
        founder_ids = self.get_founding_ids()
        last_gen_ids = self.get_final_ids()
        demography = demographic_models[self.params.demographic_model](
            self.params)
        ped_tc = msprime.PedigreeBuilder(demography=demography,
            individuals_metadata_schema=tskit.MetadataSchema.permissive_json())
        # IND TABLE
        ind_flags = np.zeros(n, dtype=np.uint32)
        x = np.full(n, -1, dtype=np.float32)
        x[founder_ids] = self.founding_gen_x
        x[last_gen_ids] = self.last_gen_x
        x_offsets = np.arange(n + 1, dtype=np.uint32)
        parents = np.ravel(self.get_parents()).astype(np.int32)
        parents_offset = np.arange(n + 1, dtype=np.uint32) * 2
        subpops = self.get_subpops().astype(float)
        # METADATA
        metadata_column = [{"subpop": subpops[i]} for i in np.arange(n)]
        encoded_metadata = [
            ped_tc.individuals.metadata_schema.validate_and_encode_row(row)
            for row in metadata_column]
        metadata, metadata_offset = tskit.pack_bytes(encoded_metadata)
        ped_tc.individuals.append_columns(flags=ind_flags, location=x,
            location_offset=x_offsets, parents=parents,
            parents_offset=parents_offset, metadata=metadata,
            metadata_offset=metadata_offset)
        # NODES TABLE
        ind_flags[self.arr[:, self._t] == 0] = tskit.NODE_IS_SAMPLE
        node_flags = np.repeat(ind_flags, 2)
        node_times = np.repeat(self.get_t(), 2).astype(np.float64)
        pop_methods = {"three_pop": self.get_three_pop_inds,
                       "one_pop": self.get_one_pop_inds}
        ind_pops = pop_methods[self.params.demographic_model]()
        node_pops = np.repeat(ind_pops, 2)
        node_inds = np.repeat(np.arange(n, dtype=np.int32), 2)
        ped_tc.nodes.append_columns(flags=node_flags, time=node_times,
            population=node_pops, individual=node_inds)
        ped_tc = ped_tc.finalise(sequence_length=self.params.seq_length)
        return ped_tc

    def get_three_pop_inds(self):
        """Get populations under the 3-population demography regime"""
        n = len(self)
        ind_pops = np.zeros(n, dtype=np.int32)
        subpops = self.get_subpops()
        ind_pops[subpops == 1] = 1
        ind_pops[subpops == 8] = 2
        return ind_pops

    def get_one_pop_inds(self):
        n = len(self)
        ind_pops = np.zeros(n, dtype=np.int32)
        return ind_pops


def get_one_pop_demography(params):
    """Construct a basic demography with a single population"""
    demography = msprime.Demography()
    demography.add_population(name="pop0", initial_size=params.K)
    return demography


def get_three_pop_demography(params):
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


demographic_models = {"three_pop" : get_three_pop_demography,
                      "one_pop" : get_one_pop_demography}


def explicit_coalescent(tc, params):
    """Execute a coalescence simulation over a pedigree table collection using
    the "fixed pedigree" msprime sim_ancestry model"""
    ts = msprime.sim_ancestry(initial_state=tc, model="fixed_pedigree",
                              recombination_rate=params.recombination_rate)
    return ts


def reconstructive_coalescent(ts0, params):
    """Continue an explicit coalescence simulation"""
    demography = demographic_models[params.demographic_model](params)
    ts = msprime.sim_ancestry(initial_state=ts0, demography=demography,
                              model="dtwf", recombination_rate=
                              params.recombination_rate)
    return ts


class SampleGen(pop_model.FullPedigree):
    """A class to store arrays of sample organisms, eg organisms in the last
    generation who were sampled for coalescent simulation.

    Used to keep track of sample organism genotypes, positions etc
    """

    def __init__(self, arr, params, t):
        super().__init__(arr, params)
        self.t = t

    @classmethod
    def from_full_pedigree_sample(cls, sample_pedigree, t):
        arr = sample_pedigree.arr[sample_pedigree.arr[:,
                                  sample_pedigree._t] == t]
        params = sample_pedigree.params
        return cls(arr, params, t)

    @classmethod
    def from_abbrev_pedigree_sample(cls, sample_pedigree):
        gen_ids = sample_pedigree.last_gen_ids
        arr = sample_pedigree.last_gen[gen_ids]
        pedigree_arr = sample_pedigree.arr[sample_pedigree.arr[:,
                                  sample_pedigree._t] == 0]
        pedigree_ids = pedigree_arr[:, sample_pedigree._id]
        arr[:, cls._id] = pedigree_ids
        arr[:, cls._parents] = sample_pedigree.arr[
            np.ix_(pedigree_ids, sample_pedigree._parents)]
        params = sample_pedigree.params
        t = 0
        return cls(arr, params, t)


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
    """
    The class of multi-window diversity and divergence analyses
    """

    def __init__(self, params, last_gen_sample, tc):
        self.params = params
        self.sample_zones = get_sample_zones(params)
        self.sample_sets = []
        self.subpop_sample_sets = []
        self.pi = None
        self.pi_XY = None
        self.subpop_pi = None
        window_kb = self.params.seq_length / 1000
        self.Mb = self.params.seq_length * self.params.n_windows / 1e6
        print(f"sampling {self.params.n_windows} x {window_kb} " 
              f"kb regions for {self.Mb} Mb total")
        self.run(last_gen_sample, tc)

    @classmethod
    def from_full_pedigree(cls, full_pedigree_sample):
        last_gen = SampleGen.from_full_pedigree_sample(full_pedigree_sample,
                                                       t=0)
        tc = full_pedigree_sample.get_tc()
        params = full_pedigree_sample.get_tc()
        return cls(params, last_gen, tc)

    @classmethod
    def from_abbrev_pedigree(cls, abbrev_pedigree_sample):
        last_gen = SampleGen.from_abbrev_pedigree_sample(
            abbrev_pedigree_sample)
        tc = abbrev_pedigree_sample.get_tc()
        params = abbrev_pedigree_sample.params
        return cls(params, last_gen, tc)

    def run(self, last_gen_sample, tc):
        n_bins = self.params.n_sample_bins
        type = self.params.multiwindow_type
        if type == "subpops":
            self.get_subpop_sample_sets(last_gen_sample)
            n_subpops = pop_model.PedigreeLike.n_subpops
            self.subpop_pi = np.zeros((self.params.n_windows, n_subpops, n_bins))
        else:
            self.get_sample_sets(last_gen_sample)
            self.pi = np.zeros((self.params.n_windows, n_bins))
            self.pi_XY = np.zeros((self.params.n_windows, n_bins, n_bins))
        for i in np.arange(self.params.n_windows):
            if type == "unrooted":
                pi, pi_XY = self.get_unrooted_window(tc)
                self.pi[i, :] = pi
                self.pi_XY[i, :, :] = pi_XY
            elif type == "rooted":
                pi, pi_XY = self.get_rooted_window(tc)
                self.pi[i, :] = pi
                self.pi_XY[i, :, :] = pi_XY
            elif type == "subpops":
                pi = self.get_subpop_window(tc)
                self.subpop_pi[i, :, :] = pi
            print(f"window {i} complete @ " + get_time_string())
        print("Multi-window sampling complete @ " + get_time_string())

    def get_mean_pi(self):
        return np.mean(self.pi, axis=0)

    def get_mean_subpop_pi(self):
        return np.mean(self.subpop_pi, axis=0)

    def get_mean_pi_XY(self):
        return np.mean(self.pi_XY, axis=0)

    def get_sample_sets(self, sample_gen):
        x = sample_gen.get_x()
        ids = (sample_gen.get_ids()).astype(np.int32)
        for bin_ in self.sample_zones:
            ind_ids = ids[np.where((x > bin_[0]) & (x <= bin_[1]))[0]]
            self.sample_sets.append(SampleSet.from_ind_ids(ind_ids))

    def get_subpop_sample_sets(self, sample_gen):
        """
        Return a list of sample sets
        """
        x = sample_gen.get_x()
        ids = sample_gen.get_ids().astype(np.int32)
        subpop_idx = sample_gen.get_subpop_idx()
        for subpop in np.arange(9):
            subpop_sets = []
            for lims in self.sample_zones:
                inds = ids[np.where((x > lims[0]) & (x <= lims[1])
                                    & (subpop_idx == subpop))[0]]
                subpop_sets.append(SampleSet.from_ind_ids(inds))
            self.subpop_sample_sets.append(subpop_sets)

    def get_rooted_window(self, tc):
        """
        Perform explicit and reconstructive coalescence simulations and
        compute diversity over spatial sample sets
        """
        ts = explicit_coalescent(tc, self.params)
        ts = reconstructive_coalescent(ts, self.params)
        pi = self.get_diversities(ts, self.sample_sets)
        pi_XY = self.get_divergences(ts, self.sample_sets)
        return pi, pi_XY

    def get_subpop_window(self, tc):
        """
        Perform explicit and reconstructive coalescence simulations and
        compute diversity and divergence using subpopulation sample sets
        """
        ts = explicit_coalescent(tc, self.params)
        ts = reconstructive_coalescent(ts, self.params)
        n_subpops = pop_model.PedigreeLike.n_subpops
        pi = np.zeros((n_subpops, self.params.n_sample_bins))
        for subpop in np.arange(n_subpops):
            subpop_set = self.subpop_sample_sets[subpop]
            lengths = np.array([len(s) for s in subpop_set])
            idx = np.where(lengths > 0)[0]
            nonzero_subpop_sets = []
            for i in idx:
                nonzero_subpop_sets.append(subpop_set[i])
            pi[subpop, idx] = self.get_diversities(ts, nonzero_subpop_sets)
        return pi

    def get_unrooted_window(self, tc):
        """Simulate coalescence over only the explicit sample pedigree, leaving
        multiple roots, and compute diversities and divergences"""
        ts = explicit_coalescent(tc, self.params)
        pi = self.get_diversities(ts, self.sample_sets)
        pi_XY = self.get_divergences(ts, self.sample_sets)
        return pi, pi_XY

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

    def save_pi(self, filename):
        file = open(filename, 'w')
        header = str(vars(self.params))
        mean = self.get_mean_pi()
        np.savetxt(
            file, mean, delimiter=' ', newline='\n', header=header)
        file.close()

    def save_subpop_pi(self, filename):
        shape = (self.params.n_windows,
                 self.params.n_sample_bins * pop_model.PedigreeLike.n_subpops)
        mean = self.get_mean_subpop_pi()
        file = open(filename, 'w')
        header = str(vars(self.params))
        np.savetxt(file, mean, delimiter=' ', newline='\n', header=header)
        file.close()

    def save_pi_XY(self, filename):
        shape = (self.params.n_windows, self.params.n_sample_bins ** 2)
        mean = self.get_mean_pi_XY()
        file = open(filename, 'w')
        header = str(vars(self.params))
        np.savetxt(file, mean, delimiter=' ', newline='\n', header=header)
        file.close()

    def save_all(self, prefix, suffix):
        """
        Save each diversity/divergence array which has been created as a .txt
        file

        :param prefix: the parameter file name
        :paramtype prefix: string
        :param suffix: the cluster and process id
        "paramtype suffix: string
        """
        type = self.params.multiwindow_type
        if type == "unrooted":
            filename1 = prefix + "_pi_" + suffix + ".txt"
            self.save_pi(filename1)
            filename2 = prefix + "_pi_XY_" + suffix + ".txt"
            self.save_pi_XY(filename2)
        elif type == "rooted":
            filename1 = prefix + "_pi_" + suffix + ".txt"
            self.save_pi(filename1)
            filename2 = prefix + "_pi_XY_" + suffix + ".txt"
            self.save_pi_XY(filename2)
        elif type == "subpops":
            filename = prefix + "_subpop_pi_" + suffix + ".txt"
            self.save_subpop_pi(filename)


def get_time_string():
    return str(time.strftime("%H:%M:%S", time.localtime()))


#params = pop_model.Params(10_000, 20, 0.1)
#params.history_type = "AbbrevPedigree"
#trial = pop_model.Trial(params)
#sample = AbbrevSamplePedigree.from_trial(trial)

#windows = MultiWindow.from_abbrev_pedigree(sample)
#sample = SamplePedigree.from_trial(trial)
#tc = sample.get_tc()
