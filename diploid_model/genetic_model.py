import numpy as np

#import networkx as nx

import tskit

import msprime

import tskit

from constants import Const

import diploid_pop_model as dip


class SamplePedigree(dip.PedigreeLike):

    factor = 0.7

    def __init__(self, pedigree, params):
        super().__init__()
        self.params = params
        self.g0 = 0
        self.g = params.g
        self.t = self.g
        self.max = int(pedigree.get_total_N() * self.factor)
        self.arr = np.zeros((self.max, self.n_cols))
        self.sample_n = params.sample_n
        self.n_bins = params.n_sample_bins
        self.sample_bins = np.linspace(0, 1, self.n_bins + 1)
        self.sample_ranges = np.zeros((self.n_bins, 2))
        self.sample_ranges[:, :] = np.arange(0, 1, 1 / self.n_bins)[:, None]
        self.sample_ranges[:, 1] += 1 / self.n_bins
        self.k0 = 0
        self.k1 = 0
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
        sample_ids = gen_sample[:, self._i]
        self.k1 = len(gen_sample)
        self.arr[self.k0:self.k1] = gen_sample
        t_vec = np.arange(self.g0, self.g)
        for t in t_vec:
            self.t = t
            old_sample = gen_sample
            parent_idx = np.unique(old_sample[:, self._parents]).astype(np.int32)
            parent_idx = np.delete(parent_idx, np.where(parent_idx[:] == -1))
            gen_sample = pedigree.arr[parent_idx]
            self.k0 = self.k1
            self.k1 += len(gen_sample)
            self.enter_sample(gen_sample)
        self.trim_pedigree()
        self.sort()
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

    def sort(self):
        """Sort the sample pedigree array by id"""
        self.arr = self.arr[self.arr[:, self._i].argsort()]

    def get_n_organisms(self):
        """Return the total number of organisms recorded in the pedigree"""
        return np.shape(self.arr)[0]

    @staticmethod
    def sort_arr(arr):
        """Sort a Struc-style array by x position"""
        x = arr[:, self._x]
        return arr[x.argsort()]

    @staticmethod
    def remap_ids(vec, old_ids, new_ids):
        """Remap the ids of individuals in the pedigree from an increasing
        integer vector with gaps to an increasing integer vector without
        gaps
        """
        idx = np.searchsorted(old_ids, vec[vec != -1])
        return new_ids[idx]

    def make_pedigree_tc(self, get_metadata=True):
        """Build a pedigree table collection from the sample pedigree array
        """
        if params.demog_model == "onepop":
            demog = make_onepop_demog(params)
        elif params.demog_model == "threepop":
            demog = make_threepop_demog(params)
        ped_tc = msprime.PedigreeBuilder(demography=demog,
            individuals_metadata_schema=tskit.MetadataSchema.permissive_json())
        # for the individuals table
        ind_flags = np.zeros(self.n_organisms, dtype=np.uint32)
        x_locs = pedigree[:, self.x]
        x_offsets = np.arange(N + 1, dtype=np.uint32)
        parents = np.ravel(pedigree[:, 4:6]).astype(np.int32)
        parents_offset = np.arange(N + 1, dtype=np.uint32) * 2
        # METADATA
        if get_metadata == True:
            metadata_column = [
                {"sex": ind[0], "genotype": (ind[6], ind[7], ind[8], ind[9])}
                for ind in pedigree
            ]
            encoded_metadata = [
                ped_tc.individuals.metadata_schema.validate_and_encode_row(row)
                for row in metadata_column
            ]
            metadata, metadata_offset = tskit.pack_bytes(encoded_metadata)
            ped_tc.individuals.append_columns(
                flags=ind_flags,
                location=x_locs,
                location_offset=x_offsets,
                parents=parents,
                parents_offset=parents_offset,
                metadata=metadata,
                metadata_offset=metadata_offset
            )
        else:
            ped_tc.individuals.append_columns(
                flags=ind_flags,
                location=x_locs,
                location_offset=x_offsets,
                parents=parents,
                parents_offset=parents_offset,
            )
        # for the nodes table
        ind_flags[pedigree[:, 3] == 0] = tskit.NODE_IS_SAMPLE
        node_flags = np.repeat(ind_flags, 2)
        node_times = np.repeat(pedigree[:, 3], 2).astype(np.float64)
        ind_pops = np.zeros(N, dtype=np.int32)
        if params.demog_model == "onepop":
            pass
        elif params.demog_model == "threepop":
            idx1 = np.where((pedigree[:, 4] == -1) & (pedigree[:, 6] == 1))
            ind_pops[idx1] = 1
            idx2 = np.where((pedigree[:, 4] == -1) & (pedigree[:, 6] == 2))
            ind_pops[idx2] = 2
        node_pops = np.repeat(ind_pops, 2)
        node_inds = np.repeat(np.arange(N, dtype=np.int32), 2)

        ped_tc.nodes.append_columns(
            flags=node_flags,
            time=node_times,
            population=node_pops,
            individual=node_inds
        )

        ped_tc = ped_tc.finalise(sequence_length=params.seq_length)
        return (ped_tc)


class AbbrevSamplePedigree:

    def __init__(self):
        pass


trial = dip.Trial(dip.params)
sample = SamplePedigree.from_trial(trial)
