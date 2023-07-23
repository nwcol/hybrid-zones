import numpy as np

#import networkx as nx

import tskit

import msprime

import tskit

from constants import Const

import diploid_pop_model as dip


class SamplePedigree(dip.PedigreeType):

    factor = 0.7

    def __init__(self, pedigree, params):
        self.params = params
        self.g0 = 0
        self.g = params.g
        self.t = self.g
        self.max = int(pedigree.get_total_N() * self.factor)
        self.arr = np.zeros((self.max, Struc.n_cols))
        self.sample_n = params.sample_n
        self.n_bins = params.n_sample_bins
        self.sample_bins = np.linspace(0, 1, self.n_bins + 1)
        self.sample_ranges = np.zeros((self.n_bins, 2))
        self.sample_ranges[:, :] = np.arange(0, 1, 1 / self.n_bins)[:, None]
        self.sample_ranges[:, 1] += 1 / self.n_bins
        self.k0 = 0
        self.k1 = 0
        self.get_sample(pedigree)

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
        sample_ids = gen_sample[:, Struc.i]
        self.k1 = len(gen_sample)
        self.arr[self.k0:self.k1] = gen_sample
        t_vec = np.arange(self.g0, self.g)
        for t in t_vec:
            self.t = t
            old_sample = gen_sample
            parent_idx = np.unique(old_sample[:, Struc.parents]).astype(np.int32)
            parent_idx = np.delete(parent_idx, np.where(parent_idx[:] == -1))
            gen_sample = pedigree.arr[parent_idx]
            self.k0 = self.k1
            self.k1 += len(gen_sample)
            self.enter_sample(gen_sample)
        self.trim_pedigree()
        self.sort()
        old_ids = self.arr[:, Struc.i].astype(np.int32)
        new_ids = np.arange(self.get_N(), dtype=np.int32)
        self.arr[:, Struc.i] = new_ids
        for i in [4, 5]:
            self.arr[self.arr[:, i] != -1, i] = self.remap_ids(
                self.arr[:, i], old_ids, new_ids)
        print("Pedigree sampling complete")

    def sample_gen_0(self, pedigree):
        """Sample n organisms in n_sample_bins bins from the most recent
        generation of a Pedigree instance
        """
        last_g = np.min(pedigree.arr[:, Struc.t])
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
        aux = np.zeros((new_max, Struc.n_cols))
        self.arr = np.vstack((self.arr, aux))
        self.max = new-max
        print("pedigree expanded " + str(newcap) + " rows at est. utilization "
              + str(utilization))

    def trim_pedigree(self):
        """Eliminate excess rows from a pedigree array"""
        self.arr = self.arr[0:self.k1]
        excess = self.max - self.k1
        frac = np.round((1 - excess / self.max) * 100, 2)
        print("pedigree trimmed of " + str(excess) + " excess rows, "
              + str(frac) + "% utilization")

    def sort(self):
        """Sort the sample pedigree array by id"""
        self.arr = self.arr[self.arr[:, Struc.i].argsort()]

    def get_N(self):
        """Return the total number of organisms recorded in the pedigree"""
        return np.shape(self.arr)[0]

    @staticmethod
    def sort_arr(arr):
        """Sort a Struc-style array by x position"""
        x = arr[:, Struc.x]
        return arr[x.argsort()]

    @staticmethod
    def remap_ids(vec, old_ids, new_ids):
        """Remap the ids of individuals in the pedigree from an increasing
        integer vector with gaps to an increasing integer vector without
        gaps

        Arguments
        ------------
        vec : 1d np array, shape (N)
            The vector of ids to be remapped. Typically a vector of parent ids.

        old_ids : 1d np array, shape (N)
            The values to map from, which positionally correspond to new_ids

        new_ids : 1d np array, shape (N)
            The values to map to

        Returns
        ------------
        out : 1d np array, shape (N)
            The remapped version of vec.
        """
        idx = np.searchsorted(old_ids, vec[vec != -1])
        return new_ids[idx]


trial = dip.Trial(dip.params)
sample = SamplePedigree.from_trial(trial)
