import matplotlib

import matplotlib.pyplot as plt

import numpy as np

import scipy.optimize as opt

from hybzones.constants import Constants

from hybzones import parameters

from hybzones import util


class GenotypeArr:
    time_ax = 0
    space_ax = 1
    genotype_ax = 2

    def __init__(self, arr, params, t, bin_size):
        self.arr = arr
        self.params = params
        self.t = t
        self.g = params.g
        self.bin_size = bin_size

    @classmethod
    def initialize(cls, params, bin_size=0.01):
        n_bins = util.get_n_bins(bin_size)
        t_dim = params.g + 1
        arr = np.zeros((t_dim, n_bins, Constants.n_genotypes), dtype=np.int32)
        return cls(arr, params, t_dim, bin_size)

    @classmethod
    def from_generation(cls, generation_table, bin_size=0.01):
        """
        Get a SubpopArr of time dimension 1, recording a single generation
        """
        bin_edges, n_bins = util.get_bins(bin_size)
        arr = np.zeros((1, n_bins, Constants.n_genotypes), dtype=np.int32)
        x = generation_table.cols.x
        genotype = generation_table.cols.genotype
        for i in np.arange(Constants.n_genotypes):
            arr[0, :, i] = np.histogram(x[genotype == i],
                                        bins=bin_edges)[0]
        params = generation_table.params
        t = generation_table.t
        return cls(arr, params, t, bin_size)

    @classmethod
    def from_pedigree(cls, pedigree_table, bin_size=0.01):
        """Get a SubpopArr recording population densities in a Pedigree of
        time dimension pedigree.g + 1
        """
        t_dim = pedigree_table.g + 1
        n_bins = util.get_n_bins(bin_size)
        arr = np.zeros((t_dim, n_bins, Constants.n_genotypes), dtype=np.int32)
        for t in np.arange(pedigree_table.g + 1):
            generation_table = pedigree_table.get_generation(t)
            arr[t, :, :] = GenotypeArr.from_generation(generation_table).arr[0]
        params = pedigree_table.params
        t = 0
        return cls(arr, params, t, bin_size)

    @classmethod
    def load_txt(cls, filename):
        file = open(filename, 'r')
        string = file.readline()
        params = parameters.Params.from_string(string)
        raw_arr = np.loadtxt(file, dtype=np.int32)
        file.close()
        shape = np.shape(raw_arr)
        t_dim = shape[0]
        n_genotypes = Constants.n_genotypes
        bin_size = shape[1] // n_genotypes
        new_shape = (t_dim, bin_size, n_genotypes)
        arr = np.reshape(raw_arr, new_shape)
        t_now = 0
        return cls(arr, params, t_now, bin_size)

    def __repr__(self):
        """
        Return a string description of the instance

        :return:
        """
        return (f"GenotypeArr of {len(self)} generations, t = {self.t}, "
                f"g = {self.g}, n organisms = {self.size}")

    def __str__(self):
        """
        Return a more detailed summary

        :return:
        """
        # write this
        return 0

    def __len__(self):
        """
        Return the number of generations recorded in the SubpopArr e.g. the
        length of the zeroth 'time' axis

        :return: length
        """
        return np.shape(self.arr)[0]

    def __getitem__(self, index):
        """
        Return the generation or generations at the times or mask designated
        by index
        """
        arr = self.arr[[index]]
        params = self.params
        bin_size = self.bin_size
        return AlleleArr(arr, params, index, bin_size)

    def enter_generation(self, generation):
        t = generation.t
        self.arr[t, :, :] = GenotypeArr.from_generation(generation).arr[0]

    @property
    def size(self):
        """
        Return the total number of organisms recorded in the array
        """
        return np.sum(self.arr)

    @property
    def generation_sizes(self):
        """
        Return a vector of population sizes for each recorded generation
        """
        return np.sum(np.sum(self.arr, axis=1), axis=1)

    @property
    def densities(self):
        """
        Return the total population densities in each generation
        """
        return np.sum(self.arr, axis=2)

    def save_txt(self, filename):
        """
        Reshape the array to be 2d with shape (generations, n_bins * n_geno.)
        and save it as a .txt file
        """
        shape = np.shape(self.arr)
        reshaped = self.arr.reshape(shape[0], shape[1] * shape[2])
        file = open(filename, 'w')
        header = str(vars(self.params))
        np.savetxt(file, reshaped, delimiter=' ', newline='\n', header=header,
                   fmt="%1.1i")
        file.close()
        print("SubpopArr saved at " + filename)

    def get_generation_size(self, t):
        """
        Return the population size at generation t
        """
        return np.sum(self.arr[t])

    def get_hybrid_densities(self, t):
        """
        Compute the sum of densities of the subpopulations with one or more
        heterozygous loci at generation t
        """
        return np.sum(self.arr[t, :, 1:8], axis=1)

    def get_densities(self, t):
        """
        Return a vector of whole population bin densities in generation t
        """
        return np.sum(self.arr[t], axis=1)

    def get_subplot(self, sub, t=0):
        """
        Plot genotype densities for a single generation on a subplot

        :param sub:
        :param t:
        :return:
        """
        b = util.get_bin_mids(self.bin_size)
        n_vec = self.get_densities(t)
        sub.plot(b, n_vec, color="black", linestyle='dashed', linewidth=2,
                 marker="x")
        sub.plot(b, self.get_hybrid_densities(t), color='green',
                 linestyle='dashed', linewidth=2, marker="x")
        c = Constants.genotype_colors
        for i in np.arange(9):
            sub.plot(b, self.arr[t, :, i], color=c[i], linewidth=2, marker="x")
        y_max = self.params.K * 1.35 * self.bin_size
        n = str(self.get_generation_size(t))
        if len(self) == 1:
            time = self.t
        else:
            time = t
        title = "t = " + str(time) + " n = " + n
        util.setup_space_plot(sub, y_max, "subpop density", title)

    def plot_density(self, t=0):
        """
        Make a plot of the densities of each subpopulation across space
        at index (time) t
        """
        fig = plt.figure(figsize=Constants.plot_size)
        sub = fig.add_subplot(111)
        self.get_subplot(sub, t)
        sub.legend(["N", "Hyb"] + Constants.subpop_legend, fontsize=8,
                   bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
        fig.show()
        return fig

    def plot_size_history(self, log=True):
        """
        Make a plot of per-genotype population sizes over time
        """
        n_vec = self.generation_sizes
        fig = plt.figure(figsize=(8, 6))
        sub = fig.add_subplot(111)
        times = np.arange(self.g + 1)
        sub.plot(times, n_vec, color="black")
        for i in np.arange(9):
            sub.plot(times, np.sum(self.arr[:, :, i], axis=1),
                     color=Constants.genotype_colors[i], linewidth=2)
        sub.set_xlim(0, np.max(times))
        sub.invert_xaxis()
        if log:
            sub.set_yscale("log")
        else:
            y_lim = np.round(self.params.K * 1.1)
            sub.set_ylim(0, y_lim)
        sub.set_xlabel("t before present")
        sub.set_ylabel("population size")
        sub.legend(["N"] + Constants.subpop_legend, fontsize=8)
        fig.show()

    def plot_history(self, plot_int):
        """
        Plot several generations on a single figure to provide snapshots of
        simulation history
        """
        snaps = np.arange(self.g, -1, -plot_int)
        n_figs = len(snaps)
        if n_figs in Constants.shape_dict:
            n_rows, n_cols = Constants.shape_dict[n_figs]
        else:
            n_rows = 2
            n_cols = (n_figs + 1) // 2
        plot_shape = (n_rows, n_cols)
        size = (n_cols * 4, n_rows * 3)
        figure, axs = plt.subplots(n_rows, n_cols, figsize=size, sharex='all')
        figure.tight_layout(pad=3.0)
        figure.subplots_adjust(right=0.9)
        for i in np.arange(n_figs):
            t = snaps[i]
            index = np.unravel_index(i, plot_shape)
            ax = axs[index]
            self.get_subplot(ax, t=t)
        if n_figs < plot_shape[0] * plot_shape[1]:
            index = np.unravel_index(n_figs, plot_shape)
            allele_arr = AlleleArr.from_subpop_arr(self[0])
            ax = axs[index]
            allele_arr.get_subplot(ax)
        figure.legend(["N", "Hyb"] + Constants.subpop_legend, fontsize=10,
                      loc='right', borderaxespad=0, fancybox=False,
                      framealpha=1, edgecolor="black")
        figure.show()
        return figure


class AlleleArr:

    time_axis = 0
    space_axis = 1
    locus_axis = 2
    allele_axis = 3

    def __init__(self, arr, params, t, bin_size):
        self.arr = arr
        self.params = params
        self.t = t
        self.g = params.g
        self.bin_size = bin_size

    @classmethod
    def from_generation(cls, generation_table, bin_size=0.01):
        """
        Get an AlleleArr of time dimension 1, recording the allele distribution
        in a single Generation
        """
        bins, n_bins = util.get_bins(0.01)
        x = generation_table.cols.x
        alleles = generation_table.cols.alleles
        loci = np.array([[0, 1], [0, 1], [2, 3], [2, 3]])
        arr = np.zeros((1, n_bins, 2, 2), dtype=np.int32)
        for i in np.arange(4):
            j, k = np.unravel_index(i, (2, 2))
            a = i % 2 + 1
            arr[0, :, j, k] = (
                    np.histogram(x[alleles[:, loci[i, 0]] == a], bins)[0]
                    + np.histogram(x[alleles[:, loci[i, 1]] == a], bins)[0])
        params = generation_table.params
        t = generation_table.t
        return cls(arr, params, t, bin_size)

    @classmethod
    def from_pedigree(cls, pedigree_table, bin_size=0.01):
        """
        Derive an AlleleArr from an entire pedigree generation by generation

        :param pedigree_table:
        :param bin_size:
        :return:
        """
        t_dim = pedigree_table.g + 1
        n_bins = util.get_n_bins(bin_size)
        arr = np.zeros((t_dim, n_bins, 2, 2), dtype=np.int32)
        for t in np.arange(t_dim):
            generation = pedigree_table.get_generation(t)
            arr[t, :, :, :] = AlleleArr.from_generation(generation).arr
        params = pedigree_table.params
        t = pedigree_table.t
        return cls(arr, params, t, bin_size)

    @classmethod
    def from_subpop_arr(cls, subpop_arr):
        """
        Convert data from a SubpopArr into an AlleleArr
        """
        manifold = Constants.allele_manifold
        arr = np.sum(subpop_arr.arr[:, :, :, None, None] * manifold, axis=2)
        params = subpop_arr.params
        t = subpop_arr.t
        bin_size = subpop_arr.bin_size
        return cls(arr, params, t, bin_size)

    def __repr__(self):
        return (f"AlleleArr of {len(self)} generations, t = {self.t}, "
                f"g = {self.g}, holding {self.n_alleles} alleles from "
                f"{self.size} organisms")

    def __str__(self):
        pass

    def __len__(self):
        """Return the number of generations represented in the array"""
        return np.shape(self.arr)[0]

    def __getitem__(self, index):
        """
        Return the generation or generations at the times or mask designated
        by index
        """
        arr = self.arr[[index]]
        params = self.params
        bin_size = self.bin_size
        return AlleleArr(arr, params, index, bin_size)

    @property
    def n_alleles(self):
        """
        Return the total number of alleles held in the array
        """
        return np.sum(self.arr)

    @property
    def size(self):
        """
        Return the total number of organisms represented in the array
        """
        return np.sum(self.arr) // 4

    @property
    def allele_densities(self):
        """
        Return an array of total allele counts per bin and time
        """
        return np.sum(self.arr, axis=3)

    @property
    def densities(self):
        """
        Return an array of organism counts per bin and time
        """
        return np.sum(self.arr, axis=(2, 3)) // 4

    @property
    def freq(self):
        """
        Return an array of allele frequencies
        """
        return self.arr / self.allele_densities[:, :, :, np.newaxis]

    @property
    def generation_freq(self):
        """
        Return the total allele frequencies for each generation
        """
        counts = np.sum(self.arr, axis=(1, 3))[:, :, None]
        return np.sum(self.arr, axis=1) / counts

    def get_size(self, t):
        """
        Return the population size of generation t
        """
        return np.sum(self.arr[t]) // 4

    def get_allele_density(self, t):
        """
        Return a vector holding the number of loci represented in each
        spatial bin at time t
        """
        return np.sum(self.arr[t, :, :, :], axis=2)

    def get_freq(self, t):
        """
        Return an array of allele frequencies at time t
        """
        n_loci = self.allele_densities
        return self.arr[t] / n_loci[t, :, :, np.newaxis]

    def get_subplot(self, sub, t=0):
        freqs = self.get_freq(t)
        bin_mids = util.get_bin_mids(self.bin_size)
        for i in np.arange(3, -1, -1):
            j, k = np.unravel_index(i, (2, 2))
            sub.plot(bin_mids, freqs[:, j, k],
                     color=Constants.allele_colors[i], linewidth=2,
                     label=Constants.allele_legend[i], marker="x")
        title = "t = " + str(self.t) + " n = " + str(self.get_size(t))
        util.setup_space_plot(sub, 1.01, "allele freq", title)

    def plot_freq(self, t=0):
        """
        Make a plot of the densities of each subpopulation across space
        at index (time) t
        """
        fig = plt.figure(figsize=Constants.plot_size)
        sub = fig.add_subplot(111)
        self.get_subplot(sub, t)
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
        fig.show()
        return fig

    def plot_freq_history(self):
        """
        Make a plot of allele frequencies over time
        """
        generation_freq = self.generation_freq
        fig = plt.figure(figsize=(8, 6))
        sub = fig.add_subplot(111)
        times = np.arange(self.g + 1)
        for i in np.arange(3, -1, -1):
            j, k = np.unravel_index(i, (2, 2))
            sub.plot(times, generation_freq[:, j, k],
                     color=Constants.allele_colors[i], linewidth=2,
                     label=Constants.allele_legend[i])
        sub.set_xlim(0, np.max(times))
        sub.invert_xaxis()
        sub.set_ylim(-0.01, 1.01)
        sub.set_xlabel("t before present")
        sub.set_ylabel("population size")
        sub.legend(fontsize=8)
        fig.show()

    def plot_history(self, plot_int):
        snaps = np.arange(self.g, -1, -plot_int)
        n_figs = len(snaps)
        if n_figs in Constants.shape_dict:
            n_rows, n_cols = Constants.shape_dict[n_figs]
        else:
            n_rows = 2
            n_cols = (n_figs + 1) // 2
        plot_shape = (n_rows, n_cols)
        size = (n_cols * 4, n_rows * 3)
        figure, axs = plt.subplots(n_rows, n_cols, figsize=size, sharex='all')
        figure.tight_layout(pad=3.0)
        figure.subplots_adjust(right=0.9)
        for i in np.arange(n_figs):
            t = snaps[i]
            index = np.unravel_index(i, plot_shape)
            ax = axs[index]
            self.get_subplot(ax, t=t)
        if n_figs < plot_shape[0] * plot_shape[1]:
            index = np.unravel_index(n_figs, plot_shape)
            figure.delaxes(axs[index])
        figure.legend(Constants.allele_legend, fontsize=10, loc='right',
                      borderaxespad=0, fancybox=False, framealpha=1,
                      edgecolor="black")
        figure.show()
        return figure


class ClinePars:

    def __init__(self, x_vec, k_vec, params, bin_size):
        if len(x_vec) != len(k_vec):
            raise AttributeError("x and k vector lengths do not match")
        self.x_vec = x_vec
        self.k_vec = k_vec
        self.params = params
        self.bin_size = bin_size

    @classmethod
    def from_pedigree(cls, pedigree_table):
        allele_arr = AlleleArr.from_pedigree(pedigree_table)
        return cls.from_allele_arr(allele_arr)

    @classmethod
    def from_genotype_arr(cls, genotype_arr):
        allele_arr = AlleleArr.from_subpop_arr(genotype_arr)
        return cls.from_allele_arr(allele_arr)

    @classmethod
    def from_allele_arr(cls, allele_arr):
        allele_freq = allele_arr.get_freq()
        a2_freq = allele_freq[:, :, 0, 1]
        x = util.get_bin_mids(allele_arr.bin_size)
        params = allele_arr.params
        t_dim = params.g + 1
        x_vec = np.zeros(t_dim)
        k_vec = np.zeros(t_dim)
        for t in np.arange(t_dim):
            try:
                cline_opt = cls.optimize_logistic(x, a2_freq[t])
                k_vec[t] = cline_opt[0][0]
                x_vec[t] = cline_opt[0][1]
            except:
                k_vec[t] = -1
                x_vec[t] = -1
        bin_size = allele_arr.bin_size
        return cls(x_vec, k_vec, params, bin_size)

    def __len__(self):
        return len(self.x_vec)

    @staticmethod
    def logistic_fxn(x, k, x_0):
        return 1 / (1.0 + np.exp(-k * (x - x_0)))

    @classmethod
    def optimize_logistic(cls, x, y):
        return opt.curve_fit(cls.logistic_fxn, x, y)

    def plot(self):
        length = len(self)
        t = np.arange(length)
        fig, axs = plt.subplots(2, 1, figsize=(8, 7), sharex='all')
        x_ax, k_ax = axs[0], axs[1]
        k_ax.plot(t, self.k_vec, color="black", linewidth=2)
        x_ax.plot(t, np.full(length, 0.5), color="red", linestyle="dashed")
        x_ax.plot(t, self.x_vec, color="black", linewidth=2)
        k_ax.set_ylim(0, 200)
        x_ax.set_ylim(0, 1)
        k_ax.set_xlim(0, length)
        axs[1].set_xlabel("generations before present")
        x_ax.set_ylabel("x_0")
        k_ax.set_ylabel("k")
        x_ax.set_title("cline parameter x_0")
        k_ax.set_title("cline parameter k")
        x_ax.invert_xaxis()
        fig.suptitle("Cline Parameters")
        fig.tight_layout(pad=1.0)
        fig.show()

    def plot_clines(self, n=10):
        """Plot the cline approximation at n even intervals in time"""
        snaps = np.linspace(len(self) - 1, 0, n, dtype=np.int32)
        x = util.get_bin_mids(self.bin_size)
        fig = plt.figure(figsize=(8, 6))
        sub = fig.add_subplot(111)
        colors = matplotlib.cm.YlGnBu(np.linspace(0.2, 1, n))
        for i in np.arange(n):
            t = snaps[i]
            y = self.logistic_fxn(x, self.k_vec[t], self.x_vec[t])
            sub.plot(x, y, color=colors[i], linewidth=2)
        sub = util.setup_space_plot(sub, 1.01, "$A^2$ cline", "Clines")
        sub.legend(snaps)
        fig.show()
