import matplotlib

import matplotlib.pyplot as plt

import numpy as np

import time

import os

import scipy.optimize as opt

from hybzones.constants import Constants

from hybzones import parameters

from hybzones import util


if __name__ == "__main__":
    plt.rcParams['figure.dpi'] = 100
    matplotlib.use('Qt5Agg')


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
        self.densities = np.sum(self.arr, axis=2)

    @classmethod
    def initialize(cls, params, bin_size=0.01):
        n_bins = util.get_n_bins(bin_size)
        t_dim = params.g + 1
        arr = np.zeros((t_dim, n_bins, Constants.n_genotypes), dtype=np.int32)
        return cls(arr, params, t_dim, bin_size)

    @classmethod
    def from_generation(cls, generation_table, bin_size=0.01, exclusive=True):
        """
        Get a SubpopArr of time dimension 1, recording a single generation

        :param exclusive: if True, exclude individuals killed by fitness
        """
        if exclusive:
            generation_table = generation_table.filter_deceased()
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
    def from_pedigree(cls, pedigree_table, bin_size=0.01, exclusive=True):
        """
        Get a GenotypeArr recording population densities in a Pedigree of
        time dimension pedigree.g + 1
        """
        t_dim = pedigree_table.g + 1
        n_bins = util.get_n_bins(bin_size)
        arr = np.zeros((t_dim, n_bins, Constants.n_genotypes), dtype=np.int32)
        for t in np.arange(pedigree_table.g + 1):
            generation_table = pedigree_table.get_generation(t)
            arr[t, :, :] = GenotypeArr.from_generation(generation_table,
                bin_size=bin_size, exclusive=exclusive).arr[0]
        params = pedigree_table.params
        t = 0
        return cls(arr, params, t, bin_size)

    @classmethod
    def load_txt(cls, filename):
        file = open(filename, 'r')
        string = file.readline()
        params = parameters.Params.from_string(string[1:])
        raw_arr = np.loadtxt(file, dtype=np.int32)
        file.close()
        shape = np.shape(raw_arr)
        t_dim = shape[0]
        n_genotypes = Constants.n_genotypes
        n_bins = shape[1] // n_genotypes
        new_shape = (t_dim, n_bins, n_genotypes)
        arr = np.reshape(raw_arr, new_shape)
        t_now = 0
        bin_size = 1 / n_bins
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
        if type(index) == int:
            arr = self.arr[[index]]
        else:
            arr = self.arr[index]
        params = self.params
        bin_size = self.bin_size
        return GenotypeArr(arr, params, index, bin_size)

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

    def get_history_subplot(self, sub, log=True):
        """
        Make a plot of per-genotype population sizes over time
        """
        n_vec = self.generation_sizes
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

    def plot_history(self, n_snaps=10):
        """
        Plot several generations on a single figure to provide snapshots of
        simulation history
        """
        snaps = np.linspace(self.g, 0, n_snaps + 1).astype(np.int32)
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

    def plot_mean_density(self, ymax=None):
        geno_sums = np.sum(self.arr, axis=2)
        mean = np.mean(geno_sums, axis=0)
        std = np.std(geno_sums, axis=0)
        fig = plt.figure(figsize=(8,6))
        sub = fig.add_subplot()
        x = util.get_bin_mids(self.bin_size)
        sub.errorbar(x, mean, yerr=std, marker="x", color="black")
        if not ymax:
            ymax = self.params * 1.25
        util.setup_space_plot(sub, ymax, "mean", "mean pop")
        fig.show()

    def plot_density_over_time(self, n=11, ymax=None):
        snaps = np.linspace(self.g - 1, 0, n).astype(np.int32)
        if n in Constants.shape_dict:
            n_rows, n_cols = Constants.shape_dict[n]
        else:
            n_rows = 2
            n_cols = (n + 1) // 2
        plot_shape = (n_rows, n_cols)
        size = (n_cols * 4, n_rows * 3)
        figure, axs = plt.subplots(n_rows, n_cols, figsize=size, sharex='all')
        figure.tight_layout(pad=3.0)
        figure.subplots_adjust(right=0.9)
        bin_size = self.bin_size
        x = util.get_bin_mids(bin_size)
        if not ymax:
            ymax = self.params.K * 1.35 * bin_size
        for i in np.arange(n):
            t = snaps[i]
            index = np.unravel_index(i, plot_shape)
            ax = axs[index]
            densities = self.densities[t]
            mean = np.mean(densities, axis=0)
            std = np.std(densities, axis=0)
            ax.errorbar(x, mean, yerr=std, color="black")
            title = "t = " + str(t) + " mean: " + str(
                np.round(np.sum(mean), 1))
            util.setup_space_plot(ax, ymax, "mean density", title)
        figure.suptitle("Mean pop. density at time intervals")
        if n < plot_shape[0] * plot_shape[1]:
            # if there is a free plot, plot all-time mean pop
            index = np.unravel_index(n, plot_shape)
            ax = axs[index]
            mean = np.mean(self.densities, axis=0)
            std = np.std(self.densities, axis=0)
            ax.errorbar(x, mean, yerr=std, color="black")
            title = "all-time mean: " + str(np.round(np.sum(mean), 1))
            util.setup_space_plot(ax, ymax, "mean density", title)
        figure.show()


class GenotypeArrSummary:

    def __init__(self, genotype_arr, snapshot_int=100):
        self.genotype_arr = genotype_arr
        self.params = genotype_arr.params
        self.length = len(genotype_arr.arr)
        self.bin_size = genotype_arr.bin_size
        self.snapshot_t = np.arange(0, self.length+snapshot_int-1,snapshot_int)
        self.snapshot_int = snapshot_int
        self.n_snapshots = len(self.snapshot_t)
        self.sub_arr = genotype_arr[self.snapshot_t]
        self.sub_arr.g = self.n_snapshots
        self.allele_arr = AlleleArr.from_subpop_arr(self.sub_arr)
        cline_pars = ClinePars.from_allele_arr(self.allele_arr)
        self.x_vec = cline_pars.x_vec
        self.k_vec = cline_pars.k_vec
        # expressed in snapshots
        self.fixation_t = self.allele_arr.detect_fixation()
        if self.fixation_t["A1"]:
            self.end_time = self.fixation_t["A1"]
            self.fixed_allele = 0
            self.fix_time = self.fixation_t["A1"]
        elif self.fixation_t["A2"]:
            self.end_time = self.fixation_t["A2"]
            self.fixed_allele = 1
            self.fix_time = self.fixation_t["A2"]
        else:
            self.end_time = 0
            self.fixed_allele = -1
            self.fix_time = None

    @property
    def var_x_through_time(self):
        return np.var(self.x_vec[self.end_time:])

    @property
    def std_x_through_time(self):
        return np.std(self.x_vec[self.end_time:])

    @property
    def mean_x_through_time(self):
        return np.mean(self.x_vec[self.end_time:])

    @property
    def min_x(self):
        return np.min(self.x_vec[self.end_time:])

    @property
    def max_x(self):
        return np.max(self.x_vec[self.end_time:])

    @property
    def vel_x(self):
        """
        Get a vector of the velocity of the cline center at each snapshot,
        normalized by dividing by the snapshot interval
        """
        steps = np.zeros(self.n_snapshots)
        steps[:] = np.nan
        for i in np.arange(self.end_time, self.n_snapshots - 1):
            step = (self.x_vec[i] - self.x_vec[i + 1]) / self.snapshot_int
            steps[i] = step
        steps /= self.snapshot_int
        return steps

    @property
    def speed_x(self):
        return np.abs(self.vel_x)

    @property
    def mean_vel_x(self):
        vel_x = self.vel_x
        return np.mean(vel_x[self.end_time:])

    @property
    def mean_speed_x(self):
        speed_x = self.speed_x
        return np.mean(speed_x[self.end_time:])

    @property
    def slope_k(self):
        steps = np.zeros(self.n_snapshots)
        steps[:] = np.nan
        for i in np.arange(self.end_time, self.n_snapshots - 1):
            step = (self.k_vec[i] - self.k_vec[i + 1]) / self.snapshot_int
            steps[i] = step
        slope_k = np.mean(steps)
        return slope_k


class SummaryCollection:

    """
    Intended as a container for GenotypeArrSummary instances sharing the same
    parameter set. summarizes them in some key statistics
    """

    def __init__(self, directory, snapshot_int=100):
        #base_dir = os.getcwd().replace(r"\hybzones", "") + r"\hybzones"
        #directory = base_dir + "\\data\\" + directory + "\\"
        file_names = os.listdir(directory)
        file_names = [directory + filename for filename in file_names]
        self.summaries = []
        for file_name in file_names:
            genotype_arr = GenotypeArr.load_txt(file_name)
            self.summaries.append(GenotypeArrSummary(genotype_arr,
                                                     snapshot_int=snapshot_int))
        self.snapshot_t = self.summaries[0].snapshot_t
        self.n_snapshots = self.summaries[0].n_snapshots
        self.params = self.summaries[0].params
        self.bin_size = self.summaries[-1].bin_size
        self.n = len(self.summaries)

    def __getitem__(self, idx):
        return self.summaries[idx]

    @property
    def fix_frac(self):
        """
        Return the fraction of trials where signal fixation occured
        """
        i = 0
        for summary in self.summaries:
            if summary.fix_time:
                i += 1
        return i / len(self.summaries)

    @property
    def mean_fix_time(self):
        """
        Return the mean time to fixation among trials where fixation occured
        """
        fix_times = []
        for summary in self.summaries:
            if summary.fix_time:
                fix_times.append(summary.snapshot_t[int(summary.fix_time)])
        return np.mean(fix_times)

    @property
    def fix_indices(self):
        fix_indices = []
        for summary in self.summaries:
            if summary.fix_time:
                fix_indices.append(int(summary.fix_time))
        return fix_indices

    def get_density_at_t(self, t):
        """
        Get a vector of mean densities across trials at a given time
        """
        densities = []
        for summary in self.summaries:
            densities.append(summary.genotype_arr.densities[t])
        return np.mean(densities, axis=0)

    def std_density_at_t(self, t):
        """
        Get a vector of mean densities across trials at a given time
        """
        densities = []
        for summary in self.summaries:
            densities.append(summary.genotype_arr.densities[t])
        return np.std(densities, axis=0)

    @property
    def mean_density(self):
        """
        Return a vector of mean densities across trials and time
        """
        densities = []
        for summary in self.summaries:
            densities.append(np.mean(summary.genotype_arr.densities, axis=0))
        return np.mean(densities, axis=0)

    @property
    def std_density(self):
        """
        Return a vector of mean densities across trials and time
        """
        densities = []
        for summary in self.summaries:
            densities.append(np.mean(summary.genotype_arr.densities, axis=0))
        return np.std(densities, axis=0)

    @property
    def mean_cline_k(self):
        mean_k = np.zeros(self.n_snapshots, dtype=np.float32)
        mean_k[:] = np.nan
        for t in np.arange(self.n_snapshots):
            this_k = []
            for summary in self.summaries:
                if t >= summary.end_time:
                    this_k.append(summary.k_vec[t])
            if len(this_k) > 0:
                mean_k[t] = np.mean(this_k)
        return mean_k

    @property
    def min_cline_k(self):
        min_k = np.zeros(self.n_snapshots, dtype=np.float32)
        min_k[:] = np.nan
        for t in np.arange(self.n_snapshots):
            this_k = []
            for summary in self.summaries:
                if t >= summary.end_time:
                    this_k.append(summary.k_vec[t])
            if len(this_k) > 0:
                min_k[t] = np.min(this_k)
        return min_k

    @property
    def max_cline_k(self):
        max_k = np.zeros(self.n_snapshots, dtype=np.float32)
        max_k[:] = np.nan
        for t in np.arange(self.n_snapshots):
            this_k = []
            for summary in self.summaries:
                if t >= summary.end_time:
                    this_k.append(summary.k_vec[t])
            if len(this_k) > 0:
                max_k[t] = np.max(this_k)
        return max_k

    @property
    def std_cline_k(self):
        std_k = np.zeros(self.n_snapshots, dtype=np.float32)
        std_k[:] = np.nan
        for t in np.arange(self.n_snapshots):
            this_k = []
            for summary in self.summaries:
                if t >= summary.end_time:
                    this_k.append(summary.k_vec[t])
            if len(this_k) > 0:
                std_k[t] = np.std(this_k)
        return std_k

    @property
    def mean_cline_x(self):
        mean_x = np.zeros(self.n_snapshots, dtype=np.float32)
        mean_x[:] = np.nan
        for t in np.arange(self.n_snapshots):
            this_x = []
            for summary in self.summaries:
                if t >= summary.end_time:
                    this_x.append(summary.x_vec[t])
            if len(this_x) > 0:
                mean_x[t] = np.mean(this_x)
        return mean_x

    @property
    def std_cline_x(self):
        std_x = np.zeros(self.n_snapshots, dtype=np.float32)
        std_x[:] = np.nan
        for t in np.arange(self.n_snapshots):
            this_x = []
            for summary in self.summaries:
                if t >= summary.end_time:
                    this_x.append(summary.x_vec[t])
            if len(this_x) > 0:
                std_x[t] = np.var(this_x)
        return std_x

    @property
    def max_cline_x(self):
        max_x = np.zeros(self.n_snapshots, dtype=np.float32)
        max_x[:] = np.nan
        for t in np.arange(self.n_snapshots):
            this_x = []
            for summary in self.summaries:
                if t >= summary.end_time:
                    this_x.append(summary.x_vec[t])
            if len(this_x) > 0:
                max_x[t] = np.max(this_x)
        return max_x

    @property
    def min_cline_x(self):
        min_x = np.zeros(self.n_snapshots, dtype=np.float32)
        min_x[:] = np.nan
        for t in np.arange(self.n_snapshots):
            this_x = []
            for summary in self.summaries:
                if t >= summary.end_time:
                    this_x.append(summary.x_vec[t])
            if len(this_x) > 0:
                min_x[t] = np.min(this_x)
        return min_x

    @property
    def mean_cline_speed(self):
        mean_speed = np.zeros(self.n_snapshots, dtype=np.float32)
        mean_speed[:] = np.nan
        for t in np.arange(self.n_snapshots):
            speeds = []
            for summary in self.summaries:
                if t >= summary.end_time:
                    speeds.append(summary.speed_x[t])
            if len(speeds) > 0:
                mean_speed[t] = np.mean(speeds)
        return mean_speed

    @property
    def std_cline_speed(self):
        std_speed = np.zeros(self.n_snapshots, dtype=np.float32)
        std_speed[:] = np.nan
        for t in np.arange(self.n_snapshots):
            speeds = []
            for summary in self.summaries:
                if t >= summary.end_time:
                    speeds.append(summary.speed_x[t])
            if len(speeds) > 0:
                std_speed[t] = np.std(speeds)
        return std_speed


class OverCollection:

    """
    Intended as a container for SummaryCollections; to be used in comparing
    statistics trial groups using different parameter sets and plotting said
    statistics
    """

    def __init__(self, directory, snapshot_int=100):
        base_dir = os.getcwd().replace(r"\hybzones", "") + r"\hybzones"
        full_dir_name = base_dir + "\\data\\" + directory + "\\"
        short_dir_names = os.listdir(full_dir_name)
        dir_names = [full_dir_name + dir_name + "\\" for dir_name in short_dir_names]
        self.collections = []
        self.names = short_dir_names
        self.params = []
        self.n = 0
        for dir_name in dir_names:
            self.collections.append(SummaryCollection(dir_name,
                snapshot_int=snapshot_int))
            self.params.append(self.collections[-1].params)
            self.n += 1
            print(f"collection #{self.n} loaded @{util.get_time_string()}")
        self.snapshot_t = self.collections[-1].snapshot_t
        self.n_snapshots = self.collections[-1].n_snapshots
        self.bin_size = self.collections[-1].bin_size
        self.colors = matplotlib.cm.gist_rainbow(np.linspace(0, 1, self.n))
        self.length = self.snapshot_t[-1]

    def __getitem__(self, i):
        return self.collections[i]

    def plot_cline_pars(self, title=None, simple=False, colls=None):
        fig, axs = plt.subplots(2, 1, figsize=(6, 8), sharex='all')
        x_ax, k_ax = axs[0], axs[1]
        length = self.snapshot_t[-1]
        times = self.snapshot_t
        for i, collection in enumerate(self.collections):
            if colls:
                if i not in colls:
                    continue
            k = collection.mean_cline_k
            if not simple:
                k_std = collection.std_cline_k
                k_min = collection.min_cline_k
                k_max = collection.max_cline_k
                k_ax.plot(times, k - k_std, color=self.colors[i], linewidth=1)
                k_ax.plot(times, k + k_std, color=self.colors[i], linewidth=1)
                k_ax.plot(times, k_max, color=self.colors[i], linewidth=1,
                          linestyle="dotted")
                k_ax.plot(times, k_min, color=self.colors[i], linewidth=1,
                          linestyle="dotted")
            k_ax.plot(times, k, color=self.colors[i], linewidth=2,
                      label=self.names[i])
            x = collection.mean_cline_x
            if not simple:
                x_std = collection.std_cline_x
                x_min = collection.min_cline_x
                x_max = collection.max_cline_x
                x_ax.plot(times, x - x_std, color=self.colors[i], linewidth=1)
                x_ax.plot(times, x + x_std, color=self.colors[i], linewidth=1)
                x_ax.plot(times, x_max, color=self.colors[i], linewidth=1,
                          linestyle="dotted")
                x_ax.plot(times, x_min, color=self.colors[i], linewidth=1,
                          linestyle="dotted")
            x_ax.plot(times, x, color=self.colors[i], linewidth=2)
        for i, collection in enumerate(self.collections):
            if colls:
                if i not in colls:
                    continue
            if len(collection.fix_indices) > 0:
                idx = collection.fix_indices
                t = self.snapshot_t[idx]
                k_ax.scatter(t, collection.mean_cline_k[idx], marker="x",
                             color=self.colors[i])
                x_ax.scatter(t, collection.mean_cline_x[idx], marker="x",
                             color=self.colors[i])
        k_ax.set_ylim(0, 200)
        x_ax.set_ylim(0, 1)
        k_ax.set_xlim(0, length)
        axs[1].set_xlabel("generations before present")
        x_ax.set_ylabel("x_0")
        k_ax.set_ylabel("k")
        x_ax.set_title("cline parameter x_0")
        k_ax.set_title("cline parameter k")
        x_ax.invert_xaxis()
        if not title:
            title = ""
        fig.suptitle("Cline Parameters " + str(title))
        fig.tight_layout(pad=1.0)
        fig.legend()
        fig.show()

    def plot_densities(self, snapshots=11, ymax=None, colls=None, title=None,
                       simple=True):
        snaps = np.linspace(self.length, 0, snapshots).astype(np.int32)
        if snapshots in Constants.shape_dict:
            n_rows, n_cols = Constants.shape_dict[snapshots]
        else:
            n_rows = 3
            n_cols = (snapshots + 1) // 3
        plot_shape = (n_rows, n_cols)
        size = (n_cols * 4, n_rows * 3)
        figure, axs = plt.subplots(n_rows, n_cols, figsize=size, sharex='all')
        figure.tight_layout(pad=3.0)
        figure.subplots_adjust(right=0.9)
        x = util.get_bin_mids(self.bin_size)
        if not ymax:
            ymax = self.params[0].K * 1.35 * self.bin_size
        for i in np.arange(snapshots):
            t = snaps[i]
            index = np.unravel_index(i, plot_shape)
            ax = axs[index]
            for k, collection in enumerate(self.collections):
                if colls:
                    if k not in colls:
                        continue
                density = collection.get_density_at_t(t)
                ax.plot(x, density, color=self.colors[k], linewidth=2)
                if not simple:
                    std = collection.std_density_at_t(t)
                    ax.plot(x, density + std, color=self.colors[k], linewidth=1)
                    ax.plot(x, density - std, color=self.colors[k], linewidth=1)
            title = "t = " + str(t)
            util.setup_space_plot(ax, ymax, "mean density", title)
        figure.suptitle("Mean pop. density at time intervals")
        if snapshots < plot_shape[0] * plot_shape[1]:
            # if there is a free plot, plot all-time mean densities
            index = np.unravel_index(snapshots, plot_shape)
            ax = axs[index]
            for k, collection in enumerate(self.collections):
                if colls:
                    if k not in colls:
                        continue
                density = collection.mean_density
                ax.plot(x, density, color=self.colors[k], linewidth=2,
                        label=self.names[k])
                if not simple:
                    std = collection.std_density
                    ax.plot(x, density + std, color=self.colors[k], linewidth=1)
                    ax.plot(x, density - std, color=self.colors[k], linewidth=1)
            title = "all-time means"
            util.setup_space_plot(ax, ymax, "mean density", title)
        figure.legend()
        if title:
            figure.suptitle(title)
        figure.show()

    def plot_fixations(self, colls=None, title=None):
        fig = plt.figure(figsize=(8, 6))
        sub = fig.add_subplot(111)
        length = len(self.snapshot_t)
        x_locs = np.arange(length)
        times = self.snapshot_t
        real_length = times[-1]
        for i, collection in enumerate(self.collections):
            if colls:
                if i not in colls:
                    continue
            n = collection.n
            values = np.zeros(length)
            index = np.sort(collection.fix_indices)
            for idx in index:
                values[:idx] += 1
            values /= n
            plt.plot(times, values, color=self.colors[i], label=self.names[i])
        for i, collection in enumerate(self.collections):
            if colls:
                if i not in colls:
                    continue
            if len(collection.fix_indices) > 0:
                idx = collection.fix_indices
                t = self.snapshot_t[idx]
                sub.scatter(t, collection.mean_cline_k[idx], marker="x",
                             color=self.colors[i])
        sub.set_ylim(-0.01, 1.01)
        sub.set_xlim(0, real_length)
        sub.set_xlabel("generations before present")
        sub.set_ylabel("% signal fixation")
        sub.invert_xaxis()
        fig.legend()
        if title:
            fig.title(title)
        fig.show()

    def plot_cline_speed(self, colls=None, title=None):
        fig = plt.figure(figsize=(8, 6))
        sub = fig.add_subplot(111)
        length = len(self.snapshot_t)
        times = self.snapshot_t
        real_length = times[-1]
        for i, collection in enumerate(self.collections):
            if colls:
                if i not in colls:
                    continue
            mean = collection.mean_cline_speed
            std = collection.std_cline_speed
            sub.plot(times, mean, color=self.colors[i], linewidth=2,
                     label=self.names[i])
            #sub.plot(times, mean + std, color=self.colors[i], linewidth=1)
            #sub.plot(times, mean - std, color=self.colors[i], linewidth=1)
        for i, collection in enumerate(self.collections):
            if colls:
                if i not in colls:
                    continue
            if len(collection.fix_indices) > 0:
                idx = collection.fix_indices
                t = times[idx]
                sub.scatter(t, collection.mean_cline_speed[idx], marker="x",
                            color=self.colors[i])
        sub.set_xlim(0, real_length)
        sub.set_xlabel("generations before present")
        sub.set_ylabel("cline speed per generation")
        sub.invert_xaxis()
        fig.legend()
        if title:
            fig.title(title)
        fig.show()

    


class AlleleArr:

    time_axis = 0
    space_axis = 1
    locus_axis = 2
    allele_axis = 3

    """
    [[A1, A2],
     [B1, B2]], 
    """

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

    def detect_fixation(self):
        """
        Retrieve a dictionary of allele fixation times. If an allele is not
        fixed, its key is left as None
        """
        fixations = {"A1": None, "A2": None, "B1": None, "B2": None}
        coords = {"A1": (0, 0), "A2": (0, 1), "B1": (1, 0), "B2": (1, 1)}
        length = len(self.arr)
        freq = self.generation_freq
        for allele in fixations:
            i, j = coords[allele]
            fix_time = length - np.searchsorted(np.flip(freq[:, i, j]), 1)
            if fix_time < length:
                fixations[allele] = fix_time
        return fixations


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
        allele_freq = allele_arr.freq
        a2_freq = allele_freq[:, :, 0, 1]
        x = util.get_bin_mids(allele_arr.bin_size)
        params = allele_arr.params
        t_dim = len(allele_arr.arr)
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
        fig, axs = plt.subplots(2, 1, figsize=(6, 8), sharex='all')
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


class GenotypeArrCollection:

    """
    Object for group analysis of several genotype arrays. It is expected
    that all genotype arrays will share an identical params instance ....
    """

    def __init__(self, genotype_arrs, snapshot_int=100):
        self.genotype_arrs = genotype_arrs
        self.params = self.genotype_arrs[0].params
        self.length = len(genotype_arrs[0].arr)
        self.bin_size = self.genotype_arrs[0].bin_size
        self.snapshot_t = np.arange(0, self.length+snapshot_int-1,snapshot_int)
        self.n_snapshots = len(self.snapshot_t)
        self.sub_arrs = []
        for arr in self.genotype_arrs:
            sub_arr = arr[self.snapshot_t]
            sub_arr.g = self.n_snapshots
            self.sub_arrs.append(sub_arr)
        self.allele_arrs = self.get_allele_arrs()
        self.cline_pars = self.get_cline_pars()
        self.n_trials = len(genotype_arrs)

    @classmethod
    def load_directory(cls, directory, snapshot_int=100):
        base_dir = os.getcwd().replace(r"\hybzones", "") + r"\hybzones"
        directory = base_dir + "\\data\\" + directory + "\\"
        file_names = os.listdir(directory)
        file_names = [directory + filename for filename in file_names]
        genotype_arrs = []
        for file_name in file_names:
            genotype_arrs.append(GenotypeArr.load_txt(file_name))
        return cls(genotype_arrs, snapshot_int=snapshot_int)

    def check_params(self):
        pass

    def append_arr(self, genotype_arr):
        self.genotype_arrs.append(genotype_arr)

    def append_list(self, genotype_arrs):
        for arr in genotype_arrs:
            self.genotype_arrs.append(arr)

    def get_cline_pars(self):
        cline_pars = []
        for arr in self.sub_arrs:
            cline_pars.append(ClinePars.from_genotype_arr(arr))
        return cline_pars

    def get_allele_arrs(self):
        allele_arrs = []
        for arr in self.sub_arrs:
            allele_arrs.append(AlleleArr.from_subpop_arr(arr))
        return allele_arrs

    def plot_cline_pars(self, title=None):
        n = len(self.cline_pars)
        bin_size = self.cline_pars[0].bin_size
        params = self.params
        length = self.n_snapshots
        x_arr = np.zeros((length, n))
        k_arr = np.zeros((length, n))
        for i, pars in enumerate(self.cline_pars):
            x_arr[:, i] = pars.x_vec
            k_arr[:, i] = pars.k_vec
        mean_x = np.zeros(length)
        std_x = np.zeros(length)
        mean_k = np.zeros(length)
        std_k = np.zeros(length)
        for i in np.arange(length):
            mean_x[i] = np.mean(x_arr[i, x_arr[i, :] > 0])
            std_x[i] = np.std(x_arr[i, x_arr[i, :] > 0])
            mean_k[i] = np.mean(k_arr[i, k_arr[i, :] > 0.5])
            std_k[i] = np.std(k_arr[i, k_arr[i, :] > 0.5])
        #
        fig, axs = plt.subplots(2, 1, figsize=(6, 8), sharex='all')
        x_ax, k_ax = axs[0], axs[1]
        k_ax.plot(self.snapshot_t, mean_k-std_k, color="black", linewidth=1)
        k_ax.plot(self.snapshot_t, mean_k+std_k, color="black", linewidth=1)
        k_ax.plot(self.snapshot_t, mean_k, color="black", linewidth=2)
        x_ax.plot(self.snapshot_t, np.full(length, 0.5), color="red",
                  linestyle="dashed")
        x_ax.errorbar(self.snapshot_t, mean_x-std_x, color="black", linewidth=1)
        x_ax.errorbar(self.snapshot_t, mean_x+std_x, color="black", linewidth=1)
        x_ax.errorbar(self.snapshot_t, mean_x, color="black", linewidth=2)
        #
        fixations = []
        for arr in self.allele_arrs:
            fixations.append(arr.detect_fixation())
        for fixes in fixations:
            if fixes["A1"] or fixes["A2"]:
                if fixes["A1"]:
                    a = "A1"
                    i = 0
                elif fixes["A2"]:
                    a = "A2"
                    i = 1
                idx = fixes[a]
                t = self.snapshot_t[idx]
                k_ax.scatter(t, mean_k[idx]+std_k[idx]+5, marker="x",
                             color=Constants.allele_colors[i])
                x_ax.scatter(t, mean_x[idx]+std_x[idx]+0.02, marker="x",
                             color=Constants.allele_colors[i])
        #
        k_ax.set_ylim(0, 200)
        x_ax.set_ylim(0, 1)
        k_ax.set_xlim(0, self.length)
        axs[1].set_xlabel("generations before present")
        x_ax.set_ylabel("x_0")
        k_ax.set_ylabel("k")
        x_ax.set_title("cline parameter x_0")
        k_ax.set_title("cline parameter k")
        x_ax.invert_xaxis()
        if not title:
            title = ""
        fig.suptitle("Cline Parameters " + str(title))
        fig.tight_layout(pad=1.0)
        fig.show()

    def plot_clines(self, n=10):
        """
        Plot the cline approximation at n even intervals in time
        """
        n += 1
        length = len(self.cline_pars[0].x_vec)
        snaps = np.linspace(length - 1, 0, n, dtype=np.int32)
        xrange = util.get_bin_mids(self.cline_pars[0].bin_size)
        fig = plt.figure(figsize=(8, 6))
        sub = fig.add_subplot(111)
        colors = matplotlib.cm.YlGnBu(np.linspace(0.2, 1, n))
        for i in np.arange(n):
            this_x = []
            this_k = []
            t = snaps[i]
            for pars in self.cline_pars:
                this_x.append(pars.x_vec[t])
                this_k.append(pars.k_vec[t])
            k = np.mean(this_k)
            x = np.mean(this_x)
            std_x = np.std(this_x)
            y = ClinePars.logistic_fxn(xrange, k, x)
            sub.plot(xrange, y, color=colors[i], linewidth=2)
            sub.errorbar(x, 0.5, xerr=std_x, color=colors[i], capsize=3,
                         linewidth=2)
        sub = util.setup_space_plot(sub, 1.01, "$A^2$ cline", "Clines")
        sub.legend(snaps)
        fig.show()

    def plot_cline_means(self, n=10):
        """
        Take the avg A2 cline from allele arrs and plot it
        """
        n += 1
        snaps = np.linspace(self.length - 1, 0, n).astype(np.int32)
        if n in Constants.shape_dict:
            n_rows, n_cols = Constants.shape_dict[n]
        else:
            n_rows = 2
            n_cols = (n + 1) // 2
        plot_shape = (n_rows, n_cols)
        size = (n_cols * 4, n_rows * 3)
        figure, axs = plt.subplots(n_rows, n_cols, figsize=size, sharex='all')
        figure.tight_layout(pad=3.0)
        figure.subplots_adjust(right=0.9)
        x = util.get_bin_mids(self.allele_arrs[0].bin_size)
        c1 = Constants.allele_colors[1]
        c2 = Constants.allele_colors[3]
        for i in np.arange(n):
            t = snaps[i]
            j = np.searchsorted(self.snapshot_t, t)
            index = np.unravel_index(i, plot_shape)
            ax = axs[index]
            a2_freqs = []
            b2_freqs = []
            for arr in self.allele_arrs:
                a2_freqs.append(arr.freq[j, :, 0, 1])
                b2_freqs.append(arr.freq[j, :, 1, 1])
            a2 = np.mean(a2_freqs, axis=0)
            a2_std = np.std(a2_freqs, axis=0)
            b2 = np.mean(b2_freqs, axis=0)
            b2_std = np.std(b2_freqs, axis=0)
            ax.plot(x, b2+b2_std, color=c2)
            ax.plot(x, b2-b2_std, color=c2)
            ax.plot(x, b2, marker="x", linewidth=2, color=c2)
            ax.plot(x, a2+a2_std, color=c1)
            ax.plot(x, a2-a2_std, color=c1)
            ax.plot(x, a2, marker="x", linewidth=2, color=c1)
            title = "t = " + str(t)
            util.setup_space_plot(ax, 1.01, "allele frequency", title)
        figure.show()

    def plot_all_clines(self, n=10):
        n += 1
        snaps = np.linspace(self.length - 1, 0, n).astype(np.int32)
        if n in Constants.shape_dict:
            n_rows, n_cols = Constants.shape_dict[n]
        else:
            n_rows = 2
            n_cols = (n + 1) // 2
        plot_shape = (n_rows, n_cols)
        size = (n_cols * 4, n_rows * 3)
        figure, axs = plt.subplots(n_rows, n_cols, figsize=size, sharex='all')
        figure.tight_layout(pad=3.0)
        figure.subplots_adjust(right=0.9)
        x = util.get_bin_mids(self.allele_arrs[0].bin_size)
        c1 = Constants.allele_colors[1]
        c2 = Constants.allele_colors[3]
        colors = matplotlib.cm.hsv(np.linspace(0, 1, self.n_trials))
        for i in np.arange(n):
            t = snaps[i]
            j = np.searchsorted(self.snapshot_t, t)
            index = np.unravel_index(i, plot_shape)
            ax = axs[index]
            for k, arr in enumerate(self.allele_arrs):
                ax.plot(x, arr.freq[j, :, 0, 1], linewidth=2, color=colors[k])
            title = "t = " + str(t)
            util.setup_space_plot(ax, 1.01, "allele frequency", title)
        figure.show()

    def plot_pop_density(self, n=10):
        """
        Plot mean population densities
        """
        n += 1
        snaps = np.linspace(self.length - 1, 0, n).astype(np.int32)
        if n in Constants.shape_dict:
            n_rows, n_cols = Constants.shape_dict[n]
        else:
            n_rows = 2
            n_cols = (n + 1) // 2
        plot_shape = (n_rows, n_cols)
        size = (n_cols * 4, n_rows * 3)
        figure, axs = plt.subplots(n_rows, n_cols, figsize=size, sharex='all')
        figure.tight_layout(pad=3.0)
        figure.subplots_adjust(right=0.9)
        bin_size = self.allele_arrs[0].bin_size
        x = util.get_bin_mids(bin_size)
        ymax = self.genotype_arrs[0].params.K * 1.35 * bin_size
        for i in np.arange(n):
            t = snaps[i]
            index = np.unravel_index(i, plot_shape)
            ax = axs[index]
            densities = []
            for arr in self.genotype_arrs:
                densities.append(arr.densities[t])
            mean = np.mean(densities, axis=0)
            std = np.std(densities, axis=0)
            ax.errorbar(x, mean, yerr=std, color="black")
            title = "t = " + str(t) + " mean: " + str(np.round(np.sum(mean),1))
            util.setup_space_plot(ax, ymax, "mean density", title)
        figure.suptitle("Mean pop. density at time intervals")
        if n < plot_shape[0] * plot_shape[1]:
            # if there is a free plot, plot all-time mean pop
            index = np.unravel_index(n, plot_shape)
            ax = axs[index]
            densities = []
            for arr in self.genotype_arrs:
                densities.append(arr.densities)
            all_arr = np.mean(densities, axis=0)
            mean = np.mean(all_arr, axis=0)
            std = np.std(all_arr, axis=0)
            ax.errorbar(x, mean, yerr=std, color="black")
            title = "all-time mean: " + str(np.round(np.sum(mean), 1))
            util.setup_space_plot(ax, ymax, "mean density", title)
        figure.show()

    def plot_histories(self, title=None):
        """
        Plot the subpop size histories of each array in the instance
        """
        n = len(self.genotype_arrs)
        if n in Constants.shape_dict:
            n_rows, n_cols = Constants.shape_dict[n]
        else:
            n_rows = 2
            n_cols = (n + 1) // 2
        plot_shape = (n_rows, n_cols)
        size = (n_cols * 4, n_rows * 3)
        figure, axs = plt.subplots(n_rows, n_cols, figsize=size, sharex='all')
        figure.tight_layout(pad=3.0)
        figure.subplots_adjust(right=0.9)
        for i in np.arange(n):
            index = np.unravel_index(i, plot_shape)
            ax = axs[index]
            self.genotype_arrs[i].get_history_subplot(ax)
        figure.legend(["N"] + Constants.subpop_legend, fontsize=6,
                      loc='right', borderaxespad=0.005, fancybox=False,
                      framealpha=1, edgecolor="black")
        if title:
            figure.suptitle(title)
        figure.show()


class GenotypeArrRange:
    """
    Class for anaalyzing large groups of genotype arrs sorted into lists
    by parameter set.

    """

    def __init__(self, root_dir, snapshot_int=500):
        """
        Load every
        """
        self.collections = self.load_directory(root_dir, snapshot_int)
        self.param_list = [coll.params for coll in self.collections]
        lengths = list(set([coll.length for coll in self.collections]))
        if len(lengths) == 1:
            self.length = lengths[0]

    @staticmethod
    def load_directory(root_dir, snapshot_int):
        base_dir = os.getcwd().replace(r"\hybzones", "") + r"\hybzones"
        full_dir_name = base_dir + "\\data\\" + root_dir + "\\"
        dir_names = os.listdir(full_dir_name)
        dir_names = [full_dir_name + x for x in dir_names if ".txt" not in x]
        collections = []
        for dir_name in dir_names:
            collections.append(GenotypeArrCollection.load_directory(
                dir_name, snapshot_int=snapshot_int))
        return collections


class RaggedArr:
    """Class to hold a ragged set of vectors in nested lists; 3d"""

    def __init__(self, shape):
        """shape must be a tuple of size 2"""
        self.shape = shape
        self.arr = [[np.array([], dtype=np.float32)
                     for j in np.arange(shape[1])] for i in np.arange(shape[0])
                    ]
        self.means = None
        self.stds = None
        self.sums = None

    def enter_vec(self, vec, i, j):
        """Enter a vector at coords (i, j)"""
        self.arr[i][j] = vec

    def get_statistics(self):
        self.means = np.zeros(self.shape, dtype=np.float32)
        self.stds = np.zeros(self.shape, dtype=np.float32)
        self.sums = np.zeros(self.shape, dtype=np.float32)
        for i in np.arange(self.shape[0]):
            for j in np.arange(self.shape[1]):
                if len(self.arr[i][j] > 0):
                    self.means[i, j] = np.mean(self.arr[i][j])
                    self.stds[i, j] = np.std(self.arr[i][j])
                self.sums[i, j] = np.sum(self.arr[i][j])

    def get_n_elements(self):
        length = 0
        for i in np.arange(self.shape[0]):
            for j in np.arange(self.shape[1]):
                length += len(self.arr[i][j])
        return length


class MatingHistograms:
    """Kind of a messed up class. Centered on two large arrays of dimensions
    (subpop, sex, x_bin, histogram) which hold histograms of reproductive success
    across space and time
    """
    n_fecundity_bins = 20
    n_pairing_bins = 10

    def __init__(self, n_arr, pairing_hist, fecundity_hist, female_pairing_arr,
                 male_pairing_arr, female_fecundity_arr, male_fecundity_arr,
                 params, bin_size):
        self.n_arr = n_arr
        self.pairing_hist = pairing_hist
        self.fecundity_hist = fecundity_hist
        self.female_pairing_arr = female_pairing_arr
        self.male_pairing_arr = male_pairing_arr
        self.female_fecundity_arr = female_fecundity_arr
        self.male_fecundity_arr = male_fecundity_arr
        self.params = params
        self.bin_size = bin_size
        self.bin_mids = np.arange(bin_size/2, 1 + bin_size/2, bin_size)

    @classmethod
    def from_pedigree(cls, pedigree, bin_size = 0.02):
        """Get the histograms via analysis of a pedigree. Computes histograms
        """
        g = pedigree.params.g
        female_ids = pedigree.get_female_idx()
        male_ids = pedigree.get_male_idx()
        t = pedigree.get_t()
        female_t = t[female_ids]
        male_t = t[male_ids]
        female_ids = female_ids[female_t > 0]
        male_ids = male_ids[male_t > 0]
        max_founder_id = np.max(pedigree.get_gen_idx(g))
        pairings = pedigree.get_parents()[max_founder_id + 1:]
        mat_ids = np.sort(pairings[:, 0])
        pat_ids = np.sort(pairings[:, 1])
        female_fecundity = (np.searchsorted(mat_ids, female_ids + 1)
                              - np.searchsorted(mat_ids, female_ids))
        male_fecundity = (np.searchsorted(pat_ids, male_ids + 1)
                            - np.searchsorted(pat_ids, male_ids))
        unique_pairings = np.unique(pairings, axis=0)
        unique_mat = np.sort(unique_pairings[:, 0])
        unique_pat = np.sort(unique_pairings[:, 1])
        female_pairings = (np.searchsorted(unique_mat, female_ids + 1)
                            - np.searchsorted(unique_mat, female_ids))
        male_pairings = (np.searchsorted(unique_pat, male_ids + 1)
                          - np.searchsorted(unique_pat, male_ids))
        subpop_idx = pedigree.get_subpop_idx()
        female_subpop_idx = subpop_idx[female_ids]
        male_subpop_idx = subpop_idx[male_ids]
        x = pedigree.get_x()
        x_bins = np.arange(0, 1, bin_size)
        n_bins = len(x_bins)
        female_x_idx = np.searchsorted(x_bins, x[female_ids]) - 1
        male_x_idx = np.searchsorted(x_bins, x[male_ids]) - 1
        fecundity_bins = np.arange(cls.n_fecundity_bins + 1)
        n_pops = pedigree.n_subpops
        fecundity_hist = np.zeros((n_pops, 2, n_bins, cls.n_fecundity_bins))
        pairing_bins = np.arange(cls.n_pairing_bins + 1)
        pairing_hist = np.zeros((n_pops, 2, n_bins, cls.n_pairing_bins))
        n_arr = np.zeros((n_pops, 2, n_bins))

        bin_females = []
        bin_males = []
        for j in np.arange(n_bins):
            bin_females.append(np.where(female_x_idx == j))
            bin_males.append(np.where(male_x_idx == j))

        shape = (n_pops, n_bins)
        female_pairing_arr = RaggedArr(shape)
        male_pairing_arr = RaggedArr(shape)
        female_fecundity_arr = RaggedArr(shape)
        male_fecundity_arr = RaggedArr(shape)

        for i in np.arange(n_pops):
            subpop_females = np.where(female_subpop_idx == i)
            subpop_males = np.where(male_subpop_idx == i)
            for j in np.arange(n_bins):
                f_idx = np.intersect1d(bin_females[j], subpop_females)
                m_idx = np.intersect1d(bin_males[j], subpop_males)

                female_pairing_arr.enter_vec(female_pairings[f_idx], i, j)
                male_pairing_arr.enter_vec(male_pairings[m_idx], i, j)
                female_fecundity_arr.enter_vec(female_fecundity[f_idx], i, j)
                male_fecundity_arr.enter_vec(male_fecundity[m_idx], i, j)

                fecundity_hist[i, 0, j, :] = np.histogram(
                    female_fecundity[f_idx], bins=fecundity_bins)[0]
                fecundity_hist[i, 1, j, :] = np.histogram(
                    male_fecundity[m_idx], bins=fecundity_bins)[0]
                pairing_hist[i, 0, j, :] = np.histogram(
                    female_pairings[f_idx], bins=pairing_bins)[0]
                pairing_hist[i, 1, j, :] = np.histogram(
                    male_pairings[m_idx], bins=pairing_bins)[0]
                n_arr[i, 0, j] = len(f_idx)
                n_arr[i, 1, j] = len(m_idx)
        return cls(n_arr, pairing_hist, fecundity_hist, female_pairing_arr,
                 male_pairing_arr, female_fecundity_arr, male_fecundity_arr,
                 pedigree.params, bin_size)

    def get_statistics(self):
        self.female_pairing_arr.get_statistics()
        self.male_pairing_arr.get_statistics()
        self.female_fecundity_arr.get_statistics()
        self.male_fecundity_arr.get_statistics()

        self.pairing_statistics()
        self.fecundity_statistics()

        self.plot_pairing_statistics()
        self.plot_pairing_heatmap()
        #self.plot_fecundity_statistics()
        #self.fecundity_heatmap()

    def pairing_statistics(self):
        nonzeros = np.nonzero(self.pairing_hist)
        self.norm_pairings = np.copy(self.pairing_hist)
        self.norm_pairings[nonzeros] = self.norm_pairings[nonzeros] / \
                                  self.n_arr[nonzeros[:3]]
        pairing_bins = np.arange(self.n_pairing_bins)
        self.mean_pairings = np.sum(self.norm_pairings * pairing_bins, axis=3)

    def fecundity_statistics(self):
        nonzeros = np.nonzero(self.fecundity_hist)
        self.norm_fecundity = np.copy(self.fecundity_hist)
        self.norm_fecundity[nonzeros] = self.norm_fecundity[nonzeros] / \
                                  self.n_arr[nonzeros[:3]]
        fecundity_bins = np.arange(self.n_fecundity_bins)
        self.mean_fecundity = np.sum(self.norm_fecundity * fecundity_bins,
                                     axis=3)

    def plot_pairing_statistics(self):
        fig, axs = plt.subplots(3, 3, figsize=(13, 10))
        for i in np.arange(Constants.n_genotypes):
            ax = axs[np.unravel_index(i, (3, 3))]
            color = colors.to_rgb(Constants.genotype_colors[i])
            femcolor = [c * 0.5 for c in color]
            ax.errorbar(self.bin_mids, self.female_pairing_arr.means[i, :],
                        yerr=self.female_pairing_arr.stds[i, :],
                        color=femcolor, capsize=2,
                        label="female " + Constants.subpop_legend[i])
            ax.errorbar(self.bin_mids+0.002,
                        self.male_pairing_arr.means[i, :],
                        yerr=self.male_pairing_arr.stds[i, :],
                        color=color, capsize=2,
                        label="male " + Constants.subpop_legend[i])
            ax = util.setup_space_plot(ax, 2.5, "n pairings",
                                       Constants.subpop_legend[i])
            ax.legend(fontsize="8")
        fig.suptitle("Mating Success")
        fig.tight_layout(pad=1.0)
        fig.show()

    def plot_pairing_heatmap(self):
        fig0, axs0 = plt.subplots(3, 3, figsize=(12, 9))
        pairing_bins = np.arange(self.n_pairing_bins + 1) - 0.5
        bins = np.arange(0, 1 + self.bin_size, self.bin_size)
        X, Y = np.meshgrid(bins, pairing_bins)
        subpop_n = np.sum(self.n_arr, axis=2)
        subpop_pairing0 = np.sum(self.female_pairing_arr.sums, axis=1)
        for i in np.arange(Constants.n_genotypes):
            ax = axs0[np.unravel_index(i, (3, 3))]
            Z = self.norm_pairings[i, 0, :, :]
            Z = np.rot90(np.fliplr(Z))
            p0 = ax.pcolormesh(X, Y, Z, vmin=0, vmax=1, cmap='plasma')
            ax.set_ylim(-0.5, 4.5)
            failures = np.sum(self.pairing_hist[i, 0, :, 0])
            successrate = np.round(1 - failures / subpop_n[i, 0], 2)
            ax.set_title(f"{Constants.subpop_legend[i]} "
                         f"ratio {np.round(subpop_pairing0[i] / subpop_n[i, 0], 2)}"
                         f" successrate {successrate}")
            fig0.colorbar(p0)
        fig0.suptitle("Female number of pairings")
        fig0.tight_layout(pad=1.0)
        fig0.show()

        fig1, axs1 = plt.subplots(3, 3, figsize=(12, 9))
        subpop_pairing1 = np.sum(self.male_pairing_arr.sums, axis=1)
        for i in np.arange(Constants.n_genotypes):
            ax = axs1[np.unravel_index(i, (3, 3))]
            Z = self.norm_pairings[i, 1, :, :]
            Z = np.rot90(np.fliplr(Z))
            p0 = ax.pcolormesh(X, Y, Z, vmin=0, vmax=1, cmap='plasma')
            ax.set_ylim(-0.5, 4.5)
            failures = np.sum(self.pairing_hist[i, 1, :, 0])
            successrate = np.round(1 - failures / subpop_n[i, 0], 2)
            ax.set_title(f"{Constants.subpop_legend[i]} "
                         f"ratio {np.round(subpop_pairing1[i] / subpop_n[i, 1], 2)}"
                         f" successrate {successrate}")
            fig1.colorbar(p0)
        fig1.suptitle("Male number of pairings")
        fig1.tight_layout(pad=1.0)
        fig1.show()

    def plot_fecundity_statistics(self):
        fig, axs = plt.subplots(3, 3, figsize=(13, 10))
        for i in np.arange(Constants.n_genotypes):
            ax = axs[np.unravel_index(i, (3, 3))]
            color = colors.to_rgb(Constants.genotype_colors[i])
            femcolor = [c * 0.5 for c in color]
            ax.errorbar(self.bin_mids, self.female_fecundity_arr.means[i, :],
                        yerr=self.female_fecundity_arr.stds[i, :],
                        color=femcolor, capsize=2,
                        label="female " + Constants.subpop_legend[i])
            ax.errorbar(self.bin_mids+0.002,
                        self.male_fecundity_arr.means[i, :],
                        yerr=self.male_fecundity_arr.stds[i, :],
                        color=color, capsize=2,
                        label="male " + Constants.subpop_legend[i])
            ax = util.setup_space_plot(ax, 6, "n offspring",
                                       Constants.subpop_legend[i])
            ax.legend(fontsize="8")
        fig.suptitle("Fecundity")
        fig.tight_layout(pad=1.0)
        fig.show()

    def fecundity_heatmap(self):
        fig0, axs0 = plt.subplots(3, 3, figsize=(12, 9))
        fecundity_bins = np.arange(self.n_fecundity_bins + 1) - 0.5
        bins = np.arange(0, 1 + self.bin_size, self.bin_size)
        X, Y = np.meshgrid(bins, fecundity_bins)
        subpop_n = np.sum(self.n_arr, axis=2)
        subpop_fecs = np.sum(self.female_fecundity_arr.sums, axis=1)
        for i in np.arange(Constants.n_genotypes):
            ax = axs0[np.unravel_index(i, (3, 3))]
            Z = self.norm_fecundity[i, 0, :, :]
            Z = np.rot90(np.fliplr(Z))
            p0 = ax.pcolormesh(X, Y, Z, vmin=0, vmax=1, cmap='plasma')
            ax.set_ylim(-0.5, 7.5)
            ax.set_title(f"{Constants.subpop_legend[i]} "
                         f"ratio {np.round(subpop_fecs[i] / subpop_n[i, 0], 2)}")
            fig0.colorbar(p0)
        fig0.suptitle("Female fecundities")
        fig0.show()

#col = OverCollection("fitness2")


#example = GenotypeArr.load_txt("c:/hybzones/data/fitness/group4/fitness_group4_arr_13756400_4.txt")
#group1 = GenotypeArrCollection.load_directory(r"fitness/group1")
#group2 = SummaryCollection(r"fitness1\\group2")
#group3 = GenotypeArrCollection.load_directory(r"fitness/group3")
#group4 = GenotypeArrCollection.load_directory(r"fitness/group4")
#group5 = GenotypeArrCollection.load_directory(r"fitness/group5")
#group6 = GenotypeArrCollection.load_directory(r"fitness/group6")
#group7 = GenotypeArrCollection.load_directory(r"fitness/group7")
#group8 = GenotypeArrCollection.load_directory(r"fitness/group8")
#group9 = GenotypeArrCollection.load_directory(r"fitness/group9")


#why do trials with only 1 extant group have std with magnitude?

