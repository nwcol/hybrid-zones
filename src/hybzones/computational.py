import numpy as np

import matplotlib.pyplot as plt

import time

import scipy.optimize as opt

from hybzones import parameters

from hybzones.parameters import Params

"""
Assumptions.

We partition the population into males and females, but we assume that 
everywhere the sex ratio is exactly 1:1

Things to add to increase the sophisitication of the model
-assortation
-think about whether your model of assortation is even correct
-multi-bin dispersal
-across-bin mating
-dispersal fxns (nonrandom)
-migration
"""

space_plotsize = (10, 6)  # or (8, 6)


class Model:

    def __init__(self, params, plot_int):
        self.params = params
        self.G = params.g
        self.g = params.g
        self.x = np.arange(0.005, 1.005, 0.01)
        self.edges = np.arange(0, 1.01, 0.01)
        self.compute_m(params)
        self.compute_b(params)
        self.get_matrix()
        self.cube = get_master_cube()
        self.large_c_matrix = get_large_c_matrix(params)
        self.run(plot_int)

    def compute_m(self, params):
        """compute bin migration rates.

        Because I don't want to try to integrate normal distributions and etc,
        m is currently estimated rather than explicitly computed
        """
        n = 100_000
        points = np.random.uniform(low=0, high=0.01, size=n)
        d = np.random.normal(0, scale=params.delta, size=n)
        points += d

        self.m = len(points[points > 0.01]) / n
        self.m_3 = len(points[points > 0.03]) / n
        self.m_2 = len(points[points > 0.02]) / n - self.m_3
        self.m_1 = len(points[points > 0.01]) / n - self.m_3 - self.m_2

    def compute_b(self, params):
        """compute b, the fraction of matings which occur between adjacent bins
        """
        n = 100_000
        points = np.random.uniform(low=0, high=0.01, size=n)
        d = np.random.normal(0, scale=params.beta, size=n)
        points += d
        rights = points[points > 0.01]
        self.b = len(rights) / n

    def get_matrix(self):
        """create the matrix of subpopulations. axis 0 is x, axis 1 is pop,
        axis 2 is sex (0 females, 1 males).
        """
        self.matrix = np.zeros((len(self.x), 9, 2), dtype=np.float32)
        edges1 = (np.array(self.params.spawnlimit_A1B1) * 100).astype(np.int32)
        edges2 = (np.array(self.params.spawnlimit_A2B2) * 100).astype(np.int32)
        self.matrix[:edges1[1], 0, :] = 1
        self.matrix[edges2[0]:, 8, :] = 1
        self.standardize()

    def run(self, plot_int=1):

        time1 = time0 = time.time()
        report_int = self.g / 10
        self.history = np.zeros((self.g + 1, 100, 9, 2))

        while self.g >= 0:
            self.mate()
            self.diffuse()
            self.migration()
            self.fitness()
            if self.g % plot_int == 0:
                self.plot()
            if self.g % report_int == 0:
                time1 = self.report(time0, time1, report_int)
            self.history[self.g, :, :, :] = self.matrix
            self.g -= 1

        self.compute_allele_matrix()
        self.allele_plot(0)
        self.plot_history()
        self.cline_analysis()
        self.plot_clinepars()

    def report(self, time0, time1, report_int):
        g = str(self.g)
        time2 = time.time()
        gen_time = (time2 - time1) / report_int
        time1 = time.time()
        dur = time1 - time0
        timenow = time.strftime("%H:%M:%S", time.localtime())
        print(
            "g " + g + " complete, runtime = " + str(np.round(dur, 2)) + " s"
            + ", averaging " + str(np.round(gen_time, 4)) + "s/gen, @ " +
            str(timenow)
        )
        return (time1)

    def reset(self):
        self.get_matrix()
        self.g = self.params.g

    def mate111(self):

        # in_bin = self.matrix * (1 - 2 * self.b)
        # out_bin = self.matrix * self.b

        mass = np.zeros((100, 9, 9))
        for i in np.arange(100):
            mass[i, :, :] = np.outer(self.matrix[i, :, 0],
                                     self.matrix[i, :, 1])
            # axis 1 females axis 2 males
            # preference and reweighting

            mass[i, :, :] *= self.large_c_matrix
            # females = np.repeat(self.matrix[i, :, 0, None], 9, axis = 1)
            # males = np.repeat(self.matrix[i, None, :, 1], 9, axis = 0)
            # mass[i, :, :] = self.large_c_matrix * females * males

        self.standardize_mass(mass)

        resultant = np.zeros((100, 9, 9, 9))
        resultant[:, :, :, :] = mass[:, :, :, None]
        resultant *= self.cube
        res = np.sum(np.sum(resultant, axis=1), axis=1)
        self.matrix[:, :, 0] = self.matrix[:, :, 1] = res[:, :]
        # self.standardize()

        """
        mass = np.zeros((100, 9, 9))
        for i in np.arange(100):
            mass[i, :, :] = self.large_c_matrix
            mass[i, :, :] *= self.matrix[i, :, 1]
            mass[i, :, :] *= self.matrix[i, :, 0, None]

        self.standardize_mass(mass)

        resultant = np.zeros((100, 9, 9, 9))
        resultant[:, :, :, :] = mass[:, :, :, None]
        resultant *= self.cube
        res = np.sum(np.sum(resultant, axis = 1), axis = 1)
        self.matrix[:, :, 0] = self.matrix[:, :, 1] = res[:, :]
        """

    def mate(self):

        mothers = self.matrix[:, :, 0]
        fathers = self.matrix[:, :, 1]
        mass = np.zeros((100, 9, 9))
        mass[:, :, :] = fathers[:, None, :]
        mass[:, :, :, ] *= self.large_c_matrix
        S = np.sum(mass, axis=2)
        S_idx = np.where(S != 0)
        mass[S_idx[0], S_idx[1], :] /= S[S_idx[0], None, S_idx[1]]
        mass *= mothers[:, :, None]
        resultant = np.zeros((100, 9, 9, 9))
        resultant[:, :, :, :] = mass[:, :, :, None]
        resultant *= self.cube
        res = np.sum(np.sum(resultant, axis=1), axis=1)
        self.matrix[:, :, 0] = self.matrix[:, :, 1] = res[:, :]
        self.standardize()

    def standardize_mass(self, mass):
        sums = np.sum(np.sum(mass, axis=1), axis=1)
        mass /= sums[:, None, None]
        return (mass)

    def standardize(self):
        sums0 = np.sum(self.matrix[:, :, 0], axis=1)
        self.matrix[:, :, 0] /= sums0[:, None]
        sums1 = np.sum(self.matrix[:, :, 1], axis=1)
        self.matrix[:, :, 1] /= sums1[:, None]

    def diffuse(self):

        nonmigrants = self.matrix * (1 - 2 * self.m)
        migrants = self.matrix * self.m
        migrants_1 = self.matrix * self.m_1
        migrants_2 = self.matrix * self.m_2
        migrants_3 = self.matrix * self.m_3
        nonmigrants[1:] += migrants[:-1]
        nonmigrants[:-1] += migrants[1:]
        nonmigrants[0] += migrants[0]
        nonmigrants[99] += migrants[99]
        self.matrix = nonmigrants

    def migration(self):

        if self.params.edge_fxn == "flux":
            self.matrix[0, :, :] *= 1 - self.m
            self.matrix[99, :, :] *= 1 - self.m
            self.matrix[0, 0, :] += self.m
            self.matrix[99, 8, :] += self.m

    def fitness(self):
        """exert the effects of extrinsic (environmental) fitness on a generation.

        This function uses logarithmic functions to compute the additive reductions
        in fitness for each allele, and sums them to get fitnesses for each
        organism. Organisms which are killed by fitness effects are flagged with
        flag = -1, preventing them from mating.

        Arguments
        ------------
        gen : np array

        params : Params instance

        Returns
        ------------
        gen : np array
        """
        if self.params.extrinsic_fitness == True:
            s_1 = self.s_1()
            s_2 = self.s_2()
            s_1 = np.repeat(s_1[:, None], 3, axis=1)[:, :, None]
            s_2 = np.repeat(s_2[:, None], 3, axis=1)[:, :, None]
            if self.params.female_fitness == True:
                idx = [0, 1]
            else:
                idx = [1]
            self.matrix[:, 0:3, idx] *= (1 - 2 * s_1)
            self.matrix[:, 3:6, idx] *= (1 - s_1 - s_2)
            self.matrix[:, 6:9, idx] *= (1 - 2 * s_2)
            self.standardize()

    def s_1(self):
        params = self.params
        x = self.x
        s1 = params.mu - params.mu / (
                    1 + np.exp(-params.k_1 * (x - params.mid_1)))
        return (s1)

    def s_2(self):
        params = self.params
        x = self.x
        s2 = params.mu - params.mu / (
                    1 + np.exp(-params.k_2 * (x - params.mid_2)))
        return (s2)

    def plot(self):
        matrix = np.mean(self.matrix, axis=2)
        fig = plt.figure(figsize=space_plotsize)
        sub = fig.add_subplot(111)
        sub.plot(self.x, np.sum(matrix[:, 1:8], axis=1), color='green',
                 linestyle='dashed'
                 )
        colors = pars.subpop_colors
        for i in np.arange(9):
            sub.plot(self.x, matrix[:, i], color=colors[i], linewidth=2)
        plt.xticks(np.arange(0, 1.1, 0.1)), plt.yticks(np.arange(0, 1.1, 0.1))
        plt.xlabel("x coordinate")
        sub.set_ylim(-0.01, 1.01), sub.set_xlim(-0.01, 1.01)
        plt.title(str(self.g))
        plt.legend(["Hyb"] + pars.subpop_legend, fontsize=8,
                   bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)

    def compute_allele_matrix(self):
        history = np.mean(self.history, axis=3)
        shape = np.shape(history)
        self.allele_matrix = np.zeros((shape[0], shape[1], 2, 2))
        factor = np.array([
            [[1, 0], [1, 0]],
            [[1, 0], [0.5, 0.5]],
            [[1, 0], [0, 1]],
            [[0.5, 0.5], [1, 0]],
            [[0.5, 0.5], [0.5, 0.5]],
            [[0.5, 0.5], [0, 1]],
            [[0, 1], [1, 0]],
            [[0, 1], [0.5, 0.5]],
            [[0, 1], [0, 1]]
        ])
        self.allele_matrix = np.sum(
            history[:, :, :, None, None] * factor, axis=2
        )

    def allele_plot(self, g):
        fig = plt.figure(figsize=space_plotsize)
        sub = fig.add_subplot(111)
        colors = pars.allele_colors
        sub.plot(
            self.x, self.allele_matrix[g, :, 1, 0], color=colors[2],
            linewidth=2, label="$B^1$"
        )
        sub.plot(
            self.x, self.allele_matrix[g, :, 1, 1], color=colors[3],
            linewidth=2, label="$B^2$"
        )
        sub.plot(
            self.x, self.allele_matrix[g, :, 0, 0], color=colors[0],
            linewidth=2, label="$A^1$"
        )
        sub.plot(self.x, self.allele_matrix[g, :, 0, 1], color=colors[1],
                 linewidth=2, label="$A^2$"
                 )
        plt.xlabel("x coordinate")
        plt.ylabel("allele frequency")
        sub.set_ylim(-0.01, 1.01), sub.set_xlim(-0.01, 1.01)
        plt.xticks(np.arange(0, 1.1, 0.1)), plt.yticks(np.arange(0, 1.1, 0.1))
        plt.title(str(g))
        handles, labels = plt.gca().get_legend_handles_labels()
        order = [2, 3, 0, 1]
        plt.legend(
            [handles[idx] for idx in order], [labels[idx] for idx in order],
            bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0
        )

    def plot_history(self, log=True):
        history = np.mean(self.history, axis=3)
        length = np.shape(history)[0]
        g_range = np.arange(length)
        fig = plt.figure(figsize=(8, 6))
        sub = fig.add_subplot(111)
        colors = pars.subpop_colors
        sub.plot(g_range, np.full(length, 1), color="black", linewidth=2)
        for i in np.arange(9):
            sub.plot(g_range, np.sum(history[:, :, i], axis=1) / 100, color=
            colors[i], linewidth=2)
        sub.set_xlim(0, length)
        sub.invert_xaxis()
        if log == True:
            sub.set_yscale("log")
        else:
            sub.set_ylim(-0.01, 1.01)
        plt.xlabel("generation before present")
        plt.ylabel("population size")
        plt.legend(["N"] + pars.subpop_legend, fontsize=8)

    def cline_analysis(self):
        cline = self.allele_matrix[:, :, 0, 1]
        pars = []
        for g in np.arange(self.G + 1):
            try:
                popt, pcov = opt.curve_fit(logistic_fxn, self.x, cline[g, :])
                pars.append(popt)
            except:
                pars.append(np.array([-1, -1]))
        self.pars = np.vstack(pars)

    def plot_clinepars(self):
        xrange = np.arange(self.G + 1)
        fig, axs = plt.subplots(2, 1, figsize=(6, 6))
        x_ax, k_ax = axs[0], axs[1]
        x_ax.plot(xrange, self.pars[:, 1], color="black")
        k_ax.plot(xrange, self.pars[:, 0], color="red")
        kmax = np.round(np.max(self.pars[:-5, 0]) * 1.2, decimals=-1)
        kmax = 100
        k_ax.set_ylim(0, kmax)
        k_ax.set_xlim(0, self.G), x_ax.set_xlim(0, self.G)
        k_ax.invert_xaxis(), x_ax.invert_xaxis()
        x_ax.set_ylim(0, 1)
        k_ax.set_xlabel("generations bp")
        x_ax.set_ylabel("x_0"), k_ax.set_ylabel("k")


def logistic_fxn(x, k, x0):
    return 1 / (1.0 + np.exp(-k * (x - x0)))


def get_large_c_matrix(params):
    c_matrix = np.swapaxes(params.c_matrix, 0, 1)
    large_c_matrix = np.zeros((9, 9))
    for i in np.arange(3):
        large_c_matrix[[i, i + 3, i + 6], :] = np.repeat(c_matrix[i], 3)
    return (large_c_matrix)


def get_possible_gametes(genotype):
    gametes = np.zeros((4, 2))
    gametes[0, :] = [genotype[0], genotype[2]]
    gametes[1, :] = [genotype[1], genotype[2]]
    gametes[2, :] = [genotype[0], genotype[3]]
    gametes[3, :] = [genotype[1], genotype[3]]
    return (gametes)


def get_all_gametes(genotypes):
    gametes = np.zeros((9, 4, 2))
    for i in np.arange(9):
        gametes[i, :, :] = get_possible_gametes(genotypes[i, :])
    return (gametes)


def get_master_cube():
    genotypes = np.array(
        [[1, 1, 1, 1], [1, 1, 1, 2], [1, 1, 2, 2], [1, 2, 1, 1], [1, 2, 1, 2],
         [1, 2, 2, 2], [2, 2, 1, 1], [2, 2, 1, 2], [2, 2, 2, 2]]
    )
    gametes = get_all_gametes(genotypes)
    allele_sums = np.array(
        [[2, 2], [2, 3], [2, 4], [3, 2], [3, 3], [3, 4], [4, 2], [4, 3],
         [4, 4]]
    )
    master_cube = np.zeros((9, 9, 9))
    for i in np.arange(9):
        for j in np.arange(9):
            gametes0 = gametes[i, :, :]
            gametes1 = gametes[j, :, :]
            zygotes = np.zeros((4, 4, 4))
            # ax0 female ax1 male ax2 content
            zygotes[:, :, 0] = gametes0[:, 0]
            zygotes[:, :, 2] = gametes0[:, 1]
            zygotes[:, :, 1] = gametes1[:, 0, None]
            zygotes[:, :, 3] = gametes1[:, 1, None]
            sums = np.zeros((4, 4, 2))
            sums[:, :, 0] = zygotes[:, :, 0] + zygotes[:, :, 1]
            sums[:, :, 1] = zygotes[:, :, 2] + zygotes[:, :, 3]
            index = np.zeros((4, 4))
            for k in np.arange(4):
                for z in np.arange(4):
                    index[k, z] = np.where(
                        (allele_sums == sums[k, z]).all(axis=1)
                    )[0]
            idx = np.unique(index).astype(np.int32)
            n = len(idx)
            counts = np.zeros(n)
            for k in np.arange(n):
                counts[k] = np.sum(index == idx[k])
            counts /= 16
            master_cube[i, j, idx] = counts
    mega_cube = np.zeros((100, 9, 9, 9))
    mega_cube[:] = master_cube
    return (mega_cube)






