"""
The Parameter class
"""

import json

import numpy as np

import scipy

import matplotlib.pyplot as plt

plt.rcParams['figure.dpi'] = 400

class Params:
    """Determines the initial state, constant parameters and runtime of a
    simulation.

    Population Model Attributes
    ---------------------------
    N_A1B1 : int
        the  number of individuals with traits A = 1, B = 1, eg genotype
        A1 A1 B1 B1 in the founding generation

    N_A1B2 : int
        init. number of individuals with traits A = 1, B = 2 eg A1 A1 B2 B2

    N_A2B1 : int
        init. number of individuals with traits A = 2, B = 1 eg A2 A2 B1 B1

    N_A2B2 : int
        init. number of individuals with traits A = 2, B = 2 eg A2 A2 B2 B2

    K : int
        the total carrying capacity of the space

    r : float
        the logistic growth rate parameter for the population

    g : int
        the number of generations

    pref_model : str
        the model of signal-dependent female mating preferences. options:
            "full_null"
            "assortative_null"
            "intermediate"
            "undesirable"
            "distinct"
            "aarons"

    c_matrix : np array
        a 3x3 matrix containing the coefficients of assortation

    beta : float
        a parameter representing mating and interaction range. female mate
        selection is a normally distributed sample with stdev beta

    bound : float
        the maximum interaction distance. by default, bound = 4 * beta

    mating_fxn : str
        the model of mating to be used. default "gaussian"
            "uniform_mating"
            "gaussian_mating"

    delta : float
        the st dev of individual dispersal, given that dispersal is random.

    dispersal_fxn : str
        the model of dispersal: "scale" "shift" "random"

    edge_fxn : str
        the model used to handle interactions at the edges of space, eg what
        occurs if an individuals' dispersal removes it from the bound [0, 1]
            "closed" "flux"

    d_scale : float
        the scale applied to the dispersal effects in the scale and shift
        dispersal models

    spawnlimit_A1B1 : tuple
        the initial spatial bounds in which A = 1 B = 1 individuals may appear

    spawnlimit_A1B2 : tuple
        the initial spatial bounds in which A = 1 B = 2 individuals may appear

    spawnlimit_A2B1 : tuple
        the initial spatial bounds in which A = 2 B = 1 individuals may appear

    spawnlimit_A2B2 : tuple
        the initial spatial bounds in which A = 2 B = 2 individuals may appear

    density_bound : float
        the range within which individuals influence each other ecologically

    intrinsic_fitness : bool
        if True, A = H individuals eg heterozygotes for the signal allele are
        affected by reduced relative fitness

    H_fitness : float
        the relative fitness of individuals with A = H signal trait

    extrinsic_fitness : bool
        if True, extrinsic, eg environmental fitness, is modelled. if False,
        the fitness parameters are not used and there is no env. fitness

    female_fitness : bool
        if True, females are affected by extrinsic, eg environmental, fitness

    k_1 : float
        the slope of the logistic fitness function for the A1 allele

    k_2 : float
        the slope of the logistic fitness function for the A2 allele

    mu : float
        the maximum reduction in fitness/allele.

    mid_1 : float
        the inflection point/midpoint of the fitness curve for the A1 allele

    mid_2 : float
        the midpoint of the fitness curve for the A2 allele


    Script Instruction Attributes
    ----------------------------
    Instructions for running on the CHTC server. These parameters are applied
    through the "diploid_model_script".

    task : str
        instructs the script what function (process) to execute. Options:

        "return_pedigree" : return a pedigree array file

        "return_pop_matrix" : return a pop_matrix array file

        "return_tree_structre" : execute a coalescence simulation and return
            the tree structure

        "diversity_summary" : execute a coalescence simulation on a pedigree
            and return diversity data


    save_pedigree : bool
        if True, save the complete simulation pedigree, which is a very very
        large array- order of tens of gigabytes

    coalesc_x_range : list of lists

    coalesc_n_vec : list of ints

    """

    def __init__(self, N, g, c):
        """
        Arguments
        ------------
        N : int
            The number of individuals in the founding generation
                note that N = 0 is permitted and is a special case, where migra
                tion seeds the founding generation at the edges

        g : int
            The number of generations to simulate

        c : float
            the level of trait-based assortation for the simulation
        """

        # population parameters
        self.N_A1B1 = N // 2
        self.N_A1B2 = 0
        self.N_A2B1 = 0
        self.N_A2B2 = N // 2

        self.K = N
        self.r = 0.5
        self.g = g

        # mating parameters
        self.pref_model = "undesirable"
        self.set_c(c, self.pref_model)
        self.beta = 0.005
        self.bound = self.beta * 4
        self.mating_model = "gaussian"

        # dispersal parameters
        self.delta = 0.01
        self.dispersal_model = "random"
        self.edge_model = "closed"
        self.d_scale = 2

        # space parameters
        self.spawnlimit_A1B1 = (0, 0.5)
        self.spawnlimit_A1B2 = (0, 0.5)
        self.spawnlimit_A2B1 = (0.5, 1)
        self.spawnlimit_A2B2 = (0.5, 1)

        self.density_bound = 0.005

        # fitness parameters
        self.intrinsic_fitness = False
        self.H_fitness = 1
        self.extrinsic_fitness = False
        self.female_fitness = False
        self.k_1 = -20
        self.k_2 = -self.k_1
        self.mu = 0.2
        self.mid_1 = 0.5
        self.mid_2 = 0.5

        # general
        self.history_type = "Pedigree"
        self.task = "return_pop_matrix"
        self.bug_check = None

        # coalescence and genetic parameters
        self.sample_n = 100
        self.n_sample_bins = 10
        self.n_windows = 10

        self.demographic_model = "threepop"
        self.mig_rate = 1e-4
        self.seq_length = 1e4
        self.recombination_rate = 1e-8
        self.u = 1e-8

    def set_c(self, c, pref_model):
        """Declare the parameter set's c-parameter and the accompanying
        heterozygote preference model, which is represented by a 3x3 matrix
        of assortation parameters

        Arguments
        ------------
        c : float
            the assortative parameter between organisms of opposite signal
            and preference

        pref_model : str
            the model which determines the assortative parameters between
            heterozygote organisms
        """
        self.pref_model = pref_model
        self.c = c
        c_matrix = np.full((3, 3), 1.0, dtype=np.float32)
        if pref_model == "full_null":
            pass
        elif pref_model == "assortative_null":
            c_matrix[0, 2] = c_matrix[2, 0] = c
        elif pref_model == "intermediate":
            c_matrix[0, 2] = c_matrix[2, 0] = c
            c_matrix[1, 0] = c_matrix[1, 2] = (1 + c) / 2
            c_matrix[0, 1] = c_matrix[2, 1] = (1 + c) / 2
        elif pref_model == "undesirable":
            c_matrix[1:, 0] = c
            c_matrix[1, 1] = c
            c_matrix[:2, 2] = c
        elif pref_model == "distinct":
            c_matrix[1:, 0] = c
            c_matrix[0, 1] = c_matrix[2, 1] = c
            c_matrix[:2, 2] = c
        elif pref_model == "semiundesirable":
            c_matrix[0, 2] = c
            c_matrix[2, 0] = c
            c_matrix[1, :] = (1 + c) / 2
        self.c_matrix = c_matrix
        self.print_c()

    def print_c(self):
        """Print out a table of c parameters (assortation parameters) for the
        parameter set
        """
        printout = np.zeros((4, 4), dtype="U16")
        printout[0, :] = ["            ", "pref B = 1", "pref B = H",
                          "pref B = 2"]
        printout[1:, 0] = ["signal A = 1", "signal A = H", "signal A = 2"]
        l = len(printout[0, 1])
        s = " "
        strings = [s * (l - len(str(i))) + str(i) for i in
                   np.ravel(self.c_matrix)]
        printout[1:, 1:] = np.reshape(strings, (3, 3))
        print(printout)

    def plot_fitness(self):
        """Make a plot of allelic fitness reductions for A1, A2 and the
        additive fitnesses of the possible signal genotypes A1A1, A1A2, A2A2,
        given a Params class isntance.

        Parameters
        ------------
        mu : float
            the maximum reduction in fitness/allele. symmetric for A1, A2

        k_1, k_2 : float
            the slope of the logistic fitness reduction curves for A1, A2.
            k_1 should be negative, k_2 positive. default -20, 20

        mid_1, mid_2 : float
            the inflection points of the logistic fitness reduction curves.
            default 0.5, 0.5
        """
        x = np.linspace(0, 1, 1000)
        s1 = self.mu - self.mu / (1 + np.exp(-self.k_1 * (x - self.mid_1)))
        s2 = self.mu - self.mu / (1 + np.exp(-self.k_2 * (x - self.mid_2)))
        fig = plt.figure(figsize=(8, 6))
        sub = fig.add_subplot(111)
        sub.plot(x, s1, color="red", linestyle="dashed", label="s_1")
        sub.plot(x, s2, color="blue", linestyle="dashed", label="s_2")
        sub.plot(x, 1 - 2 * s1, color="red", label="1 - 2s_1")
        sub.plot(x, 1 - s1 - s2, color="purple", label="1 - s_1 - s_2")
        sub.plot(x, 1 - 2 * s2, color="blue", label="1 - 2s_2")
        sub.set_ylim(-0.01, 1.01), sub.set_xlim(-0.01, 1.01)
        sub.legend(loc="lower left")

    def plot_pref_model(self):
        fig = plt.figure(figsize=(8, 6))
        sub = fig.add_subplot(111)
        x = [0, 0.5, 1]
        sub.plot(x, self.c_matrix[:, 0], color="red")
        sub.plot(x, self.c_matrix[:, 1], color="green")
        sub.plot(x, self.c_matrix[:, 2], color="blue")
        a = sub.get_xticks().tolist()
        a[0] = "B = 11"
        a[1] = "B = 12 = 21"
        a[2] = "B = 22"
        sub.set_xticklabels(a)

    def compute_r(self):
        c_matrix = np.copy(self.c_matrix)
        normed = c_matrix / np.sum(c_matrix, axis=0)
        mating_n = np.ravel((normed * 10_000).astype(np.int32))
        t = np.array([[0, 0], [0.5, 0], [1, 0], [0.5, 0], [0.5, 0.5], [0.5, 1],
                      [1, 0], [1, 0.5], [1, 1]])
        matings = np.zeros((30_000, 2))
        i_0 = i_1 = 0
        for i in np.arange(9):
            i_1 += mating_n[i]
            matings[i_0:i_1, :] = t[i]
            i_0 += mating_n[i]
        r = scipy.stats.pearsonr(matings[:, 0], matings[:, 1])[0]
        return (r)

    def plot_scale(self, center=0.5):
        """Make a plot of normal distributions with scales self.beta,
        self.delta to allow visualization of the scale of interactions in the
        space.
        """
        fig = plt.figure(figsize=(8, 6))
        sub = fig.add_subplot(111)
        x = np.linspace(0, 1, 1000)
        beta = compute_pd(x - center, self.beta)
        delta = compute_pd(x - center, self.delta)
        beta /= np.max(beta)
        delta /= np.max(delta)
        sub.plot(x, beta, color='red')
        sub.plot(x, delta, color='orange')
        sub.set_ylim(-0.01, 1.01), sub.set_xlim(-0.01, 1.01)
        plt.xticks(np.arange(0, 1.1, 0.1))
        plt.yticks(np.arange(0, 1.1, 0.1))

    def print_genetic_params(self):
        genetic_params = ["sample_sizes", "sample_bins", "n_windows",
                          "demog_model", "mig_rate", "seq_length",
                          "recomb_rate", "u"]
        for x in genetic_params:
            print(f"{x : <20}" + str(getattr(self, x)))

    def save(self, filename):
        param_dict = vars(self)
        c_list = [float(x) for x in list(np.ravel(self.c_matrix))]
        param_dict['c_matrix'] = c_list
        filename = "param_files/" + filename
        file = open(filename, 'w')
        json.dump(param_dict, file, indent=0)
        self.c_matrix = np.reshape(self.c_matrix, (3, 3)).astype(np.float32)
        print("params file written")

    @classmethod
    def load(cls, filename):
        file = open(filename, 'r')
        param_dict = json.load(file)
        param_instance = cls(param_dict["K"], param_dict["g"], param_dict["c"])
        for param in param_dict:
            setattr(param_instance, param, param_dict[param])
        param_instance.c_matrix = np.reshape(param_instance.c_matrix, (3, 3))
        return (param_instance)

    @staticmethod
    def from_dict(param_dict):
        params = Params(0, 0, 0)
        for param in param_dict:
            setattr(params, param, param_dict[param])
        return (params)
