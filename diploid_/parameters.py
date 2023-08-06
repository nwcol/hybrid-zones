"""
The Parameter class
"""

import json

import numpy as np

import matplotlib.pyplot as plt

from diploid import math_fxns

from diploid import plot_util

plt.rcParams['figure.dpi'] = 400


class Params:
    """Determines the initial state, constant parameters and runtime of a
    simulation.

    Population Model Attributes
    ---------------------------
    N : int
        the total initial population size

    Subpop_n : list of ints, length 9
        the numbers of each subpopulation which which to populate the space
        A1A1B1B1 has index 0, A2A2B2B2 has index 1

    subpop_lims : list len 9 of lists,
            default [[0, 0.5,], [], ..., [], [0.5, 1]]
        gives the boundaries within which each subpopulation is allowed to
        spawn.

    K : int, defualt N
        the total carrying capacity of the space

    r : float, default 0.5
        the logistic growth rate parameter for the population

    g : int
        the number of generations

    c : float
        defines the level of assortative mating between the 'pure' subpops
        A and B; also parameterizes all other levels of assortation via the
        preference model

    pref_model : str
        the model of signal-dependent female mating preferences. options:
            "null" "intermediate" "undesirable" "semi_undesirable"
            "distinct"
        further documentation is forthcoming

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
        self.N = N
        self.subpop_n = [N//2, 0, 0, 0, 0, 0, 0, 0, N//2]
        self.subpop_lims = [[0, 0.5], [], [], [], [], [], [], [], [0.5, 1]]
        self.K = N
        self.r = 0.5
        self.g = g

        # mating parameters
        self.c = c
        self.pref_model = "undesirable"
        self.beta = 0.005
        self.bound = self.beta * 4
        self.mating_model = "gaussian"

        # dispersal parameters
        self.delta = 0.01
        self.dispersal_model = "random"
        self.edge_model = "closed"
        self.d_scale = 2
        self.density_bound = 0.005

        # fitness parameters
        self.intrinsic_fitness = False
        self.hyb_fitness = 1
        self.extrinsic_fitness = False
        self.female_fitness = False
        self.k_1 = -20
        self.k_2 = -self.k_1
        self.mu = 0.2
        self.mid_1 = 0.5
        self.mid_2 = 0.5

        # general
        self.history_type = "Pedigree"
        self.task = "get_pedigree_table"
        self.bug_check = None

        # coalescence and genetic parameters
        self.sample_n = 100
        self.n_sample_bins = 10
        self.lower_t_cutoff = None
        self.upper_t_cutoff = None
        self.n_windows = 10
        self.mig_rate = 1e-4
        self.seq_length = 1e4
        self.recombination_rate = 1e-8
        self.u = 1e-8
        self.demographic_model = "one_pop"
        self.multiwindow_type = "rooted"

    @classmethod
    def load(cls, filename):
        """Load a dictionary representation of a params class instance from a
        .json file, initialize a new instance and fill its fields with the
        values from the file
        """
        file = open(filename, 'r')
        param_dict = json.load(file)
        file.close()
        params = cls(param_dict["K"], param_dict["g"], param_dict["c"])
        for param in param_dict:
            setattr(params, param, param_dict[param])
        return params

    @classmethod
    def from_dict(cls, param_dict):
        params = cls(param_dict["K"], param_dict["g"], param_dict["c"])
        for field in param_dict:
            setattr(params, field, param_dict[field])
        return params

    @classmethod
    def from_string(cls, string):
        "Convert a string into a params instance"
        param_dict = eval(string[1:])
        params = cls.from_dict(param_dict)
        return params

    def __str__(self):
        return f"Parameters: K = {self.K}, N = {self.N}, g = {self.g}, \
                c = {self.c} with {self.pref_model} model."

    def __repr__(self):
        prototype = Params(10_000, 10, 0.1)
        proto_dict = dict(vars(prototype))
        self_dict = dict(vars(self))
        difs = []
        for parameter in self_dict:
            if self_dict[parameter] != proto_dict[parameter]:
                if parameter not in ["K", "g", 'c']:
                    difs.append(parameter)
        string = ""
        for dif in difs:
            string += f"params.{dif} = {self_dict[dif]} \n"
        return f"Params({self.K}, {self.g}, {self.c}) \n" + string

    def save(self, filename):
        """Write a dictionary representation of the params class instance in
        a .json file
        """
        param_dict = vars(self)
        filename = "param_files/" + filename
        file = open(filename, 'w')
        json.dump(param_dict, file, indent=0)
        file.close()
        print("params file written to " + filename)

    def parse(self):
        """Check to make sure that parameters stored as lists are the proper
        length and that values are not out of range"""
        if len(self.subpop_n) != 9:
            raise Exception("Invalid length: %d for subpop_n!" %
                            len(self.subpop_n))
        if len(self.subpop_lims) != 9:
            raise Exception("Invalid length: %d for subpop_lims!" %
                            len(self.subpop_lims))
        if self.pref_model not in c_matrix_methods:
            raise Exception(f"{self.pref_model} is not a valid pref_model!")

    def get_c_matrix(self):
        """Get the matrix of biases to mating probabilities given by the
        parameters "c" and "pref_model"
        """
        if self.pref_model in c_matrix_methods:
            c_matrix = c_matrix_methods[self.pref_model](self.c)
        else:
            name = self.pref_model
            raise Exception("Preference model is not implemented" % name)
        return c_matrix

    def print_c(self):
        """Print out a table of the assortative parameters defined by the
        instance
        """
        printout = np.zeros((4, 4), dtype="U16")
        printout[0, :] = ["            ", "pref B = 1", "pref B = H",
                          "pref B = 2"]
        printout[1:, 0] = ["signal A = 1", "signal A = H", "signal A = 2"]
        l = len(printout[0, 1])
        c_matrix = self.get_c_matrix()
        strings = [" " * (l - len(str(i))) + str(i) for i in
                   np.ravel(c_matrix)]
        printout[1:, 1:] = np.reshape(strings, (3, 3))
        print(printout)

    def plot_fitness(self):
        """Make a plot of allelic fitness reductions for A1, A2 and the
        additive fitness of the possible signal genotypes A1A1, A1A2, A2A2,
        given a Params class instance.
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
        sub = plot_util.setup_space_plot(sub, 1.01, "relative fitness",
                                         "environmental fitness")
        sub.legend(loc="lower left")

    def plot_scale(params, center=0.5):
        """Make a plot of normal distributions with scales self.beta and self.delta
        to give a visualization of the scale of interactions in the space.
        """
        fig = plt.figure(figsize=Const.plot_size)
        sub = fig.add_subplot(111)
        x = np.linspace(0, 1, 1000)
        beta = math_fxns.compute_pd(x - center, params.beta)
        delta = math_fxns.compute_pd(x - center, params.delta)
        beta /= np.max(beta)
        delta /= np.max(delta)
        sub.plot(x, beta, color='red')
        sub.plot(x, delta, color='orange')
        sub = plot_util.setup_space_plot(sub, 1.01, "normalized density",
                                         "scale")
        fig.show()

    ### outdated configuration stuff. update when you can!

    def estimate_memory(params, out_type):
        """
        Make an estimate of the maximum number of Gb of memory required to run
        a simulation
        """
        objects = []
        pop_arr_size = estimate_pop_arr_size(params)
        objects.append(pop_arr_size)

        if out_type == "pop_arr":
            pass

        elif out_type == "n_roots":
            ped_size = estimate_max_pedigree_size(params)
            tc_size = estimate_tc_size(0.8 * get_E_inds(params))
            ts_size = tc_size * 10  # a total guess
            objects.append(ped_size)
            objects.append(tc_size)
            objects.append(ts_size)

        Gb = np.sum(objects)
        return (Gb)

    def estimate_pop_arr_size(params):
        """Get an estimate of the number the megabytes of memory associated with
        a pop array and acessory data of given size. Each element takes 8 bytes
        of memory, and each individual is composed of 8 elements
        """
        itemsize = 8
        rows = 8
        inds = get_E_inds(params)
        size = itemsize * rows * inds
        Gb = size / 1_000_000_000
        return (Gb)

    def estimate_max_pedigree_size(params):
        """
        The maximum size a pedigree can take. Pedigrees will never be this large in
        practice though.
        """
        itemsize = 8
        rows = 8
        E_util = 0.8
        pedigree_inds = E_util * get_E_inds(params)
        size = itemsize * rows * pedigree_inds
        Gb = size / 1_000_000_000
        return (Gb)

    def estimate_tc_size(pedigree_inds):

        node_rows = 6
        node_element_size = 4  # bytes
        nodes_size = node_rows * node_element_size * pedigree_inds * 2
        ind_rows = 6
        ind_element_size = 4
        ind_size = ind_rows * ind_element_size * pedigree_inds * 2
        tc_size = nodes_size + ind_size
        tc_size /= 1_000_000_000
        return (tc_size)

    def get_E_inds(params):
        """
        Estimate the expected number of individuals in a simulation trial
        """
        E_inds = params.K * (params.g + 1)
        return (E_inds)


def get_null_c_matrix(c):
    c_matrix = np.array([[1, 1, 1],
                         [1, 1, 1],
                         [1, 1, 1]], dtype=np.float32)
    return c_matrix


def get_intermediate_c_matrix(c):
    h = (1 + c) / 2
    c_matrix = np.array([[1, h, c],
                         [h, 1, h],
                         [c, h, 1]],  dtype=np.float32)
    return c_matrix


def get_undesirable_c_matrix(c):
    c_matrix = np.array([[1, 1, c],
                         [c, c, c],
                         [c, 1, 1]],  dtype=np.float32)
    return c_matrix


def get_semi_undesirable_c_matrix(c):
    h = (1 + c) / 2
    c_matrix = np.array([[1, 1, c],
                         [h, h, h],
                         [c, 1, 1]], dtype=np.float32)
    return c_matrix


def get_distinct_c_matrix(c):
    c_matrix = np.array([[1, c, c],
                         [c, 1, c],
                         [c, c, 1]],  dtype=np.float32)
    return c_matrix


# map the string names of preference models to the functions which implement
# them
c_matrix_methods = {"null" : get_null_c_matrix,
                    "intermediate" : get_intermediate_c_matrix,
                    "undesirable" : get_undesirable_c_matrix,
                    "semi_undesirable": get_semi_undesirable_c_matrix,
                    "distinct" : get_distinct_c_matrix}
