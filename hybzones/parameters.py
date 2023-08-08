import json

import numpy as np

import matplotlib.pyplot as plt

from diploid import math_fxns

from diploid import plot_util

plt.rcParams['figure.dpi'] = 400


class Params:

    def __init__(self, N, g, c):

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

        # scripting
        self.task = "get_pedigree_table"

        # genetics parameters
        self.sample_sizes = [50, 50, 50, 50, 50, 50, 50, 50, 50, 50]
        self.sample_bins = [[0.0, 0.1], [0.1, 0.2], [0.2, 0.3], [0.3, 0.4],
                            [0.4, 0.5], [0.5, 0.6], [0.6, 0.7], [0.7, 0.8],
                            [0.8, 0.9], [0.9, 1.0]]
        self.time_cutoffs = (None, None)
        self.n_windows = 10
        self.mig_rate = 1e-4
        self.seq_length = 1e4
        self.recombination_rate = 1e-8
        self.u = 1e-8
        self.demographic_model = "one_pop"
        self.rooted = True

    @classmethod
    def load(cls, filename):
        """
        Load a dictionary representation of a params class instance from a
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
        """
        Convert a string into a params instance

        :param string:
        :return:
        """
        return cls.from_dict(eval(string))

    @classmethod
    def from_arr(cls, arr):
        """
        Convert a 1d array of characters into a string and then into a params
        instance

        :param arr:
        :return:
        """
        string = "".join(list(arr))
        return cls.from_string(string)

    @property
    def as_string(self):
        """
        Return a string representation of the params instance

        :return:
        """
        return str(vars(self))

    @property
    def as_arr(self):
        """
        Return an array of dtype U1 holding the parameter string as single
        characters

        :return:
        """
        return np.array(list(self.as_string), dtype="U1")

    @property
    def as_dict(self):
        return dict(vars(self))

    def __str__(self):
        _dict = self.as_dict
        _out = [f"{key :.<25} {str(_dict[key])}" for key in _dict]
        out = "\n".join(_out)
        return out

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
        return f"params = Params({self.K}, {self.g}, {self.c}) \n" + string

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
        """
        Make a plot of allelic fitness reductions for A1, A2 and the
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
        """
        Make a plot of normal distributions with scales self.beta and self.delta
        to give a visualization of the scale of interactions in the space.
        """
        fig = plt.figure(figsize=(8,6))
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
